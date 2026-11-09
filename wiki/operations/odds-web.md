---
type: reference
title: odds-web, the read-only web UI
description: What odds web serve runs, the database role and session it reads through, the report data behind the portfolio page, the status strip and the data-health page, its cache and failure bounds, the vendored scripts, and how deploy.sh and systemd run it.
sources:
  - odds_cli.commands.web
  - odds_web.app
  - odds_web.db
  - odds_web.cache
  - odds_web.portfolio
  - odds_web.status
  - odds_web.coverage
  - odds_web.closes
  - odds_web.role_check
  - odds_lambda.health_readers.source_freshness
  - odds_lambda.health_readers.last_heartbeats
  - odds_lambda.health_readers.schedule_heartbeat_name
  - odds_lambda.scheduling.schedule_store.read_schedules
  - odds_lambda.storage.readers.OddsReader.get_quota_burn
  - odds_lambda.candidate_rules.inputs.rule_quotes
  - odds_lambda.candidate_rules.build_market_state
  - odds_core.close.close_status
  - odds_analytics.candidate_report.candidate_report
  - odds_core.config.DatabaseConfig
  - odds-web.service
  - scripts/odds_web_grants.sql
  - deploy.sh
status: stable
---

# odds-web, the read-only web UI

`odds-web` (`packages/odds-web`, import name `odds_web`) is a single-user, server-rendered web UI on the production host. It shows the forward tests of the candidates ([Candidate registry](../evaluation/candidate-registry.md)) and the health of the data behind their verdicts, reachable from the host and over the tailnet. It writes nothing, and it calls the code that scores candidates rather than re-deriving a number ([ADR-0020](../../docs/adr/0020-read-only-web-ui-on-the-production-host.md)).

## Running it

`odds web serve [--port 8765]` (`odds_cli.commands.web.serve`) builds the app with `odds_web.app.create_app_from_settings` and runs uvicorn on `odds_cli.commands.web.LOOPBACK` (`127.0.0.1`). There is no host option, so only the host and `tailscale serve` reach it. It exits 1 without starting when `WEB_DATABASE_URL` is unset (`odds_web.db.WebDatabaseNotConfigured`). The command imports the web stack only when it runs, so no other `odds` command loads it.

Systemd runs it as `odds-web.service` beside `odds-scheduler.service`, with `CPUQuota` and `MemoryMax` so it cannot starve the scheduler, and no systemd dependency on the scheduler in either direction. `deploy.sh` restarts it after the scheduler ([Redeploy](../scheduling/howto-run-scheduler-service.md#redeploy)). The one-time host steps are in [Set up the odds-web host](howto-set-up-odds-web-host.md).

## Database access

The app never uses `odds_core.database.async_session_maker`, which binds to `DATABASE_URL` only. `odds_web.db.web_database_from_settings` builds its own engine from `odds_core.config.DatabaseConfig.web_url` (`WEB_DATABASE_URL`) and every reader takes a session from that factory (`odds_web.db.WebDatabase`).

- **The role.** `WEB_DATABASE_URL` names the `odds_web` role, which `scripts/odds_web_grants.sql` limits to `SELECT` on the tables the pages read: `events`, `odds_snapshots`, `prices`, `paper_trades`, `candidate_decisions`, `fetch_logs`, `alert_history`, `mlb_probable_pitchers` and the APScheduler tables. It grants nothing else and sets no default privileges, so a table a reader starts using must be added to the file. The file also sets the role's `default_transaction_read_only`.
- **The engine.** `odds_web.db.build_web_database` opens at most `odds_web.db.POOL_SIZE` connections. Every transaction on it is `READ ONLY` and carries `odds_web.db.STATEMENT_TIMEOUT_MS` as its statement timeout (`odds_web.db.restrict_transactions`), so a DSN that is not the restricted role still cannot write through the app.
- **`odds copy` and `--db`.** `odds_cli.db_override.override_database_url` swaps the database name in `WEB_DATABASE_URL` like the agent's DSN.

## Report data and the cache

`odds_analytics.candidate_report.candidate_report` is the one read of the candidates' results: it loads the paper trades and the first decision of every version once, and returns a `odds_analytics.candidate_report.CandidateSummary` per candidate version ([what it holds](../evaluation/candidate-registry.md#odds-candidates-report)). `odds candidates report`, `odds_mcp.server.get_portfolio` and the portfolio page all call it. The page runs the bootstrap on a one-thread executor, so at most one runs at a time (`odds_web.app.create_app`).

`odds_web.cache.SingleFlightCache` holds each section's result for `odds_web.cache.TTL_SECONDS`. Concurrent requests share one load. A failed load keeps the last good value and its time, and is not retried for `odds_web.cache.FAILURE_BACKOFF_SECONDS`, so a down database sees one query per window. The failure is logged with its traceback, which `odds web serve` sends to journald under systemd, and has an `odds_web.cache.FailureKind`:

- **`unavailable`**: the database did not answer (one of `odds_web.cache.DATABASE_ERRORS`: a driver error such as a refused connection or a statement timeout, a pool timeout, or a network error). It is logged as a warning, `web_section_unavailable`.
- **`fault`**: any other exception, a defect in odds-web or in the code it calls, including a query the database rejects (`odds_web.cache.QUERY_ERRORS`: bad SQL, or a table the `odds_web` role has no grant on). It is logged as an error, `web_section_failed`.

Each section is rendered on its own, so a fault in one leaves the others and the status strip standing. Above its data a section says when the data was loaded ("as of 14:05 UTC"). After a failure it says instead "Database unavailable" or "odds-web error (KeyError), see the server log", followed by the time of the data it still shows, dimmed, or by "no data loaded yet".

## Portfolio page

`odds_web.app.create_app` serves `/` and `/fragments/portfolio`, the same section, which htmx re-fetches every 60 seconds. `odds_web.portfolio.portfolio_rows` builds one row per candidate version in the report, agents included:

- **Scope.** The registry entry's sport and market, or `agent` for an agent, and "not current" for a version the registry does not hold.
- **Next checkpoint.** The bets with a close against the next multiple of `odds_analytics.candidate_report.CHECKPOINT_SIZE`, capped at `odds_analytics.candidate_report.KILL_AT_BETS`. An agent with no bet that has a close shows none, and so does a version whose verdict has fired.
- **Forest plot.** `odds_analytics.candidate_report.CandidateSummary.current`: the mean EV over every bet with a close, its 95% interval, and the promote interval at the family size now, drawn against a dashed zero on one axis of ±`odds_web.portfolio.AXIS_PERCENT` percent for every row. An interval past the axis is clipped and marked. The dot is green for a non-negative mean and red for a negative one, and those are the only uses of green and red on the page. A version with fewer than `odds_analytics.candidate_report.MIN_CURRENT_EVENTS` events with a close has no estimate, and says why.
- **P&L.** `odds_analytics.candidate_report.CandidateSummary.pnl` in units of a 1.0 stake, in no colour and never beside the EV on one axis.
- **Verdict.** `odds_analytics.candidate_report.CandidateSummary.verdict` with the checkpoint it rests on (`odds_analytics.candidate_report.CandidateSummary.checkpoint`, frozen once reached). It has no colour. A version with trades and none with a close shows "no close". Because the verdict fires only at a checkpoint, `promote` and `kill` never appear below `odds_analytics.candidate_report.CHECKPOINT_SIZE` bets with a close.

The current estimate and the verdict's checkpoint are separate fields: the estimate moves with every bet, and the checkpoint is the one the verdict is read at ([Verdict](../evaluation/candidate-registry.md#odds-candidates-report)).

## Status strip

Every page loads the strip from `/fragments/status` when it opens and every 60 seconds after. `odds_web.status.load_status` joins three reads, each in one session:

- **Jobs.** One chip per schedule in the scheduler's store, read from the APScheduler tables by `odds_lambda.scheduling.schedule_store.read_schedules` rather than by starting a scheduler, which would write the store's metadata. A chip shows the time since the job's last `heartbeat:<job>` row (`odds_lambda.health_readers.last_heartbeats`, under the name `odds_lambda.health_readers.schedule_heartbeat_name` gives), and its tooltip the next run and whether a worker holds the job. It is late past the job's `odds_core.config.AlertConfig.heartbeat_expectations` hours, and unknown with no heartbeat.
- **Sources.** One chip per market of each bootstrapped collection source and sport, with the time since its last snapshot and the state `odds_lambda.health_readers.source_freshness` gives it: late when stale, idle with no fixture inside the source's horizon, unknown with no history ([Stale-data check](health-monitor.md#stale-data-check)).
- **Odds API quota.** One chip per key with its balance (`odds_lambda.storage.readers.OddsReader.get_latest_quota_by_key`) and the days its burn projection leaves (`odds_lambda.storage.readers.OddsReader.get_quota_burn`); unknown under a day of readings since the key's last reset, and "quota unknown" with no reading.

The judgements are the health monitor's own, so the strip and the alerts never disagree on a threshold. The strip uses no green or red: a late chip is amber and an unknown one a hollow dot.

## Data-health page

`/data` holds two sections, each also served alone at `/fragments/coverage` and `/fragments/closes` for htmx to refresh every 60 seconds.

**Decision-window coverage** (`odds_web.coverage.candidate_coverage`) draws a heatmap for each active mechanical candidate at the version the registry holds: one row per event of its sport kicking off from `odds_web.coverage.PAST` (7 days) ago to `odds_web.coverage.UPCOMING` (24 hours) ahead, one column per `odds_web.coverage.BUCKET_MINUTES` (10) minutes before kick-off from `odds_web.coverage.LEAD_IN_MINUTES` (60) before the earliest decision time, with the decision times marked. Per event and market it reads the rule's quotes once over every bucket's window (`odds_lambda.candidate_rules.inputs.rule_quotes`), then builds the market state per bucket with `odds_lambda.candidate_rules.build_market_state`, the rule's `sharp_inputs` and its maximum input age, from only the snapshots stored by the bucket's time. A bucket at a decision the version recorded takes the decision's time and its recorded `known_at` instead, so it shows what the run saw. A cell shows whether the rule's state has its inputs, and its tooltip which inputs were fresh: retail, from a state needing no sharp input, and each sharp input, from a state needing only that one. A bucket still ahead is left empty. The page changes nothing the rule hash covers.

**Close coverage by week** (`odds_web.closes.close_coverage`) counts, for each sport and market, every (event, market, line) a paper trade is on, for events kicking off in the last `odds_web.closes.WEEKS` (8) weeks, by kick-off week and by `odds_core.close.close_status`: live Betfair close, the sport's fallback named by its source (FDUK or Polymarket), pending within `odds_core.close.close_grace`, no close, or unsupported for a market the sport has no close for, such as MLB `h2h`.

## Static files

`odds_web/static/vendor/htmx.min.js` is htmx 2.0.11 as published on npm, byte for byte, under the Zero-Clause BSD licence (`0BSD`, which asks for no notice). It is committed, so the page loads nothing from a CDN and the repo has no Node build.

## Related

- [Reporting surfaces](reporting-surfaces.md)
- [Databases and environments](../foundations/databases-and-environments.md#the-web-database-role)
- [Set up the odds-web host](howto-set-up-odds-web-host.md)
