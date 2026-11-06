---
type: reference
title: odds-web, the read-only web UI
description: What odds web serve runs, the database role and session it reads through, the report data behind the portfolio page, its cache and failure bounds, the vendored scripts, and how deploy.sh and systemd run it.
sources:
  - odds_cli.commands.web
  - odds_web.app
  - odds_web.db
  - odds_web.cache
  - odds_web.portfolio
  - odds_web.role_check
  - odds_analytics.candidate_report.candidate_report
  - odds_core.config.DatabaseConfig
  - odds-web.service
  - scripts/odds_web_grants.sql
  - scripts/setup-odds-web-host.sh
  - deploy.sh
status: stable
---

# odds-web, the read-only web UI

`odds-web` (`packages/odds-web`, import name `odds_web`) is a single-user, server-rendered web UI on the production host. It shows the forward tests of the candidates ([Candidate registry](../evaluation/candidate-registry.md)), reachable from the host and over the tailnet. It writes nothing, and it calls the code that scores candidates rather than re-deriving a number ([ADR-0020](../../docs/adr/0020-read-only-web-ui-on-the-production-host.md)).

## Running it

`odds web serve [--port 8765]` (`odds_cli.commands.web.serve`) builds the app with `odds_web.app.create_app_from_settings` and runs uvicorn on `odds_cli.commands.web.LOOPBACK` (`127.0.0.1`). There is no host option, so only the host and `tailscale serve` reach it. It exits 1 without starting when `WEB_DATABASE_URL` is unset (`odds_web.db.WebDatabaseNotConfigured`). The command imports the web stack only when it runs, so no other `odds` command loads it.

Systemd runs it as `odds-web.service` beside `odds-scheduler.service`, with `CPUQuota` and `MemoryMax` so it cannot starve the scheduler, and no systemd dependency on the scheduler in either direction. `deploy.sh` restarts it after the scheduler ([Redeploy](../scheduling/howto-run-scheduler-service.md#redeploy)). The one-time host steps are in [Set up the odds-web host](howto-set-up-odds-web-host.md).

## Database access

The app never uses `odds_core.database.async_session_maker`, which binds to `DATABASE_URL` only. `odds_web.db.web_database_from_settings` builds its own engine from `odds_core.config.DatabaseConfig.web_url` (`WEB_DATABASE_URL`) and every reader takes a session from that factory (`odds_web.db.WebDatabase`).

- **The role.** `WEB_DATABASE_URL` names the `odds_web` role, which `scripts/odds_web_grants.sql` limits to `SELECT` on the tables the pages read: `events`, `odds_snapshots`, `prices`, `paper_trades`, `candidate_decisions`, `fetch_logs`, `alert_history` and the APScheduler tables. It grants nothing else and sets no default privileges, so a table a reader starts using must be added to the file. The file also sets the role's `default_transaction_read_only`.
- **The engine.** `odds_web.db.build_web_database` opens at most `odds_web.db.POOL_SIZE` connections. Every transaction on it is `READ ONLY` and carries `odds_web.db.STATEMENT_TIMEOUT_MS` as its statement timeout (`odds_web.db.restrict_transactions`), so a DSN that is not the restricted role still cannot write through the app.
- **`odds copy` and `--db`.** `odds_cli.db_override.override_database_url` swaps the database name in `WEB_DATABASE_URL` like the agent's DSN.

## Report data and the cache

`odds_analytics.candidate_report.candidate_report` is the one read of the candidates' results: it loads the paper trades and the first decision of every version once, and returns a `odds_analytics.candidate_report.CandidateSummary` per candidate version ([what it holds](../evaluation/candidate-registry.md#odds-candidates-report)). `odds candidates report`, `odds_mcp.server.get_portfolio` and the portfolio page all call it. The page runs the bootstrap on a one-thread executor, so at most one runs at a time (`odds_web.app.create_app`).

`odds_web.cache.SingleFlightCache` holds each section's result for `odds_web.cache.TTL_SECONDS`. Concurrent requests share one load. A failed load keeps the last good value and its time, and is not retried for `odds_web.cache.FAILURE_BACKOFF_SECONDS`, so a down database sees one query per window. A section whose load failed renders "Unavailable" with the time of its last good data (or "never") instead of failing the page, over a stale copy of the data when there is one.

## Portfolio page

`odds_web.app.create_app` serves `/` and `/fragments/portfolio`, the same section, which htmx re-fetches every 60 seconds. `odds_web.portfolio.portfolio_rows` builds one row per candidate version in the report, agents included:

- **Scope.** The registry entry's sport and market, or `agent` for an agent, and "not current" for a version the registry does not hold.
- **Next checkpoint.** The bets with a close against the next multiple of `odds_analytics.candidate_report.CHECKPOINT_SIZE`, capped at `odds_analytics.candidate_report.KILL_AT_BETS`. An agent with no bet that has a close shows none, and so does a version whose verdict has fired.
- **Forest plot.** `odds_analytics.candidate_report.CandidateSummary.current`: the mean EV over every bet with a close, its 95% interval, and the promote interval at the family size now, drawn against a dashed zero on one axis of ±`odds_web.portfolio.AXIS_PERCENT` percent for every row. An interval past the axis is clipped and marked. The dot is green for a non-negative mean and red for a negative one, and those are the only uses of green and red on the page. A version with fewer than `odds_analytics.candidate_report.MIN_CURRENT_EVENTS` events with a close has no estimate, and says why.
- **P&L.** `odds_analytics.candidate_report.CandidateSummary.pnl` in units of a 1.0 stake, in no colour and never beside the EV on one axis.
- **Verdict.** `odds_analytics.candidate_report.CandidateSummary.verdict` with the checkpoint it rests on (`odds_analytics.candidate_report.CandidateSummary.checkpoint`, frozen once reached). It has no colour. A version with trades and none with a close shows "no close". Because the verdict fires only at a checkpoint, `promote` and `kill` never appear below `odds_analytics.candidate_report.CHECKPOINT_SIZE` bets with a close.

The current estimate and the verdict's checkpoint are separate fields: the estimate moves with every bet, and the checkpoint is the one the verdict is read at ([Verdict](../evaluation/candidate-registry.md#odds-candidates-report)).

## Static files

`odds_web/static/vendor/htmx.min.js` is htmx 2.0.11 as published on npm, byte for byte, under the Zero-Clause BSD licence (`0BSD`, which asks for no notice). It is committed, so the page loads nothing from a CDN and the repo has no Node build.

## Related

- [Reporting surfaces](reporting-surfaces.md)
- [Databases and environments](../foundations/databases-and-environments.md#the-web-database-role)
- [Set up the odds-web host](howto-set-up-odds-web-host.md)
