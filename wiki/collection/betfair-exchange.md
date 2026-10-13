---
type: component
title: Betfair Exchange client and job
description: "The Betfair client, cert loading and adapter, the fetch-betfair-exchange job's cadence on AWS and locally, and what price is actually stored as betfair_exchange."
sources:
  - odds_lambda.betfair.client.BetfairExchangeClient
  - odds_lambda.betfair.adapter.betfair_book_to_bookmaker_entry
  - odds_lambda.betfair.cert_loader.resolve_cert_paths
  - odds_lambda.betfair.constants.SPORT_CONFIG
  - odds_lambda.jobs.fetch_betfair_exchange.main
  - odds_lambda.jobs.fetch_betfair_exchange.ingest_sport
  - odds_lambda.jobs.fetch_betfair_exchange.CADENCE
  - odds_lambda.scheduling.jobs._JOB_CRON_MAP
  - odds_core.config.BetfairConfig
  - deployment/aws/terraform/lambda.tf
  - deployment/aws/terraform/eventbridge.tf
status: stable
---

# Betfair Exchange client and job

Package `packages/odds-lambda/odds_lambda/betfair/`, driven by the fetch-betfair-exchange job. It writes snapshots under bookmaker key `betfair_exchange`, so existing readers and the [sharp benchmark](../modelling/sharp-benchmark.md) fallback consume them unchanged. On the sharp-reference question: Betfair `sharp_prob` correlates 0.9994 with Pinnacle (a ~1.1 pp level shift) and is the sharp fallback ([decision-hybrid-sharp-reference](../modelling/decision-hybrid-sharp-reference.md)).

## Client and adapter

- `odds_lambda.betfair.client.BetfairExchangeClient` wraps `betfairlightweight` (read-only, delayed app key). `login()` uses cert login when both cert paths are set, else `login_interactive()` (dev only; prompts for 2FA). `list_events(sport_cfg, lookahead_hours)` filters by competition; `list_match_odds_books` fetches the catalogue then `EX_BEST_OFFERS` books in batches of 25, returning `odds_lambda.betfair.client.BetfairBook` / `odds_lambda.betfair.client.BetfairRunner` dataclasses. `odds_lambda.betfair.client.betfair_session` is the login/logout context manager.
- `odds_lambda.betfair.constants.SPORT_CONFIG` holds per-sport config: EPL (`MATCH_ODDS`, market key `1x2`, draw runner) and MLB (market key `h2h`, `"Away @ Home"` event names). `competition_id` is pinned; name hints are a fallback.
- `odds_lambda.betfair.adapter`: `odds_lambda.betfair.adapter.resolve_teams` parses the event name; `odds_lambda.betfair.adapter.betfair_book_to_bookmaker_entry` emits one bookmaker entry; `odds_lambda.betfair.adapter.build_raw_data` wraps it with `source: betfair_api`. Runner names pass through `odds_core.team.normalize_team_name` after stripping MLB pitcher annotations such as "(TBD)" ([events](../storage/events.md#matchers)).
- `odds_lambda.betfair.cert_loader.resolve_cert_paths`: SSM SecureString params (`cert_pem_ssm_param` + `key_pem_ssm_param`) are written to `/tmp/betfair.crt` and `/tmp/betfair.key` (mode 600, cached across warm starts); otherwise `cert_file` / `cert_key` pass through. Terraform reads the SSM cert params only when `betfair_enabled` is true.
- `odds_core.config.BetfairConfig` (`BETFAIR_` prefix; [configuration](../foundations/configuration.md)): `enabled` defaults to false; enabling without username, password and app key fails validation. `lookahead_hours` defaults to 168; `sports` defaults to every key in `odds_lambda.betfair.constants.SPORT_CONFIG`.

## The fetch-betfair-exchange job

`odds_lambda.jobs.fetch_betfair_exchange.main(ctx)` logs in, lists upcoming Match Odds markets, converts each book to a `betfair_exchange` bookmaker entry and writes snapshots via `odds_lambda.storage.writers.OddsWriter.store_odds_snapshot` ([odds-storage](../storage/odds-storage.md)). It does nothing unless `settings.betfair.enabled` (`BETFAIR_ENABLED`).

### Where it runs

- AWS: per-sport rule on the scheduler Lambda, created only when Terraform `betfair_enabled` is true (default false); Betfair credentials and SSM cert parameter names are set in `deployment/aws/terraform/lambda.tf` ([aws-infrastructure](../operations/aws-infrastructure.md)). The dev deploy invokes `fetch-betfair-exchange-epl` as a smoke test ([ci-cd-workflows](../operations/ci-cd-workflows.md)).
- Local: in default `bootstrap_jobs`, installed as cron `*/30 * * * *` for every sport in `odds_lambda.betfair.constants.SPORT_CONFIG` (`odds_lambda.scheduling.jobs._JOB_CRON_MAP` is the source; [local-scheduler](../scheduling/local-scheduler.md)).

### Flow

1. Resolve target sports: `ctx.sport`, else `settings.betfair.sports` or every `odds_lambda.betfair.constants.SPORT_CONFIG` key.
2. Pre-schedule with `odds_lambda.scheduling.decision.decide_forward_resilient` and `odds_lambda.jobs.fetch_betfair_exchange.CADENCE`: 15 min under 3 h, 30 min at 3-12 h, 2 h beyond or with no games, 1 h database fallback. Always-run (no gate); the overnight window applies only to single-sport runs ([self-scheduling](../scheduling/self-scheduling.md)).
3. Inside `odds_core.alerts.job_alert_context`, login, `odds_lambda.jobs.fetch_betfair_exchange.ingest_sport` per sport, logout in `finally`. Blocking client calls run in `asyncio.to_thread`; the job tags `api_request_id` `betfair_api_<time>`.
4. `odds_lambda.jobs.fetch_betfair_exchange.ingest_sport` skips in-play, CLOSED/INACTIVE and under-populated books, resolves team names (`odds_lambda.betfair.adapter.resolve_teams`), calls `find_or_create_event` and stores the snapshot, then adds one sport-level `odds_core.models.FetchLog` row without quota ([odds-api-ingestion](odds-api-ingestion.md#fetch_logs)).
5. If events were seen but zero snapshots stored, `send_job_warning("job_empty:<job>")`. An empty event list is treated as a venue gap and does not alert.
6. Post-schedule.

## Hazards

- **Only best back is the price.** Outcome `price` is `best_back` converted to American (`odds_core.odds_math.decimal_to_american`, which truncates; [odds-math-and-devigging](../foundations/odds-math-and-devigging.md)); best lay, sizes and last traded price live only in `betfair_meta.runners`. Runners without a back price are dropped, and a book with fewer outcomes than expected returns `None` (nothing stored).
- **Delayed key**: prices lag 1-180 s (`odds_core.config.BetfairConfig` docstring); in-play and suspended books are still fetched (`betfair_meta.inplay`, `market_status`), and markets suspended pre-match are stored and counted in `books_suspended`.
- **Betfair failures count as fetch failures.** `odds_core.models.FetchLog` rows have `success = snapshots_stored > 0` and no quota. `odds_lambda.health_monitor.HealthMonitor` reads the latest 10 rows regardless of source, so Betfair rows count toward `consecutive_failures` and success rate, and `status quota` can show nothing when the newest row is Betfair ([odds-api-client-and-quota](odds-api-client-and-quota.md), [alerts-and-health](../operations/alerts-and-health.md)).
- **Cron floor is replaced.** Local cron and self-scheduling share one APScheduler id, so the first self-schedule replaces the `*/30` cron ([local-scheduler](../scheduling/local-scheduler.md)).
- **Assert, not config error.** `odds_lambda.jobs.fetch_betfair_exchange.main` asserts `username`, `password` and `app_key` are set when enabled; missing values fail inside `assert`.
- **Unknown runner names fall back to the normalized name** with a warning, so an alias gap yields an outcome that matches neither team.
- **Market key differs by sport** (`1x2` vs `h2h`) and from the Odds API's football `h2h` ([sources-and-keys](sources-and-keys.md)).
- **Three writers.** `betfair_exchange` is also written by FDUK (`BFE*`) and by OddsPortal's Betfair row ([howto-load-historical-odds](howto-load-historical-odds.md)).
- Whether the job and the SSM cert parameters exist in AWS cannot be checked from the repo: `odds scheduler list-jobs --backend aws`.

## Related

- [Jobs and deployment matrix](../scheduling/jobs-and-deployment-matrix.md)
- [Self-scheduling](../scheduling/self-scheduling.md)
- [OddsPortal scraper and job](oddsportal.md)
