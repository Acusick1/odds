---
type: component
title: fetch-betfair-exchange job
summary: "Polls Betfair Exchange best back/lay prices per sport on a kickoff-proximity cadence: self-scheduling on AWS, a cron floor locally."
sources:
  - docs/ARCHITECTURE.md#betfair-exchange-job
  - docs/DEPLOYMENT.md#job-deployment-matrix
  - packages/odds-lambda/odds_lambda/jobs/fetch_betfair_exchange.py::main
  - packages/odds-lambda/odds_lambda/jobs/fetch_betfair_exchange.py::ingest_sport
  - packages/odds-lambda/odds_lambda/jobs/fetch_betfair_exchange.py::CADENCE
verified_at: 7887039
---

# fetch-betfair-exchange job

`packages/odds-lambda/odds_lambda/jobs/fetch_betfair_exchange.py::main(ctx)` logs into the Betfair Exchange API ([Betfair client](betfair-exchange-client.md)), lists upcoming Match Odds markets, converts the best back and lay to a `betfair_exchange` bookmaker entry and writes snapshots via `OddsWriter.store_odds_snapshot`. It does nothing unless `settings.betfair.enabled` (`BETFAIR_ENABLED`).

## Where it runs

- AWS: per-sport rule on the scheduler Lambda, created only when Terraform `betfair_enabled` is true; Betfair credentials and SSM cert parameter names are set in `deployment/aws/terraform/lambda.tf` ([AWS infrastructure](aws-infrastructure.md)).
- Local: in default `bootstrap_jobs`, installed as cron `*/30 * * * *` for every sport in `SPORT_CONFIG` (`packages/odds-lambda/odds_lambda/scheduling/jobs.py::_JOB_CRON_MAP` is the source; see [local scheduler](local-scheduler.md)).

## Flow

1. Resolve target sports: `ctx.sport`, else `settings.betfair.sports` or every `SPORT_CONFIG` key.
2. Pre-schedule with `decide_forward_resilient` and `CADENCE`: 15 min under 3 h, 30 min at 3-12 h, 2 h beyond or with no games, 1 h database fallback. Always-run (no gate); the overnight window applies only to single-sport runs ([decision engine](scheduling-decision-engine.md)).
3. Inside `job_alert_context`, login, `ingest_sport` per sport, logout in `finally`. Blocking client calls run in `asyncio.to_thread`.
4. `ingest_sport` skips in-play, CLOSED/INACTIVE and under-populated books, resolves team names (`resolve_teams`), calls `find_or_create_event` and stores the snapshot (`api_request_id="betfair_api_<time>"`), then adds one sport-level `FetchLog` row.
5. If events were seen but zero snapshots stored, `send_job_warning("job_empty:<job>")`. An empty event list is treated as a venue gap and does not alert.
6. Post-schedule.

## Hazards

- Betfair writes `FetchLog` rows with `success = snapshots_stored > 0` and no quota value. `HealthMonitor` reads the latest 10 rows regardless of source, so Betfair failures count toward `consecutive_failures` and success rate ([table fetch_logs](table-fetch-logs.md), [health monitor](health-monitor.md)).
- Local cron and self-scheduling share one APScheduler id, so the cron floor may be replaced after the first run ([local scheduler](local-scheduler.md)).
- `main` asserts `username`, `password` and `app_key` are set when enabled; missing values fail inside `assert` rather than a configuration error.
- In-play detection depends on `book.inplay`; markets suspended pre-match are still stored and counted in `books_suspended`.

## Related

- [Self-scheduling](self-scheduling.md), [job registry](job-registry.md).
