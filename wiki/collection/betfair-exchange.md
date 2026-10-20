---
type: reference
title: Betfair Exchange client and job
description: "The Betfair client, cert loading and adapter, the fetch-betfair-exchange job's cadence and where it runs, and what price is actually stored as betfair_exchange."
sources:
  - odds_lambda.betfair.client.BetfairExchangeClient
  - odds_lambda.betfair.adapter.betfair_book_to_bookmaker_entry
  - odds_lambda.betfair.cert_loader.resolve_cert_paths
  - odds_lambda.betfair.constants.SPORT_CONFIG
  - odds_lambda.jobs.fetch_betfair_exchange.main
  - odds_lambda.jobs.fetch_betfair_exchange.ingest_sport
  - odds_lambda.jobs.fetch_betfair_exchange.CADENCE
  - odds_lambda.scheduling.jobs._JOB_SPORTS
  - odds_core.config.BetfairConfig
  - deployment/aws/terraform/lambda.tf
  - deployment/aws/terraform/eventbridge.tf
status: stable
---

# Betfair Exchange client and job

Package `packages/odds-lambda/odds_lambda/betfair/`, driven by the fetch-betfair-exchange job. It writes snapshots under bookmaker key `betfair_exchange`, so existing readers and the [sharp benchmark](../modelling/sharp-benchmark.md) fallback consume them unchanged. It is the sharp-reference fallback when Pinnacle has no price ([ADR-0003](../../docs/adr/0003-hybrid-sharp-reference.md)).

## Client and adapter

- `odds_lambda.betfair.client.BetfairExchangeClient` wraps `betfairlightweight` (read-only, delayed app key). `odds_lambda.betfair.client.BetfairExchangeClient.login` uses cert login when both cert paths are set, else `login_interactive()` (dev only; prompts for 2FA). `list_events(sport_cfg, lookahead_hours)` filters by competition; `odds_lambda.betfair.client.BetfairExchangeClient.list_match_odds_books` fetches the catalogue then `EX_BEST_OFFERS` books in batches of 25, returning `odds_lambda.betfair.client.BetfairBook` / `odds_lambda.betfair.client.BetfairRunner` dataclasses. `odds_lambda.betfair.client.betfair_session` is the login/logout context manager.
- `odds_lambda.betfair.constants.SPORT_CONFIG` holds per-sport config: EPL (`MATCH_ODDS`, market key `1x2`, draw runner) and MLB (market key `h2h`, `"Away @ Home"` event names). `competition_id` is pinned; name hints are a fallback.
- `odds_lambda.betfair.adapter`: `odds_lambda.betfair.adapter.resolve_teams` parses the event name; `odds_lambda.betfair.adapter.betfair_book_to_bookmaker_entry` emits one bookmaker entry; `odds_lambda.betfair.adapter.build_raw_data` wraps it with `source: betfair_api`. Runner names pass through `odds_core.team.normalize_team_name` after `odds_lambda.betfair.adapter.strip_pitcher_annotation` removes the MLB pitcher annotation, such as "(TBD)" or the 22-character-truncated "(D Rasm" that has no closing parenthesis ([event-matching](../storage/event-matching.md)).
- `odds_lambda.betfair.cert_loader.resolve_cert_paths`: SSM SecureString params (`cert_pem_ssm_param` + `key_pem_ssm_param`) are written to `/tmp/betfair.crt` and `/tmp/betfair.key` (mode 600, cached across warm starts); otherwise `cert_file` / `cert_key` pass through. Terraform reads the SSM cert params only when `betfair_enabled` is true.
- `odds_core.config.BetfairConfig` (`BETFAIR_` prefix; [configuration](../foundations/configuration.md)): `enabled` defaults to false; enabling without username, password and app key fails validation. `lookahead_hours` defaults to 168; `sports` defaults to every key in `odds_lambda.betfair.constants.SPORT_CONFIG`.

## The fetch-betfair-exchange job

`odds_lambda.jobs.fetch_betfair_exchange.main(ctx)` logs in, lists upcoming Match Odds markets, converts each book to a `betfair_exchange` bookmaker entry and writes snapshots via `odds_lambda.storage.writers.OddsWriter.store_odds_snapshot` ([odds-writer](../storage/odds-writer.md)). It does nothing unless `settings.betfair.enabled` (`BETFAIR_ENABLED`).

### Where it runs

- Production (mini-PC): in default `bootstrap_jobs`, self-scheduling for every sport in `odds_lambda.betfair.constants.SPORT_CONFIG` (its `odds_lambda.scheduling.jobs._JOB_SPORTS` allowlist is derived from that map, as the `fetch-polymarket-odds` allowlist is from `odds_lambda.polymarket_sports.MATCH_ODDS_SPORTS`). A scheduler restart keeps the pending run; with none, bootstrap runs the job once to start the chain ([bootstrap](../scheduling/local-scheduler.md#bootstrap-jobs-and-cron-table)).
- AWS (dormant): per-sport rule on the scheduler Lambda, created only when Terraform `betfair_enabled` is true (default false); Betfair credentials and SSM cert parameter names are set in `deployment/aws/terraform/lambda.tf` ([aws-infrastructure](../operations/aws-infrastructure.md)). The dev deploy invokes `fetch-betfair-exchange-epl` as a smoke test ([ci-cd-workflows](../operations/ci-cd-workflows.md)).

### Flow

1. Resolve target sports: `ctx.sport`, else `settings.betfair.sports` or every `odds_lambda.betfair.constants.SPORT_CONFIG` key.
2. Pre-schedule with `odds_lambda.scheduling.decision.decide_forward_resilient` and `odds_lambda.jobs.fetch_betfair_exchange.CADENCE`: 15 min under 3 h, 30 min at 3-12 h, 2 h beyond or with no games, 1 h database fallback. Its `close_lead` of 5 min moves the next run to 5 min before a kickoff whenever the cadence would step past that time, so every kickoff gets a snapshot at the last [ADR-0012](../../docs/adr/0012-retail-ev-against-betfair-exchange-close.md) decision time, inside the live close window. Always-run (no gate); the overnight window applies only to single-sport runs ([decision-engine](../scheduling/decision-engine.md#close-run)).
3. Inside `odds_core.alerts.job_alert_context`, login, `odds_lambda.jobs.fetch_betfair_exchange.ingest_sport` per sport, logout in `finally`. Blocking client calls run in `asyncio.to_thread`; the job tags `api_request_id` `betfair_api_<time>`. On the local scheduler a run past `odds_lambda.scheduling.runner.JOB_TIMEOUT` is cancelled and alerted, and a failed run that booked nothing is re-booked ([Local scheduler](../scheduling/local-scheduler.md#scheduling-primitives)).
4. `odds_lambda.jobs.fetch_betfair_exchange.ingest_sport` skips in-play, CLOSED/INACTIVE and under-populated books, resolves team names (`odds_lambda.betfair.adapter.resolve_teams`), calls `odds_lambda.storage.writers.OddsWriter.find_or_create_event` (which returns the canonical event, the Odds API one when it exists, and never merges) and stores the snapshot, then adds one sport-level `odds_core.models.FetchLog` row without quota ([fetch logs](odds-ingestion-service.md#fetch-logs)).
5. Still inside the alert context, `odds_lambda.collection_calendar.require_rows_for_fixtures` raises `odds_lambda.collection_calendar.EmptyCollectionError` for a sport with zero stored snapshots and a SCHEDULED fixture within 72 hours for EPL or 6 hours for MLB (`odds_lambda.jobs.fetch_betfair_exchange.SOURCE`); the message carries the run's errors, which sends the critical alert and skips the heartbeat. A shorter horizon than the other sources, because markets for later fixtures may not be posted yet ([empty collection](../operations/alerting-and-heartbeats.md#empty-collection)).
6. Post-schedule.

## Adapter behaviour

- Outcome `price_decimal` is `best_back` unconverted, with no `price` key (`odds_core.odds_math.PRICE_DECIMAL_KEY`, [odds-math](../foundations/odds-math.md)). Best lay, sizes and last traded price live only in `betfair_meta.runners`. Runners without a back price are dropped, and a book with fewer outcomes than expected returns `None` (nothing stored).
- The app key is the delayed one: prices lag 1-180 s (`odds_core.config.BetfairConfig` docstring). In-play and suspended books are still fetched (`betfair_meta.inplay`, `market_status`), and markets suspended pre-match are stored and counted in `books_suspended`.
- Unknown runner names fall back to the normalized name with a warning, so an alias gap yields an outcome that matches neither team ([event-matching](../storage/event-matching.md)).
- The market key differs by sport (`1x2` vs `h2h`), and `betfair_exchange` is also written by FDUK (`BFE*`) and by OddsPortal's Betfair row ([sport-and-market-keys](sport-and-market-keys.md#market-key-dialects)).
- Its `odds_core.models.FetchLog` rows have `success = snapshots_stored > 0` and no quota; how readers treat them is on [health-monitor](../operations/health-monitor.md) and [odds status](../operations/health-monitor.md#odds-status).
- Whether the job and the SSM cert parameters exist in AWS cannot be checked from the repo: `odds scheduler list-jobs --backend aws`.

## Related

- [Job deployment matrix](../scheduling/job-deployment-matrix.md)
- [Decision engine](../scheduling/decision-engine.md)
- [OddsPortal scraper](oddsportal-scraper.md)
