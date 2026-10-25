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
  - odds_lambda.betfair.kickoff.next_betfair_kickoff
  - odds_lambda.betfair.kickoff.MAX_BETFAIR_START_LEAD
  - odds_lambda.betfair.kickoff.next_close_target
  - odds_lambda.betfair.kickoff.parse_betfair_start
  - odds_lambda.jobs.fetch_betfair_exchange.CLOSE_LEAD
  - odds_lambda.scheduling.jobs._JOB_SPORTS
  - odds_core.config.BetfairConfig
  - deployment/aws/terraform/lambda.tf
  - deployment/aws/terraform/eventbridge.tf
mentions:
  - odds_core.alerts.job_alert_context
status: stable
---

# Betfair Exchange client and job

Package `packages/odds-lambda/odds_lambda/betfair/`, driven by the fetch-betfair-exchange job. It writes snapshots under bookmaker key `betfair_exchange`, so existing readers and the [sharp benchmark](../modelling/sharp-benchmark.md) fallback consume them unchanged. It is the sharp-reference fallback when Pinnacle has no price ([ADR-0003](../../docs/adr/0003-hybrid-sharp-reference.md)).

## Client and adapter

- `odds_lambda.betfair.client.BetfairExchangeClient` wraps `betfairlightweight` (read-only, delayed app key). `odds_lambda.betfair.client.BetfairExchangeClient.login` uses cert login when both cert paths are set, else `login_interactive()` (dev only; prompts for 2FA). `list_events(sport_cfg, lookahead_hours, lookback_hours)` filters by competition over a window that starts `odds_lambda.jobs.fetch_betfair_exchange.LOOKBACK_HOURS` before now, because Betfair's start time for a game can pass before the game's own and the market can still be pre-play, and a `odds_lambda.betfair.client.BetfairBook.market_start_time` is `None` when the catalogue has none; `odds_lambda.betfair.client.BetfairExchangeClient.list_match_odds_books` fetches the catalogue then `EX_BEST_OFFERS` books in batches of 25, returning `odds_lambda.betfair.client.BetfairBook` / `odds_lambda.betfair.client.BetfairRunner` dataclasses. `odds_lambda.betfair.client.betfair_session` is the login/logout context manager.
- `odds_lambda.betfair.constants.SPORT_CONFIG` holds per-sport config: EPL (`MATCH_ODDS`, market key `1x2`, draw runner) and MLB (market key `h2h`, `"Away @ Home"` event names). `competition_id` is pinned; name hints are a fallback.
- `odds_lambda.betfair.adapter`: `odds_lambda.betfair.adapter.resolve_teams` parses the event name; `odds_lambda.betfair.adapter.betfair_book_to_bookmaker_entry` emits one bookmaker entry; `odds_lambda.betfair.adapter.build_raw_data` wraps it with `source: betfair_api`. Runner names are the full team names, but the halves of a baseball event name are cut at a varying length (16 to 22 characters seen in production) and carry the probable pitchers ("(D Rasm") or a doubleheader suffix ("- Game 1"). `odds_lambda.betfair.adapter.canonical_teams` therefore takes each team from the one runner its half matches, using the event name only to decide home and away, and falls back to `odds_lambda.betfair.adapter.clean_team_name` when no runner matches uniquely. Doubleheader names seen in production put the label after the home team, with or without a dash ("St. Louis Cardinals @ Cincinnati Reds Game 1", "... @ Pittsburgh Pirates - Game 1"); `odds_lambda.betfair.adapter.doubleheader_game_number` reads N from it, or `None` when the name has no numbered label. Names then pass through `odds_core.team.normalize_team_name` ([event-matching](../storage/event-matching.md)).
- `odds_lambda.betfair.cert_loader.resolve_cert_paths`: SSM SecureString params (`cert_pem_ssm_param` + `key_pem_ssm_param`) are written to `/tmp/betfair.crt` and `/tmp/betfair.key` (mode 600, cached across warm starts); otherwise `cert_file` / `cert_key` pass through. Terraform reads the SSM cert params only when `betfair_enabled` is true.
- `odds_core.config.BetfairConfig` (`BETFAIR_` prefix; [configuration](../foundations/configuration.md)): `enabled` defaults to false; enabling without username, password and app key fails validation. `lookahead_hours` defaults to 168; `sports` defaults to every key in `odds_lambda.betfair.constants.SPORT_CONFIG`.

## The fetch-betfair-exchange job

`odds_lambda.jobs.fetch_betfair_exchange.main(ctx)` logs in, lists upcoming Match Odds markets, converts each book to a `betfair_exchange` bookmaker entry and writes snapshots, with their odds rows, via `odds_lambda.storage.writers.OddsWriter.store_odds_snapshot` ([odds-writer](../storage/odds-writer.md#store_odds_snapshot)). It does nothing unless `settings.betfair.enabled` (`BETFAIR_ENABLED`).

### Where it runs

- Production (local scheduler): in the default `bootstrap_jobs`, self-scheduling for every sport in `odds_lambda.betfair.constants.SPORT_CONFIG` (its `odds_lambda.scheduling.jobs._JOB_SPORTS` allowlist is derived from that map, as the `fetch-polymarket-odds` allowlist is from `odds_lambda.polymarket_sports.MATCH_ODDS_SPORTS`; other allowlists, such as EPL-only `fetch-oddsportal-results`, are written out). A scheduler restart keeps the pending run; with none, bootstrap runs the job once to start the chain ([bootstrap](../scheduling/local-scheduler.md#bootstrap-jobs-and-cron-table)).
- AWS (dormant): per-sport rule on the AWS Lambda function, created only when Terraform `betfair_enabled` is true (default false); Betfair credentials and SSM cert parameter names are set in `deployment/aws/terraform/lambda.tf` ([aws-infrastructure](../operations/aws-infrastructure.md)). The dev deploy invokes `fetch-betfair-exchange-epl` as a smoke test ([ci-cd-workflows](../operations/ci-cd-workflows.md)).

### Flow

1. Resolve the target sport: `ctx.sport` is required (`odds_lambda.scheduling.jobs.require_sport`), and a sport outside `odds_lambda.betfair.constants.SPORT_CONFIG` logs `betfair_unknown_sport` and returns.
2. Pre-schedule with `odds_lambda.collection_calendar.decide_collection_run` and `odds_lambda.jobs.fetch_betfair_exchange.CADENCE`: 15 min under 3 h, 30 min at 3-12 h, 2 h beyond, 1 h database fallback. With no fixture within 28 days the run is dormant: a 6 h check-in, its heartbeat, no login, the same decision for every collector ([dormancy](../scheduling/decision-engine.md#dormancy)). Its lead `odds_lambda.jobs.fetch_betfair_exchange.CLOSE_LEAD` (5 min) moves the next run to 5 min before a kickoff whenever the cadence would step past that time, so every kickoff gets a snapshot at the last [ADR-0012](../../docs/adr/0012-retail-ev-against-betfair-exchange-close.md) decision time, inside the live close window. The job passes `odds_lambda.betfair.kickoff.next_betfair_kickoff` as its kickoff lookup, so the cadence and the close run also count the Betfair start in an event's latest stored snapshot (`betfair_meta.market_start_time`) while it is still ahead and earlier than `commence_time` (`odds_lambda.betfair.kickoff.next_close_target`). Betfair can turn a market in play at its own start, an hour before first pitch for the 2026-10-24 World Series game (inferred from the job's logs), so a close run lands before the market goes in play; once that start passes, the event's `commence_time` is the target again and keeps its own close run. A Betfair start more than `odds_lambda.betfair.kickoff.MAX_BETFAIR_START_LEAD` (3 h) before `commence_time` is ignored as a mismatched listing, a later one never delays a run, and a missing or malformed one counts as none (`odds_lambda.betfair.kickoff.parse_betfair_start`). Only the snapshots tagged `betfair_api_` in `api_request_id` of events starting within 3 h of the next one are read. The run before an earlier Betfair start is still more than `odds_core.betfair_close.LIVE_CLOSE_WINDOW` before `commence_time` when the gap exceeds 30 min, so it is not the [close](../modelling/clv-target-functions.md#retail-ev-rows). The sport's overnight window applies to the cadence, not to the close run ([lead runs](../scheduling/decision-engine.md#lead-runs)).
3. Inside [`odds_core.alerts.job_alert_context`](../operations/alerts.md#delivery) (a failure alerts once per outage), login, `odds_lambda.jobs.fetch_betfair_exchange.ingest_sport` for the sport, logout in `finally`. Blocking client calls run in `asyncio.to_thread`; the job tags `api_request_id` `betfair_api_<time>`. On the local scheduler a run past `odds_lambda.scheduling.runner.JOB_TIMEOUT` is cancelled and alerted, and a failed run that booked nothing is re-booked ([Local scheduler](../scheduling/local-scheduler.md#scheduling-primitives)).
4. `odds_lambda.jobs.fetch_betfair_exchange.ingest_sport` skips in-play, CLOSED/INACTIVE and under-populated books and stores every other book with prices, including a pre-play book past its Betfair start (counted in `books_pre_play_past_start`), because the last pre-play price is the [close](../modelling/clv-target-functions.md#retail-ev-rows) whatever Betfair's start says. It resolves team names (`odds_lambda.betfair.adapter.resolve_teams`) and the doubleheader game number, finds or creates each book's event through `odds_lambda.live_events.LiveEventFinder` (for MLB, one MLB schedule request per run resolves every game to its `gamePk` first, and a game left unresolved gets no event or snapshot; [live MLB events](../storage/event-matching.md#live-mlb-events)), which never merges, and stores the snapshot, whose `hours_until_commence` and `fetch_tier` come from the event's `commence_time`, not Betfair's start ([odds-writer](../storage/odds-writer.md#store_odds_snapshot)), then adds one sport-level `odds_core.models.FetchLog` row without quota ([fetch logs](odds-ingestion-service.md#fetch-logs)).
5. Still inside the alert context, a run that dropped MLB games sends the `mlb_games_dropped:<job>` warning (`odds_lambda.live_events.alert_dropped_games`, [alerts](../operations/alerts.md#alert-history)), which never fails the run. Then `odds_lambda.collection_calendar.require_rows_for_fixtures` raises `odds_lambda.collection_calendar.EmptyCollectionError` for a sport with zero stored snapshots and a SCHEDULED fixture within 72 hours for EPL or 6 hours for MLB (`odds_lambda.jobs.fetch_betfair_exchange.SOURCE`); a fixture the run identified but did not price (a book in play, a closed or under-populated book past its Betfair start, or a suspended book with no prices) leaves the calendar, matched by identity rather than start time: the MLB `gamePk` for MLB (`odds_lambda.collection_calendar.MlbGameKey`; not when the resolver only inferred it from a started doubleheader sibling, since that row can be the started game's own), the event within ±2 h otherwise (`odds_lambda.collection_calendar.EventKey`; `odds_lambda.live_events.LiveEventFinder.fixture_key`, which never creates an event; `odds_lambda.jobs.fetch_betfair_exchange.IngestionStats.accounted_fixtures`), with a `betfair_empty_run_check_discounts_unpriced` warning on an empty run; a book with no start time identifies nothing, and a fixture beyond those still raises, including one whose only book is open and pre-play past its Betfair start but has no prices. The message carries the run's errors, which sends the critical alert and skips the heartbeat. A shorter horizon than the other sources, because markets for later fixtures may not be posted yet ([empty collection](../operations/alerting-and-heartbeats.md#empty-collection)).
6. Post-schedule.

## Adapter behaviour

- Outcome `price_decimal` is `best_back` unconverted, with no `price` key (`odds_core.odds_math.PRICE_DECIMAL_KEY`, [odds-math](../foundations/odds-math.md)). Best lay, sizes and last traded price live only in `betfair_meta.runners`. Runners without a back price are dropped, and a book with fewer outcomes than expected returns `None` (nothing stored).
- The app key is the delayed one: prices lag 1-180 s (`odds_core.config.BetfairConfig` docstring). In-play and suspended books are still fetched (`betfair_meta.inplay`, `market_status`), and markets suspended pre-match are stored and counted in `books_suspended`. `betfair_meta.market_start_time` is Betfair's start for the market (ISO, or null), which can differ from the event's `commence_time` and can move.
- Unknown runner names fall back to the normalized name with a warning, so an alias gap yields an outcome that matches neither team ([event-matching](../storage/event-matching.md)).
- The market key differs by sport (`1x2` vs `h2h`), and `betfair_exchange` is also written by FDUK (`BFE*`) and by OddsPortal's Betfair row ([sport-and-market-keys](sport-and-market-keys.md#market-key-dialects)).
- Its `odds_core.models.FetchLog` rows have `success = snapshots_stored > 0` and no quota; how readers treat them is on [health-monitor](../operations/health-monitor.md) and [odds status](../operations/health-monitor.md#odds-status).
- Whether the job and the SSM cert parameters exist in AWS cannot be checked from the repo: `odds scheduler list-jobs --backend aws`.

## Related

- [Job deployment matrix](../scheduling/job-deployment-matrix.md)
- [Decision engine](../scheduling/decision-engine.md)
- [OddsPortal scraper](oddsportal-scraper.md)
