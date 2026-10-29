---
type: reference
title: Polymarket match prices as odds snapshots
description: The Polymarket client and settings, how Polymarket's three-way match markets become polymarket odds snapshots, the history grid and its drop rules, attaching matches to events, the backfill command and the fetch-polymarket-odds job.
sources:
  - odds_lambda.polymarket_sports.POLYMARKET_SPORTS
  - odds_lambda.polymarket_sports.PolymarketSportConfig
  - odds_lambda.polymarket_sports.MATCH_ODDS_SPORTS
  - odds_lambda.polymarket_odds
  - odds_lambda.polymarket_odds.TIER_STEPS
  - odds_lambda.polymarket_odds.MAX_HORIZON
  - odds_lambda.polymarket_odds.MAX_STALENESS
  - odds_lambda.polymarket_odds.SUM_TOLERANCE
  - odds_lambda.polymarket_odds.parse_match_event
  - odds_lambda.polymarket_odds.listed_game
  - odds_lambda.storage.event_resolver.attach_event
  - odds_lambda.storage.event_resolver.attach_events
  - odds_lambda.polymarket_odds.grid_offsets
  - odds_lambda.polymarket_odds.resample
  - odds_lambda.polymarket_odds.build_envelope
  - odds_lambda.polymarket_fetcher.PolymarketClient.get_events
  - odds_lambda.polymarket_fetcher.PolymarketClient.get_all_events
  - odds_lambda.polymarket_fetcher.PolymarketClient
  - odds_lambda.polymarket_fetcher.PolymarketClient.get_price_history
  - odds_lambda.polymarket_fetcher.PolymarketClient.get_price_series
  - odds_lambda.polymarket_fetcher.MAX_CONCURRENT_REQUESTS
  - odds_core.config.PolymarketConfig
  - odds_cli.commands.polymarket
  - odds_cli.commands.polymarket.discover
  - odds_lambda.polymarket_fetcher.MAX_HISTORY_WINDOW
  - odds_lambda.polymarket_odds_backfill.backfill_sport
  - odds_lambda.jobs.fetch_polymarket_odds
  - odds_lambda.jobs.fetch_polymarket_odds.CADENCE
  - odds_lambda.jobs.fetch_polymarket_odds.SOURCE
  - odds_lambda.jobs.fetch_polymarket_odds.ingest_sport
  - odds_cli.commands.polymarket.backfill_odds
  - odds_lambda.storage.writers.OddsWriter.store_envelope
  - odds_core.odds_math.PRICE_DECIMAL_KEY
  - odds_core.bookmakers
  - odds_lambda.polymarket_odds.MAX_QUOTE_SPREAD
  - odds_lambda.jobs.fetch_polymarket_odds.EmptyListingError
mentions:
  - odds_core.alerts.job_alert_context
status: stable
---

# Polymarket match prices as odds snapshots

Polymarket's EPL match markets are stored as `odds_core.models.OddsSnapshot` ([columns and constraints](../storage/odds-tables.md#odds_snapshots)) rows with `polymarket` price rows ([price tables](../storage/price-tables.md)), so a training config can list `polymarket` in `sharp_bookmakers` ([sharp benchmark](../modelling/sharp-benchmark.md)). The decision is [ADR-0013](../../docs/adr/0013-collect-polymarket-epl-as-sharp-reference.md), which supersedes [ADR-0007](../../docs/adr/0007-deprioritise-polymarket.md); Polymarket is still not a UK trading venue. Nothing stores NBA Polymarket data. Why the venue is usable for EPL is in [Polymarket market character](polymarket-market-research.md#market-character-by-sport).

## Client and settings

`odds_lambda.polymarket_fetcher.PolymarketClient` uses aiohttp with a tenacity retry (3 attempts) and no auth. Its base URLs are `odds_core.config.PolymarketConfig.gamma_base_url` and `odds_core.config.PolymarketConfig.clob_base_url`; the settings class (env prefix `POLYMARKET_`) also holds `odds_core.config.PolymarketConfig.enabled`, the only other setting. Gamma `GET /events` serves discovery (`odds_lambda.polymarket_fetcher.PolymarketClient.get_events`, below). CLOB `GET /prices-history` (`market` = token id, `interval`, `fidelity`, optional `startTs`/`endTs`, returning `{history: [{t, p}]}`) serves history through `odds_lambda.polymarket_fetcher.PolymarketClient.get_price_history`, which `odds_lambda.polymarket_fetcher.PolymarketClient.get_price_series` wraps for explicit time ranges ([history grid](#history-grid)). Batch calls cap at `odds_lambda.polymarket_fetcher.MAX_CONCURRENT_REQUESTS` (10). `odds polymarket discover` (`odds_cli.commands.polymarket.discover`) lists the active NBA events currently on Polymarket through `odds_lambda.polymarket_fetcher.PolymarketClient.get_events` and stores nothing.

## Sport config and discovery

`odds_lambda.polymarket_sports.POLYMARKET_SPORTS` maps a sport key to its Gamma series ID, optional tag ID, ticker prefix, market key and whether a match has three markets (`odds_lambda.polymarket_sports.PolymarketSportConfig`). EPL is series `10188`, no tag, ticker prefix `epl`, market key `1x2`. `odds_lambda.polymarket_sports.MATCH_ODDS_SPORTS` lists the sports with three-way match markets; the odds path accepts only those, and NBA (a single two-outcome market per game) is outside it.

`odds_lambda.polymarket_fetcher.PolymarketClient.get_events` fetches one Gamma page for a sport (at most 100 events) and `odds_lambda.polymarket_fetcher.PolymarketClient.get_all_events` pages until an empty page and keeps one event per Polymarket id, because offset paging can repeat an event when the listing shifts. Gamma keeps `active=true` on closed events, so listing closed events passes `active=None`. The series mixes match events with prop events (corners, exact score, player props), so `odds_lambda.polymarket_odds.parse_match_event` keeps only events whose slug is `{prefix}-{home3}-{away3}-YYYY-MM-DD`. It classifies the event's markets by slug suffix (`-{home3}`, `-draw`, `-{away3}`), not by title or order, because titles differ between seasons. An event with an unknown or missing market is skipped with a warning. Team names come from the event's `teams` entries (`ordering` home or away) and go through `odds_lambda.polymarket_odds.canonical_team`, which drops a trailing ` FC` or ` AFC` and then applies `odds_core.team.normalize_team_name`.

## Snapshot shape

`odds_lambda.polymarket_odds.build_envelope` emits one bookmaker entry, key `polymarket`, one market under the sport's market key, with outcomes home team, `Draw`, away team. Prices are the reciprocals of the three Yes probabilities, stored as `price_decimal` (`odds_core.odds_math.PRICE_DECIMAL_KEY`) with no `price` key; the exact probabilities and their sum also sit in `polymarket_meta`, which odds parsers ignore. Live snapshots also keep each outcome's best bid and ask there. The Yes prices carry no overround, so the pipeline's proportional devig only removes the small residual from the sum. The envelope's `source` key and the snapshot's `source` column are `polymarket` ([envelope format](../storage/odds-tables.md#envelope-format)). The market key `1x2` matches the other EPL three-way writers for exchange data ([market-key dialects](sport-and-market-keys.md#market-key-dialects)).

A snapshot holds only `polymarket`, so it carries no retail prices to build a decision row from, and `polymarket` is not in `odds_core.bookmakers.UK_RETAIL_BOOKMAKERS`, the book set of `retail_ev` rows and of the candidate rules' retail prices. `odds_analytics.feature_groups.collect_event_data` leaves snapshots of source `polymarket` out of an event's bundle, so neither the samplers nor the raw target's closing-snapshot choice (`odds_analytics.feature_groups.raw_closes`) see them. The retail-EV row builder merges sources per decision time and takes its baseline and close only from Betfair; `polymarket` enters a row only as the optional `polymarket_at_t` features, from the latest snapshot at or before t ([retail-EV features](../modelling/clv-target-functions.md#retail-ev-features)). `odds_core.bookmakers.NON_BETTABLE_BOOKMAKERS` lists `polymarket` with Pinnacle, so retail screening never reports it as a retail book ([retail edges](../agent/mcp-market-tools.md#retail-edges)), and `odds_core.bookmakers.ZERO_MARGIN_BOOKMAKERS` skips the validator's vig floor for it. Its key has no alias in the Polymarket entry of `odds_core.bookmakers.BOOKMAKER_ALIASES_BY_SOURCE`.

## History grid

History is the CLOB `/prices-history` midpoint for each market's Yes token ([research](https://github.com/Acusick1/odds/blob/research/polymarket-epl-price-history/research/polymarket-epl-price-history.md), observed behaviour, not documented):

- `interval=max` returns nothing for old markets, so `odds_lambda.polymarket_fetcher.PolymarketClient.get_price_series` requests explicit `startTs`/`endTs` windows of at most `odds_lambda.polymarket_fetcher.MAX_HISTORY_WINDOW` (14 days; 20 days is rejected) at fidelity 1 and merges them.
- The series is a one-minute grid with the last price carried forward, not trades.

`odds_lambda.polymarket_odds.resample` samples that series as-of onto a grid anchored on kick-off (`odds_lambda.polymarket_odds.grid_offsets`), from kick-off back to `odds_lambda.polymarket_odds.MAX_HORIZON` (7 days). The spacing is the fetch tier's step (`odds_lambda.polymarket_odds.TIER_STEPS`): 5 minutes in the closing tier (0-3 h), 15 minutes in pregame and sharp (3-24 h), 30 minutes in early (1-3 days), 1 hour in opening (3-7 days), about 313 points per match. Each tier's range holds multiples of its step, so 1, 3, 6, 12, 24, 72 and 168 hours are grid points. The live job's cadence uses the same steps, which makes live and historical snapshots comparable, and the grid is dense where most volume trades. Volume is mostly in the last 24 hours and the 1-minute source gives no reason to go finer than 5 minutes (inferred).

A grid time is dropped, and counted, when:

- any outcome has no observation within `odds_lambda.polymarket_odds.MAX_STALENESS` (30 minutes) at or before it, or a price is outside (0, 1) (`missing`);
- the three prices sum further than `odds_lambda.polymarket_odds.SUM_TOLERANCE` (0.03) from 1 (`off_sum`). The research found every match within 0.03 from 3 days out and 33 of 446 beyond it at 7 days, apparently placeholder quotes on empty books (inferred), so the filter mostly removes 7-day points.

No grid time after the earlier of the event's `commence_time` and Polymarket's start time is used, so a differing kick-off cannot leak in-play prices. A snapshot's tier is derived from its time against the event's `commence_time` (`odds_lambda.tier_utils.snapshot_tier`).

## Attaching to events

Polymarket creates no events. `odds_lambda.storage.event_resolver.attach_event` finds the existing event of a match (`odds_lambda.polymarket_odds.listed_game` builds the listing from the Polymarket event id, the canonical teams and Polymarket's start time) and records the Polymarket event id (`pm_event_id`) as the event's `polymarket` external id ([Event resolver](../storage/event-resolver.md#attaching-without-creating)). It takes the fixture lock, then matches by that id, else by sport and teams within ±2 h, the nearest event winning. The job attaches its matches as one batch (`odds_lambda.storage.event_resolver.attach_events`); the backfill attaches one at a time. A match with no event, or whose id is refused (a second Gamma match for a fixture whose event holds another Polymarket id), is counted as unattached and writes nothing; the refusal is logged once per event and id. An event for a match with none is created later by a source that creates events. 

## Backfill command

`odds polymarket backfill-odds` (`odds_cli.commands.polymarket.backfill_odds`, which calls `odds_lambda.polymarket_odds_backfill.backfill_sport`) takes `--sport` (default `soccer_epl`), `--since` (default 2025-08-01, the first EPL market), `--limit` and `--dry-run`.

- It lists closed events from Gamma and attaches each to an event of the database `DATABASE_URL` points at ([attaching](#attaching-to-events)). It uses the event only when it is `FINAL` and its kick-off is at or after `--since`.
- It skips an event (by its integer id) that already holds a history snapshot, found as a `polymarket` snapshot whose payload is its own envelope without the bid and ask keys (a live run's payload lists its envelopes, which always carry those keys; `odds_lambda.polymarket_odds_backfill._history_event_ids`), so a rerun adds nothing. It commits after each attachment and each event, and runs each attachment and each event's writes in a savepoint, so a match that fails is rolled back, recorded in the report's errors, and the others stand.
- It writes `odds_core.models.OddsSnapshot` rows through `odds_lambda.storage.shaped_snapshots.store_shaped_snapshot`: source `polymarket`, market `1x2`, time basis `observed` (the history is an as-of resample at our grid times), one `mid` price row per outcome, and its own payload holding its envelope ([writing price rows](../storage/price-tables.md#writing-price-rows)). A snapshot already stored is skipped and its payload discarded; the counts are of snapshots inserted.
- It prints the counts: matches, attached, skipped, written, snapshots, dropped points by reason, finished events without a match, matches without an event, and matches that failed. `--dry-run` fetches and resamples but writes nothing, not even the external ids.

> [!WARNING]
> The command writes to whatever `DATABASE_URL` points at. Production is loaded only from the production host ([databases](../foundations/databases-and-environments.md#environments)).

## The fetch-polymarket-odds job

`odds_lambda.jobs.fetch_polymarket_odds.main` stores the current midpoint of every open match within `odds_lambda.polymarket_odds.MAX_HORIZON` that attaches to an event. It is per sport (`fetch-polymarket-odds-epl`; `ctx.sport` is required and must be in the allowlist), allowlisted to `odds_lambda.polymarket_sports.MATCH_ODDS_SPORTS`, one of the default bootstrap jobs ([ADR-0013](../../docs/adr/0013-collect-polymarket-epl-as-sharp-reference.md)); it self-schedules, and a scheduler restart keeps its pending run ([where each job runs](../scheduling/job-deployment-matrix.md#where-each-job-runs)). It does nothing when `odds_core.config.PolymarketConfig.enabled` is false.

1. Pre-schedule with `odds_lambda.collection_calendar.decide_collection_run` (`stored_only`, timed from the events' `commence_time`) and `odds_lambda.jobs.fetch_polymarket_odds.CADENCE`, derived from `odds_lambda.polymarket_odds.TIER_STEPS`: 5 min closing, 15 min pregame and sharp, 30 min early, 1 h opening ([decision engine](../scheduling/decision-engine.md#cadence)). With no stored SCHEDULED event within 28 days the run is dormant: a 6 h check-in and its heartbeat, no Gamma call ([dormancy](../scheduling/decision-engine.md#dormancy)). The sport's overnight window applies, and the job passes no [leads](../scheduling/decision-engine.md#lead-runs).
2. Inside `odds_core.alerts.job_alert_context`, `odds_lambda.jobs.fetch_polymarket_odds.ingest_sport` lists open events, keeps matches that have not started, attaches them to events ([attaching](#attaching-to-events)), and stores a snapshot through `odds_lambda.storage.writers.OddsWriter.store_envelope` under the event's integer id with source `polymarket`, market `1x2` and time basis `observed`. Its price rows are `mid` (the midpoint), `back` (the ask) and `lay` (the bid) per outcome, all as decimal odds 1/p, and the run's envelopes are archived as one payload ([writing price rows](../storage/price-tables.md#writing-price-rows)). Each snapshot is written in its own savepoint, so a failing match is recorded in the run's errors and the others stand. A snapshot already stored for that time is skipped, and when nothing was inserted the run's payload is discarded; `snapshots_stored` counts inserts.
3. The price is the midpoint of Gamma's `bestBid` and `bestAsk` (a match with any midpoint outside (0, 1) has no prices), the same quantity as the history series (observed: `/midpoint` equals the last history point). A match is skipped and counted when its event has started (`snapshot_time >= commence_time`), when a bid or ask is missing, when any outcome's spread exceeds `odds_lambda.polymarket_odds.MAX_QUOTE_SPREAD` (0.03, an empty book), or when the prices are off the sum tolerance.
4. Each sport's run writes an `odds_core.models.FetchLog` row (`success` is whether a snapshot was stored), ([fetch logs](odds-ingestion-service.md#fetch-logs)).
5. Failures alert: a sport that raises, or that Gamma lists no upcoming match for while our database has SCHEDULED events within 5 days (`odds_lambda.jobs.fetch_polymarket_odds.LISTING_CHECK_HORIZON`, `odds_lambda.jobs.fetch_polymarket_odds.EmptyListingError`), is logged as a failed fetch, the job reschedules, and then it raises inside `odds_core.alerts.job_alert_context`, which sends the critical alert (once per outage), and skips the heartbeat. If matches were seen but none stored, `send_job_warning("job_empty:<job>")`. On the local scheduler a run past `odds_lambda.scheduling.runner.JOB_TIMEOUT` is cancelled and alerted, and a failed run that booked nothing is re-booked ([Local scheduler](../scheduling/local-scheduler.md#scheduling-primitives)).

Its `odds_lambda.jobs.fetch_polymarket_odds.SOURCE` feeds the per-source [stale-data check](../operations/health-monitor.md#stale-data-check), which ignores it until the job is in `bootstrap_jobs` and has stored a first `observed` snapshot under source `polymarket`. The job writes a heartbeat, expected every 11 hours (`fetch-polymarket-odds-epl` in `odds_core.config.AlertConfig.heartbeat_expectations`) to cover the 22:00-06:00 UTC overnight pause. The job attaches to existing events and never creates one, so a match the Odds API or OddsPortal has not yet synced is skipped until its event exists.

## Related

- [Polymarket market character](polymarket-market-research.md)
- [Odds snapshot tables](../storage/odds-tables.md)
- [Sharp benchmark](../modelling/sharp-benchmark.md)
