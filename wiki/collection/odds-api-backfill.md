---
type: reference
title: Odds API historical backfill
description: "Acquisition manifests, the causal call planner, the executor that runs its calls, odds discover and odds backfill."
sources:
  - odds_cli.commands.discover
  - odds_cli.commands.backfill
  - odds_analytics.backfill_plan
  - backfills/odds-api-epl-2020-26.yaml
  - odds_analytics.backfill_executor.BackfillExecutor
  - odds_analytics.backfill_executor.BackfillExecutor.execute
  - odds_lambda.data_fetcher.TheOddsAPIClient.get_historical_odds
status: stable
---

# Odds API historical backfill

Fills in historical Odds API snapshots (the paid `/historical/odds` endpoint, 10 units per region per market per call; the free tier has no historical endpoints; [odds-api-client](odds-api-client.md#cost-per-call)) and final scores. A replay backtest needs FINAL events plus snapshots before decision time (`odds_analytics.candidate_backtest.final_events` selects the events; [replay backtest](../evaluation/candidate-backtest.md)). Code: `packages/odds-cli/odds_cli/commands/{discover,backfill}.py` with logic in odds-analytics. The steps are on [howto-backfill-odds-api-history](howto-backfill-odds-api-history.md). It is separate from the EPL scripts ([historical-ingest-scripts](historical-ingest-scripts.md)); neither is scheduled.

## Acquisition manifests

A manifest in `backfills/` fixes what a historical purchase requests, validated by `odds_analytics.backfill_plan.Manifest`. A frozen manifest is never edited; a top-up is a new manifest. `backfills/odds-api-epl-2020-26.yaml` holds:

- **Content:** the sport, the seasons as disjoint `[start, end)` kick-off ranges, the markets in the Odds API's keys (`h2h`, and `totals` with its line), each once, and either named bookmakers or regions, not both.
- **Coverage bands:** each band (`odds_analytics.backfill_plan.Band`) is a range of leads before kick-off in whole minutes, from `min_lead_minutes` to `max_lead_minutes` inclusive, with a max input age. It promises that a decision at any whole minute in the range finds a snapshot at most the max age old, so a candidate deciding at any of those leads with at least that max input age has inputs, since it reads the latest snapshot in `[t - max age, t]` at decision time `t`, both ends inclusive (`odds_lambda.candidate_rules.market_state.load_market_state`). Bands do not overlap.
- **Snapshot grid:** steps by date, each a snapshot interval from its start until the next step's start (10 minutes from 2020-06-06, 5 from 2022-09-01).
- **Probe:** the snapshot times to request before the full run and what each must return: fixtures at least `min_fixture_lead_hours` ahead, the `required_bookmakers` on each market, and an `x-requests-last` of `units_per_call`. Validation rejects a lead shorter than the earliest band's farthest lead plus its max age, a units figure that differs from the manifest's, and required books or markets the manifest does not request. The probe also checks that the snapshot interval matches the grid.

Granularity and billing are the Odds API's documented values until the probe confirms them ([odds-api-client](odds-api-client.md#cost-per-call)).

## Planning calls

`odds_analytics.backfill_plan.plan` takes a manifest and fixtures and returns the fewest calls, each a `odds_analytics.backfill_plan.Call` with its time and the fixture bands it serves. One call returns the whole league, so the planner works on decision times, not fixtures: it merges every fixture's bands into stretches of time, each needing the strictest max age of the bands that cover it. Walking forward through decision minutes, at the first one the last call is too old for, it adds a call at the latest grid snapshot at or before that minute (`odds_analytics.backfill_plan.Manifest.floor_to_grid`). Every decision minute therefore has a call at most its max age old, and no fewer grid calls achieve that. A call serves a fixture's band (`odds_analytics.backfill_plan.targets`) when it lies between the band's farthest decision less its max age and its nearest decision. A max age with no grid snapshot inside it, or a decision before the first snapshot, raises `ValueError`.

`odds_analytics.backfill_plan.load_fixtures` reads the season's FINAL events of the manifest's sport.

A call costs `odds_analytics.backfill_plan.Manifest.units_per_call`: 10 units per market per region, where each group of up to 10 named bookmakers counts as one region.

## Commands

- `odds discover games --start D --end D` calls `odds_lambda.data_fetcher.TheOddsAPIClient.get_historical_events` per day and resolves each listed game to its event and creates a SCHEDULED one only when none matches (`odds_cli.commands.discover`; [Odds API events](../storage/event-matching.md#odds-api-events)). The default window is yesterday to today. Events already in the database, stored by any source, keep their id, status, scores, teams, MLB `gamePk` and acknowledgements, and their kick-off once it has passed, since the [kick-off authority](../storage/kickoff-authority.md) refuses a start at or before now after the stored kick-off ([events](../storage/events.md#writing-events)). It prints the events inserted, updated and dropped, and commits every 100 games.
- `odds backfill plan --manifest M [--season S ...] [--cost]` plans the manifest's calls for each season (`odds_cli.commands.backfill.plan_backfill`) and makes no API call. It prints each call's time and the number of fixture bands it serves. With `--cost` it prints a table of calls and units per season: one row for all bands, then a row per band with the calls that band adds to all the others, so band rows do not sum to the total.
- `odds backfill execute --manifest M [--season S ...] [--dry-run] [--no-skip-existing]` plans each selected season and runs the calls with `odds_analytics.backfill_executor.BackfillExecutor` (`odds_cli.commands.backfill.execute_backfill`). It prints the call count, the units that many calls cost (`odds_analytics.backfill_plan.Manifest.units_per_call` each) and the key's balance (or that it is unknown), asks for confirmation unless `--dry-run`, shows a progress bar and ends with a summary table. See [the executor](#executing-calls).
- `odds backfill scores --start D --end D [--dry-run]` fills missing scores from `nba_team_game_logs`, and sets FINAL ([event-status-lifecycle](event-status-lifecycle.md)). Home/away comes from the matchup string (`vs.` home, `@` away). The game logs come from `odds nba-stats fetch` ([nba-injuries-and-stats](nba-injuries-and-stats.md)).
- `odds backfill status` prints counts of events with snapshots and the earliest/latest snapshot time. It counts all `odds_snapshots`, not only backfilled ones.

## Executing calls

`odds_analytics.backfill_executor.BackfillExecutor.execute` runs a plan's calls in order. The request comes from the manifest: its sport (whose key pays), its markets and its bookmakers or regions, whichever it names (`odds_lambda.data_fetcher.TheOddsAPIClient.get_historical_odds` sends only that one). For each call:

- **Skip.** Unless `skip_existing` is off, a call is skipped when every fixture it serves already has an `odds_api` snapshot of each requested market (soccer's `h2h` stored as `1x2`) in `(call time - grid step, call time]`, the snapshot the call would return (`odds_analytics.backfill_plan.Manifest.grid_step_at`). The skip makes a rerun after a failure cheap.
- **Dry run.** A dry run makes no API call and writes nothing; it counts the calls that would be made.
- **Fetch.** One request at the call's time. The response carries the provider's own snapshot time, the latest at or before the requested time, and the call's cost from `x-requests-last` (`odds_core.api_models.HistoricalOddsResponse`).
- **Store.** Every returned game that resolves to a stored event is stored, whether or not the plan names it, with source `odds_api`, time basis `observed` and the provider's snapshot time, the same meaning as a live fetch ([Writing price rows](../storage/price-tables.md#writing-price-rows)). The games resolve in fixture-lock order (`odds_lambda.live_events.lock_order`), whatever order the API lists them in, through a finder built from the response's games, so an MLB call reads the MLB schedule once ([Odds API events](../storage/event-matching.md#odds-api-events)); a dropped game is collected and reported once at the end through `odds_lambda.live_events.alert_dropped_games`.
- **Log.** Each call writes a `odds_core.models.FetchLog` row with the key's remaining balance and used credits from `x-requests-remaining` and `x-requests-used`, which `odds_lambda.storage.readers.OddsReader.get_latest_quota_by_key` then sees. A call's rows and fetch log are committed together. A failed call rolls back and writes a failed fetch log in its own transaction, with the last balance the client saw, so the balance survives. The call's cost (`x-requests-last`) is not in `fetch_logs`; it is in the structured log (`units_used`) and summed in the run's result.
- **Pace.** The run sleeps 1 s after each paid call.

Any exception from a call stops the run after it is reported, since every call is paid. The result counts calls made, skipped and would-be-made, the units used (the sum of the responses' `x-requests-last`), the fixtures and snapshots stored, the dropped games and the last quota remaining (`odds_analytics.backfill_executor.BackfillResult`).

## Limits

- The manifest planner refuses a kick-off between whole minutes (`odds_analytics.backfill_plan.Fixture`), since it steps decisions a whole minute at a time. No stored EPL kick-off has seconds; 115 NBA kick-offs do (production, 2026-10-31), so an NBA manifest fails until they are handled.
- The backfill's calls spend the key of the sport the manifest names, so an NBA manifest spends NBA's key (index 1), the key that is also `fetch-odds-nba`'s ([keys](odds-api-client.md#retry-and-keys)).
- `basketball_nba` is still literal in `odds backfill scores` (`odds_cli.commands.backfill`, which reads game logs and spends no units). `odds discover games` requires `--sport`. `odds backfill plan` and `odds backfill execute` take the manifest's sport, and `odds backfill plan --cost` and `odds backfill execute` print that sport's key balance (the key's newest recorded quota, reported as unknown when no recent fetch recorded one).
- Historical responses for soccer carry the three-way market as `1x2`, relabelled, and bookmaker keys renamed to pipeline keys, by `odds_lambda.data_fetcher.TheOddsAPIClient.get_historical_odds` ([sport-and-market-keys](sport-and-market-keys.md#market-key-dialects)).
- Final scores are not available from the Odds API for games older than 3 days (`odds_cli.commands.discover` docstring), hence `backfill scores`.
- `odds validate` reports which fetch tiers a game's snapshots cover (`odds_lambda.storage.tier_validator.TierCoverageValidator`, whose discovery check asks `/scores` for the sport's games on a day of the league's calendar; [fetch-tier](../storage/fetch-tier.md#the-tier-a-snapshot-covers)).

## Related

- [Backfill Odds API history](howto-backfill-odds-api-history.md)
- [Odds API quota accounting](odds-api-quota.md)
- [Results and status jobs](results-jobs.md)
