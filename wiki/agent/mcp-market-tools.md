---
type: reference
title: MCP odds, prediction and retail-edge tools
description: get_upcoming_fixtures, get_current_odds, get_odds_history, get_predictions and find_retail_edges, with inputs, anchor-relative price resolution, ranking and dispersion statistics, and the limits of their prices.
sources:
  - odds_mcp.server.get_upcoming_fixtures
  - odds_mcp.server.get_current_odds
  - odds_mcp.server.get_odds_history
  - odds_mcp.server.get_predictions
  - odds_mcp.server.find_retail_edges
  - odds_analytics.utils.per_book_market_holds
  - odds_lambda.storage.readers.OddsReader
  - odds_core.prediction_models.Prediction
status: stable
---

# MCP odds, prediction and retail-edge tools

Five read-only tools defined in `odds_mcp.server`, part of the [odds-mcp server](mcp-server.md). Unknown event ids return `{"error": "Event '<id>' not found"}`. The first four read stored data; `odds_mcp.server.find_retail_edges` is the [betting agent](betting-agent.md)'s flagship screener, about 290 lines.

## `get_upcoming_fixtures(league="soccer_epl", days_ahead=7)`

`SCHEDULED` events with `commence_time` between now and now plus `days_ahead` (clamped to 1-30), ordered by kickoff, then event ID, via `odds_lambda.storage.readers.OddsReader.get_events_by_date_range`. Each item: `id`, `sport_key`, `sport_title`, `home_team`, `away_team`, `commence_time`, status, scores and `completed_at`. Live events are excluded, so a fixture disappears once the update-status job marks it `LIVE`; that job must be running for the status to change ([status lifecycle](../collection/event-status-lifecycle.md)).

## `get_current_odds(event_id, market, include_raw_data=False, lookback_hours=2.0)`

Resolves the latest price per `(bookmaker_key, outcome_name, point)`:

1. Anchor on the newest snapshot that contains `market` (`get_latest_snapshot(market=...)`).
2. Walk snapshots in `[anchor_time - lookback_hours, anchor_time]` newest first; the first price seen for each tuple wins (`odds_lambda.storage.readers.OddsReader.get_latest_book_prices`). This lets a Betfair-only snapshot coexist with an older retail snapshot.

Response: `event` plus `snapshot` with `anchor_snapshot_id`, `snapshot_time` (newest contributing snapshot), `anchor_created_at`, `anchor_fetch_tier`, `anchor_hours_until_commence`, `bookmaker_count` (distinct books actually returned), `lookback_hours`, and `odds[]`. `book_age_seconds` and the lookback window are relative to the anchor snapshot, not the wall clock, so if scraping stops, ages still look small; compare `snapshot_time` with the current time to judge staleness. Per-book prices in one response can come from different snapshots, so they are not co-temporal; each row's `snapshot_time` says which. Each odds row carries `bookmaker_key`, `outcome_name`, `price` (derived American quote), `price_decimal` (the stored source price), `point`, `last_update` (the book's own timestamp, falling back to snapshot time), `snapshot_time` and `book_age_seconds`. With no snapshot, or no `market` odds in the window, `snapshot` is `null` with a `message`. `include_raw_data` adds the anchor's `raw_data`.

## `get_odds_history(event_id, market)`

Every snapshot for the event in time order, each with the extracted `market` odds instead of raw JSON, plus `snapshot_count`. There is no pagination or limit, so a long-lived event returns a large payload ([odds reader](../storage/odds-reader.md)).

## `get_predictions(event_id, limit=5, since_hours=None)`

Reads stored `predictions` rows newest first; it does not run the model. `total_matching` counts rows after the `since_hours` filter, `returned` the limited list (`limit` floored at 1). Each row: `id`, `snapshot_id`, `model_name`, `model_version`, `predicted_clv`, `created_at`. Rows are written only by inline scoring, which runs only on a local scheduler: one row per snapshot, with `model_version` an S3 ETag ([prediction scoring](../modelling/prediction-scoring.md), [model hub](../evaluation/clv-model-findings.md)). The tool returns whatever the scorer wrote; no row exists for events the scoring job never reached..

## Market vocabulary

`market` is a `MarketKey`: `h2h`, `1x2`, `totals` or `spreads`, matched exactly against the stored key. EPL OddsPortal and Betfair data is stored as `1x2`, MLB as `h2h` ([sport and market keys](../collection/sport-and-market-keys.md#market-key-dialects)). A key with no data yields an empty result, not an error.

## Retail edges

`odds_mcp.server.find_retail_edges` ranks retail books that price an outcome longer than the sharp reference ([sharp benchmark](../modelling/sharp-benchmark.md)). Books in `sharp_bookmakers` and in `odds_core.bookmakers.NON_BETTABLE_BOOKMAKERS` (Pinnacle and `polymarket`, which a UK bettor cannot use) are never retail rows. Exchanges are retail rows unless the sharp list names them.

Inputs: `event_id`, `market` (`h2h`, `1x2`, `totals`, `spreads`), `sharp_bookmakers` (list, JSON-array string or comma string; default `["pinnacle", "betfair_exchange"]`), `sharp_lookback_hours=2.0`, `lookback_hours=2.0` (retail).

Computation:

1. Anchor on the event's latest snapshot containing `market`; both windows end at that snapshot's time.
2. Sharp prices: `odds_lambda.storage.readers.OddsReader.get_sharp_prices`, newest first, per outcome, taking the first sharp book in priority order that has a price.
3. Retail prices: `odds_lambda.storage.readers.OddsReader.get_latest_book_prices`, the newest price per `(book, outcome, point)`. Any book outside the sharp list is retail.
4. Per `(outcome, point)` bucket: `divergence = retail_implied_prob - sharp_implied_prob` per book, with implied probabilities taken as `1 / price_decimal`; `median_divergence` and `dispersion_stddev` (population standard deviation, needs 2+ books); `z_score = (divergence - median) / stddev`, set only when the bucket has 3+ books and a nonzero stddev; `best_retail` (lowest implied probability) and `worst_retail`.
5. `market_hold` per book, only for `h2h` and `1x2`, from `odds_analytics.utils.per_book_market_holds` over decimal prices, which drops books missing any required outcome.

Output: `event`, `snapshot_time` (newest contributing retail snapshot), `sharp_bookmakers` (the input list, not the resolved books), `per_outcome[]`, and `retail_edges[]`: rows with negative divergence only, sorted by divergence ascending, then `z_score` ascending (null last), then book, outcome and point. Each row carries `price` (derived American quote), `price_decimal`, `snapshot_time` and `book_age_seconds`. Dispersion statistics mix prices from different snapshots, so weigh `book_age_seconds` when sizing a stake. Empty cases return `message` strings: no snapshots, no `market` odds in the window, or no retail books. The tool depends on `odds-analytics` only for `odds_analytics.utils.per_book_market_holds` ([workspace packages](../foundations/workspace-packages.md)).

Divergence is vig-inclusive on both sides, with no devigging ([devigging](../foundations/devigging.md)). A retail book with a larger margin looks worse on every outcome; use `market_hold` to separate margin from divergence. The same holds for the `sharp_prob` and `retail_sharp_diff` model features, and the three code copies of the sharp-price logic differ ([code copies](../modelling/sharp-benchmark.md#code-copies)).

Sharp prices are keyed by outcome only, while retail rows are bucketed by `(outcome, point)`, so on `totals` and `spreads` one sharp line is compared with retail prices at other lines (#439).

Under the default hybrid sharp list, outcomes resolve independently, so one outcome's sharp price can be Pinnacle and another's Betfair; read `sharp_snapshot_time` and `sharp_age_seconds` per row.

`required_outcomes` for the hold calculation is the set of outcomes present among retail odds, so an outcome no retail book carries is not required.

## Related

- [Sport context tools](mcp-sport-context-tools.md), [Match briefs](match-briefs.md), [Paper trading](paper-trading.md)
- [Odds reader](../storage/odds-reader.md)
