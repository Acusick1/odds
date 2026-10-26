---
type: reference
title: MCP odds and retail-edge tools
description: get_upcoming_fixtures, get_current_odds, get_odds_history and find_retail_edges, with inputs, anchor-relative price resolution, ranking and dispersion statistics, and the limits of their prices.
sources:
  - odds_mcp.server.get_upcoming_fixtures
  - odds_mcp.server.get_current_odds
  - odds_mcp.server.get_odds_history
  - odds_mcp.server.find_retail_edges
  - odds_mcp.prices.BOOK_SIDES
  - odds_mcp.prices.BookPrice
  - odds_mcp.prices.book_quotes
  - odds_mcp.prices.latest_priced_snapshot
  - odds_mcp.prices.latest_book_prices
  - odds_mcp.prices.resolve_sharp_prices
  - odds_mcp.prices.outcome_label
  - odds_core.quotes.quotes
  - odds_analytics.utils.per_book_market_holds
status: stable
---

# MCP odds and retail-edge tools

Four read-only market-data tools (none reads paper trades), part of the [odds-mcp server](mcp-server.md). Unknown event ids return `{"error": "Event '<id>' not found"}`. The first three read stored data; `odds_mcp.server.find_retail_edges` is the [betting agent](betting-agent.md)'s flagship screener, about 290 lines.

The tools read prices from the `prices` table through `odds_core.quotes.quotes` ([reading quotes](../storage/price-tables.md#reading-quotes)), in `odds_mcp.prices`, and treat the event id as opaque: every source's snapshots hang on the one event of a match ([event resolver](../storage/event-resolver.md)). A book's price is its `back` price, except Polymarket's, which is its `mid` (`odds_mcp.prices.BOOK_SIDES`, `odds_mcp.prices.book_quotes`); lay and traded prices are not surfaced. Outcomes are shown as the event's home and away team names, `Draw`, `Over` and `Under` (`odds_mcp.prices.outcome_label`), and `point` is the stored line. Bookmaker keys are the canonical keys the writers store ([bookmakers](../storage/price-tables.md#reading-an-envelope)), so the same book has one key across sources.

## `get_upcoming_fixtures(league="soccer_epl", days_ahead=7)`

`SCHEDULED` events with `commence_time` between now and now plus `days_ahead` (clamped to 1-30), ordered by kickoff, then event ID, via `odds_lambda.storage.readers.OddsReader.get_events_by_date_range`. Each item: `id`, `sport_key`, `sport_title`, `home_team`, `away_team`, `commence_time`, status, scores and `completed_at`. Live events are excluded, so a fixture disappears once the update-status job marks it `LIVE`; that job must be running for the status to change ([status lifecycle](../collection/event-status-lifecycle.md)).

## `get_current_odds(event_id, market, include_raw_data=False, lookback_hours=2.0)`

Resolves the latest price per `(bookmaker_key, outcome_name, point)`:

1. Anchor on the newest snapshot of `market` that holds a book price (`odds_mcp.prices.latest_priced_snapshot`, one query on `odds_snapshots` for a snapshot with a `back` or `mid` price row; time descending, lowest id first).
2. Take the quotes of snapshots in `[anchor_time - lookback_hours, anchor_time]`, walk them newest first, and keep the first price seen for each tuple (`odds_mcp.prices.latest_book_prices`). This lets a Betfair-only snapshot coexist with an older retail snapshot.

Response: `event` plus `snapshot` with `anchor_snapshot_id`, `snapshot_time` (newest contributing snapshot), `anchor_created_at`, `anchor_source` and `anchor_time_basis` (the anchor's `odds_core.vocabulary.Source` and `odds_core.vocabulary.TimeBasis`, which says what its time means), `anchor_hours_until_commence` (the event's kick-off minus the anchor time, negative once it has started), `bookmaker_count` (distinct books actually returned), `lookback_hours`, and `odds[]`. `book_age_seconds` and the lookback window are relative to the anchor snapshot, not the wall clock, so if scraping stops, ages still look small; compare `snapshot_time` with the current time to judge staleness. Per-book prices in one response can come from different snapshots, so they are not co-temporal; each row's `snapshot_time` says which. Each odds row carries `bookmaker_key`, `market_key`, `outcome_name`, `price` (derived American quote), `price_decimal` (the stored source price), `point`, `last_update` (the price's `changed_at`, falling back to snapshot time), `snapshot_time` and `book_age_seconds`. With no snapshot, or no `market` odds in the window, `snapshot` is `null` with a `message`.

## `get_odds_history(event_id, market)`

Every snapshot of `market` that holds a book price, in time order, each with its prices instead of raw JSON, plus `snapshot_count`. A snapshot carries `id`, `event_id`, `snapshot_time`, `created_at`, `bookmaker_count`, `source`, `time_basis`, `hours_until_commence` and `odds[]`. There is no pagination or limit, so a long-lived event returns a large payload ([odds reader](../storage/odds-reader.md)).

## Market vocabulary

`market` is a `MarketKey`: `h2h`, `1x2`, `totals` or `spreads`, matched exactly against the stored key. EPL OddsPortal and Betfair data is stored as `1x2`, MLB as `h2h` ([sport and market keys](../collection/sport-and-market-keys.md#market-key-dialects)). A key with no data yields an empty result, not an error.

## Retail edges

`odds_mcp.server.find_retail_edges` ranks retail books that price an outcome longer than the sharp reference ([sharp benchmark](../modelling/sharp-benchmark.md)). Books in `sharp_bookmakers` and in `odds_core.bookmakers.NON_BETTABLE_BOOKMAKERS` (Pinnacle and `polymarket`, which a UK bettor cannot use) are never retail rows. Exchanges are retail rows unless the sharp list names them.

Inputs: `event_id`, `market` (`h2h`, `1x2`, `totals`, `spreads`), `sharp_bookmakers` (list, JSON-array string or comma string; default `["pinnacle", "betfair_exchange"]`), `sharp_lookback_hours=2.0`, `lookback_hours=2.0` (retail).

Computation:

1. Anchor on the event's latest snapshot of `market` holding a book price; both windows end at that snapshot's time.
2. Sharp prices: `odds_mcp.prices.resolve_sharp_prices`, newest first, per outcome, taking the first sharp book in priority order that has a price.
3. Retail prices: `odds_mcp.prices.latest_book_prices`, the newest price per `(book, outcome, point)`. Any book outside the sharp list is retail.
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
