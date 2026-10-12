---
type: component
title: find_retail_edges tool
summary: "The agent's retail-vs-sharp screener: inputs, ranking, dispersion statistics, vig-inclusive divergence and its dependence on the hold maths."
sources:
  - packages/odds-mcp/odds_mcp/server.py::find_retail_edges
  - packages/odds-analytics/odds_analytics/utils.py::per_book_market_holds
  - packages/odds-lambda/odds_lambda/storage/readers.py::OddsReader
  - docs/BETTING_AGENT.md#mcp-tools
verified_at: 7887039
---

# find_retail_edges tool

`packages/odds-mcp/odds_mcp/server.py::find_retail_edges` is the [betting agent](betting-agent.md)'s flagship screener, about 290 lines in the [odds-mcp server](mcp-server.md). It ranks retail books that price an outcome longer than the sharp reference ([definition](retail-sharp-divergence.md)).

## Inputs

`event_id`, `market` (`h2h`, `1x2`, `totals`, `spreads`), `sharp_bookmakers` (list, JSON-array string or comma string; default `["pinnacle", "betfair_exchange"]`), `sharp_lookback_hours=2.0`, `lookback_hours=2.0` (retail).

## Computation

1. Anchor on the event's latest snapshot containing `market`; both windows end at that snapshot's time.
2. Sharp prices: `OddsReader.get_sharp_prices`, newest first, per outcome, taking the first sharp book in priority order that has a price.
3. Retail prices: `OddsReader.get_latest_book_prices`, the newest price per `(book, outcome, point)`. Any book not in the sharp list is retail.
4. Per `(outcome, point)` bucket: `divergence = retail_implied_prob - sharp_implied_prob` per book; `median_divergence` and `dispersion_stddev` (population standard deviation, needs 2+ books); `z_score = (divergence - median) / stddev`, set only when the bucket has 3+ books and a nonzero stddev; `best_retail` (lowest implied probability) and `worst_retail`.
5. `market_hold` per book, only for `h2h` and `1x2`, from `per_book_market_holds`, which drops books missing any required outcome.

## Output

`event`, `snapshot_time` (newest contributing retail snapshot), `sharp_bookmakers` (the input list, not the resolved books), `per_outcome[]`, and `retail_edges[]`: rows with negative divergence only, sorted by divergence ascending, then `z_score` ascending (null last), then book, outcome and point. Each row carries `snapshot_time` and `book_age_seconds` ([provenance](mcp-market-data-tools.md)). Empty cases return `message` strings: no snapshots, no `market` odds in the window, or no retail books.

## Hazards

- Divergence is vig-inclusive on both sides and the docstring does not say so ([Devigging](devigging.md)). A retail book with a larger margin looks worse on every outcome; use `market_hold`.
- Sharp prices are keyed by outcome name only, while retail rows are bucketed by `(outcome, point)`. On `totals` and `spreads` with several lines, one sharp price is compared with retail prices at other lines.
- Under the default hybrid list, outcomes resolve independently, so one outcome's sharp price can be Pinnacle and another's Betfair; read `sharp_snapshot_time` and `sharp_age_seconds` per row.
- Dispersion stats mix non-co-temporal prices; weigh `book_age_seconds` when sizing a stake.
- `required_outcomes` for the hold calculation is the set of outcomes present among retail odds, so an outcome no retail book carries is not required.
- The module depends on `odds-analytics` only for `per_book_market_holds` ([odds-analytics](package-odds-analytics.md)).
- The statistics are inline in the tool, so they are only testable through the whole tool.

## Related

- [Sharp benchmark](sharp-benchmark.md)
- [Odds reader](odds-reader.md)
