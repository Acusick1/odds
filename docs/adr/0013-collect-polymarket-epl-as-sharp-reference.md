---
status: accepted
---

# Collect Polymarket EPL prices as a sharp reference

Store Polymarket's EPL match prices as `polymarket` odds snapshots, historical and live. Polymarket stays closed to us as a trading venue. This supersedes [ADR-0007](0007-deprioritise-polymarket.md).

## Context

[ADR-0007](0007-deprioritise-polymarket.md) stopped extending the Polymarket pipeline. It rested on three records: thin EPL books ($10K-$100K per match, AMM-driven), UK trading blocked, and CLOB price history that expires after about 30 days. Under map #506 the EPL evidence was re-measured against the public Gamma and CLOB APIs on 2026-10-18, at no Odds API cost:

- **Liquidity (#509).** Since 2025-08 the median match trades about $0.9M before kick-off (range $0.33M-$4.0M over nine complete matches), mostly in the last 24 hours. Spreads are 1-2 cents. The earlier "thin" figure has no recorded method and could not be reproduced ([findings](https://github.com/Acusick1/odds/blob/research/polymarket-epl-liquidity-2026/research/polymarket-epl-liquidity-2026.md)).
- **History (#518).** `/prices-history` with explicit `startTs`/`endTs` windows of at most 14 days returned a one-minute series for all 449 finished matches back to 2025-08-15. `interval=max` is empty for old markets, which is probably what the 30-day retention observation had used (inferred). The three Yes prices sum to within 0.03 of 1 from 3 days out ([findings](https://github.com/Acusick1/odds/blob/research/polymarket-epl-price-history/research/polymarket-epl-price-history.md)).
- **Prices.** The series are midpoints and sit within 0.35-0.5 pp per outcome of Betfair Exchange from 3 days out to kick-off. That is not a CLV test and does not show independent information.
- **Access.** UK trading is still blocked, an external fact the repo cannot verify.

## Decision

Collect Polymarket EPL match prices as a sharp reference, not a trading venue.

- **Storage.** Prices are `polymarket` bookmaker entries on `odds_snapshots` rows, in the market key `1x2`, so they fit the existing snapshot model. The NBA tables and jobs are unchanged.
- **History.** `odds polymarket backfill-odds` loads finished matches from 2025-08 onward onto a kick-off-anchored grid.
- **Live.** The `fetch-polymarket-odds` job runs for sports that have three-way Polymarket match markets. It is one of the default bootstrap jobs.
- **Role.** `polymarket` is a reference, never a retail price. `odds_mcp.server.find_retail_edges` excludes `odds_core.bookmakers.NON_BETTABLE_BOOKMAKERS` (Pinnacle and Polymarket, which a UK bettor cannot use) and leaves exchanges to its `sharp_bookmakers` list, as before. Decision-row sampling ignores snapshots that hold only a prediction market.

## Considered options

| Choice | Taken | Rejected, and why |
|---|---|---|
| Where to store | `polymarket` entries on `odds_snapshots` (market key `1x2`) | The existing `polymarket_*` tables: they are NBA-shaped (two outcomes per market, an NBA ticker parser, manual event linking) and the feature pipeline and sharp-reference readers read `odds_snapshots` |
| History depth | 7 days, with a drop rule for prices off the sum | 3 days only: the sum rule is reliable from 3 days out and 7-day points fail it in about 7% of matches, but the retail rows of [ADR-0012](0012-retail-ev-against-betfair-exchange-close.md) run to 7 days out and the dropped points are counted, not hidden |
| Collection | Backfill and a live job | Backfill only: the live job's best bid and ask cannot be backfilled, and the live midpoint keeps the series current for the fresh evaluation season. The live job stores midpoints and best bid and ask, not order books |
| Exchanges in retail screening | Leave to `sharp_bookmakers` | Always exclude them: an exchange is bettable from the UK, so excluding it is the caller's choice. Only Pinnacle and Polymarket are excluded unconditionally |

## Consequences

- `fetch-polymarket-odds` is in `odds_core.config.SchedulerConfig.bootstrap_jobs`, which ends ADR-0007's "no Polymarket job runs by default" for EPL; the NBA jobs stay out. `odds_core.config.AlertConfig.heartbeat_expectations` holds a `fetch-polymarket-odds-epl` entry (11 h, covering the EPL overnight pause). Finished matches are backfilled from `/prices-history`, but the live best bid and ask exist only from the date the job first runs.
- The client is sport-agnostic through `odds_lambda.polymarket_sports.POLYMARKET_SPORTS`; the NBA ids moved out of `odds_core.config.PolymarketConfig`.
- Whether Polymarket adds information beyond Betfair Exchange is untested. A `polymarket` snapshot holds no other bookmaker, so using it as the sharp side of a decision row needs a pipeline that merges sources per decision time.
- History starts at 2025-08 and adds no earlier training range.
- Order books cannot be replayed, so only prices are stored.

## Evidence

- Issues: #509, #518, #519, map #506.
- Findings: the two research notes linked above, on branches `research/polymarket-epl-liquidity-2026` and `research/polymarket-epl-price-history`.
- Code and docs: `odds_lambda.polymarket_sports.POLYMARKET_SPORTS`, `odds_lambda.polymarket_odds`, [Polymarket odds snapshots](../../wiki/collection/polymarket-odds-snapshots.md).
