---
status: proposed
---

# Give the live close no fallback from the other Betfair Exchange feeds

The live close stays the Betfair job's price, then FDUK's closing price, as [ADR-0012](0012-retail-ev-against-betfair-exchange-close.md) orders them. Neither OddsPortal's Betfair Exchange row nor the Odds API's `betfair_exchange` price is added between them. This adds to ADR-0012 and does not supersede it.

## Context

ADR-0012's live close is the last Betfair job snapshot that is not in-play within 30 minutes before kick-off (`odds_core.close.is_live_close`), within the [ADR-0014](0014-ignore-wide-exchange-prices.md) overround cap, else the FDUK Betfair Exchange closing price. FDUK has `1x2` only and publishes days after the match. If the Betfair job stopped for good, `1x2` scoring would wait on FDUK and totals would have no close (#735).

The Betfair Exchange price is also collected two other ways:

- the Betfair row in each OddsPortal live snapshot, on `1x2` and over/under 2.5;
- the Odds API's `betfair_ex_uk` on the `fetch-odds` job, on `h2h` (stored as `1x2`), with no totals (#707).

#735 set the test: on live EPL matches with a Betfair job close, take each alternative's last price before kick-off and apply #730's thresholds to `Δp = p_alternative − p_live`, both devigged proportionally. Every outcome must have `|mean Δp| ≤ 0.3 pp` and `sd ≤ 1.0 pp`. Every probability band with at least 50 rows must have `|mean Δp| ≤ 0.5 pp`. A band under 50 rows is inconclusive.

## Decision

- **No alternative feed becomes a fallback.** Both fail #730's thresholds on every market they cover, whatever the age cap.
- **The live close order is unchanged:** the Betfair job, else FDUK (ADR-0012), else Polymarket for NBA `h2h` ([ADR-0019](0019-nba-h2h-close-betfair-with-polymarket-fallback.md)).
- **The candidate report keeps reporting the source of each close** (`odds_analytics.candidate_report.ScoredBet.close_source`). No new source can appear in it.

## Considered options

- **OddsPortal's Betfair row as a fallback on `1x2`.**
  - Its home and away sd are 1.5 to 2.7 pp under every age cap, against the 1.0 pp limit.
  - Within 10 minutes of kick-off, 10 of its 37 matches differ from the live close by more than 2 pp on some outcome. Its price is observed about 1 minute after the live close in those matches. So the gap is not price movement; OddsPortal's Betfair row lags the exchange (inferred).
- **OddsPortal's Betfair row as the totals close.** Totals have no fallback at all, so a looser bar could be argued for. But its mean is −0.56 pp on over and +0.56 pp on under, outside the 0.3 pp limit, with sd 1.47 pp, on 30 matches. Both bands are inconclusive, under 50 rows.
- **The Odds API's `betfair_exchange` price on `1x2`.** `fetch-odds` runs every 2 hours within 3 hours of an EPL kick-off (`odds_lambda.jobs.fetch_odds.CADENCE`), so it rarely fetches near kick-off. None of 149 matches has a price within 10 minutes and 6 within 30 minutes. Its last price is a median 70 minutes out, and its home and away sd exceed 1.6 pp.
- **A fallback that passes only on trimmed data.** Dropping matches more than 2 pp apart brings OddsPortal's `1x2` sd under 1.0 pp. The trim was chosen after seeing the data, so it cannot decide the test.

## Consequences

- If the Betfair job stops for good, EPL `1x2` scoring waits up to 7 days for FDUK (`odds_core.close.FDUK_CLOSE`), and totals have no close.
- The [backtest substitute close](0023-odds-api-betfair-substitute-close-for-epl-backtests.md) is not affected. It uses the Odds API's historical (`source_stamped`) snapshots, which the test did not read.
- The comparison can be re-run as matches accumulate. A pass would be a new ADR superseding this one.

## Evidence

- Issues: #735 (scope), #730 (thresholds), #707 (no Odds API Betfair totals).
- Comparison: `uv run python scripts/compare_live_close_alternatives.py --to 2026-11-28`, read-only, on production data on 2026-11-28 ([live-close alternatives comparison](../../wiki/evaluation/live-close-alternatives-comparison.md)). There are 152 FINAL EPL matches from 2026-05-01, 149 with a `1x2` live close and 31 with a totals live close.

  Each cell is `mean Δp / sd Δp` in pp and n, then the verdict. A band is bounded by the live close probability.

  | Source, market, age cap | home or over | draw | away or under | p < 0.25 | 0.25–0.5 | p ≥ 0.5 | matched |
  |---|---|---|---|---|---|---|---|
  | OddsPortal `1x2`, none | −0.16 / 2.68, fail | +0.10 / 1.10, fail | +0.06 / 2.33, fail | +0.63 (93), fail | −0.05 (137), pass | −1.13 (46), inconclusive | 92 of 149 |
  | OddsPortal `1x2`, 30 min | −0.33 / 1.66, fail | +0.14 / 0.70, pass | +0.18 / 1.58, fail | +0.34 (65), pass | −0.04 (103), pass | −0.53 (33), inconclusive | 67 of 149 |
  | OddsPortal `1x2`, 10 min | −0.27 / 1.82, fail | +0.22 / 0.64, pass | +0.04 / 1.69, fail | +0.41 (32), inconclusive | −0.06 (64), pass | −0.64 (15), inconclusive | 37 of 149 |
  | OddsPortal totals 2.5, 10 or 30 min | −0.56 / 1.47, fail | | +0.56 / 1.47, fail | none | +0.70 (30), inconclusive | −0.70 (30), inconclusive | 30 of 31 |
  | Odds API `1x2`, none | −0.41 / 1.73, fail | +0.41 / 0.75, fail | −0.00 / 1.64, fail | +0.49 (40), inconclusive | −0.06 (68), pass | −0.89 (18), inconclusive | 42 of 149 |
  | Odds API `1x2`, 30 min | −0.45 / 1.13, fail | +0.30 / 0.37, fail | +0.14 / 0.96, pass | +0.32 (10), inconclusive | −0.00 (3), inconclusive | −0.64 (5), inconclusive | 6 of 149 |

- Time before kick-off of the prices used, with no cap:
  - OddsPortal `1x2`: median 15.8 minutes. 30 of the 149 matches (20.1%) have a price within 5 minutes and 37 (24.8%) within 10 minutes.
  - OddsPortal totals: median 4.4 minutes. 30 of 31 are within 10 minutes.
  - Odds API: median 70.0 minutes, none within 10 minutes.
- With a 10-minute cap, the Odds API has no price for any match.
