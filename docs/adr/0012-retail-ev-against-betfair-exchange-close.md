---
status: accepted
---

# Predict a retail bet's EV against the Betfair Exchange close

The next EPL CLV model predicts the expected value of a bet at a UK retail price, measured against the Betfair Exchange closing price. It replaces the bet365 close-minus-snapshot delta of [ADR-0002](0002-bet365-clv-target.md) and [ADR-0011](0011-bet365-clv-target-on-like-for-like-evidence.md).

## Context

The bet365 own-movement target failed its keep test, and the model was suspended (map #494; [EPL CLV re-baseline](../../wiki/evaluation/exp-epl-clv-rebaseline.md)). That target predicts how far one book's line moves. It does not measure the thing a bet earns, which is the gap between the price taken and the fair price at the close. Bets can be placed at any time before kick-off, so the decision time is an input, not a fixed horizon (map #506).

## Decision

Agreed 2026-10-18 (#515, under the map #506):

- **Rows.** One row per event, outcome (home, draw and away), UK retail book, and retail snapshot at time t, from 7 days before kick-off up to kick-off. All UK retail books we collect are included, because the user can bet with each of them. Hours to kick-off, book and outcome are features.
- **Target.** `EV = p_close × retail_decimal_odds(t) − 1`. Retail odds are used raw, margin included, because that is the price a bet gets. `p_close` is the Betfair Exchange price at the last snapshot before kick-off, devigged proportionally across the three back prices (`odds_core.odds_math.devig_probabilities`). Rows without a Betfair Exchange close are dropped.
- **Weights.** Each row's loss weight is `(1 / retail_odds²) × w_event`. The first factor makes the squared error a squared probability error, so longshots don't dominate. `w_event` gives each event's rows a total weight of 1, so densely scraped events don't dominate either.
- **Baseline.** `p_BFE_fair(t) × retail_odds(t) − 1`, where `p_BFE_fair(t)` is the devigged Betfair Exchange price at or before t. This assumes the exchange's current price is its close, and has no parameters.
- **Keep bar** for a pre-registered fresh season, which has to pass both conditions:
  1. Weighted out-of-sample MSE beats the baseline's, with an event-clustered bootstrap 95% CI on the gain that excludes zero.
  2. Bets with predicted EV > 0 realise a higher mean EV against the Betfair Exchange close than the bets the baseline picks.

## Considered options

- **Target.**
  - The probability gap `p_close − 1/odds` has the same sign as EV, but it is not in return units.
  - Retail devigged, `p_close − p_retail_fair`, measures mispricing net of margin, not bettable value.
- **Close reference.**
  - Pinnacle with a Betfair Exchange fallback, as in [ADR-0003](0003-hybrid-sharp-reference.md), would switch reference mid-history, because FDUK Pinnacle ends 2026-01-08.
  - An average of sharp prices would change its mix by season.
  - Betfair Exchange has continuous coverage: FDUK closing prices from 2024-25, plus the live Betfair job.
- **Retail rows.** Taking only the best price across books biases the target upward, because it is a maximum over noisy books.
- **Horizons.**
  - A fixed horizon grid discards data, and FDUK history has only two points per event.
  - Unweighted snapshots let events scraped every few minutes dominate the loss.
- **Weights.**
  - A `1/odds` weight only halves the longshot dominance.
  - Unweighted EV MSE is dominated by longshots.
- **Keep bar.** An MSE test alone ignores whether the picks pay. A betting test alone is noisier.
- **Devig.** Power or Shin devigging barely changes the result at an exchange overround of about 0.5%. A back–lay midpoint exists only in the live Betfair data, not in FDUK history.

## Consequences

- The training pipeline needs a new target type, three outcomes per event (`FeatureConfig.outcome` is home or away today), row weights, and retail-book rows. None of this exists yet.
- The sharp *features* may still use other sharp sources, such as Pinnacle or Polymarket at t. Only the target's close is fixed to Betfair Exchange.
- Historical retail prices at arbitrary horizons are scarce. FDUK has an opening price (1–4 days out) and a closing price per book, and live collection supplies the rest. So training rows are dense only where live collection ran.
- ADR-0002 and ADR-0011 are superseded. Their target stays in the code until the new one is built.

## Evidence

- Issues #494, #506, #515.
- [EPL CLV re-baseline](../../wiki/evaluation/exp-epl-clv-rebaseline.md), [Pinnacle against bet365](../../wiki/evaluation/exp-epl-clv-pinnacle-retest.md).
- FDUK coverage research (#511), Polymarket price-history research (#518).
