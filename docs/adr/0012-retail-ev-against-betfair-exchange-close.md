---
status: accepted
---

# Predict a retail bet's EV against the Betfair Exchange close

The keep bar below no longer decides whether a model is kept; [Goal and strategy](../../wiki/foundations/goal-and-strategy.md) does.

The next EPL CLV model predicts the expected value of a bet at a UK retail price, measured against the Betfair Exchange closing price. It replaces the bet365 close-minus-snapshot delta of [ADR-0002](0002-bet365-clv-target.md) and [ADR-0011](0011-bet365-clv-target-on-like-for-like-evidence.md).

## Context

The bet365 own-movement target failed its keep test, and the model was suspended (map #494; [EPL CLV re-baseline](https://github.com/Acusick1/odds/blob/1ca09fe/wiki/evaluation/exp-epl-clv-rebaseline.md)). ADR-0011 was only ever proposed. That target predicts how far one book's line moves. It does not measure the thing a bet earns, which is the gap between the price taken and the fair price at the close. Bets can be placed at any time before kick-off, so the decision time is an input, not a fixed horizon (map #506).

## Decision

Agreed 2026-10-18 (#515, under the map #506):

- **Rows.**
  - There is one row per event, outcome (home, draw and away), UK retail book, and retail snapshot at time t.
  - t runs from 7 days before kick-off up to 5 minutes before it. FDUK closing-price rows are excluded, because at the close the baseline equals the target.
  - Every UK retail book we collect is included, because the user can bet with each of them.
  - Hours to kick-off, book and outcome are features.
- **Close.**
  - `p_close` is the Betfair Exchange back price, devigged proportionally across the three outcomes (`odds_core.odds_math.devig_probabilities`).
  - It is taken from the last live Betfair job snapshot that is not in-play and falls within 30 minutes before kick-off. Where there is none, the FDUK closing price is used.
  - The FDUK closing price is treated as taken at kick-off, so it is never an "at or before t" price for any row.
  - Rows without a close are dropped.
- **Target.**
  - `EV = p_close × retail_decimal_odds(t) − 1`.
  - Retail odds are used raw, margin included, because that is the price a bet gets.
- **Baseline.**
  - `p_BFE_fair(t) × retail_odds(t) − 1`. `p_BFE_fair(t)` is the devigged Betfair Exchange price from the latest live Betfair job or FDUK opening snapshot at or before t, so the exchange's current price stands in for its close. The baseline has no parameters.
  - A row is dropped if that snapshot is more than 2 hours older than t, or if there is none. The model and the baseline are scored on the same rows.
- **Weights.**
  - Each row's loss weight is `(1 / retail_odds²) × w_event`, with `w_event = 1 / (rows in the event)`.
  - The first factor makes the squared EV error a squared probability error, because retail odds are known at t.
  - The second factor stops densely scraped events dominating.
- **Keep bar.** A model is kept only if it passes both conditions on a fresh season. Pre-registration names the season and its freeze date.
  1. Its weighted out-of-sample MSE beats the baseline's, with an event-clustered bootstrap 95% CI on the gain that excludes zero.
  2. Its picks earn more than the baseline's. Take the total `w_event`-weighted realised EV, against `p_close`, over rows with predicted EV > 0. The model's total minus the baseline's must have an event-clustered bootstrap 95% CI that excludes zero. Totals stay defined when the two pick sets differ in size or when one of them is empty.

## Considered options

- **Target.**
  - The probability gap `p_close − 1/odds` has the same sign as EV, but it is not in return units.
  - Retail devigged, `p_close − p_retail_fair`, measures mispricing net of margin, not bettable value.
- **Close reference.**
  - Pinnacle with a Betfair Exchange fallback, as in [ADR-0003](0003-hybrid-sharp-reference.md), would switch reference mid-history, because FDUK Pinnacle ends 2026-01-08.
  - An average of sharp prices would change its mix by season.
  - Betfair Exchange has continuous coverage: FDUK closing prices from 2024-25, plus the live Betfair job.
- **Retail rows.** Taking only the best price across books would bias the target upward, because it is a maximum over noisy books (inferred).
- **Horizons.**
  - A fixed horizon grid discards data, and FDUK history has only two points per event.
  - Unweighted snapshots let events scraped every few minutes dominate the loss.
- **Weights.**
  - A `1/odds` weight still leaves the squared error scaled by odds, so longshots would count more.
  - Unweighted EV MSE scales the squared error by odds², so longshots would dominate.
- **Keep bar.**
  - An MSE test alone ignores whether the picks pay.
  - A betting test alone is noisier (inferred).
  - Comparing mean EV at a matched number of picks was rejected. Totals need no choice of N.
- **Stale exchange prices.**
  - Falling back to Polymarket keeps more rows, but the baseline would then mix two sources.
  - With no age cap, the baseline would weaken on old rows and flatter the model.
- **Devig.** Power or Shin devigging should barely change the result at an exchange overround of about 0.5% (inferred). A back–lay midpoint exists only in the live Betfair data, not in FDUK history.

## Consequences

- The training pipeline needs a new target type, three outcomes per event (`FeatureConfig.outcome` has no draw), row weights, and retail-book rows. None of this exists yet.
- The sharp *features* may still use other sharp sources, such as Pinnacle or Polymarket at t. Only the target's close is fixed to Betfair Exchange.
- Training rows exist only from 2024-25 onward, where Betfair Exchange closing prices exist.
- Before the live Betfair job, the 2-hour cap keeps few OddsPortal rows.
- The hours-to-kick-off feature needs true snapshot times. FDUK opening prices are stamped 24 h out, but are really taken 1–4 days out (#516).
- Historical retail prices at arbitrary horizons are scarce. FDUK has an opening price (1–4 days out) and a closing price per book, and live collection supplies the rest. So training rows are dense only where live collection ran.
- ADR-0002 and ADR-0011 are superseded. Their target stays in the code until the new one is built.

## Evidence

- Issues #494, #506, #515.
- [EPL CLV re-baseline](https://github.com/Acusick1/odds/blob/1ca09fe/wiki/evaluation/exp-epl-clv-rebaseline.md), [Pinnacle against bet365](https://github.com/Acusick1/odds/blob/1ca09fe/wiki/evaluation/exp-epl-clv-pinnacle-retest.md).
- FDUK coverage research (#511), Polymarket price-history research (#518).
