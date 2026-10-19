---
type: explanation
title: CLV and the training target
description: What CLV means here, the retail-EV target against the Betfair Exchange close and its baseline and weights, the older close-minus-snapshot delta, and how sampling changes a target.
sources:
  - odds_analytics.retail_ev.build_event_rows
  - odds_analytics.retail_ev.RetailEvRow
  - odds_analytics.retail_ev.RetailEvDataset
  - odds_analytics.retail_ev_metrics
  - experiments/scripts/retail_ev_exchange_overround.py
  - odds_analytics.sequence_loader.calculate_devigged_bookmaker_target
status: stable
---

# CLV and the training target

CLV (closing line value) is how much better a price is than the fair price at the close. The training pipeline has two kinds of target. `target_type: retail_ev` predicts what a bet at a UK retail price is worth against the devigged Betfair Exchange close, as [ADR-0012](../../docs/adr/0012-retail-ev-against-betfair-exchange-close.md) decides. The older targets, `devigged_bookmaker` and `raw`, regress how far one line moves between a decision snapshot and the close; the suspended EPL model uses the first. The functions, dispatch and config fields are on [Target functions](clv-target-functions.md).

## The retail-EV target

ADR-0012 measures what a bet earns, which is the gap between the price taken and the fair price at the close, rather than how far one book's line moves. A bet can be placed at any time, so decision time is an input.

- **Rows.** `odds_analytics.retail_ev.build_event_rows` makes one `odds_analytics.retail_ev.RetailEvRow` for each outcome (home, draw, away), UK retail book and retail snapshot at time t, from 7 days to 5 minutes before kick-off. One row joins prices from several snapshots and sources: the retail price at t, the exchange and Polymarket prices and the earlier retail and exchange history at or before t, and the close. Every feature, including the optional Polymarket, dynamics, schedule and standings groups, uses only data known at t ([retail-EV features](clv-target-functions.md#retail-ev-features)). Each row also carries the horizon bucket and season that reports slice by ([target functions](clv-target-functions.md#retail-ev-rows)).
- **Target.** `EV = p_close × retail_decimal(t) − 1`, with the retail price raw, margin included, because that is the price a bet gets.
- **Close.** `p_close` is the devigged Betfair Exchange back price from the last live Betfair job snapshot that is not in-play within 30 minutes before kick-off, else the FDUK closing price, treated as taken at kick-off. Events with neither are dropped. Betfair Exchange is the only close because it covers every season from 2024-25, while FDUK Pinnacle ends in January 2026 (ADR-0012).
- **Baseline.** `p_BFE_fair(t) × retail_decimal(t) − 1` treats the latest exchange price at or before t, from the live Betfair job or an FDUK opening snapshot, as the close. It has no parameters. A row whose exchange price is more than 2 hours older than t, or absent, is dropped, so the model and the baseline are scored on the same rows.
- **FDUK opening rows.** Their t is FDUK's latest documented collection time, an upper bound on when the price was taken, which can be a day or more earlier ([historical ingest scripts](../collection/historical-ingest-scripts.md#snapshot-shape)). Another source's price at t can therefore postdate the retail price. So an FDUK opening row pairs only with the Betfair Exchange price in its own snapshot, for the baseline as well; with none, or only a wide one, the row is dropped (`odds_analytics.retail_ev.DropCounts.rows_without_own_exchange`). It also takes no Polymarket price, lagged exchange price, schedule or standings. The evidence is #548: there, Polymarket prices at the FDUK stamp predicted those rows' residual, and shifting them back 3 hours removed the gain. ADR-0012 does not address this.
- **Thin exchange books.** An exchange snapshot whose back overround exceeds `odds_analytics.retail_ev.MAX_EXCHANGE_OVERROUND` (5%) counts as no price, for the baseline and the close; a row falls back to the latest narrow price within the 2-hour cap. This is proposed in [ADR-0014](../../docs/adr/0014-ignore-wide-exchange-prices.md), which adds to ADR-0012. `experiments/scripts/retail_ev_exchange_overround.py` (events selected by close, any status) reproduces its evidence: on the development data every such book is 3 to 7 days out, and at that horizon books over 5% put the exchange price a standard deviation of 7.5 to 16 pp from the close, against 2.7 to 4.7 pp for narrower ones; left in, they would weaken the baseline where the model can see `exchange_overround`.
- **Weights.** Each row's loss weight is `(1 / retail_decimal²) × w_event`, with `w_event = 1 / (rows in the event)` (`odds_analytics.retail_ev.RetailEvDataset`). The first factor turns the squared EV error into a squared probability error, so longshots do not dominate; the second stops densely scraped events dominating (ADR-0012).

Rows exist only from 2024-25, where FDUK carries Betfair Exchange closing prices, and are dense only where live collection ran: before the live Betfair job, the 2-hour cap keeps few rows other than the FDUK opening snapshot's own books ([data sources](../collection/data-sources.md)). That history is development data. A model is kept only on a pre-registered fresh season, by the two keep-bar tests in `odds_analytics.retail_ev_metrics`: a weighted-MSE gain over the baseline, and a gain in the total `w_event`-weighted realised EV of rows predicted EV > 0, each with an event-clustered bootstrap 95% CI that excludes zero. The clusters are events because an event's rows share one close `(inferred)`. Simulated on resampled development events, one season of 380 events gives the MSE test 80% power at a gain of about 4.5% to 6% of the baseline's weighted MSE; the pick test's power depends more on the shape of a model's edge than on its gain ([keep-bar power](../evaluation/exp-retail-ev-power.md)).

## The close-minus-snapshot delta

`devigged_bookmaker` regresses `devigged_fair_close - devigged_fair_at_snapshot` for one bookmaker (`target_bookmaker`). A positive value means the price at the snapshot was too low and the line moved up (`odds_analytics.sequence_loader.calculate_devigged_bookmaker_target` returns closing probability minus snapshot probability for the configured `outcome`). Why a delta rather than the closing price is `(inferred)`: the decision needs the size of the move, and a delta is stationary and roughly mean-zero.

The suspended model's target is the devigged bet365 delta, recorded in [ADR-0002](../../docs/adr/0002-bet365-clv-target.md) on NBA runs the map #494 voids, and superseded with [ADR-0011](../../docs/adr/0011-bet365-clv-target-on-like-for-like-evidence.md) by ADR-0012. On EPL, scored on the same events, folds and features, a Pinnacle-target model has no out-of-sample skill (R² about 0 against the training-fold mean), against about 0.023 for bet365. The difference is within bootstrap noise ([Pinnacle against bet365](../evaluation/exp-epl-clv-pinnacle-retest.md)). Pinnacle's EPL lines move almost as much as bet365's, with a target variance 91% of bet365's. The NBA runs are on [Target bookmaker and data volume experiments](../evaluation/exp-target-and-data-volume.md). The target bookmaker's snapshot price is also an input to that model's strongest feature, `retail_sharp_diff` ([sharp benchmark](sharp-benchmark.md)).

## Sampling changes a delta target

The delta is measured from the sampled snapshot to the close ([decision time](look-ahead-bias.md)), so the same event can yield different targets under different samplers. A target measured from a closing-tier snapshot has std 0.0016 against 0.054 at the sharp tier, so any R² near 0.6 on it is an artefact, not skill ([NBA timing and injury experiments](../evaluation/exp-nba-timing-and-injuries.md#nba-injury-features-exp-6-6b)). The retail-EV target has no sampler: every retail snapshot in the window is a row, and the horizon is a feature.

## Related

- [Training config](training-config.md), [training pipeline](training-pipeline.md) and [cross-validation](cross-validation.md): how a target is trained and evaluated
- [CLV model findings](../evaluation/clv-model-findings.md): what the model can predict
- [Look-ahead bias](look-ahead-bias.md): inputs must precede the decision time
