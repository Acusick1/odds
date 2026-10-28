---
type: explanation
title: As-of-t feature groups on the retail-EV target
description: The October 2026 exploratory rerun of the price-only retail-EV protocol with the Polymarket, dynamics, schedule and standings groups added one at a time and together, against the price-only model, on live-collected and FDUK-only events, per fold and after a Bonferroni correction, with the timing leak an earlier version of the Polymarket group had on FDUK opening rows.
sources:
  - experiments/scripts/retail_ev_as_of_t_groups.py
  - experiments/scripts/retail_ev_price_only.py
  - experiments/configs/xgboost_epl_retail_ev_price_only.yaml
  - experiments/results/retail_ev_as_of_t_groups/results.csv
  - experiments/results/retail_ev_as_of_t_groups/bonferroni.csv
  - experiments/results/retail_ev_as_of_t_groups/summary.json
  - experiments/results/retail_ev_as_of_t_groups/folds.csv
  - experiments/results/retail_ev_as_of_t_groups/feature_importance.csv
  - odds_analytics.retail_ev.RetailEvFeatures
  - odds_analytics.retail_ev.RetailEvInput.decision_time_is_exact
  - odds_analytics.training.config.RetailEvFeatureGroupName
  - odds_analytics.retail_ev_metrics.mse_gain
  - odds_analytics.retail_ev_metrics.pick_ev_gain
status: stable
---

# As-of-t feature groups on the retail-EV target

This run repeats the protocol of [Price-only models on the retail-EV target](exp-epl-retail-ev-price-only.md) (#540) with the as-of-t feature groups of #545 ([retail-EV features](../modelling/clv-target-functions.md#retail-ev-features)). Each group is added alone, and then all four together. Each variant's models are compared with the price-only model of the same arm and seed, on the same rows.

**This is development data, and the results are exploratory.** No keep-bar claim is made. Only a live forward test promotes a model ([goal and strategy](../foundations/goal-and-strategy.md)). The run reads events kicking off up to 2026-07-31 and leaves 2026-27 untouched.

**Findings.**

- **On the live-collected events, no group beats the price-only model.** These are the 69 events whose rows come from live collection at their true times. Every residual-arm gain over price-only has a 95% CI that includes zero, and all lie between −29.4 and +6.9 × 10⁻⁶ weighted MSE.
- **On the FDUK-only events, no group beats price-only either.** Those rows carry none of the new columns except `retail_dispersion`.
- **No residual-arm gain over price-only excludes zero, even before correction.** Over all events, no group's residual-arm weighted-MSE gain over price-only excludes zero at 95%. The Bonferroni-corrected CIs at level 1 − 0.05/10, over the 10 tests of five variants and two residual seeds, exclude zero in no slice (`bonferroni.csv`).
- **An earlier version of `polymarket_at_t` leaked time on FDUK opening rows.** On the same protocol it had shown a gain of 60 to 66 over price-only, about 84% of it from those rows. The probe below traces that gain to Polymarket prices from after the real FDUK collection time.
- **The direct arm stays below the baseline in every variant.** Some direct-arm gains over the direct price-only model do exclude zero. Polymarket's picks gain +0.085 event units with both seeds. On the live events, `dynamics` gains +64.4 and +101.7 in weighted MSE, and seed 7 gains +155.7 with Polymarket and +82.4 with all four. But they improve on a direct price-only model that is itself 60 to 77 below the baseline overall and 394 to 443 below it on the live events. The direct variants stay 50 to 126 below the baseline overall, so the conclusion stands.

## Setup

- **Script.** The script `experiments/scripts/retail_ev_as_of_t_groups.py` builds each variant from `experiments/configs/xgboost_epl_retail_ev_price_only.yaml` with `feature_groups` set. It runs the functions of `experiments/scripts/retail_ev_price_only.py`, which take every event in the date range with a close and slice rows by the source their close is stored under: the same tuning range, scoring folds, arms and seeds, with each variant's Optuna studies under their own name. The residual arm in this run was the script's target transform on mean-one weights, as on [price-only](exp-epl-retail-ev-price-only.md#setup); the script sets the pipeline's `retail_ev_arm` on the fixed weight scale, so a rerun differs slightly ([rerun](exp-epl-retail-ev-price-only.md#rerun-with-the-pipeline-residual-arm-and-a-fixed-weight-scale)). It writes `experiments/results/retail_ev_as_of_t_groups/`.
- **Rows.** All variants share 112,455 rows over 720 events, and the script checks this. 108,594 rows over 340 events are scored, in five 68-event folds from 2025-08-15 to 2026-05-24.
  - The 69 live-collected events hold 103,269 rows, mostly OddsPortal live snapshots. Their 44 OddsPortal live events fall in fold 4 (2026-02-28 onward).
  - The 271 FDUK-only events hold 5,325 rows, one FDUK opening snapshot each.
  - There are 195 fewer rows than in the price-only run of #540. These are FDUK opening rows whose own snapshot has no usable exchange price, which the builder drops ([CLV target](../modelling/clv-target.md#the-retail-ev-target)). The price-only variant therefore differs slightly from #540.
  - `summary.json` records the git SHA, `git_dirty` (false for this run), the script and config hashes, and `data_sha256`. Its `git_sha` `d9c035d` is the commit the run was made from before the branch was rebased onto main. It has the same code as `68091b8`: no change to `packages/`, `experiments/scripts/` or `experiments/configs/`.
- **FDUK opening times are upper bounds.** An FDUK opening row's t is FDUK's latest documented collection time, and the price was taken at or before it. Its horizon is therefore a lower bound: 1 to 3 days for most matches and under 12 hours for Friday and Tuesday evening kick-offs ([historical ingest scripts](../collection/historical-ingest-scripts.md#snapshot-shape)). Such a row pairs only with its own snapshot's exchange price. It gets no Polymarket price, lagged exchange price, schedule or standings (`odds_analytics.retail_ev.RetailEvInput.decision_time_is_exact`, set by `odds_analytics.retail_ev.is_exact_time` from the retail snapshot's source and time basis).
- **Tuning sees almost none of the new columns.** Hyperparameters are tuned on the 2024-25 FDUK opening rows, where every new column except `retail_dispersion` is NaN (`summary.json`, `coverage_tuning`). The new columns are learnt only when the frozen hyperparameters are refitted on each scoring fold's training rows.
- **Coverage on scored rows** (`coverage_scored`):
  - Polymarket, schedule and standings columns: 94%, which is every row that isn't an FDUK opening row. `pm_spread` is dropped as all NaN, because the backfill stores no bid or ask.
  - Exchange moves: 86 to 88%. Consensus moves at 1, 3 and 6 hours: 79%, 65% and 46%.
  - `retail_dispersion`: 100%.
- **Metrics.** These are the price-only page's metrics: the weighted-MSE gain in units of 10⁻⁶, and the pick-EV gain in event units. Both come with event-clustered bootstrap 95% CIs from 2,000 resamples. "Over price-only" compares a variant's model with the price-only model of the same arm and seed. `bonferroni.csv` gives the residual arm's gains over price-only with CIs at level 1 − 0.05/10, for all events and for each event type.

## Live-collected events

Residual-arm weighted-MSE gain on the 69 live-collected events (103,269 rows; baseline weighted MSE 4608.8) (`results.csv`, slice `event_type`):

| Variant | Seed 42 over price-only | Seed 7 over price-only | Seed 42 over baseline | Seed 7 over baseline |
|---|---|---|---|---|
| price-only | | | 26.0 [−96.3, 145.9] | 22.6 [−106.2, 147.3] |
| `polymarket_at_t` | −6.2 [−21.0, 9.9] | −29.0 [−85.7, 27.0] | 19.8 [−100.0, 136.8] | −6.4 [−172.8, 155.9] |
| `dynamics` | 6.6 [−15.6, 28.9] | 4.2 [−34.0, 47.4] | 32.6 [−77.9, 143.2] | 26.8 [−104.8, 159.0] |
| `schedule_as_of_t` | 3.1 [−9.2, 16.2] | −27.8 [−85.5, 31.8] | 29.1 [−93.2, 148.9] | −5.2 [−176.2, 163.5] |
| `standings_as_of_t` | −2.5 [−14.9, 10.2] | −29.4 [−86.1, 30.8] | 23.5 [−98.2, 142.6] | −6.8 [−177.4, 161.1] |
| all four | 6.9 [−14.3, 27.6] | −2.0 [−39.7, 43.8] | 32.9 [−79.1, 142.8] | 20.6 [−120.9, 164.0] |

- **Picks.** Every residual-arm pick gain over price-only on these events is between −0.008 and 0.007 event units, and every one has a CI that includes zero.

## FDUK-only events

Residual-arm weighted-MSE gain on the 271 FDUK-only events (5,325 rows; baseline weighted MSE 3111.2):

| Variant | Seed 42 over price-only | Seed 7 over price-only | Seed 42 over baseline | Seed 7 over baseline |
|---|---|---|---|---|
| price-only | | | 53.7 [−12.9, 120.0] | 40.7 [−31.1, 112.5] |
| `polymarket_at_t` | 0.8 [−8.1, 8.8] | 7.8 [−30.6, 43.8] | 54.5 [−13.0, 120.1] | 48.6 [−39.6, 136.1] |
| `dynamics` | 8.4 [−3.6, 21.4] | 24.5 [−11.1, 58.9] | 62.1 [−1.8, 124.2] | 65.2 [−10.8, 140.1] |
| `schedule_as_of_t` | 5.1 [−1.9, 12.4] | 8.5 [−30.2, 47.1] | 58.8 [−9.2, 124.7] | 49.2 [−43.9, 140.5] |
| `standings_as_of_t` | 4.5 [−3.3, 12.3] | 6.1 [−32.4, 45.4] | 58.2 [−10.2, 124.8] | 46.8 [−41.9, 135.5] |
| all four | 10.2 [−3.8, 25.2] | 31.2 [−6.1, 66.7] | 63.9 [−0.6, 126.1] | 71.9 [−12.2, 152.7] |

- **Gains on rows without the columns.** These rows have no Polymarket, schedule or standings values, so the variants differ from price-only only through the trees fitted on the other events' rows.
- **Picks.** With all four groups, seed 7's picks earn less than price-only's: −0.042 [−0.072, −0.014] event units.

## All events and folds

Residual-arm gain over price-only on all 340 events. The price-only model gains 47.9 [−14.7, 105.0] and 37.0 [−27.8, 97.5] over the baseline, whose weighted MSE is 3421.7:

| Variant | Seed 42 | Seed 7 | Bonferroni CI, seed 42 | Bonferroni CI, seed 7 |
|---|---|---|---|---|
| `polymarket_at_t` | −0.6 [−7.9, 6.6] | 0.2 [−32.2, 32.3] | [−10.8, 10.4] | [−43.0, 47.1] |
| `dynamics` | 8.1 [−3.4, 19.7] | 20.3 [−12.2, 49.7] | [−8.0, 25.5] | [−25.4, 63.5] |
| `schedule_as_of_t` | 4.7 [−1.4, 10.9] | 0.9 [−33.3, 34.5] | [−3.6, 13.0] | [−51.3, 48.3] |
| `standings_as_of_t` | 3.0 [−3.2, 9.7] | −1.2 [−35.5, 32.4] | [−5.9, 13.6] | [−50.9, 48.8] |
| all four | 9.5 [−3.4, 22.0] | 24.3 [−9.0, 56.1] | [−10.6, 28.4] | [−24.2, 68.7] |

Per fold, residual gain over price-only, seed 42 then seed 7. Each fold has 68 events; fold 4 holds 101,991 of the rows:

| Fold | `polymarket_at_t` | `dynamics` | `schedule_as_of_t` | `standings_as_of_t` | all four |
|---|---|---|---|---|---|
| 0 | −1.9; −11.8 | 11.1; −6.2 | 5.1; −4.8 | 2.9; −9.6 | 7.7; 0.3 |
| 1 | −3.4; −0.1 | −13.7; 9.5 | 3.0; −19.0 | −1.5; −10.1 | −2.8; 24.1 |
| 2 | −4.7; −33.9 | −4.9; −23.6 | −0.1; −19.5 | −7.5; −46.8 | −8.3; −32.1 |
| 3 | 6.1; 76.4 | 25.0; 93.4 | 13.5; 83.7 | 16.8; 85.3 | 24.9; 104.5 |
| 4 | 0.6; −29.4 | 22.3; 27.7 | 2.0; −35.7 | 4.3; −25.0 | 25.7; 24.3 |

- **Fold CIs that exclude zero.** Four do, all positive: seed 7 in fold 3 for `dynamics` (93.4 [10.5, 181.0]), `standings_as_of_t` (85.3 [0.8, 171.7]) and all four (104.5 [14.3, 201.3]), and all four with seed 42 in fold 4 (25.7 [1.3, 51.0]). Seed 7's fold-3 gain appears in every variant, including `polymarket_at_t` (76.4 [−12.7, 161.2]). So it is more likely a property of seed 7's refits in that fold than of any one group (inferred).
- **Picks over all events.** With all four groups, seed 7's residual-arm picks earn less than price-only's: −0.049 [−0.084, −0.021] event units. No other residual-arm pick gain over price-only excludes zero.
- **Feature importance.** In the all-groups models, `retail_dispersion` ranks third, at 0.110 of total gain, behind `sharp_prob` and `consensus_prob` (`feature_importance.csv`).

## The FDUK timing leak

An earlier version of `polymarket_at_t` read the latest Polymarket price at or before every row's t, including FDUK opening rows. The Polymarket grid has a point exactly on the FDUK stamp, so `pm_age_hours` was 0 on 96% of the FDUK-only rows. But FDUK took its price at or before that stamp, possibly much earlier. A Polymarket price at the stamp can therefore include the market's moves after FDUK's real collection. The models picked those moves up: Polymarket's moves in the hours before the stamp correlated 0.17 with the residual on FDUK rows, but not on live rows `(inferred)`.

The independent review of #548 tested this with a robustness probe on the earlier features. It shifted every Polymarket lookup back by k hours, refitted the frozen residual-arm hyperparameters of the earlier run, and scored each shift against price-only. CIs come from 1,000 event-clustered resamples:

| Shift k | Seed 42, all | Seed 7, all | Seed 42, FDUK-only | Seed 42, live |
|---|---|---|---|---|
| 0 h | 60.0 [22.8, 105.3] | 65.8 [−0.6, 146.5] | 63.4 [15.9, 112.5] | 46.8 [−26.8, 127.6] |
| 1 h | 26.7 [−5.0, 60.9] | 13.7 [−44.5, 77.1] | 29.3 [−6.8, 64.5] | 16.7 [−52.1, 92.3] |
| 2 h | 19.3 [−9.9, 52.5] | 4.7 [−51.1, 65.6] | 24.7 [−10.0, 60.1] | −1.7 [−74.1, 71.3] |
| 3 h | 16.3 [−13.6, 48.0] | −3.3 [−63.5, 60.3] | 16.1 [−18.9, 55.3] | 16.9 [−40.1, 73.7] |
| 6 h | 10.2 [−14.6, 36.5] | −0.4 [−50.1, 50.8] | 11.2 [−19.9, 42.5] | 6.5 [−37.4, 52.7] |
| 12 h | 6.2 [−19.5, 31.3] | −4.6 [−55.1, 44.0] | 5.5 [−24.9, 34.8] | 8.7 [−41.3, 56.5] |

A 1-hour shift removes 54 to 79% of the gain, and after it no gain has a CI that excludes zero. A real Polymarket signal would not depend on the price being the one stamped exactly at the FDUK collection time.

## Related

- [Price-only models on the retail-EV target](exp-epl-retail-ev-price-only.md)
- [Decision time and look-ahead bias](../modelling/look-ahead-bias.md#no-future-data-layer-by-layer)
- [Target functions: retail-EV features](../modelling/clv-target-functions.md#retail-ev-features)
- [Polymarket match prices as odds snapshots](../collection/polymarket-odds-snapshots.md)
