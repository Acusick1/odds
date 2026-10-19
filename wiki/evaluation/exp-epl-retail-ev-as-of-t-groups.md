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
  - odds_analytics.retail_ev.RetailEvRow.decision_time_is_exact
  - odds_analytics.training.config.RetailEvFeatureGroupName
  - odds_analytics.retail_ev_metrics.mse_gain
  - odds_analytics.retail_ev_metrics.pick_ev_gain
status: stable
---

# As-of-t feature groups on the retail-EV target

This run repeats the protocol of [Price-only models on the retail-EV target](exp-epl-retail-ev-price-only.md) (#540) with the as-of-t feature groups of #545 ([retail-EV features](../modelling/clv-target-functions.md#retail-ev-features)). Each group is added alone, and then all four together. Each variant's models are compared with the price-only model of the same arm and seed, on the same rows.

**This is development data, and the results are exploratory.** No keep-bar claim is made: ADR-0012 judges a model only on a pre-registered fresh season. The run reads events kicking off up to 2026-07-31 and leaves 2026-27 untouched.

**Findings.**

- **On the live-collected events, no group beats the price-only model.** These are the 69 events whose rows come from live collection at their true times. Every residual-arm gain over price-only has a 95% CI that includes zero. The closest is `dynamics` with seed 7: +47.3 [−0.8, 98.0] × 10⁻⁶ weighted MSE.
- **On the FDUK-only events, no group beats price-only either.**
- **Nothing survives a Bonferroni correction.** Over the 10 tests of five variants and two residual seeds, no gain over price-only excludes zero at level 1 − 0.05/10, overall or within either event type (`bonferroni.csv`). Uncorrected, only `dynamics` seed 7 overall excludes zero: +28.0 [3.3, 53.4].
- **An earlier version of `polymarket_at_t` leaked time on FDUK opening rows.** On the same protocol it had shown a gain of 60 to 66 over price-only, about 84% of it from those rows. The probe below traces that gain to Polymarket prices from after the real FDUK collection time. With the fix, those rows get no Polymarket features, and the group's gain is −1.5 and +3.2.
- **The direct arm stays below the baseline in every variant.**

## Setup

- **Script.** The script `experiments/scripts/retail_ev_as_of_t_groups.py` builds each variant from `experiments/configs/xgboost_epl_retail_ev_price_only.yaml` with `feature_groups` set. It runs the functions of `experiments/scripts/retail_ev_price_only.py`: the same tuning range, scoring folds, arms and seeds, with each variant's Optuna studies under their own name. It writes `experiments/results/retail_ev_as_of_t_groups/`.
- **Rows.** All variants share the 112,650 price-only rows, and the script checks this. 108,789 rows over 340 events are scored, in five 68-event folds from 2025-08-15 to 2026-05-24.
  - The 69 live-collected events hold 103,464 rows, mostly OddsPortal live snapshots. Their 44 OddsPortal live events fall in fold 4 (2026-02-28 onward).
  - The 271 FDUK-only events hold 5,325 rows, one FDUK opening snapshot each.
  - `summary.json` records the git SHA, `git_dirty`, the script and config hashes, and `data_sha256`.
  - The price-only variant reproduces the #540 numbers exactly.
- **FDUK opening times are upper bounds.** An FDUK opening row's t is FDUK's latest documented collection time, and the price was taken at or before it. Its horizon is therefore a lower bound: 1 to 3 days for most matches and under 12 hours for Friday and Tuesday evening kick-offs ([historical ingest scripts](../collection/historical-ingest-scripts.md#snapshot-shape)). Such rows get no column that reads another source's price at t, which is the Polymarket group and the exchange moves (`odds_analytics.retail_ev.RetailEvRow.decision_time_is_exact`).
- **Tuning sees few of the new features.** Hyperparameters are tuned on 2024-25 FDUK opening rows only. Those rows have no Polymarket price and no moves, so only `retail_dispersion`, the schedule and the standings columns are populated there (`summary.json`, `coverage_tuning`). The new columns are learnt only when the frozen hyperparameters are refitted on each scoring fold's training rows.
- **Coverage on scored rows** (`coverage_scored`):
  - Polymarket columns: 93.8%, which is every non-FDUK-opening row. `pm_spread` is dropped as all NaN, because the backfill stores no bid or ask.
  - Exchange moves: 86 to 88%. Consensus moves at 1, 3 and 6 hours: 79%, 64% and 46%.
  - `retail_dispersion` and schedule: 100%. Standings: 99.8%.
- **Metrics.** These are the price-only page's metrics: the weighted-MSE gain in units of 10⁻⁶, and the pick-EV gain in event units. Both come with event-clustered bootstrap 95% CIs from 2,000 resamples. "Over price-only" compares a variant's model with the price-only model of the same arm and seed. `bonferroni.csv` gives the residual arm's gains over price-only with CIs at level 1 − 0.05/10, separately for all events and for each event type.

## Live-collected events

Residual-arm weighted-MSE gain on the 69 live-collected events (103,464 rows) (`results.csv`, slice `event_type`):

| Variant | Seed 42 over price-only | Seed 7 over price-only | Seed 42 over baseline | Seed 7 over baseline |
|---|---|---|---|---|
| price-only | | | 28.8 [−93.7, 147.7] | −16.5 [−196.1, 161.4] |
| `polymarket_at_t` | −8.2 [−22.0, 6.1] | 12.8 [−43.4, 67.7] | 20.6 [−98.1, 137.3] | −3.8 [−168.5, 159.3] |
| `dynamics` | 6.0 [−16.4, 28.3] | 47.3 [−0.8, 98.0] | 34.8 [−75.0, 145.6] | 30.8 [−113.3, 173.8] |
| `schedule_as_of_t` | −50.3 [−118.7, 9.1] | −9.3 [−121.6, 95.7] | −21.5 [−146.1, 97.3] | −25.9 [−170.2, 116.9] |
| `standings_as_of_t` | −25.1 [−119.5, 68.2] | 28.2 [−109.8, 164.0] | 3.7 [−90.6, 99.9] | 11.6 [−80.4, 102.1] |
| all four | −81.2 [−206.7, 23.1] | −6.7 [−155.9, 135.3] | −52.4 [−191.7, 69.7] | −23.2 [−146.6, 87.4] |

- **Picks.** Every pick gain over price-only on these events is between −0.009 and 0.002 event units. With all four groups, the residual picks earn less than the baseline's: −0.013 [−0.030, −0.002] for both seeds.

## FDUK-only events

Residual-arm weighted-MSE gain on the 271 FDUK-only events (5,325 rows):

| Variant | Seed 42 over price-only | Seed 7 over price-only | Seed 42 over baseline | Seed 7 over baseline |
|---|---|---|---|---|
| price-only | | | 54.0 [−12.5, 120.6] | 48.1 [−45.7, 141.4] |
| `polymarket_at_t` | 0.3 [−8.1, 8.2] | 0.7 [−25.7, 28.1] | 54.3 [−13.0, 119.6] | 48.8 [−39.6, 136.4] |
| `dynamics` | 8.3 [−3.6, 21.0] | 22.9 [−2.8, 52.5] | 62.2 [−1.7, 124.7] | 71.0 [−9.6, 150.6] |
| `schedule_as_of_t` | 2.7 [−25.8, 32.5] | −0.4 [−42.5, 40.3] | 56.6 [−10.2, 121.0] | 47.7 [−22.1, 116.1] |
| `standings_as_of_t` | −3.3 [−46.3, 41.8] | −9.6 [−72.2, 53.8] | 50.7 [0.9, 103.9] | 38.5 [−4.6, 84.1] |
| all four | 9.7 [−55.8, 76.1] | −5.3 [−89.5, 77.7] | 63.7 [−12.0, 140.4] | 42.8 [−26.7, 112.8] |

- **Polymarket.** It changes almost nothing on these rows, which now carry no Polymarket price.
- **Picks.** With all four groups, the residual picks beat the baseline: 0.058 [0.016, 0.106] and 0.056 [0.016, 0.102] event units. That isn't significantly better than price-only: 0.025 [−0.036, 0.086] and 0.044 [−0.012, 0.105].

## All events and folds

Residual-arm gain over price-only on all 340 events. The price-only model gains 48.8 [−13.8, 105.6] and 34.7 [−52.3, 115.9] over the baseline, whose weighted MSE is 3419.6:

| Variant | Seed 42 | Seed 7 | Bonferroni CI, seed 42 | Bonferroni CI, seed 7 |
|---|---|---|---|---|
| `polymarket_at_t` | −1.5 [−8.6, 5.4] | 3.2 [−21.9, 26.9] | [−11.3, 9.0] | [−33.5, 37.8] |
| `dynamics` | 7.8 [−3.6, 19.4] | 28.0 [3.3, 53.4] | [−8.6, 25.2] | [−4.4, 64.0] |
| `schedule_as_of_t` | −8.3 [−34.6, 18.0] | −2.2 [−42.0, 39.0] | [−43.7, 26.7] | [−54.6, 59.1] |
| `standings_as_of_t` | −7.8 [−47.5, 34.6] | −1.7 [−58.9, 58.3] | [−67.1, 56.8] | [−80.1, 89.4] |
| all four | −9.1 [−66.3, 51.1] | −5.6 [−76.1, 67.8] | [−96.8, 76.4] | [−99.4, 103.8] |

Per fold, residual gain over price-only, seed 42 then seed 7. Each fold has 68 events; fold 4 holds 102,186 of the rows:

| Fold | `polymarket_at_t` | `dynamics` | `schedule_as_of_t` | `standings_as_of_t` | all four |
|---|---|---|---|---|---|
| 0 | −2.8; −13.4 | 12.5; 19.1 | 12.1; 24.4 | 59.5; 22.4 | 117.7; 73.2 |
| 1 | −3.3; 15.1 | −13.7; 11.9 | 1.3; 15.6 | −33.0; −8.4 | −42.1; −14.9 |
| 2 | −4.5; 11.7 | −4.7; 21.5 | −9.9; 26.1 | 58.0; 81.1 | 74.8; 110.1 |
| 3 | 5.7; 3.3 | 24.3; 35.8 | −7.0; −49.6 | −97.1; −97.4 | −130.2; −179.0 |
| 4 | −2.4; −0.9 | 20.1; 50.9 | −37.2; −26.6 | −25.7; −6.2 | −64.0; −16.3 |

Two fold CIs exclude zero: `dynamics` seed 7 in fold 4 (+50.9 [2.2, 106.4]), and `standings_as_of_t` seed 42 in fold 3 (−97.1 [−193.4, −5.4]).

- **Picks over all events.** With all four groups, the residual arm's picks beat the baseline by 0.045 [0.001, 0.096] and 0.043 [0.000, 0.093] event units. Neither is significantly better than price-only.
- **Feature importance.** In the all-groups models, the standings goal rate and goal difference and the home rest days rank right behind `sharp_prob` and `consensus_prob` (`feature_importance.csv`). Those groups gain nothing alone, so these splits fit noise (inferred).

## The FDUK timing leak

An earlier version of `polymarket_at_t` read the latest Polymarket price at or before every row's t, including FDUK opening rows. The Polymarket grid has a point exactly on the FDUK stamp, so `pm_age_hours` was 0 on 96% of the FDUK-only rows. But FDUK took its price at or before that stamp, possibly much earlier. A Polymarket price at the stamp can therefore include the market's moves after FDUK's real collection. The models picked those moves up: Polymarket's moves in the hours before the stamp correlated 0.17 with the residual on FDUK rows, but not on live rows.

The independent review of #548 tested this with a robustness probe on the earlier features. It shifted every Polymarket lookup back by k hours, refitted the frozen residual-arm hyperparameters of the earlier run, and scored each shift against price-only. CIs come from 1,000 event-clustered resamples:

| Shift k | Seed 42, all | Seed 7, all | Seed 42, FDUK-only | Seed 42, live |
|---|---|---|---|---|
| 0 h | 60.0 [22.8, 105.3] | 65.8 [−0.6, 146.5] | 63.4 [15.9, 112.5] | 46.8 [−26.8, 127.6] |
| 1 h | 26.7 [−5.0, 60.9] | 13.7 [−44.5, 77.1] | 29.3 [−6.8, 64.5] | 16.7 [−52.1, 92.3] |
| 2 h | 19.3 [−9.9, 52.5] | 4.7 [−51.1, 65.6] | 24.7 [−10.0, 60.1] | −1.7 [−74.1, 71.3] |
| 3 h | 16.3 [−13.6, 48.0] | −3.3 [−63.5, 60.3] | 16.1 [−18.9, 55.3] | 16.9 [−40.1, 73.7] |
| 6 h | 10.2 [−14.6, 36.5] | −0.4 [−50.1, 50.8] | 11.2 [−19.9, 42.5] | 6.5 [−37.4, 52.7] |
| 12 h | 6.2 [−19.5, 31.3] | −4.6 [−55.1, 44.0] | 5.5 [−24.9, 34.8] | 8.7 [−41.3, 56.5] |

A 1-hour shift removes 55 to 80% of the gain, and a 3-hour shift removes all of it. A real Polymarket signal would not depend on the price being the one stamped exactly at the FDUK collection time.

## Related

- [Price-only models on the retail-EV target](exp-epl-retail-ev-price-only.md)
- [Decision time and look-ahead bias](../modelling/look-ahead-bias.md#no-future-data-layer-by-layer)
- [Target functions: retail-EV features](../modelling/clv-target-functions.md#retail-ev-features)
- [Polymarket match prices as odds snapshots](../collection/polymarket-odds-snapshots.md)
