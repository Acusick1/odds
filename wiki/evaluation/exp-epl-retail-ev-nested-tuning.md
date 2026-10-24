---
type: explanation
title: Nested tuning on the as-of-t columns
description: The October 2026 exploratory rerun of the residual retail-EV models with hyperparameters re-tuned for every scoring fold by nested walk-forward inside 2025-26, so tuning sees the Polymarket and live-collected rows; price-only, Polymarket at t and all four as-of-t groups against the baseline, price-only and the frozen 2024-25 settings, on live-collected and FDUK-only events, per fold, after a Bonferroni correction, and where the tuned settings sit.
sources:
  - experiments/scripts/retail_ev_nested_tuning.py
  - experiments/configs/xgboost_epl_retail_ev_price_only.yaml
  - experiments/results/retail_ev_nested_tuning/results.csv
  - experiments/results/retail_ev_nested_tuning/bonferroni.csv
  - experiments/results/retail_ev_nested_tuning/summary.json
  - experiments/results/retail_ev_nested_tuning/folds.csv
  - experiments/results/retail_ev_nested_tuning/corner.csv
  - experiments/results/retail_ev_nested_tuning/trials.csv
  - experiments/results/retail_ev_nested_tuning/feature_importance.csv
  - odds_analytics.training.cross_validation.make_walk_forward_splits
  - odds_analytics.retail_ev_metrics.mse_gain
status: stable
---

# Nested tuning on the as-of-t columns

The [price-only](exp-epl-retail-ev-price-only.md) (#540) and [as-of-t groups](exp-epl-retail-ev-as-of-t-groups.md) (#545) runs tune on 2024-25 and freeze the settings. That season has only FDUK opening rows, so no tuning trial sees a Polymarket price, an exchange move or a live-collected row. The tuned residual models sit in a heavy-regularisation corner: about 100 trees, a learning rate near 0.01 and `min_child_weight` near its upper bound of 2,000. This run (#549) re-tunes inside 2025-26, on folds that carry those columns, without tuning on any row it scores.

**This is development data, and the results are exploratory.** Only a live forward test promotes a model ([goal and strategy](../foundations/goal-and-strategy.md)). The rows have been examined by #540 and #545, and the design below was chosen with their layout in view. The run reads events kicking off up to 2026-07-31 and leaves 2026-27 untouched.

**Findings.**

- **Nothing is a finding after correction.** None of the 48 tests in the Bonferroni family excludes zero (`bonferroni.csv`).
- **On the 50 live-collected scored events, no model beats the baseline.** Every residual model's point gain over the baseline is between −114 and +7 × 10⁻⁶ weighted MSE, with 95% CIs that include zero.
- **Re-tuning does worse than the frozen 2024-25 settings for price-only.** The nested price-only model's point estimates are 25 to 29 below the frozen one over all events, with CIs that include zero. On picks, seed 7's nested price-only model earns −0.026 [−0.056, −0.001] event units against frozen, outside the Bonferroni family. For the group variants, nested and frozen are level.
- **The settings move only partly out of the corner.** `min_child_weight` leaves its upper bound in most folds, but the learning rate stays near 0.01 to 0.02, and the models stay close to the baseline.
- **The new columns are barely used.** Averaged over the outer folds, Polymarket's columns take 0.4 to 1.4% of total gain, rising to 2.2 to 6.4% in fold 9, and the move columns take none.
- **One Polymarket model is the best in fold 9, but it is not evidence of Polymarket signal (inferred).** Seed 42's Polymarket model is the best of the 12 models in fold 9, the one fold tuned on dense live rows. Seed 7 does not replicate it, and that fold is 17 events in one week.

## Design

### Why nested walk-forward

Only 69 events in the development range are live-collected. 25 of them, from September 2025 to January 2026, have about 66 rows each. The other 44 run from 2026-04-18 to 2026-05-24, and 41 of those, from 2026-05-01, carry about 101,000 rows. The March 2026 events the local database holds are mostly dropped by `odds_analytics.feature_groups.filter_completed_events` (#546).

So the dense rows span about three and a half weeks. With the 7-day gap, any scheme that tunes on dense rows and then scores dense rows has fewer than 20 events on each side.

- **Tuning on the first half of 2025-26 and scoring the second half** sees the as-of-t columns on only 19 sparse events. It never sees a dense live row, which is the defect #549 sets out to fix, for the rows that matter.
- **Nested walk-forward** re-tunes every scoring fold on everything before it. The last fold's tuning includes 17 dense live events, from 2026-05-01 to 2026-05-10. It costs one Optuna study per fold, and each fold's tuning is noisier than one study on a whole season.

The run uses nested walk-forward. Only the last outer fold has dense live rows in its tuning, so the per-fold table separates it.

### Folds

- **Outer folds.** `odds_analytics.training.cross_validation.make_walk_forward_splits` with `min_train_events` 550 (the 380 events of 2024-25 plus the first 170 of 2025-26) and 17-event steps makes 10 folds. They cover the last 170 events of 2025-26, from 2025-12-26 to 2026-05-24, with no partial block. Training events that kick off within 7 days of a fold's first event are dropped.
- **Inner folds.** Each outer fold tunes only on its own training rows: every event kicking off at least 7 days before the fold. The inner validation blocks are 17 events of 2025-26, counted back from the last training event, so the most recent block is always a tuning fold. 2024-25 is only ever trained on. Each inner fold has the 7-day gap too.
  - There are 9 inner folds for the first outer fold and 18 for the last (`folds.csv`).
  - The last outer fold's inner validation holds 46 live-collected events and 26,943 live rows. Its most recent inner block is 17 dense live events. The objective is the mean fold weighted MSE, so that block counts for 1/18 of it.
  - Every earlier outer fold's inner validation holds only the 18 to 28 sparse live events, at most 1,788 rows.
- **Trials.** 30 TPE trials per study over the #540 search space (`experiments/configs/xgboost_epl_retail_ev_price_only.yaml`, in units of mean-one weights in this run), seeded by the run's seed. There are 60 studies: 3 variants × 2 seeds × 10 outer folds. The run took 14 minutes on 20 workers.

### Models and comparisons

- **Residual arm only.** XGBoost fits the target minus the martingale baseline, and the baseline is added back to its prediction ([price-only setup](exp-epl-retail-ev-price-only.md#setup)). In this run that was a script transform on mean-one weights; the script sets the pipeline's `retail_ev_arm: residual` on the fixed weight scale ([retail-EV arm](../modelling/training-config.md#retail-ev-arm)).
- **Variants.** Price-only (38 columns), price plus `polymarket_at_t` (43), and price plus all four as-of-t groups (69) ([retail-EV features](../modelling/clv-target-functions.md#retail-ev-features)). Seeds 42 and 7.
- **Frozen reference.** Every outer fold also refits the residual-arm settings #545 tuned on 2024-25, read from `experiments/results/retail_ev_as_of_t_groups/summary.json`, on the same training rows. That isolates the tuning scheme's effect on the same scored rows.
- **Metrics.** Gains are reference minus challenger in weighted MSE, in units of 10⁻⁶, from `odds_analytics.retail_ev_metrics.mse_gain`. They come with event-clustered bootstrap 95% CIs (2,000 resamples) in `results.csv`.
- **Bonferroni family.** It holds the pooled weighted-MSE tests: per variant and seed, nested against the baseline and against frozen, and the two group variants against nested price-only. That is 16 comparisons in each of 3 slices (all, live-collected, FDUK-only), so 48 tests. `bonferroni.csv` gives CIs at level 1 − 0.05/48 from 200,000 event-clustered resamples, which leaves about 104 resamples beyond each tail. The pick-EV gains in `results.csv` and the per-fold CIs are outside the family and only descriptive.
- **Provenance.** `summary.json` records the git SHA, `git_dirty`, the script and config hashes and `data_sha256`. The data hash matches #545's, so the rows are the same. The run was made from a clean tree (`git_dirty` false), and repeated runs give identical models, tuned settings and weighted-MSE results.

## The scored rows

The 170 scored events hold 104,163 rows (`summary.json`). The baseline's weighted MSE is 3984.4 over all of them (`results.csv`):

| Slice | Events | Rows | Baseline wMSE |
|---|---|---|---|
| All | 170 | 104,163 | 3984.4 |
| Live-collected | 50 | 102,009 | 5522.9 |
| FDUK-only | 120 | 2,154 | 3307.5 |

The 50 live-collected events are 6 sparse events from 2025-12-27 to 2026-01-31 and all 44 events from 2026-04-18 on. Folds 8 and 9 are entirely live-collected and hold 96,729 of the rows.

## Results

### Live-collected events

Weighted-MSE gain on the 50 live-collected events, with 95% CIs (`results.csv`, slice `event_type`):

| Variant, seed | Over baseline | Over frozen | Over nested price-only |
|---|---|---|---|
| price-only, 42 | −91.7 [−281.1, 99.0] | −57.0 [−132.5, 15.3] | |
| price-only, 7 | −82.7 [−291.8, 117.6] | −40.9 [−114.0, 25.0] | |
| `polymarket_at_t`, 42 | 6.9 [−153.6, 169.2] | 40.6 [−19.0, 101.1] | 98.6 [33.4, 174.5] |
| `polymarket_at_t`, 7 | −58.8 [−301.4, 160.2] | 44.1 [−58.3, 138.0] | 23.9 [−78.2, 111.2] |
| all four, 42 | −114.3 [−405.2, 122.5] | −100.0 [−314.9, 51.6] | −22.6 [−206.9, 120.8] |
| all four, 7 | −81.3 [−389.3, 161.5] | −47.1 [−268.2, 98.8] | 1.4 [−180.3, 138.7] |

- **Polymarket, seed 42.** Its gain over price-only is the only live-event CI that excludes zero at 95%. Its Bonferroni CI is [−5.8, 241.3], and seed 7's gain is a quarter of it. Most of it comes from fold 9 (below).
- **Picks.** Every pick-EV gain over the baseline on these events is between −0.012 and −0.005 event units, and every CI includes zero.

### FDUK-only events

Weighted-MSE gain on the 120 FDUK-only events:

| Variant, seed | Over baseline | Over frozen | Over nested price-only |
|---|---|---|---|
| price-only, 42 | 73.2 [−52.5, 197.5] | −16.2 [−54.7, 20.9] | |
| price-only, 7 | 53.0 [−74.9, 171.4] | −18.5 [−62.1, 20.4] | |
| `polymarket_at_t`, 42 | 73.3 [−37.8, 180.2] | −20.4 [−59.5, 19.0] | 0.2 [−30.4, 32.2] |
| `polymarket_at_t`, 7 | 87.9 [−53.6, 232.1] | −8.0 [−52.1, 31.5] | 34.9 [−6.5, 78.4] |
| all four, 42 | 119.2 [−10.5, 241.8] | 14.6 [−30.9, 60.4] | 46.0 [13.5, 77.6] |
| all four, 7 | 102.5 [−35.8, 241.7] | −43.0 [−97.3, 6.3] | 49.5 [5.3, 97.4] |

- **All four groups on FDUK rows.** These rows carry none of the new columns except `retail_dispersion` ([CLV target](../modelling/clv-target.md#the-retail-ev-target)). Both seeds' gains over price-only exclude zero at 95% but not after correction (`bonferroni.csv`: [−7.0, 102.5] and [−22.2, 135.3]). `retail_dispersion` takes 6 to 17% of total gain in the all-four models (`feature_importance.csv`), so it is a candidate carrier (inferred). That share is over all training rows, which by fold 9 are mostly live rows, so it does not show what drives the FDUK-only predictions. Two other routes are open. These rows' NaNs in the other new columns follow each split's default branch, so a split fitted on live rows also moves FDUK-only predictions. And `colsample_bytree` samples from 69 columns instead of 38, which changes the price-column splits. #545 saw the same direction on these events with frozen settings: +10.2 and +31.2, with CIs including zero.

### All events

| Variant, seed | Over baseline | Over frozen | Over nested price-only |
|---|---|---|---|
| price-only, 42 | 22.8 [−83.1, 131.9] | −28.7 [−64.7, 8.2] | |
| price-only, 7 | 11.5 [−96.5, 122.0] | −25.3 [−61.9, 11.2] | |
| `polymarket_at_t`, 42 | 53.1 [−38.1, 148.6] | −1.8 [−34.5, 32.9] | 30.2 [1.1, 61.6] |
| `polymarket_at_t`, 7 | 43.1 [−73.8, 165.6] | 7.9 [−34.3, 49.4] | 31.6 [−9.8, 73.8] |
| all four, 42 | 47.8 [−69.6, 167.8] | −20.4 [−91.5, 38.0] | 25.0 [−35.6, 76.4] |
| all four, 7 | 46.3 [−80.7, 177.5] | −44.2 [−116.9, 16.4] | 34.8 [−30.5, 91.8] |

No pick-EV gain over the baseline or over price-only excludes zero over all events.

### By fold

Weighted-MSE gain over the baseline per outer fold, seed 42 then seed 7 (`results.csv`, slice `fold`). Each fold has 17 events:

| Fold | First kick-off | Live events | Rows | Baseline wMSE | Price-only | `polymarket_at_t` | All four |
|---|---|---|---|---|---|---|---|
| 0 | 2025-12-26 | 1 | 405 | 2882.5 | −91.0; −147.8 | −25.0; −112.9 | −41.3; −103.0 |
| 1 | 2026-01-01 | 2 | 453 | 3698.2 | 33.9; 24.9 | −2.4; 62.0 | 36.8; 64.8 |
| 2 | 2026-01-07 | 1 | 405 | 4565.2 | 313.3; 301.1 | 292.1; 407.7 | 338.8; 454.3 |
| 3 | 2026-01-24 | 2 | 351 | 3107.9 | −218.4; −230.2 | −170.7; −260.0 | −184.1; −174.7 |
| 4 | 2026-02-01 | 0 | 261 | 2338.0 | 47.9; 27.8 | 31.2; 42.9 | 109.5; 27.5 |
| 5 | 2026-02-11 | 0 | 297 | 3771.3 | 127.6; 114.3 | 161.7; 176.1 | 211.0; 139.0 |
| 6 | 2026-02-28 | 0 | 279 | 2612.7 | 107.0; 88.7 | 69.8; 78.2 | 174.8; 103.0 |
| 7 | 2026-04-13 | 10 | 4,983 | 3308.0 | 44.3; 65.3 | 82.8; 131.0 | 105.2; 72.7 |
| 8 | 2026-05-03 | 17 | 46,014 | 7576.9 | 33.9; 100.9 | 84.0; 80.8 | 72.1; 173.2 |
| 9 | 2026-05-17 | 17 | 50,715 | 5605.7 | −176.7; −238.8 | −8.3; −185.7 | −348.5; −305.7 |

- **Fold 9 is the only fold tuned on dense live rows, and every model in it is worse than the baseline.** No fold-9 CI over the baseline excludes zero.
- **Polymarket in fold 9.** Seed 42's Polymarket model has weighted MSE 5614.0, 8.3 above the baseline's 5605.7 and the best of the 12 models in the fold. It gains 168.4 [74.7, 271.0] over nested price-only and 101.3 over the frozen price-only model (5715.3), so its lead is not only a weak nested price-only fit. Its Polymarket columns take 6.4% of total gain in that fold. Still, it is not evidence of Polymarket signal (inferred). Seed 7's Polymarket model (5791.3) is worse than both frozen price-only fits (5715.3 and 5728.2). The best of 12 models on one 17-event week is a selection, and nothing in the Bonferroni family excludes zero.
- **Folds 0 to 6** hold 6 live events between them, so they speak almost only to FDUK rows. Every model gains over the baseline in folds 2, 4, 5 and 6, and loses in folds 0 and 3.

## Where the tuned settings sit

Tuned settings per outer fold, summarised over the 10 folds (`corner.csv`, `folds.csv`). "In corner" counts folds with `min_child_weight` ≥ 1,000, at most 150 trees and a learning rate ≤ 0.015 together:

| Variant, seed | In corner | Median `min_child_weight` (min) | Median trees | Median learning rate | Median depth | Spread of prediction minus baseline, nested / frozen (pp) |
|---|---|---|---|---|---|---|
| price-only, 42 | 2 | 986 (144) | 150 | 0.014 | 3 | 0.66 / 0.51 |
| price-only, 7 | 4 | 1642 (20) | 150 | 0.013 | 3 | 0.70 / 0.54 |
| `polymarket_at_t`, 42 | 1 | 548 (198) | 150 | 0.011 | 2.5 | 0.58 / 0.53 |
| `polymarket_at_t`, 7 | 1 | 671 (29) | 300 | 0.013 | 2 | 0.74 / 0.73 |
| all four, 42 | 0 | 633 (42) | 150 | 0.016 | 3.5 | 0.81 / 0.51 |
| all four, 7 | 2 | 1531 (70) | 175 | 0.013 | 3 | 0.75 / 0.70 |

- **Partly out of the corner.** `min_child_weight` drops below 1,000 in 33 of the 60 studies, and depth reaches 4 or 5 in 14. The learning rate stays between 0.010 and 0.021 in all but five studies, and the trees stay between 100 and 400. With mean-one weights of about 150 per event, a `min_child_weight` of 500 is still about 3 events per leaf.
- **The models stay close to the baseline.** The weighted spread of prediction minus baseline is 0.3 to 1.4 pp, against 0.45 to 0.84 pp for the frozen settings and 6 to 8 pp for the target. Fold 9's all-four models are the widest, 1.1 and 1.4 pp, and are also the worst against the baseline.
- **The inner folds of 2025-26 reward a model.** Unlike #540's 2024-25 tuning, in which no trial beat the baseline, every one of the 60 studies has trials that beat the baseline on its inner folds. The best trial's margin is 21 to 83 × 10⁻⁶ in mean fold weighted MSE. That margin is optimistic, since it is the best of 30 trials on the folds it was picked on. The outer folds do not reproduce it against the frozen settings.
- **The new columns get few splits.** Averaged over the outer-fold models, Polymarket's columns take 0.4 to 1.4% of total gain, schedule and standings under 0.5%, and the exchange and consensus moves none. In fold 9, whose training rows include the first dense live events, the Polymarket columns take 2.2 to 6.4%, and schedule and standings up to 3.2% (`feature_importance.csv`, per fold). `retail_dispersion`, which every row has, takes 6 to 17%. The other columns are present on fewer than 50 of the 540 to 697 training events, so even a `min_child_weight` of a few hundred leaves little room to split on them (inferred).

## Caveats

- **Development data, examined before.** The fold layout was designed with #540's and #545's results and the live events' dates in view.
- **One fold carries the question.** Only fold 9, with 17 events from one week, is scored with tuning that saw dense live rows. Its tuning saw 17 dense live events, as one of 18 inner folds.
- **Thirty trials per study**, on inner folds of 17 events. The tuned settings move between neighbouring outer folds, for example price-only seed 7's `min_child_weight` goes from 1,664 to 33 to 20 to 1,818 over folds 0 to 3.
- **The search space is #540's.** Its lower bounds on learning rate (0.01) and trees (100) still bind in many studies.
- **The scored range is the second half of 2025-26 only**, so these numbers are not comparable to #540's and #545's over all 340 events. The frozen reference is.

## Related

- [Price-only models on the retail-EV target](exp-epl-retail-ev-price-only.md)
- [As-of-t feature groups on the retail-EV target](exp-epl-retail-ev-as-of-t-groups.md)
- [Cross-validation](../modelling/cross-validation.md), [Hyperparameter tuner](../modelling/hyperparameter-tuner.md)
- [What the CLV model can and cannot do](clv-model-findings.md)
