---
type: explanation
title: CV protocol on a common validation range
description: The October 2026 comparison of expanding, sliding-380 and sliding-760 walk-forward CV scored on the same validation events, the tuning-range event counts, and the fold layout the re-baseline uses.
sources:
  - experiments/scripts/cv_protocol_common_range.py
  - experiments/configs/xgboost_epl_cv_protocol_base.yaml
  - experiments/configs/xgboost_epl_cv_protocol_expanding.yaml
  - experiments/configs/xgboost_epl_cv_protocol_sliding_380.yaml
  - experiments/configs/xgboost_epl_cv_protocol_sliding_760.yaml
  - experiments/results/cv_protocol_common_range/results.csv
  - experiments/results/cv_protocol_common_range/event_counts.csv
  - experiments/results/cv_protocol_common_range/snapshot_coverage.csv
  - experiments/results/cv_protocol_common_range/folds.csv
  - experiments/results/cv_protocol_common_range/summary.json
  - odds_analytics.training.cross_validation.make_walk_forward_splits
status: stable
---

# CV protocol on a common validation range

This experiment picks the walk-forward protocol for the EPL CLV re-baseline (#452, on the map #494). It is the evidence for [ADR-0010](../../docs/adr/0010-expanding-walk-forward-and-evaluation-range.md). It replaces the [CV protocol grid](exp-cv-protocol-grid.md), which scored each cell on a different validation set. How folds are built is on [Cross-validation](../modelling/cross-validation.md).

## Setup

- **Script:** `experiments/scripts/cv_protocol_common_range.py`, seeded (Optuna TPE sampler 42, XGBoost `random_state` 42, bootstrap seed 42), with outputs in `experiments/results/cv_protocol_common_range/`. A second run reproduced `results.csv`, `folds.csv` and `oof_predictions.csv` byte for byte; `trials.csv` differed only in trial timestamps.
- **Config:** `experiments/configs/xgboost_epl_cv_protocol_base.yaml` is the deployed model's config with `feature_groups: [tabular]` only. It keeps the devigged bet365 target, the `[pinnacle, betfair_exchange]` sharp reference, the retail list, tier sampling at `decision_tier: sharp` and the principled search space ([tuner](../modelling/hyperparameter-tuner.md#production-search-space)). One cell config per protocol inherits from it and sets only the walk-forward window.
- **Tuning range:** only events before 2024-08-01 are loaded (`end_date: 2024-07-31`). The script fails if any row falls on or after that date. The 2024-25 and 2025-26 seasons are the evaluation range, reserved for the re-baseline (#496).
- **Cells:** expanding (`min_train_events` 380), sliding-380 and sliding-760, all with `val_step_events: 50`.
- **Tuning:** 100 Optuna trials per cell, minimising that cell's mean fold MSE over its own folds. The best trial is then refitted fold by fold through `odds_analytics.training.cross_validation.make_walk_forward_splits` and `odds_analytics.xgboost_line_movement.XGBoostLineMovementStrategy.train_from_config`, to collect the out-of-fold prediction for every validation row. The script checks that the refit reproduces the tuned objective.
- **Common validation range:** the events every cell predicts. It starts at the largest `min_train_events` (760) and ends where sliding-760's last full 50-event block ends. That gives 1,100 events and 1,100 rows, 2021-08-13 to 2024-04-27 (`summary.json`).
- **Metric:** MSE pooled over the common rows. Each cell also reports the two per-fold baselines defined on #495, both fitted on the same training rows: an OLS fit with intercept on `retail_sharp_diff`, and the training-fold mean as a floor.
- **Uncertainty:** a paired bootstrap that resamples common-range events (2,000 resamples) gives a 95% percentile CI for each cell's MSE minus the expanding cell's MSE, and for the model's MSE minus the OLS baseline's.
- **Choice rule:** take the cell with the lowest pooled MSE. If its CI against expanding includes zero, take expanding instead, because it is simpler and has more data. The rule, its reason and the cells were fixed in the ticket's design before the run (#503).

## Event counts

`experiments/results/cv_protocol_common_range/event_counts.csv` counts FINAL EPL events in the database and the events that yield a training row under the production sampler:

| Season | Events | With rows |
|---|---|---|
| 2015-16 to 2018-19 | 380 each | 0 |
| 2019-20 | 380 | 380 |
| 2020-21 | 380 | 380 |
| 2021-22 | 380 | 380 |
| 2022-23 | 380 | 380 |
| 2023-24 | 380 | 380 |

`experiments/results/cv_protocol_common_range/snapshot_coverage.csv` counts the stored snapshots per season, source and tier that quote a sharp bookmaker (Pinnacle or Betfair Exchange) and bet365:

- **2015-16 to 2018-19 yield no rows.** Their FDUK closing snapshots quote Pinnacle but never bet365, so no event has a bet365 close and none gets a target. The "expanding from 2015" cell is therefore the expanding cell, which already starts at the first season with rows.
- **2019-20 and 2020-21** are FDUK-only. Their closing snapshots carry bet365, and every row has a sharp price.
- **Every tuning-range decision row comes from the FDUK `early` snapshot.** No OddsPortal snapshot in the tuning range quotes a sharp bookmaker, and `odds_analytics.feature_groups.TierSampler` excludes closing-tier snapshots at `decision_tier: sharp` and keeps only snapshots that quote a sharp bookmaker. `hours_until_event` runs from 23 to 24.25 in every season (`event_counts.csv`). So the FDUK seasons are the same decision regime as the later ones.
- In total there are 1,900 events and 1,900 rows, one per event. That leaves 1,100 common validation events. The design kept sliding-760 only if at least about 300 remained (#503).

## Results

MSE pooled over the 1,100 common rows (`experiments/results/cv_protocol_common_range/results.csv`). Differences are ×10⁻⁶, with 95% bootstrap CIs.

| Cell | Folds | Tuned CV MSE (own folds) | Pooled MSE | OLS baseline | Mean floor | Minus expanding | Minus OLS |
|---|---|---|---|---|---|---|---|
| expanding | 30 | 0.000706 | 0.000662 | 0.000672 | 0.000696 | 0 | -10.4 [-22.8, 1.0] |
| sliding-380 | 30 | 0.000712 | 0.000667 | 0.000678 | 0.000699 | 5.4 [-4.9, 16.6] | -10.7 [-28.8, 5.1] |
| sliding-760 | 22 | 0.000659 | 0.000659 | 0.000673 | 0.000695 | -2.5 [-11.2, 6.2] | -13.3 [-28.6, 2.3] |

- **The tuned CV MSE column is not comparable across cells.** Expanding and sliding-380 average over folds from event 380, which include the 2020-21 season. Sliding-760 starts at 760, so its column equals its pooled MSE on the common range. On the common range the three cells lie within 8 × 10⁻⁶ of each other.
- Sliding-760 has the lowest pooled MSE, but its CI against expanding includes zero. Under the choice rule, **expanding is chosen**.
- Every cell beats the OLS baseline in point estimate, but every CI includes zero. This is not the keep test: the hyperparameters were tuned on these rows, and #496 runs the test on the evaluation range.

### Chosen protocol and hyperparameters

The chosen config is `experiments/configs/xgboost_epl_cv_protocol_expanding.yaml`. Its tuned hyperparameters (best trial 94 of 100, `summary.json` `chosen_best_params`) are:

| Parameter | Value |
|---|---|
| `n_estimators` | 150 |
| `max_depth` | 2 |
| `learning_rate` | 0.016540622580753447 |
| `min_child_weight` | 11 |
| `subsample` | 0.5 |
| `colsample_bytree` | 1.0 |
| `reg_lambda` | 3.164426633191584 |

Every other model field keeps its value in the base config.

## Fold layout for the re-baseline

`summary.json` `rebaseline_fold_layout` gives the layout #496 uses. The hyperparameters are frozen at the values above, and nothing is re-tuned on the evaluation range.

- **Data:** the base config with `end_date` extended over the evaluation range and `start_date: 2019-08-01`. The rows stay in chronological order, as `odds_analytics.training.cross_validation.make_walk_forward_splits` assumes.
- **Window:** `window_type: expanding`, `val_step_events: 50`.
- **First fold:** `min_train_events: 1900`, the tuning-range event count, so the first validation block starts at the first event on or after 2024-08-01. The 1,900th event must be `op_2023-2024_ARS_EVE_2024-05-19` (2024-05-19). If the tuning-range event set has changed, recount it before trusting the layout.
- **Later folds:** each fold trains on every earlier event, including earlier evaluation-range events, and the last partial block is dropped.

## Caveats

- **Tuning bias favours sliding-760.** Each cell is scored on rows its own tuner saw, so every pooled MSE is optimistic. Sliding-760's tuning folds are exactly the 1,100 common events, so its tuned objective equals its pooled MSE. Expanding and sliding-380 were tuned on 1,500 events, 400 of them outside the common range. The bias tilts the comparison towards sliding-760, which still did not clear the rule, so the choice of expanding stands. The absolute levels are not out-of-sample estimates.
- **One seed.** Seed-to-seed spread was not measured, so the gaps between cells, all within the bootstrap CIs, may be smaller than seed noise.
- **Late best trials.** Every cell's best trial came at trial 94 or 95, but the best value improved by at most 2.7 × 10⁻⁶ after trial 50 (`trials.csv`), well inside the CIs.
- **Tabular only.** The protocol was chosen on the tabular group. The match_stats group was not re-tested.
- **One decision regime.** Every tuning row is a day-before FDUK snapshot. The evaluation range adds Betfair Exchange closing odds (from 2024-25) and FDUK Pinnacle gaps after July 2025 ([data used](clv-model-findings.md#data-used)), so the sharp-reference mix there differs from the tuning range.
- **Pooled MSE reflects target variance.** The 2020-21 season's first fold has MSE 0.00118, against a common-range level of about 0.00066 (`folds.csv`). That is one reason the tuned CV MSEs of the cells that include it are higher.

## Related

- [CV protocol grid](exp-cv-protocol-grid.md), [CLV model findings](clv-model-findings.md), [Cross-validation](../modelling/cross-validation.md)
