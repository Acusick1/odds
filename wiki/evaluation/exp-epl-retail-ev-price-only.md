---
type: explanation
title: Price-only models on the retail-EV target
description: The October 2026 exploratory run of price-only XGBoost on the ADR-0012 retail-EV target against the martingale baseline and a constant shrinkage of it, on development data, with the tuning and scoring layout, the keep-bar metrics by horizon and retail source, where the gain comes from, and feature importances.
sources:
  - experiments/scripts/retail_ev_price_only.py
  - experiments/configs/xgboost_epl_retail_ev_price_only.yaml
  - experiments/configs/xgboost_epl_retail_ev.yaml
  - experiments/results/retail_ev_price_only/results.csv
  - experiments/results/retail_ev_price_only/summary.json
  - experiments/results/retail_ev_price_only/folds.csv
  - experiments/results/retail_ev_price_only/trials.csv
  - experiments/results/retail_ev_price_only/event_contributions.csv
  - experiments/results/retail_ev_price_only/feature_importance.csv
  - odds_analytics.retail_ev.retail_ev_features
  - odds_analytics.retail_ev_metrics.mse_gain
  - odds_analytics.retail_ev_metrics.pick_ev_gain
  - odds_analytics.training.cross_validation.make_walk_forward_splits
status: stable
---

# Price-only models on the retail-EV target

This is the first model run on the retail-EV target of [ADR-0012](../../docs/adr/0012-retail-ev-against-betfair-exchange-close.md) (#540, under the map #506): a UK retail bet's EV against the devigged Betfair Exchange close, one row per event, outcome, book and retail snapshot ([CLV and the training target](../modelling/clv-target.md#the-retail-ev-target)). XGBoost sees only the price features at t. It is compared with the parameter-free martingale baseline, which takes the exchange price at t as the close.

**This is development data, and the results are exploratory.** ADR-0012 judges a model only on a pre-registered fresh season. Nothing here is a keep-bar result or a claim of skill.

**Findings.**

- XGBoost fitted directly on the EV target is worse than the baseline.
- XGBoost fitted on the target minus the baseline (the residual arm) beats it by 28 to 45 × 10⁻⁶ weighted MSE, about 1% of the baseline's MSE. Neither seed's CI excludes zero.
- About ten events, nearly all FDUK-only, supply all of that gain.
- On picks, the residual arm is level with the baseline.

## Setup

- **Script and config.** `experiments/scripts/retail_ev_price_only.py` runs `experiments/configs/xgboost_epl_retail_ev_price_only.yaml` and writes `experiments/results/retail_ev_price_only/`.
  - The config inherits `experiments/configs/xgboost_epl_retail_ev.yaml`. It extends `end_date` to 2026-10-18, keeps expanding walk-forward with `min_train_events` 380 and 50-event steps, sets `n_jobs` 1, and adds the search space.
  - `summary.json` records the git SHA and the SHA-256 of the script and config.
- **Rows.** The rows come from the training pipeline (`odds_analytics.training.data_preparation.prepare_training_data_from_config` with `target_type: retail_ev`): 113,334 rows over 758 events. The script fails unless they match `odds_analytics.retail_ev.load_retail_ev_dataset` row for row.
  - The pipeline keeps only events with a final score. That is 29 events fewer than `experiments/scripts/retail_ev_target_report.py` counts, because the report does not filter on scores.
- **Features.** These are the 38 price-at-t columns of `odds_analytics.retail_ev.retail_ev_features`: hours to kick-off, retail, book-fair, consensus and exchange probabilities and their differences, the overrounds, the exchange price's age, `baseline_ev`, weekday, and one-hot outcome and book. There are no event-level groups, because `target_type: retail_ev` rejects them ([training config](../modelling/training-config.md)).
- **Loss weights.** Each row is weighted by `(1/odds²) × w_event`, scaled to mean one over all rows. The tuning subset keeps that scale, so `min_child_weight` means the same number of events in tuning as in scoring (about 150 weight units per event).
- **Tuning range.** Hyperparameters are tuned only on the 380 events of 2024-25 (3,861 rows, all from FDUK opening snapshots).
  - The tuning folds are expanding walk-forward inside that season: `min_train_events` 130, 50-event steps and the 7-day gap, which makes 5 folds.
  - Each run gets 50 TPE trials, with the Optuna study in sqlite under `experiments/optuna/`.
  - The best trial is then frozen, as [ADR-0010](../../docs/adr/0010-expanding-walk-forward-and-evaluation-range.md) does for its evaluation range. No trial saw a scored row.
- **Scoring folds.** `odds_analytics.training.cross_validation.make_walk_forward_splits` uses `min_train_events` 380, so the first validation event is the first 2025-26 event. Each validation block is 50 events, and training events that kick off within 7 days of the block's first event are dropped.
  - This gives 7 folds: 350 events and 108,969 rows, kicking off from 2025-08-15 to 2026-08-31.
  - The last 28 events (504 rows) form a partial block and are not scored.
- **Predictions.** Each scored row gets six:
  - **Baseline:** `p_BFE_fair(t) × odds − 1` (ADR-0012).
  - **Shrinkage:** baseline × β, with β fitted by weighted least squares through the origin on the fold's training rows.
  - **Direct arm, two seeds:** XGBoost fitted on the EV target.
  - **Residual arm, two seeds:** XGBoost fitted on target minus baseline, with the baseline added back to its prediction. A model that learns nothing then reproduces the baseline. This is a script-level target transform; the pipeline has no such option.
  - The seeds are 42 and 7, used both as the Optuna sampler seed and as XGBoost's `random_state`.
- **Metrics.**
  - `odds_analytics.retail_ev_metrics.mse_gain` gives baseline weighted MSE minus challenger weighted MSE.
  - `odds_analytics.retail_ev_metrics.pick_ev_gain` gives the total `w_event`-weighted realised EV of rows with predicted EV > 0, challenger minus baseline.
  - Both come with event-clustered bootstrap 95% CIs (2,000 resamples, the same resampled events for every comparison).
  - A pick total is in event units: `w_event` sums to 1 per event, so 0.27 is 0.27 of one event's worth of EV.
  - Shrinkage by β > 0 never changes the sign of a prediction, so its picks equal the baseline's and its pick gain is zero by construction.

## The scored rows

The 350 scored events contribute equal weight, but their rows are very uneven (`results.csv`):

| Slice | Events | Rows | Sum of `w_event` |
|---|---|---|---|
| FDUK-only events | 281 | 5,505 | 281 |
| Live-collected events | 69 | 103,464 | 69 |
| Retail source `fduk_opening` | 350 | 6,894 | 290.4 |
| Retail source `oddsportal_live` | 44 | 100,926 | 42.3 |
| Retail source `oddsportal` | 25 | 1,149 | 17.3 |

- **The dense rows fall in one fold.** 101,133 of the 108,969 scored rows are in the last fold, 2026-05-02 to 2026-08-31 (`folds.csv`). That fold trains on 10,974 rows, almost all FDUK opening rows.
- **No model trains on dense live history before it is scored on it.** In row count the folds test FDUK-trained models on live data. In weight, though, the live-collected events are only 69 of 350.
- **The horizon buckets are uneven in weight as well as rows.** 0–1 h has 1,998 rows but 0.7 event units, and 1–6 h has 12.3. Only 6–24 h (138.5) and 1–3 d (160.2) carry real weight, mostly from the FDUK opening price.

## Tuning

Mean fold weighted MSE on the five tuning folds, in units of 10⁻⁶ (`summary.json`, `trials.csv`):

| Run | Best trial | Best | Median trial | Hyperparameters of the best trial |
|---|---|---|---|---|
| Baseline | | 3651.5 | | none |
| Shrinkage | | 3653.0 | | none |
| Direct, seed 42 | 30 | 3776.3 | 3834.9 | 350 trees, depth 2, lr 0.0105, mcw 22, subsample 0.5, colsample 0.5, λ 0.012 |
| Direct, seed 7 | 42 | 3777.2 | 3842.3 | 400 trees, depth 2, lr 0.0117, mcw 25, subsample 0.6, colsample 0.7, λ 3.78 |
| Residual, seed 42 | 42 | 3660.5 | 3747.1 | 100 trees, depth 4, lr 0.0113, mcw 1916, subsample 0.6, colsample 0.8, λ 0.44 |
| Residual, seed 7 | 49 | 3697.5 | 3795.8 | 100 trees, depth 2, lr 0.0142, mcw 422, subsample 0.9, colsample 0.5, λ 0.88 |

**None of the 200 trials beats the baseline on the tuning folds.**

- **Direct arm.** Both seeds settle on shallow, slow-learning trees with a small `min_child_weight`. These do not reproduce the identity in `baseline_ev`.
- **Residual arm.** Both seeds settle on 100 trees at a learning rate near 0.01, which shrinks every step heavily: the residual models stay close to zero and so close to the baseline.

This is a collapse towards the baseline, not to a constant predictor. On the scoring folds, the residual models correlate 0.99 to 0.997 with the baseline. Their prediction minus the baseline has a weighted standard deviation of 0.4 to 0.65 pp, against 6.0 to 8.2 pp for the target (`folds.csv`).

The direct models are worse than that. Their predictions have a smaller spread than the baseline's (2.3 to 3.0 pp against 2.6 to 3.7 pp) and correlate only 0.88 to 0.98 with it. A piecewise-constant fit of the baseline loses more than any signal it adds (inferred).

## Results

Weighted MSE over the 350 scored events, in units of 10⁻⁶. Gains are baseline minus challenger, with the 95% event-clustered CI (`results.csv`):

| | Weighted MSE | Gain over baseline | Gain over shrinkage |
|---|---|---|---|
| Baseline | 3399.5 | 0 | |
| Shrinkage | 3397.0 | 2.5 [−2.2, 7.3] | 0 |
| Direct, seed 42 | 3470.9 | −71.5 [−170.4, 17.4] | −74.0 [−174.8, 15.5] |
| Direct, seed 7 | 3457.1 | −57.7 [−150.4, 29.0] | −60.2 [−154.7, 26.7] |
| Residual, seed 42 | 3354.9 | 44.5 [−15.8, 106.6] | 42.0 [−17.0, 102.7] |
| Residual, seed 7 | 3371.0 | 28.5 [−22.5, 78.8] | 26.0 [−23.2, 74.2] |

- **Shrinkage.** β is 1.018 to 1.029 in every fold, so the target is spread slightly wider than the baseline, not narrower. Its gain is within noise.
- **Seeds.** Seed 7 minus seed 42 is 13.8 [−5.5, 33.2] for the direct arm and −16.0 [−44.1, 9.8] for the residual arm. The seed spread is about half the residual arm's gain.

Pick totals, in event units. Each pick gain is the run's total minus the baseline's:

| | Picks (rows) | Total realised EV | Pick gain |
|---|---|---|---|
| Baseline | 1,572 | 0.274 | 0 |
| Direct, seed 42 | 748 | 0.034 | −0.240 [−0.422, −0.092] |
| Direct, seed 7 | 1,381 | 0.082 | −0.192 [−0.353, −0.058] |
| Residual, seed 42 | 1,830 | 0.295 | 0.021 [−0.036, 0.077] |
| Residual, seed 7 | 1,701 | 0.296 | 0.022 [−0.029, 0.074] |

- **Direct arm.** It loses picks with a CI that excludes zero. The worse fit costs real money on these rows.
- **Baseline.** Its picks already have positive realised total EV against the close. 0.197 of its 0.274 comes from the 1,149 `oddsportal` rows of 25 events (September 2025 to January 2026), and 0.068 from FDUK opening rows (`results.csv`, `pick_total_ev_baseline`).

### By horizon and retail source

Gain over the baseline in weighted MSE (10⁻⁶) for the residual arm, with the direct seed-42 arm for comparison:

| Slice | Events | Baseline wMSE | Residual, seed 42 | Residual, seed 7 | Direct, seed 42 |
|---|---|---|---|---|---|
| 0–1 h | 29 | 403.7 | −31.5 [−85.9, 19.8] | −24.7 [−67.9, 14.1] | −92.4 [−180.8, −14.7] |
| 1–6 h | 43 | 2448.0 | −128.6 [−254.3, 0.7] | −138.1 [−269.6, −15.8] | −295.7 [−557.7, −77.6] |
| 6–24 h | 167 | 2951.9 | 119.1 [24.9, 219.7] | 114.5 [37.4, 199.0] | −34.7 [−172.8, 89.1] |
| 1–3 d | 199 | 3383.2 | 26.7 [−72.2, 112.3] | −16.6 [−101.4, 60.6] | −81.9 [−277.0, 54.8] |
| 3–7 d | 56 | 5488.7 | −100.6 [−263.7, 37.9] | −49.1 [−164.8, 59.9] | −91.8 [−218.5, 27.0] |
| FDUK opening | 350 | 3077.5 | 52.9 [−12.3, 120.1] | 28.3 [−26.6, 85.6] | 11.6 [−56.1, 80.0] |
| OddsPortal live | 44 | 5968.0 | −30.0 [−183.6, 121.4] | 14.2 [−114.0, 125.9] | −82.0 [−291.1, 88.8] |
| OddsPortal (other) | 25 | 2248.5 | 94.0 [−217.3, 375.3] | 68.0 [−207.9, 313.1] | −1427.4 [−3067.4, −331.4] |
| Close from live Betfair | 51 | 5514.6 | −30.8 [−169.3, 105.0] | −21.1 [−131.8, 75.5] | −112.5 [−289.3, 35.1] |
| Close from FDUK | 299 | 3025.8 | 57.8 [−11.6, 122.3] | 37.2 [−22.8, 94.8] | −64.2 [−185.9, 31.1] |
| Draw | 350 | 2571.3 | 76.4 [26.8, 124.8] | 80.9 [30.9, 132.2] | 85.2 [9.5, 160.2] |

- **6–24 h.** This is the only horizon where both residual seeds gain with CIs that exclude zero. It is almost entirely FDUK opening prices, and its gain is what makes the overall gain positive.
- **Close to kick-off (1–6 h, 0–1 h).** Both arms lose. The 0–1 h rows are all live-collected; at 1–6 h about two thirds of the weight is FDUK opening prices taken late, and the rest is live rows with little training history.
- **Draws.** Every XGBoost run improves on the baseline's draw rows, while the shrinkage is worse there (−42.4 [−58.6, −26.2]). Away rows go the other way.
- **Sixteen slices per run.** Some of these CIs exclude zero by chance alone. The slices are a guide to where to look, not results.
- **2026-27.** Only 10 events are scored (`results.csv`, `season`), too few to read.

### Where the gain comes from

`event_contributions.csv` splits each run's pooled gain into per-event parts that sum to it. `summary.json` `concentration` summarises them:

| Run | Total gain (10⁻⁶) | Top 10 events | Without the top 10 | Events with positive and negative gain |
|---|---|---|---|---|
| Residual, seed 42 | 44.5 | 46.1 | −1.5 | 182 / 168 |
| Residual, seed 7 | 28.5 | 40.7 | −12.2 | 184 / 166 |
| Direct, seed 42 | −71.5 | 45.8 | −117.3 | 175 / 175 |

**The residual arm's whole gain comes from about ten events.** Nine of residual seed 42's top ten are FDUK-only events with 15 to 21 rows each; one is a live-collected event with 66 rows. The densest events do not dominate, because `w_event` caps every event at the same weight. The top ten's pick gain (0.061 and 0.056) is also larger than the whole pick gain.

## Feature importances

Mean share of total gain over the seven fold models (`feature_importance.csv`), for features above 0.5% in some run:

| Feature | Direct, seed 42 | Direct, seed 7 | Residual, seed 42 | Residual, seed 7 |
|---|---|---|---|---|
| `sharp_prob` | 0.069 | 0.090 | 0.245 | 0.387 |
| `consensus_prob` | 0.060 | 0.045 | 0.195 | 0.194 |
| `exchange_overround` | 0.028 | 0.033 | 0.115 | 0.072 |
| `hours_to_kickoff` | 0.012 | 0.014 | 0.088 | 0.076 |
| `retail_sharp_diff` | 0.190 | 0.157 | 0.055 | 0.028 |
| `book_fair_prob` | 0.038 | 0.021 | 0.053 | 0.043 |
| `baseline_ev` | 0.514 | 0.555 | 0.046 | 0.013 |
| `num_bookmakers` | 0.007 | 0.009 | 0.041 | 0.042 |
| `day_of_week` | 0.004 | 0.005 | 0.027 | 0.043 |
| `retail_implied_prob` | 0.020 | 0.014 | 0.023 | 0.039 |
| `retail_consensus_diff` | 0.039 | 0.034 | 0.036 | 0.002 |
| `outcome_draw` | 0.004 | 0.005 | 0.012 | 0.028 |

- **Direct arm.** It spends most of its splits rebuilding `baseline_ev` and `retail_sharp_diff`, the baseline's own inputs.
- **Residual arm.** It splits on the exchange probability's level (`sharp_prob`), the retail consensus, the exchange overround and the horizon. It looks for where the exchange price at t is biased as a forecast of its close: by price level, by agreement with the retail books and by the exchange book's width (inferred).
- **Book identity.** No book one-hot carries more than 0.1%.
- **Exchange age.** `exchange_age_hours` has zero gain, because nearly all of the training rows are FDUK openings, whose exchange price is read from the same snapshot.

## Caveats

- **Development data.** The rows have been examined before (#522, #538), and the slices were chosen with the data in view. Every number here is exploratory.
- **Tuning on another regime.** The tuning season has only FDUK opening rows, one snapshot per event and 1 to 4 days out. The tuned hyperparameters never saw a live-collected row.
- **One fold holds the dense data.** The live-collected rows are effectively scored once, by models trained without them. Their slices say little about a model trained on dense live history.
- **The residual arm is a script transform.** The pipeline's `retail_ev` target and its tuner fit the EV directly, which is the direct arm.
- **The search space.** The `min_child_weight` range of 20 to 2,000 was scaled to the mean-one weights, at about 0.13 to 13 events per leaf `(inferred)`. Both residual seeds chose lr near 0.01 with only 100 trees, at the edge of the space. A space allowing fewer, stronger steps, or early stopping, might find a better residual model. It might also find nothing.
- **Fifty trials per run** on five small folds. The seed spread on the residual arm (16 × 10⁻⁶) is a sizeable fraction of its gain.

## Related

- [CLV and the training target](../modelling/clv-target.md), [Cross-validation](../modelling/cross-validation.md), [Hyperparameter tuner](../modelling/hyperparameter-tuner.md), [What the CLV model can and cannot do](clv-model-findings.md), [EPL CLV re-baseline against the simple baseline](exp-epl-clv-rebaseline.md)
