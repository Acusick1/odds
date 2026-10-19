---
type: explanation
title: Price-only models on the retail-EV target
description: The October 2026 exploratory run of price-only XGBoost on the ADR-0012 retail-EV target against the martingale baseline and a constant shrinkage of it, on 2024-25 and 2025-26 development data, with the tuning and scoring layout, per-fold results, the keep-bar metrics by horizon and retail source, how the gain spreads over events, and feature importances.
sources:
  - experiments/scripts/retail_ev_price_only.py
  - experiments/configs/xgboost_epl_retail_ev_price_only.yaml
  - experiments/configs/xgboost_epl_retail_ev.yaml
  - experiments/results/retail_ev_price_only/results.csv
  - experiments/results/retail_ev_price_only/summary.json
  - experiments/results/retail_ev_price_only/folds.csv
  - experiments/results/retail_ev_price_only/fold_event_types.csv
  - experiments/results/retail_ev_price_only/trials.csv
  - experiments/results/retail_ev_price_only/event_contributions.csv
  - experiments/results/retail_ev_price_only/feature_importance.csv
  - odds_analytics.retail_ev.RetailEvFeatures
  - odds_analytics.retail_ev.RetailEvRow
  - odds_analytics.retail_ev_metrics.mse_gain
  - odds_analytics.retail_ev_metrics.pick_ev_gain
  - odds_analytics.training.cross_validation.make_walk_forward_splits
  - odds_analytics.feature_groups.filter_completed_events
status: stable
---

# Price-only models on the retail-EV target

This is the first model run on the retail-EV target of [ADR-0012](../../docs/adr/0012-retail-ev-against-betfair-exchange-close.md) (#540, under the map #506): a UK retail bet's EV against the devigged Betfair Exchange close, one row per event, outcome, book and retail snapshot ([CLV and the training target](../modelling/clv-target.md#the-retail-ev-target)). XGBoost sees only the price features at t. It is compared with the parameter-free martingale baseline, which takes the exchange price at t as the close.

**This is development data, and the results are exploratory.** ADR-0012 judges a model only on a pre-registered fresh season, so nothing here is a keep-bar result or a claim of skill. The run reads 2024-25 and 2025-26 only. 2026-27 is left untouched, because it is the candidate fresh season.

**Findings.**

- XGBoost fitted directly on the EV target is worse than the baseline.
- XGBoost fitted on the target minus the baseline (the residual arm) beats it by 35 to 49 × 10⁻⁶ weighted MSE, 1.0 to 1.4% of the baseline's MSE. Neither seed's CI excludes zero.
- The residual arm gains in 4 of the 5 folds, and is level with the baseline on the live-collected events. Its gain is spread over events rather than carried by a few.
- On picks, the residual arm is level with the baseline.

## Setup

- **Script and config.** `experiments/scripts/retail_ev_price_only.py` runs `experiments/configs/xgboost_epl_retail_ev_price_only.yaml` and writes `experiments/results/retail_ev_price_only/`. Its functions also run the [as-of-t group variants](exp-epl-retail-ev-as-of-t-groups.md).
  - The config inherits `experiments/configs/xgboost_epl_retail_ev.yaml`, whose `end_date` is 2026-07-31. It sets expanding walk-forward with `min_train_events` 380 and 68-event steps, sets `n_jobs` 1, and adds the search space.
  - `summary.json` records the git SHA, the SHA-256 of the script and config, and a SHA-256 of the rows' event IDs and targets (`data_sha256`). The script also writes `git_dirty`, whether any tracked file differed from that SHA when the run started; this run's `summary.json` predates that field.
- **Rows.** The rows come from the training pipeline (`odds_analytics.training.data_preparation.prepare_training_data_from_config` with `target_type: retail_ev`): 112,650 rows over 720 events. The current builder leaves 112,455, because it drops the 195 FDUK opening rows whose own snapshot has no usable exchange price ([CLV target](../modelling/clv-target.md#the-retail-ev-target)). The script fails unless they match `odds_analytics.retail_ev.load_retail_ev_dataset` row for row.
- **Event selection.** The pipeline and the script take every event in the date range and keep those with an ADR-0012 close, whatever their status or scores. The reported results use the 720 events with scores. Selecting by close adds the 29 FINAL events without scores (28 from March 2026), 573 rows, for kick-offs up to 2026-07-31.
- **Features.** These are the 38 price columns of `odds_analytics.retail_ev.RetailEvFeatures` with no optional groups, each from the row's own snapshot or the exchange price at or before t: hours to kick-off, retail, book-fair, consensus and exchange probabilities and their differences, the overrounds, the exchange price's age, `baseline_ev`, weekday, and one-hot outcome and book. There are no event-level groups, because `target_type: retail_ev` rejects them, and no as-of-t groups ([training config](../modelling/training-config.md)).
- **Loss weights.** Each row is weighted by `(1/odds²) × w_event`, scaled to mean one over all rows. The tuning subset keeps that scale, so `min_child_weight` means the same number of events in tuning as in scoring (about 150 weight units per event).
- **Tuning range.** Hyperparameters are tuned only on the 380 events of 2024-25 (3,861 rows, all from FDUK opening snapshots).
  - The tuning folds are expanding walk-forward inside that season: `min_train_events` 130, 50-event steps and the 7-day gap, which makes 5 folds.
  - Each run gets 50 TPE trials, with the Optuna study in sqlite under `experiments/optuna/`.
  - The best trial is then frozen, as [ADR-0010](../../docs/adr/0010-expanding-walk-forward-and-evaluation-range.md) does for its evaluation range. No trial saw a scored row.
- **Scoring folds.** `odds_analytics.training.cross_validation.make_walk_forward_splits` uses `min_train_events` 380, so the first validation event is the first 2025-26 event. Training events that kick off within 7 days of a validation block's first event are dropped.
  - The 68-event step is chosen so that the 340 scored 2025-26 events make 5 folds with no partial block. With 50-event steps, the last 40 events, which hold 40 of the 44 OddsPortal live events of April and May 2026, would fall in an unscored partial block.
  - The scored range is 340 events and 108,789 rows, kicking off from 2025-08-15 to 2026-05-24.
- **Predictions.** Each scored row gets six:
  - **Baseline:** `p_BFE_fair(t) × odds − 1` (ADR-0012).
  - **Shrinkage:** baseline × β, with β fitted by weighted least squares through the origin on the fold's training rows.
  - **Direct arm, two seeds:** XGBoost fitted on the EV target.
  - **Residual arm, two seeds:** XGBoost fitted on target minus baseline, with the baseline added back to its prediction. A model that learns nothing then reproduces the baseline. This is a script-level target transform; the pipeline has no such option.
  - The seeds are 42 and 7, used both as the Optuna sampler seed and as XGBoost's `random_state`.
- **Metrics.**
  - `odds_analytics.retail_ev_metrics.mse_gain` gives baseline weighted MSE minus challenger weighted MSE.
  - `odds_analytics.retail_ev_metrics.pick_ev_gain` gives the total `w_event`-weighted realised EV of rows with predicted EV > 0, challenger minus baseline.
  - Both come with event-clustered bootstrap 95% CIs over per-event totals (2,000 resamples, the same resampled events for every comparison).
  - A pick total is in event units: `w_event` sums to 1 per event, so 0.28 is 0.28 of one event's worth of EV.
  - Shrinkage by β > 0 never changes the sign of a prediction, so its picks equal the baseline's and its pick gain is zero by construction.
  - The horizon buckets and season labels are properties of each `odds_analytics.retail_ev.RetailEvRow`, computed from its decision time and kick-off.

## The scored rows

The 340 scored events contribute equal weight, but their rows are very uneven (`results.csv`):

| Slice | Events | Rows | Sum of `w_event` |
|---|---|---|---|
| FDUK-only events | 271 | 5,325 | 271 |
| Live-collected events | 69 | 103,464 | 69 |
| Retail source `fduk_opening` | 340 | 6,714 | 280.4 |
| Retail source `oddsportal_live` | 44 | 100,926 | 42.3 |
| Retail source `oddsportal` | 25 | 1,149 | 17.3 |

- **The dense rows fall in the last fold.** 102,186 of the 108,789 scored rows are in fold 4, 2026-02-28 to 2026-05-24 (`folds.csv`). Its 68 events are the 44 OddsPortal live events and 24 FDUK-only events kicking off from 2026-02-28 to 2026-04-19.
- **No model trains on dense live history before it is scored on it.** Fold 4 trains on 10,269 rows, almost all FDUK opening rows.
- **The horizon buckets are uneven in weight as well as rows.** 0–1 h has 1,998 rows but 0.7 event units, and 1–6 h has 11.3. Only 6–24 h (135.5) and 1–3 d (155.2) carry much weight.

## Tuning

Mean fold weighted MSE on the five tuning folds, in units of 10⁻⁶ (`summary.json`, `trials.csv`):

| Run | Best trial | Best | Median trial | Hyperparameters of the best trial |
|---|---|---|---|---|
| Baseline | | 3651.5 | | none |
| Shrinkage | | 3653.0 | | none |
| Direct, seed 42 | 44 | 3756.2 | 3836.7 | 100 trees, depth 2, lr 0.031, mcw 32, subsample 0.5, colsample 0.5, λ 0.053 |
| Direct, seed 7 | 26 | 3774.0 | 3834.9 | 250 trees, depth 2, lr 0.015, mcw 20, subsample 0.5, colsample 0.5, λ 2.40 |
| Residual, seed 42 | 22 | 3662.3 | 3752.8 | 100 trees, depth 4, lr 0.010, mcw 1982, subsample 0.6, colsample 0.7, λ 1.95 |
| Residual, seed 7 | 41 | 3690.3 | 3765.3 | 100 trees, depth 5, lr 0.018, mcw 1827, subsample 0.5, colsample 0.9, λ 0.062 |

**None of the 200 trials beats the baseline on the tuning folds.**

- **Residual arm.** The tuner pushes towards less model. Both seeds take the lower bound of 100 trees and a learning rate near the lower bound of 0.01, and set `min_child_weight` near its upper bound of 2,000 (about 13 events per leaf).
- **Direct arm.** Both seeds take depth 2 and subsample 0.5, the lower bounds of both, with `min_child_weight` near its lower bound of 20.

The residual models collapse towards the baseline, not to a constant predictor. On the scoring folds they correlate 0.984 to 0.995 with the baseline. Their prediction minus the baseline has a weighted standard deviation of 0.5 to 0.9 pp, against 6.0 to 7.7 pp for the target (`folds.csv`).

The direct models have a smaller spread than the baseline (2.3 to 2.7 pp against 2.6 to 3.6 pp) and correlate only 0.87 to 0.95 with it. A piecewise-constant fit of the baseline loses more than any signal it adds (inferred).

## Results

Weighted MSE over the 340 scored events, in units of 10⁻⁶. Gains are baseline minus challenger, with the 95% event-clustered CI (`results.csv`):

| | Weighted MSE | Gain over baseline | Gain over shrinkage |
|---|---|---|---|
| Baseline | 3419.6 | 0 | |
| Shrinkage | 3416.5 | 3.1 [−1.4, 7.7] | 0 |
| Direct, seed 42 | 3495.8 | −76.2 [−185.5, 16.8] | −79.3 [−190.8, 14.9] |
| Direct, seed 7 | 3496.1 | −76.5 [−185.0, 18.8] | −79.6 [−189.1, 16.6] |
| Residual, seed 42 | 3370.8 | 48.8 [−13.8, 105.6] | 45.7 [−14.5, 102.2] |
| Residual, seed 7 | 3384.9 | 34.7 [−52.3, 115.9] | 31.6 [−53.0, 112.2] |

- **Shrinkage.** β is 1.018 to 1.028 in every fold, so the target is spread slightly wider than the baseline, not narrower. Its gain is within noise.
- **Seeds.** Seed 7 minus seed 42 is −0.3 [−14.8, 13.2] for the direct arm and −14.1 [−44.4, 15.3] for the residual arm.

Pick totals, in event units. Each pick gain is the run's total minus the baseline's:

| | Picks (rows) | Total realised EV | Pick gain |
|---|---|---|---|
| Baseline | 1,571 | 0.275 | 0 |
| Direct, seed 42 | 638 | 0.011 | −0.265 [−0.436, −0.103] |
| Direct, seed 7 | 779 | 0.002 | −0.273 [−0.445, −0.115] |
| Residual, seed 42 | 1,848 | 0.305 | 0.029 [−0.025, 0.088] |
| Residual, seed 7 | 2,207 | 0.283 | 0.007 [−0.053, 0.071] |

- **Direct arm.** It loses on picks, with a CI that excludes zero. The worse fit costs real money on these rows.
- **Baseline.** Its picks already have positive realised total EV against the close. 0.197 of its 0.275 comes from the 1,149 `oddsportal` rows of 25 events (September 2025 to January 2026), and 0.069 from FDUK opening rows (`results.csv`, `pick_total_ev_baseline`).

### By fold

Weighted MSE per fold, in units of 10⁻⁶ (`folds.csv`). Each fold has 68 validation events:

| Fold | Validation kick-offs | Rows | Baseline | Shrinkage | Direct 42 | Direct 7 | Residual 42 | Residual 7 |
|---|---|---|---|---|---|---|---|---|
| 0 | 2025-08-15 to 2025-10-05 | 1,641 | 3164.8 | 3161.2 | 3277.2 | 3270.3 | 3100.3 | 3118.9 |
| 1 | 2025-10-05 to 2025-12-03 | 1,836 | 2875.6 | 2868.7 | 3119.9 | 3133.2 | 2828.1 | 2854.0 |
| 2 | 2025-12-03 to 2026-01-07 | 1,812 | 2734.9 | 2737.5 | 2777.8 | 2776.5 | 2749.4 | 2786.4 |
| 3 | 2026-01-07 to 2026-02-27 | 1,314 | 3467.3 | 3462.9 | 3364.0 | 3378.0 | 3355.7 | 3329.2 |
| 4 (live) | 2026-02-28 to 2026-05-24 | 102,186 | 4828.2 | 4825.0 | 4916.8 | 4899.3 | 4793.1 | 4808.6 |

- **Residual arm.** Both seeds beat the baseline in folds 0, 1, 3 and 4, and lose in fold 2.
- **Direct arm.** It beats the baseline only in fold 3.
- **The live fold.** The residual arm's gain of 35 and 20 in fold 4 does not come from the live events.

Fold 4 split by event type, weighted MSE in units of 10⁻⁶ (`fold_event_types.csv`):

| Fold 4 events | Events | Rows | Baseline | Shrinkage | Direct 42 | Direct 7 | Residual 42 | Residual 7 |
|---|---|---|---|---|---|---|---|---|
| FDUK-only | 24 | 384 | 2762.5 | 2751.8 | 2852.6 | 2803.4 | 2655.2 | 2629.2 |
| Live-collected | 44 | 101,802 | 5882.6 | 5883.2 | 5970.4 | 5969.1 | 5884.4 | 5921.0 |

- **Live events.** On the 44 live-collected events the residual arm is level with or worse than the baseline. Its fold-4 gain comes from the 24 FDUK-only events.
- **The live slices over all folds agree.** The residual gain is −1.0 and −39.7 on OddsPortal live rows, +4.5 and −30.9 where the close is from live Betfair, and +28.8 [−93.7, 147.7] and −16.5 [−196.1, 161.4] on live-collected events.

### By slice

Gain over the baseline in weighted MSE (10⁻⁶), for the residual arm, with the direct seed-42 arm for comparison:

| Slice | Events | Baseline wMSE | Residual, seed 42 | Residual, seed 7 | Direct, seed 42 |
|---|---|---|---|---|---|
| 0–1 h | 29 | 403.7 | −18.7 [−67.9, 30.5] | −75.3 [−167.2, 6.5] | −93.0 [−185.8, −12.6] |
| 1–6 h | 42 | 2623.3 | −83.8 [−195.6, 34.4] | −203.9 [−402.4, −4.1] | −254.3 [−529.4, −37.2] |
| 6–24 h | 164 | 2955.6 | 123.8 [33.5, 215.4] | 108.9 [−15.4, 233.0] | −59.9 [−222.6, 72.6] |
| 1–3 d | 194 | 3379.6 | 28.0 [−61.5, 114.6] | 23.8 [−111.4, 148.5] | −77.5 [−286.9, 59.2] |
| 3–7 d | 55 | 5621.6 | −104.0 [−229.5, 20.8] | −122.6 [−309.8, 67.9] | −77.3 [−183.8, 28.2] |
| FDUK opening | 340 | 3090.6 | 53.4 [−13.1, 117.3] | 46.1 [−46.7, 135.8] | 20.3 [−47.8, 89.5] |
| OddsPortal live | 44 | 5968.0 | −1.0 [−122.5, 118.9] | −39.7 [−227.9, 150.4] | −85.3 [−294.2, 84.7] |
| OddsPortal (other) | 25 | 2248.5 | 99.4 [−182.9, 351.6] | 40.2 [−368.5, 393.6] | −1604.9 [−3327.6, −413.1] |
| Close from live Betfair | 41 | 6166.6 | 4.5 [−115.5, 131.3] | −30.9 [−219.5, 167.6] | −73.9 [−268.7, 98.0] |
| Close from FDUK | 299 | 3025.8 | 55.1 [−11.4, 116.4] | 44.1 [−50.8, 128.1] | −76.6 [−205.1, 27.7] |
| Home | 340 | 3101.7 | 84.5 [11.6, 154.6] | 73.1 [−30.0, 174.4] | −45.9 [−133.9, 36.5] |
| Draw | 340 | 2602.6 | 83.9 [36.4, 132.3] | 97.3 [25.8, 169.2] | 95.5 [15.8, 177.9] |
| Away | 340 | 4389.1 | −32.1 [−136.7, 69.7] | −64.5 [−214.9, 74.2] | −215.1 [−418.5, −51.9] |

`results.csv` has 16 slices per run, counting the total. The two seeds of an arm are fitted on the same rows, so they are not independent tests.

- **Multiple comparisons.** Across 4 runs × 16 slices, a Bonferroni correction needs a z-score of about 3.4. Approximating the standard error from each CI's width (CI width / 3.92), residual seed 42 on draws (z ≈ 3.4) only just reaches it, and seed 7's draw slice (z ≈ 2.7) does not. No slice is a finding.
- **Strongest slices.** Draws and home rows are the strongest for the residual arm, and 6–24 h for seed 42. The near-kick-off buckets (0–1 h, 1–6 h) and 3–7 d go the other way.
- **Shrinkage on draws and home.** Its draw slice is −42.3 [−57.3, −26.8] and its home slice +26.3 [13.2, 39.1]: scaling every EV by β > 1 helps home rows and hurts draws.

### How the gain spreads over events

`event_contributions.csv` splits each run's pooled gain into per-event parts that sum to it. `summary.json` `concentration` summarises both tails. In MSE gain (10⁻⁶):

| Run | Total | Top 10 events | Bottom 10 events | Total without both tails | Median event | Events gaining / losing |
|---|---|---|---|---|---|---|
| Residual, seed 42 | 48.8 | 45.6 | −44.8 | 48.0 | 0.067 | 176 / 164 |
| Residual, seed 7 | 34.7 | 64.4 | −67.3 | 37.6 | 0.031 | 172 / 168 |
| Direct, seed 42 | −76.2 | 47.7 | −106.0 | −18.0 | 0.003 | 170 / 170 |
| Direct, seed 7 | −76.5 | 48.9 | −108.3 | −17.1 | −0.041 | 166 / 174 |

- **Residual arm.** Its two tails cancel, so its gain is spread over the events between them: trimming 10 events from each end leaves the total almost unchanged.
- **Direct arm.** Its loss is concentrated in its worst ten events.
- **FDUK-only events in the tails.** They are 10 and 9 of the residual arm's top tens, and 8 and 7 of its bottom tens. FDUK-only events are 80% of the scored events (`fduk_only_share_of_scored_events`), so this is close to the base rate and says nothing about the source.

## Feature importances

Mean share of total gain over the five fold models (`feature_importance.csv`), for features above 0.5% in some run:

| Feature | Direct, seed 42 | Direct, seed 7 | Residual, seed 42 | Residual, seed 7 |
|---|---|---|---|---|
| `sharp_prob` | 0.066 | 0.099 | 0.242 | 0.189 |
| `consensus_prob` | 0.055 | 0.051 | 0.176 | 0.182 |
| `exchange_overround` | 0.027 | 0.029 | 0.116 | 0.134 |
| `book_fair_prob` | 0.042 | 0.030 | 0.079 | 0.062 |
| `hours_to_kickoff` | 0.008 | 0.011 | 0.074 | 0.065 |
| `retail_sharp_diff` | 0.180 | 0.174 | 0.054 | 0.068 |
| `baseline_ev` | 0.532 | 0.502 | 0.052 | 0.067 |
| `num_bookmakers` | 0.006 | 0.009 | 0.042 | 0.041 |
| `book_overround` | 0.012 | 0.006 | 0.031 | 0.048 |
| `day_of_week` | 0.003 | 0.006 | 0.033 | 0.024 |
| `retail_consensus_diff` | 0.037 | 0.048 | 0.028 | 0.059 |
| `retail_implied_prob` | 0.015 | 0.018 | 0.028 | 0.022 |
| `is_weekend` | 0.002 | 0.002 | 0.015 | 0.010 |
| `outcome_draw` | 0.003 | 0.006 | 0.011 | 0.008 |

- **Direct arm.** It spends most of its splits rebuilding `baseline_ev` and `retail_sharp_diff`, the baseline's own inputs.
- **Residual arm.** It splits on the exchange probability's level (`sharp_prob`), the retail consensus, the exchange overround and the horizon. It looks for where the exchange price at t is a biased forecast of its close: by price level, by agreement with the retail books and by the exchange book's width (inferred).
- **Book identity.** No book one-hot carries more than 0.5%.
- **Exchange age.** `exchange_age_hours` has zero gain, because nearly all of the training rows are FDUK openings, whose exchange price is read from the same snapshot.

## Caveats

- **Development data.** The rows have been examined before (#522, #538), and the slices were chosen with the data in view. Every number here is exploratory.
- **The fold step was set by this data.** The 68-event step was chosen after seeing that 50-event steps leave the live events unscored.
- **Tuning on another regime.** The tuning season has only FDUK opening rows, one snapshot per event and 1 to 4 days out. The tuned hyperparameters never saw a live-collected row.
- **One fold holds the dense data.** The live-collected rows are scored once, by models trained without them.
- **The residual arm is a script transform.** The pipeline's `retail_ev` target and its tuner fit the EV directly, which is the direct arm.
- **The search space.** The `min_child_weight` range of 20 to 2,000 was scaled to the mean-one weights. Both residual seeds settled at the edges of the space that give the least model.
- **Fifty trials per run** on five small folds.

## Related

- [CLV and the training target](../modelling/clv-target.md), [Cross-validation](../modelling/cross-validation.md), [Hyperparameter tuner](../modelling/hyperparameter-tuner.md), [What the CLV model can and cannot do](clv-model-findings.md), [EPL CLV re-baseline against the simple baseline](exp-epl-clv-rebaseline.md)
