---
type: reference
title: Experiment scripts and artifacts
description: Layout of experiments/, config and script conventions, the verification artifact files, and the inventory of scripts.
sources:
  - odds_analytics.training.data_preparation.save_verification_artifacts
  - odds_analytics.training.config.MLTrainingConfig
  - experiments/scripts/retail_ev_target_report.py
  - experiments/scripts/retail_ev_power.py
  - experiments/configs/xgboost_epl_retail_ev.yaml
  - experiments/configs/xgboost_epl_retail_ev_price_only.yaml
status: stable
---

# Experiment scripts and artifacts

`experiments/` holds `scripts/` (standalone Python), `configs/` (training YAMLs) and `results/<experiment_name>/` (outputs). Steps for running and recording an experiment are on [Run and record a modelling experiment](howto-run-experiment.md).

## Script and config conventions

- Scripts load training data through `odds_analytics.feature_groups.prepare_training_data` or `odds_analytics.retail_ev.load_retail_ev_dataset`, with an `odds_analytics.training.config.MLTrainingConfig` read from `experiments/configs/` (the loader validates strategy type, tuning metric and search-space keys, target fields, the validation split a non-CV tuning run needs and the limits of the `retail_ev` residual arm, and rejects a `direction` key) ([training config](../modelling/training-config.md)).
- A YAML can name a parent in `base`; the loader deep-merges the child's keys over the parent's, so `experiments/configs/xgboost_epl_retail_ev_price_only.yaml` overrides only what differs from `experiments/configs/xgboost_epl_retail_ev.yaml`.
- The default CV protocol is not walk-forward, so a YAML must set the target and protocol itself; `experiments/configs/xgboost_epl_retail_ev.yaml` sets `target_type: retail_ev` with expanding walk-forward ([target functions](../modelling/clv-target-functions.md), [cross-validation](../modelling/cross-validation.md)).
- Scripts run as `uv run python experiments/scripts/<script>.py`.
- XGBoost configs under `experiments/configs/` have no early-stopping setting; boosting length is `n_estimators`, which the search space tunes ([training config](../modelling/training-config.md)).

## Results directories

A results directory holds CSV tables, a `summary.json` and, for the retail-EV evaluation, a manifest ([retail-EV evaluation](retail-ev-evaluation.md#outputs)). The experiment page that reports a run links its directory. `experiments/results/` holds only some runs' outputs, so a missing results directory does not mean the run did not happen.

## Verification artifacts

`odds train run` and `odds train tune` write them automatically ([training pipeline](../modelling/training-pipeline.md)). A script can call `odds_analytics.training.data_preparation.save_verification_artifacts(X, y, feature_names, output_dir, event_ids=event_ids)`, which writes:

| File | Content |
|---|---|
| `feature_stats.csv` | count, null_count, null_pct, mean, std, min, max, nunique per feature |
| `feature_sample.csv` | first 20 rows with event ids and target |
| `feature_target_corr.csv` | Pearson and Spearman correlation with the target |

## Inventory

| Script | Subject | Page |
|---|---|---|
| `retail_ev_target_report` | Row counts (events selected by close) by season, source, horizon bucket and book for the retail-EV target (the season and horizon bucket are properties of each row's `odds_analytics.retail_ev.RetailEvInput`), with the baseline's weighted MSE; trains no model | [CLV and the training target](../modelling/clv-target.md#the-retail-ev-target) |
| `retail_ev_exchange_overround` | Gap between the exchange price at t and the close by back overround, and three policies for wide exchange books | none |
| `retail_ev_price_only` | Price-only XGBoost on the retail-EV target, direct and residual arms over two seeds, against the martingale baseline and a constant shrinkage | [Price-only models on the retail-EV target](exp-epl-retail-ev-price-only.md) |
| `retail_ev_weight_scale` | Mean loss weight per event and the events a frozen `min_child_weight` stands for, under the fixed scale and under mean-one rescaling, and the frozen residual trial refitted under each weighting | [Price-only rerun](exp-epl-retail-ev-price-only.md#rerun-with-the-pipeline-residual-arm-and-a-fixed-weight-scale) |
| `retail_ev_power` | Simulated power of the retail-EV keep-bar tests on resampled fresh seasons (events with a close), for calibrated, sparse and no-information synthetic models, with a noise-seed sweep at one season | [Power of the retail-EV keep bar](exp-retail-ev-power.md) |
| `retail_ev_as_of_t_groups` | The `retail_ev_price_only` protocol with each as-of-t feature group added alone and all together, against the baseline and the price-only model | [As-of-t feature groups on the retail-EV target](exp-epl-retail-ev-as-of-t-groups.md) |
| `retail_ev_nested_tuning` | Residual-arm retail-EV models for price-only, Polymarket at t and all four as-of-t groups, re-tuned for every scoring fold by nested walk-forward inside 2025-26, against the baseline, price-only and the frozen 2024-25 settings | [Nested tuning on the as-of-t columns](exp-epl-retail-ev-nested-tuning.md) |
| `betfair_historical_analysis` | Parses Betfair historical NBA files | none |
| `betfair_nba_explore` | Live Betfair NBA market survey | none |
| `polymarket_mlb_discover`, `polymarket_mlb_orderbooks`, `polymarket_mlb_sentiment`, `polymarket_mlb_volume` | MLB Polymarket survey | [Polymarket market research](../collection/polymarket-market-research.md) |
| `show_agent_summaries` | Prints agent run summaries from `logs/agent_runs/` | [Reporting surfaces](../operations/reporting-surfaces.md) |
