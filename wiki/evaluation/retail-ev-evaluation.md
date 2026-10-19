---
type: reference
title: Retail-EV evaluation
description: The one-look evaluation of a frozen retail-EV model on a window of events, with its spec, the window-end guard, the refit blocks, the keep-bar statistics and verdict, the coverage report, the descriptive slices and the output files.
sources:
  - odds_analytics.retail_ev_evaluation
  - odds_cli.commands.evaluate
status: stable
---

# Retail-EV evaluation

`odds_analytics.retail_ev_evaluation` scores a frozen model for the [retail-EV target](../modelling/clv-target.md#the-retail-ev-target) once, on every event of a window, and returns ADR-0012's keep or retire verdict ([ADR-0012](../../docs/adr/0012-retail-ev-against-betfair-exchange-close.md)). `odds evaluate retail-ev` (`odds_cli.commands.evaluate.evaluate_retail_ev`) runs it; the steps are on [Run the retail-EV evaluation](howto-run-retail-ev-evaluation.md).

## Spec

`odds_analytics.retail_ev_evaluation.EvaluationSpec` is a YAML file that holds every choice the run makes. Unknown fields are rejected. Each field's default and description are in the class.

| Field | Meaning |
|---|---|
| `training_start` | First kick-off read, so the first training event |
| `window.start`, `window.end` | First and last kick-off scored, both inclusive and timezone-aware |
| `refit_every_events` | Window events scored by each refit |
| `gap` | A block trains on events that kick off at least this long before its first event; shorter than `odds_analytics.retail_ev.DECISION_WINDOW` is rejected |
| `feature_groups` | As-of-t groups added to the price columns ([feature groups](../modelling/feature-groups.md)) |
| `arm` | `odds_analytics.training.config.RetailEvArm`; the residual arm takes `baseline_ev` as each row's base margin |
| `model` | `odds_analytics.training.config.XGBoostConfig` parameters |
| `trial` | `odds_analytics.retail_ev_evaluation.SummaryTrial`: a run's `best_params` from an experiment `summary.json`, path relative to the spec file; they replace the `model` fields they name |
| `seed` | XGBoost `random_state`; setting `model.random_state` as well is rejected |
| `bootstrap` | Resamples and seed of the event-clustered bootstrap |

`odds_analytics.retail_ev_evaluation.EvaluationSpec.sha256` hashes the validated spec, so two files that parse to the same spec share a hash.

## What it reads

`odds_analytics.retail_ev_evaluation.load_evaluation_data` raises `odds_analytics.retail_ev_evaluation.WindowEndError` when:

- the window ends after the time the run starts, because scoring a window that has not ended is an interim look;
- any event or row it reads kicks off after `window.end` (`odds_analytics.retail_ev_evaluation.check_window_end`).

It reads the sport's events whose kick-off falls between `training_start` and `window.end`, in kick-off order, and builds their rows with `odds_analytics.retail_ev.load_retail_ev_dataset`. Team history for the schedule and standings groups is read only up to `window.end` (`odds_analytics.as_of_team_features.load_team_history`).

The columns are `odds_analytics.retail_ev.RetailEvFeatures.names` for the spec's groups. No variance filter runs, unlike the [training pipeline](../modelling/training-pipeline.md), so a book with no row keeps its all-zero column. `odds_analytics.retail_ev_evaluation.EvaluationData.fingerprint` hashes the row identities, the feature matrix, the targets, the baselines and both weights.

## Refit blocks

`odds_analytics.retail_ev_evaluation.evaluation_blocks` splits the window events that have rows into blocks of `refit_every_events` in kick-off order. The last block may be partial and is scored. Each block's model trains on every row of the events that kick off at least `gap` before the block's first event, including earlier window events. `odds_analytics.retail_ev_evaluation.xgboost_fitter` fits `odds_analytics.xgboost_line_movement.XGBoostLineMovementStrategy` with the loss weights and predicts on the target's scale.

## Statistics and verdict

`odds_analytics.retail_ev_evaluation.keep_bar` computes, over every scored row:

1. `odds_analytics.retail_ev_metrics.mse_gain`, the baseline's weighted MSE minus the model's;
2. `odds_analytics.retail_ev_metrics.pick_ev_gain`, the model's total `w_event`-weighted realised EV over rows it predicts above zero, minus the baseline's.

Both CIs are event-clustered percentile bootstraps drawn from the spec's resamples and seed. `odds_analytics.retail_ev_evaluation.KeepBar.verdict` is `keep` only if both CI lower bounds lie above zero, and `retire` otherwise.

## Coverage and slices

`odds_analytics.retail_ev_evaluation.coverage` reports:

- the window events, the scored events and rows, the window events with no row, and the window's drop counts;
- the events and rows before the window;
- the shares of scored events with live OddsPortal rows, with a live Betfair close, and with FDUK opening rows only;
- rows and events by horizon, retail source, close source, event type and outcome;
- runs of at least `odds_analytics.retail_ev_evaluation.OUTAGE_MIN_EVENTS` consecutive window events with no live OddsPortal row, and with no live Betfair close.

`odds_analytics.retail_ev_evaluation.slice_table` repeats both statistics, with CIs, picks and pick totals, by each of `odds_analytics.retail_ev_evaluation.SLICE_KEYS`. The slices are descriptive and do not enter the verdict.

## Outputs

`odds_analytics.retail_ev_evaluation.run_evaluation` writes to the output directory, and `odds_analytics.retail_ev_evaluation.claim_output_dir` refuses a directory that is not empty unless forced.

| File | Contents |
|---|---|
| `manifest.json` | Git SHA and whether tracked files differ from it, the spec and its hashes, the trial file's hash, the resolved XGBoost parameters, the retail-EV constants, the feature names, the data and prediction hashes, row, event and block counts, library versions and the run time |
| `report.json` | The verdict, both statistics with their CIs and the coverage report |
| `slices.csv` | The slice table |
| `blocks.csv` | Per block: the training cutoff, training and scored events and rows, kick-off ranges and both weighted MSEs |
| `predictions.csv.gz` | Every scored row with its block, labels, target, baseline, model prediction and weights |
