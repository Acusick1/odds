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

`odds_analytics.retail_ev_evaluation` scores a frozen model for the [retail-EV target](../modelling/clv-target.md#the-retail-ev-target) once, on every event of a window, and returns ADR-0012's keep or retire verdict ([ADR-0012](../../docs/adr/0012-retail-ev-against-betfair-exchange-close.md)). `odds evaluate retail-ev` (`odds_cli.commands.evaluate.evaluate_retail_ev`) runs it; the steps are on [Run the retail-EV evaluation](howto-run-retail-ev-evaluation.md). It is an offline check of one model on a past window; a model is promoted only by a live forward test ([goal and strategy](../foundations/goal-and-strategy.md)). A 2026-27 spec is committed as `experiments/configs/retail_ev_evaluation_2026_27.yaml`, at the tag `retail-ev-eval-2026-27`.

## Spec

`odds_analytics.retail_ev_evaluation.EvaluationSpec` is a pydantic model loaded from a YAML file (`odds_analytics.retail_ev_evaluation.EvaluationSpec.from_yaml`). It holds every choice the run makes and rejects unknown fields. Each field's default and description are in the class.

| Field | Meaning |
|---|---|
| `training_start` | First kick-off read, so the first training event |
| `window.start`, `window.end` | First and last kick-off scored, both inclusive and timezone-aware |
| `refit_every_events` | Window events scored by each refit |
| `gap` | A block trains on events that kick off at least this long before its first event; shorter than `odds_analytics.retail_ev.DECISION_WINDOW` is rejected |
| `feature_groups` | As-of-t groups added to the price columns ([feature groups](../modelling/feature-groups.md)) |
| `odds_analytics.retail_ev_evaluation.EvaluationSpec.feature_names` | The expected feature columns, in order; a run that builds any other list is refused |
| `arm` | `odds_analytics.training.config.RetailEvArm`; the residual arm takes `baseline_ev` as each row's base margin |
| `model` | `odds_analytics.training.config.XGBoostConfig` parameters |
| `trial` | `odds_analytics.retail_ev_evaluation.SummaryTrial`: a run's `best_params` from an experiment `summary.json`, path relative to the spec file; they replace the `model` fields they name |
| `seed` | XGBoost `random_state`; setting `model.random_state` as well is rejected |
| `bootstrap` | Resamples and seed of the event-clustered bootstrap |

`odds_analytics.retail_ev_evaluation.EvaluationSpec.sha256` hashes the validated spec, so two files that parse to the same spec share a hash.

## What it reads

`odds_analytics.retail_ev_evaluation.load_evaluation_data` raises `odds_analytics.retail_ev_evaluation.WindowEndError` when:

- the window ends after the time the run starts;
- any event or row it reads kicks off after `window.end` (`odds_analytics.retail_ev_evaluation.check_window_end`).

It raises `odds_analytics.retail_ev_evaluation.IncompleteWindowError` when a window event is not FINAL or has no FDUK closing snapshot (`odds_analytics.retail_ev_evaluation.incomplete_events`), unless `allow_incomplete` names that event by its integer id. Which sources an event's snapshots hold comes from their `source`, `time_basis` and time (`odds_analytics.retail_ev_evaluation.snapshot_stamps`, every market): an FDUK closing snapshot is FDUK with time basis `close`, and a live OddsPortal snapshot is OddsPortal `observed`. The run time is passed to `odds_analytics.retail_ev.load_retail_ev_dataset` as `now`.

It reads the sport's events whose kick-off falls between `training_start` and `window.end`, in kick-off order, and builds their rows with `odds_analytics.retail_ev.load_retail_ev_dataset`. Team history for the schedule and standings groups is read only up to `window.end` (`odds_analytics.as_of_team_features.load_team_history`).

The columns are `odds_analytics.retail_ev.RetailEvFeatures.names` for the spec's groups. No variance filter runs, unlike the [training pipeline](../modelling/training-pipeline.md): a book with no row keeps its all-zero column. `odds_analytics.retail_ev_evaluation.EvaluationData.fingerprint` hashes the row identities, the feature matrix, the targets, the baselines and both weights.

## Refit blocks

`odds_analytics.retail_ev_evaluation.evaluation_blocks` splits the window events that have rows into blocks of `refit_every_events` in kick-off order. The last block may be partial and is scored. Each block's model trains on every row of the events that kick off at least `gap` before the block's first event, including earlier window events. `odds_analytics.retail_ev_evaluation.xgboost_fitter` fits the model wrapper `odds_analytics.xgboost_line_movement.XGBoostLineMovementStrategy` through `odds_analytics.retail_ev.fit_retail_ev`, the fit the [retail-EV model rule](candidate-rules.md#the-retail-ev-model-rule) also uses, which takes the row arrays as given, with the loss weights and predicts on the target's own scale.

## Statistics and verdict

`odds_analytics.retail_ev_evaluation.keep_bar` computes, over every scored row:

1. `odds_analytics.retail_ev_metrics.mse_gain`, the baseline's weighted MSE minus the model's;
2. `odds_analytics.retail_ev_metrics.pick_ev_gain`, the model's total `w_event`-weighted realised EV over rows it predicts above zero, minus the baseline's.

Both CIs are event-clustered percentile bootstraps drawn from the spec's resamples and seed. `odds_analytics.retail_ev_evaluation.KeepBar.verdict` is `keep` only if both CI lower bounds lie above zero, and `retire` otherwise.

## Coverage and slices

`odds_analytics.retail_ev_evaluation.coverage` reports:

- the window events, by status, and the share whose snapshots include an FDUK closing price, a live OddsPortal snapshot, and a live Betfair snapshot within `odds_core.betfair_close.LIVE_CLOSE_WINDOW` before kick-off (the window the [Betfair close check](../operations/health-monitor.md#betfair-close-check) alerts on);
- the scored events and rows, and the events and rows before the window;
- each unscored window event with its status and those three source flags, and the window's drop counts;
- runs of at least `odds_analytics.retail_ev_evaluation.OUTAGE_MIN_EVENTS` consecutive window events with no live OddsPortal snapshot, and with no live Betfair snapshot near kick-off;
- scored rows and events by horizon, retail source and retail time basis (the retail snapshot's, such as `oddsportal` and `observed`), close source (the source the close is stored under, `betfair` or `football_data_uk`), event type (`fduk_only` when every row of the event comes from FDUK, else `live_collected`) and outcome.

`odds_analytics.retail_ev_evaluation.slice_table` repeats both statistics, with CIs, picks and pick totals, by each of `odds_analytics.retail_ev_evaluation.SLICE_KEYS`. The slices are descriptive and do not enter the verdict.

## Outputs

`odds_analytics.retail_ev_evaluation.run_evaluation` writes to the output directory. Unless forced, it refuses:

- a spec whose `spec_sha256` already appears in a `manifest.json` under `experiments/results/*/` (`odds_analytics.retail_ev_evaluation.check_one_look`);
- an output directory that is not empty (`odds_analytics.retail_ev_evaluation.check_output_dir`).

Both checks, and the data guards, run before the output directory is created. A manifest that cannot be parsed is skipped with a warning (`odds_analytics.retail_ev_evaluation.prior_runs`). The check catches an accidental rerun, not a deliberate one: `--force` or another results directory gets past it, so the frozen tag and the committed outputs are what show that a look happened once.

| File | Contents |
|---|---|
| `manifest.json` | Git SHA, its exact tag or null, and whether tracked files differ from it; whether the run was forced, the earlier manifests of the same spec and the `predictions_sha256` it overwrote; the incomplete window events and the ones allowed by ID; the spec and its hashes, the trial file's hash, the resolved XGBoost parameters, the retail-EV constants, the feature names, the data and prediction hashes, row, event and block counts, library versions and the run time |
| `report.json` | The verdict, both statistics with their CIs and the coverage report |
| `slices.csv` | The slice table |
| `blocks.csv` | Per block: the training cutoff, training and scored events and rows, kick-off ranges and both weighted MSEs |
| `predictions.csv.gz` | Every scored row with its block, labels, target, baseline, model prediction and weights |
