---
type: howto
title: Train and publish a CLV model
description: Write a config, train or tune, check the artifacts, publish, and confirm scoring.
sources:
  - odds_cli.commands.train.run_training
  - odds_cli.commands.train.run_tuning
  - odds_cli.commands.model.publish_model
  - experiments/configs/xgboost_epl_production_tabular_match_stats.yaml
status: stable
---

# Train and publish a CLV model

1. **Write a config.** Start from the production YAML `experiments/configs/xgboost_epl_production_tabular_match_stats.yaml`; its description carries no metrics, which are on [CLV model findings](../evaluation/clv-model-findings.md#deployed-model).
   - Leave boosting length to `n_estimators`; the config has no early-stopping field.
   - Set `target_type: devigged_bookmaker` and `target_bookmaker: bet365`, the production target ([ADR-0002](../../docs/adr/0002-bet365-clv-target.md)).
   - Choose the data so `target_bookmaker` belongs to the selected `data_source` (see [Choose data](#choose-data)).
   - Set `sport_key` and the CV protocol ([ADR-0004](../../docs/adr/0004-walk-forward-cv-protocol.md), [training config](training-config.md)).
   - Copy the search space from that YAML, not from an older config, which carries a space that collapses on sliding windows ([search-space experiment](../evaluation/exp-epl-feature-ablation-and-search-space.md#principled-search-space-re-test)).
   - Run `odds train validate --config <file>`.
2. **Train.** `odds train run --config <file>`.
   - Options: `--output`, `--dry-run`, `--verbose`, `--track`, `--tracking-uri`, `--sport`.
   - It runs CV, fits a final model and writes `models/xgboost_line_movement_<experiment.name>.pkl` plus a sibling `.yaml` holding the full config ([training pipeline](training-pipeline.md)).
3. **Or tune first.** `odds train tune --config <file> --n-trials 100`; the config needs a `tuning` section ([tuner](hyperparameter-tuner.md)). The best parameters go to `<config>_best.yaml`; then run step 2 on that file.
   - Do not publish the output of `--train-best`: it leaves only a metadata `.yaml` beside the `.pkl`, which scoring refuses (#437).
4. **Check the artifacts.** Read the CV metrics and `experiments/results/<experiment.name>/verification/`.
   - No all-NaN or constant features, and the expected feature count ([run an experiment](../evaluation/experiment-scripts.md#verification-artifacts)).
5. **Publish.** `odds model publish --name epl-clv-home --path models/<file>.pkl [--version v1.2] [--bucket <bucket>]`.
   - It needs AWS credentials and a config YAML beside the pickle; pass `--bucket` explicitly for any other account ([model store](model-store.md#odds-model-publish)).
   - Give each publish a distinct `--version` label: the loader reads only `latest/`, which every publish overwrites, so the labelled prefix is what you can restore from ([model store](model-store.md#layout-and-loading)).
6. **Confirm scoring.** Check `aws s3 ls s3://<bucket>/<name>/latest/` and the `model_loaded` log line.
   - Scoring runs only where a local scheduler runs `fetch-oddsportal` ([prediction scoring](prediction-scoring.md#where-it-runs)).

## Choose data

- EPL with the bet365 target and hybrid sharp reference: `data_source: null`, `sharp_bookmakers: [pinnacle, betfair_exchange]`, `closing_source_priority: [football_data_uk]`.
- NBA with the bet365 target: `data_source: oddsportal`. NBA with the Pinnacle target: `oddsapi`.
- LSTM runs also set `min_snapshots: 5` or more.
