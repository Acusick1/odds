---
type: reference
title: Experiment scripts and artifacts
description: Layout of experiments/, config and script conventions, the FINDINGS.md sections, the verification artifact files, and the inventory of scripts.
sources:
  - odds_analytics.training.data_preparation.save_verification_artifacts
  - experiments/scripts/cv_protocol_grid.py
  - experiments/configs/xgboost_epl_production_tabular_match_stats.yaml
  - experiments/results/exp4_hours_to_game/FINDINGS.md
status: stable
---

# Experiment scripts and artifacts

`experiments/` holds `scripts/` (standalone Python), `configs/` (training YAMLs) and `results/<experiment_name>/` (outputs). Steps for running and recording an experiment are on [Run and record a modelling experiment](howto-run-experiment.md).

## Script and config conventions

- Scripts load data through `odds_analytics.feature_groups.prepare_training_data` with an `odds_analytics.training.config.MLTrainingConfig` read from `experiments/configs/` ([training config](../modelling/training-config.md)).
- Code defaults are `target_type: raw` and `target_bookmaker: pinnacle`, and the default CV protocol is not walk-forward, so a YAML must set the target and protocol itself to match production ([target functions](../modelling/clv-target-functions.md), [cross-validation](../modelling/cross-validation.md), [ADR-0004](../../docs/adr/0004-walk-forward-cv-protocol.md)).
- The `experiments/configs/xgboost_epl_production_*` YAMLs carry the search space that avoids constant predictors on sliding windows; older YAMLs carry the original space ([search-space experiment](exp-epl-feature-ablation-and-search-space.md#principled-search-space-re-test)).
- Scripts run as `uv run python experiments/scripts/<script>.py`.
- The February 2026 scripts reference feature groups the code does not define and may not run ([early NBA signal search](exp-early-nba-signal-and-lstm.md#caveats)).
- Training configs under `experiments/configs/` have no early-stopping setting; boosting length is `n_estimators`, which the search space tunes.

## `FINDINGS.md`

Exemplar: `experiments/results/exp4_hours_to_game/FINDINGS.md`. Sections:

- Setup: date, git SHA, dataset size, sampling, target, exact reproduce command.
- Key Results: numbers, tables, significance.
- Interpretation: meaning, caveats, alternatives.
- Implications: next steps, go/no-go.

The git SHA records the code that produced the results; the script is committed before the run and the results after (convention set in #134). Plots are PNG and data artifacts are CSV, in the same directory.

## Verification artifacts

`odds train run` and `odds train tune` write them automatically ([training pipeline](../modelling/training-pipeline.md)). Scripts call `odds_analytics.training.data_preparation.save_verification_artifacts(X, y, feature_names, output_dir, event_ids=event_ids)`, which writes:

| File | Content |
|---|---|
| `feature_stats.csv` | count, null_count, null_pct, mean, std, min, max, nunique per feature |
| `feature_sample.csv` | first 20 rows with event ids and target |
| `feature_target_corr.csv` | Pearson and Spearman correlation with the target |

Zeroed features once hid an adapter bug ([early NBA signal search](exp-early-nba-signal-and-lstm.md#the-in_play-leak)).

## Inventory

`experiments/results/` holds only some runs' outputs (verification CSVs for several configs, `FINDINGS.md` for seven experiments), so a missing results directory does not mean the run did not happen.

| Script | Subject | Page |
|---|---|---|
| `exp1_feature_correlations`, `exp2_feature_group_isolation` | February 2026 NBA feature runs | [Early NBA signal](exp-early-nba-signal-and-lstm.md) |
| `exp4_hours_to_game`, `exp6b_injury_closing_tier` | NBA timing and injuries | [NBA timing and injuries](exp-nba-timing-and-injuries.md) |
| `exp6_learning_curve` | NBA data volume | [Target and data volume](exp-target-and-data-volume.md) |
| `exp7_backtest_simulation`, `exp7b_line_shopping` | Flat betting and line shopping | [Betting simulation](exp-betting-simulation.md) |
| `cv_protocol_grid` | CV window and step grid | [CV protocol grid](exp-cv-protocol-grid.md) |
| `hybrid_sharp_validation` | Pinnacle versus Betfair sharp reference | [EPL sharp reference](exp-epl-sharp-reference.md) |
| `lineup_signal_check` | EPL lineup features | [EPL feature ablation](exp-epl-feature-ablation-and-search-space.md#lineup-signal-check), [EPL context features](../modelling/epl-context-features.md) |
| `betfair_historical_analysis` | Parses Betfair historical NBA files | none |
| `betfair_nba_explore` | Live Betfair NBA market survey | none |
| `sharp_feature_correlation` | Pinnacle versus Betfair sharp features on overlapping events | none |
| `sharp_regime_analysis` | Pinnacle versus Betfair sharp since 2024 | none |
| `fold_diagnostics` | Per-fold composition and drift for a CV protocol | none |
| `polymarket_mlb_discover`, `polymarket_mlb_orderbooks`, `polymarket_mlb_sentiment`, `polymarket_mlb_volume` | MLB Polymarket survey | [Polymarket market research](../collection/polymarket-market-research.md) |
| `show_agent_summaries` | Prints agent run summaries from `logs/agent_runs/` | [Reporting surfaces](../operations/reporting-surfaces.md) |
