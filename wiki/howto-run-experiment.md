---
type: howto
title: Run and record a modelling experiment
summary: Script conventions, the required FINDINGS.md, verification artifacts and their checks, and updating the experiment record in the wiki's exp pages.
sources:
  - packages/odds-analytics/odds_analytics/training/data_preparation.py::save_verification_artifacts
  - experiments/scripts/cv_protocol_grid.py
  - docs/DEVELOPMENT.md#running-experiments
  - docs/DEVELOPMENT.md#conventions
  - docs/DEVELOPMENT.md#required-outputs
  - docs/DEVELOPMENT.md#verification-artifacts
  - docs/DEVELOPMENT.md#after-running
verified_at: 7887039
---

# Run and record a modelling experiment

Experiments are standalone scripts in `experiments/scripts/` writing to `experiments/results/<experiment_name>/`.

## Conventions

- Load data with the training pipeline (`prepare_training_data`, `MLTrainingConfig`) and reuse YAMLs from `experiments/configs/` ([training-pipeline](training-pipeline.md), [training-config](training-config.md)).
- The script must run as `uv run python experiments/scripts/<script>.py`.
- Commit the script before running it, then commit the results, so `FINDINGS.md` can record the git SHA of the code that produced them.

## Required outputs

- **`FINDINGS.md`** with sections Setup (date, git SHA, dataset size, sampling, target, exact reproduce command), Key Results (numbers, tables, significance), Interpretation (meaning, caveats, alternatives) and Implications (next steps, go/no-go).
- Plots as PNG and data artifacts (CSVs) in the same directory.

## Verification artifacts

`odds train run` and `odds train tune` write them automatically. Scripts call:

```python
save_verification_artifacts(X, y, feature_names, output_dir, event_ids=event_ids)
```

It writes `feature_stats.csv` (count, null_count, null_pct, mean, std, min, max, nunique), `feature_sample.csv` (first 20 rows with event ids and target) and `feature_target_corr.csv` (Pearson and Spearman against the target). Before reporting results, check that no feature is 100% NaN or constant unless expected, the feature count matches the configured groups ([feature-groups](feature-groups.md)), and correlations agree in sign with prior experiments. Zeroed features once hid an adapter bug ([exp-early-nba-signal-search](exp-early-nba-signal-search.md)).

## After running

Record the result as an `exp-` wiki page ([clv-model-findings](clv-model-findings.md) holds the dated index) and update the hub's current conclusions in place if the result changes what is known; follow the ingest steps in `wiki/SCHEMA.md`. Link the `FINDINGS.md`.

## Existing scripts

Documented results: `exp1_feature_correlations`, `exp2_feature_group_isolation`, `exp4_hours_to_game`, `exp6_learning_curve`, `exp6b_injury_closing_tier`, `exp7_backtest_simulation`, `exp7b_line_shopping`, `cv_protocol_grid`, `hybrid_sharp_validation`, `lineup_signal_check` ([features-epl-context](features-epl-context.md)). Undocumented exploration scripts: `betfair_historical_analysis` (parses Betfair historical NBA files), `betfair_nba_explore` (live Betfair NBA market survey), `sharp_feature_correlation` (Pinnacle vs Betfair sharp features on overlapping events), `sharp_regime_analysis` (Pinnacle vs Betfair sharp since 2024), `fold_diagnostics` (per-fold composition and drift for a CV protocol).

## Hazards

- `docs/DEVELOPMENT.md#after-running` still tells authors to add a row to the `docs/MODELING.md` experiment log.
- Scripts from February 2026 reference feature groups that no longer exist and may not run unchanged ([exp-early-nba-signal-search](exp-early-nba-signal-search.md)).
- `experiments/results/` holds only some runs' outputs (verification CSVs for several configs, no `FINDINGS.md`).

## Related

- [Walk-forward CV](walk-forward-cv.md)
