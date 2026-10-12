---
type: howto
title: Run and record a modelling experiment
description: Script conventions, the required FINDINGS.md and verification artifacts with their checks, and how to record the result in the wiki.
sources:
  - packages/odds-analytics/odds_analytics/training/data_preparation.py::save_verification_artifacts
  - experiments/scripts/cv_protocol_grid.py
  - docs/DEVELOPMENT.md#running-experiments
  - docs/DEVELOPMENT.md#conventions
  - docs/DEVELOPMENT.md#required-outputs
  - docs/DEVELOPMENT.md#verification-artifacts
  - docs/DEVELOPMENT.md#after-running
verified: "7887039"
status: stable
---

# Run and record a modelling experiment

Experiments are standalone scripts in `experiments/scripts/` writing to `experiments/results/<experiment_name>/`.

## Conventions

- Load data with the training pipeline (`prepare_training_data`, `MLTrainingConfig`) and reuse YAMLs from `experiments/configs/` ([training pipeline](../modelling/training-pipeline.md)).
- Set the target and the CV protocol explicitly in the YAML. Code defaults are `target_type: raw` and `target_bookmaker: pinnacle`, not the production bet365 devigged target ([CLV target](../modelling/clv-target.md)), and the default protocol is not walk-forward ([walk-forward CV](../modelling/decision-walk-forward-cv.md)).
- When starting a sliding-window run, copy the principled search space from the `experiments/configs/xgboost_epl_production_*` configs; older YAMLs carry the original space that collapses to constant predictors ([search-space experiment](exp-epl-feature-ablation-and-search-space.md#principled-search-space-re-test)).
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

It writes `feature_stats.csv` (count, null_count, null_pct, mean, std, min, max, nunique), `feature_sample.csv` (first 20 rows with event ids and target) and `feature_target_corr.csv` (Pearson and Spearman against the target). Before reporting results, check that:

- no feature is 100% NaN or constant unless expected;
- the feature count matches the configured groups ([feature pipeline](../modelling/feature-pipeline.md));
- correlations agree in sign with prior experiments.

Zeroed features once hid an adapter bug, which is why these checks exist ([early NBA signal search](exp-early-nba-signal-and-lstm.md#the-in_play-leak)).

## Record the result in the wiki

Rules for wiki pages are in [SCHEMA.md](../SCHEMA.md).

1. Put a new experiment page in the `evaluation` area. When the result is a decision (for example a target or protocol choice), put it on the matching decision page in the `modelling` area instead.
2. Link the `FINDINGS.md` from the page and cite the script and configs in `sources`.
3. Update the current conclusions of [CLV model findings](clv-model-findings.md) in place if the result changes what is known, and add the dated index line.
4. Run `uv run python scripts/wiki_lint.py --fix`, then the plain lint.

## Existing scripts

Documented results: `exp1_feature_correlations`, `exp2_feature_group_isolation`, `exp4_hours_to_game`, `exp6_learning_curve`, `exp6b_injury_closing_tier`, `exp7_backtest_simulation`, `exp7b_line_shopping`, `cv_protocol_grid`, `hybrid_sharp_validation`, `lineup_signal_check` ([EPL context features](../modelling/epl-context-features.md)). Undocumented exploration scripts: `betfair_historical_analysis` (parses Betfair historical NBA files), `betfair_nba_explore` (live Betfair NBA market survey), `sharp_feature_correlation` (Pinnacle vs Betfair sharp features on overlapping events), `sharp_regime_analysis` (Pinnacle vs Betfair sharp since 2024), `fold_diagnostics` (per-fold composition and drift for a CV protocol).

## Hazards

- `docs/DEVELOPMENT.md#after-running` still tells authors to add a row to the `docs/MODELING.md` experiment log; the wiki is the record now.
- Scripts from February 2026 reference feature groups that no longer exist and may not run unchanged ([early NBA signal search](exp-early-nba-signal-and-lstm.md#hazards)).
- `experiments/results/` holds only some runs' outputs (verification CSVs for several configs, no `FINDINGS.md`).

## Related

- [Walk-forward CV](../modelling/decision-walk-forward-cv.md), [CLV model findings](clv-model-findings.md)
