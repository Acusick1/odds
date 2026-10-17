---
type: howto
title: Run and record a modelling experiment
description: Run an experiment script, check its verification artifacts and FINDINGS.md, and record the result in the wiki.
sources:
  - odds_analytics.training.data_preparation.save_verification_artifacts
  - experiments/scripts/cv_protocol_grid.py
status: stable
---

# Run and record a modelling experiment

Conventions, file formats and the script inventory are on [Experiment scripts and artifacts](experiment-scripts.md).

1. **Write the script and config.** Put the script in `experiments/scripts/` and reuse a YAML from `experiments/configs/` ([conventions](experiment-scripts.md#script-and-config-conventions)).
   - Set `target_type`, `target_bookmaker` and the CV protocol explicitly, so the run matches production rather than the code defaults.
   - For a sliding-window run, copy the search space from `experiments/configs/xgboost_epl_production_*`.
2. **Commit the script.** Do this before the run, so `FINDINGS.md` can record the git SHA of the code that produced the results.
3. **Run it.** `uv run python experiments/scripts/<script>.py`; outputs go to `experiments/results/<experiment_name>/`.
   - Call `save_verification_artifacts(X, y, feature_names, output_dir, event_ids=event_ids)` from the script (`odds train run` and `odds train tune` already do) ([artifact files](experiment-scripts.md#verification-artifacts)).
4. **Check the artifacts before reporting results.**
   - No feature is 100% NaN or constant unless expected.
   - The feature count matches the configured groups ([feature groups](../modelling/feature-groups.md)).
   - Correlations agree in sign with prior experiments.
5. **Write `FINDINGS.md` and commit the results.** Use the [section format](experiment-scripts.md#findingsmd) with the plots and CSVs beside it.
6. **Write the experiment page.** Put a new explanation page in the `evaluation` area, link the `FINDINGS.md` from it and cite the script and configs in `sources`. Rules for wiki pages are in [SCHEMA.md](../SCHEMA.md).
   - When the result is a decision (for example a target or protocol choice), also write an ADR in `docs/adr/` ([ADR-0001](../../docs/adr/0001-wiki-replaces-docs.md) is the style reference); the experiment page holds the evidence and links the ADR.
7. **Update the hub.** Update the current conclusions of [CLV model findings](clv-model-findings.md) in place if the result changes what is known, and add the dated index line.
8. **Lint.** Run `uv run python scripts/wiki_lint.py --fix`, then the plain lint.

## Related

- [Cross-validation](../modelling/cross-validation.md), [CLV model findings](clv-model-findings.md)
