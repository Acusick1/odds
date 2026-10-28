---
type: howto
title: Run and record a modelling experiment
description: Run an experiment script, check its verification artifacts, and record the result in the wiki.
sources:
  - odds_analytics.training.data_preparation.save_verification_artifacts
  - experiments/scripts/retail_ev_price_only.py
status: stable
---

# Run and record a modelling experiment

Conventions, file formats and the script inventory are on [Experiment scripts and artifacts](experiment-scripts.md).

1. **Write the script and config.** Put the script in `experiments/scripts/` and reuse a YAML from `experiments/configs/` ([conventions](experiment-scripts.md#script-and-config-conventions)); `experiments/scripts/retail_ev_price_only.py` is the reference for a script that tunes and scores a retail-EV model and slices its rows by retail source, time basis and close source. A `tuning` section sets `metric`, which fixes the optimisation direction.
   - Set `target_type` and the CV protocol explicitly; the CV code defaults are not walk-forward. `experiments/configs/xgboost_epl_retail_ev.yaml` is the reference.
2. **Commit the script.** Do this before the run, so the results can record the git SHA of the code that produced them.
3. **Run it.** `uv run python experiments/scripts/<script>.py`; outputs go to `experiments/results/<experiment_name>/`.
   - Call `save_verification_artifacts(X, y, feature_names, output_dir, event_ids=event_ids)` from the script (`odds train run` and `odds train tune` already do) ([artifact files](experiment-scripts.md#verification-artifacts)).
4. **Check the artifacts before reporting results.**
   - No feature is 100% NaN or constant unless expected.
   - The feature count matches the configured groups ([feature groups](../modelling/feature-groups.md)).
   - Correlations agree in sign with prior experiments.
5. **Commit the results.** Keep the CSVs and `summary.json` in `experiments/results/<experiment_name>/` ([results directories](experiment-scripts.md#results-directories)).
6. **Write the experiment page.** Put a new explanation page in the `evaluation` area, cite the results directory, the script and the configs in `sources`. Rules for wiki pages are in [SCHEMA.md](../SCHEMA.md).
   - When the result is a decision (for example a target or protocol choice), also write an ADR in `docs/adr/` ([ADR-0001](../../docs/adr/0001-wiki-replaces-docs.md) is the style reference); the experiment page holds the evidence and links the ADR.
7. **Lint.** Run `uv run python scripts/wiki_lint.py --fix`, then the plain lint.

## Related

- [Cross-validation](../modelling/cross-validation.md)
