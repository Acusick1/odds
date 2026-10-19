---
type: howto
title: Run the retail-EV evaluation
description: Write the evaluation spec, run odds evaluate retail-ev once from a clean tree on a local copy of the data, and read the verdict, manifest and coverage.
sources:
  - odds_cli.commands.evaluate.evaluate_retail_ev
  - odds_analytics.retail_ev_evaluation.EvaluationSpec
  - experiments/configs/retail_ev_evaluation_dry_run.yaml
status: stable
---

# Run the retail-EV evaluation

The spec fields, the window-end guard, the statistics and the output files are on [Retail-EV evaluation](retail-ev-evaluation.md).

> [!WARNING]
> Never point `DATABASE_URL` at production. The evaluation reads a local copy ([databases](../foundations/databases-and-environments.md#environments)).

1. **Load the data locally.** For a fresh-season look, copy production once the window has ended and its last FDUK file is loaded ([copy production data](../foundations/howto-copy-production-data.md)). A dry run uses the development data already in the local database.
2. **Write the spec.** Copy `experiments/configs/retail_ev_evaluation_dry_run.yaml` and set the window, training start, refit cadence, feature groups, arm, hyperparameter trial, seed and bootstrap (`odds_analytics.retail_ev_evaluation.EvaluationSpec`).
   - A window end after the run time is refused, so the run cannot be an interim look.
3. **Commit the spec and the code.** Run from a clean tree, so the manifest's `git_sha` pins the code and `git_dirty` is false.
4. **Run it once.**

   ```bash
   uv run odds evaluate retail-ev --spec experiments/configs/<spec>.yaml --output-dir experiments/results/<name>
   ```

   - The command prints the verdict and both keep-bar statistics with their CIs.
   - A second run into the same directory is refused. `--force` overwrites it; use it only for dry runs.
5. **Check the manifest.** In `manifest.json`, confirm `git_dirty` is false, the `git_sha` is the intended commit, `spec_sha256` matches the registered spec, and the event counts are plausible.
6. **Read the coverage.** In `report.json`, the coverage section lists the window events with no row, the share of events with live collection and the collection outages. They describe power, not the verdict.
7. **Commit the outputs.** Commit every file in the output directory, so the verdict, its provenance and the predictions it scored stay together.

## Related

- [Retail-EV evaluation](retail-ev-evaluation.md)
- [Price-only models on the retail-EV target](exp-epl-retail-ev-price-only.md)
- [Power of the retail-EV keep bar](exp-retail-ev-power.md)
