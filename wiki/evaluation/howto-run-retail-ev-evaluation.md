---
type: howto
title: Run the retail-EV evaluation
description: Write the evaluation spec, run odds evaluate retail-ev once from a clean tree on a local copy of the data, and read the verdict, manifest and coverage.
sources:
  - odds_cli.commands.evaluate.evaluate_retail_ev
  - odds_analytics.retail_ev_evaluation.EvaluationSpec
  - experiments/configs/retail_ev_evaluation_dry_run.yaml
  - experiments/configs/retail_ev_evaluation_2026_27.yaml
status: stable
---

# Run the retail-EV evaluation

The spec fields, the window-end guard, the statistics and the output files are on [Retail-EV evaluation](retail-ev-evaluation.md).

> [!WARNING]
> Never point `DATABASE_URL` at production. The evaluation reads a local copy ([databases](../foundations/databases-and-environments.md#environments)).

1. **Load the data locally.** For a window that has ended, copy production once the window has ended and its last FDUK file is loaded ([copy production data](../foundations/howto-copy-production-data.md)). A dry run uses the development data already in the local database.
2. **Write the spec.** For the 2026-27 window the spec is already committed: `experiments/configs/retail_ev_evaluation_2026_27.yaml`; skip to step 3. Otherwise copy `experiments/configs/retail_ev_evaluation_dry_run.yaml` and set the window, training start, refit cadence, feature groups, arm, hyperparameter trial, seed and bootstrap (`odds_analytics.retail_ev_evaluation.EvaluationSpec`).
   - Set `odds_analytics.retail_ev_evaluation.EvaluationSpec.feature_names` to the columns the frozen model expects; the run refuses any other list.
   - A window end after the run time is refused, so the run cannot be an interim look.
3. **Commit the spec and the code.** Run from a clean tree, so the manifest's `git_sha` pins the code and `git_dirty` is false. For the 2026-27 spec, run from a checkout of the tag `retail-ev-eval-2026-27` (`git worktree add worktrees/retail-ev-look retail-ev-eval-2026-27`), with `--output-dir experiments/results/retail_ev_evaluation_2026_27`, and copy the outputs to main to commit them.
4. **Run it once.**

   ```bash
   uv run odds evaluate retail-ev --spec experiments/configs/<spec>.yaml --output-dir experiments/results/<name>
   ```

   - The command prints the verdict and both keep-bar statistics with their CIs.
   - Every window event must be FINAL with an FDUK closing snapshot. `--allow-incomplete <event id>` (the integer event id), repeated for each, scores the named events anyway; any other incomplete event is still refused, and the manifest lists both.
   - A spec already evaluated under `experiments/results/`, or a non-empty output directory, is refused. `--force` runs anyway and the manifest records it; use it only for dry runs.
5. **Check the manifest.** In `manifest.json`, confirm `git_dirty` is false, the `git_sha` is the intended commit, `spec_sha256` matches the registered spec, and the event counts are plausible.
6. **Read the coverage.** In `report.json`, the coverage section lists the window events with no row, the share of events with live collection and the collection outages. They describe power, not the verdict.
7. **Commit the outputs.** For the fresh-season look, commit every file in the output directory. For a dry run, leave `predictions.csv.gz` out; the manifest's `predictions_sha256` identifies it.

## Related

- [Retail-EV evaluation](retail-ev-evaluation.md)
- [Price-only models on the retail-EV target](exp-epl-retail-ev-price-only.md)
- [Power of the retail-EV keep bar](exp-retail-ev-power.md)
