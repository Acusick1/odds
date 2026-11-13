---
type: reference
title: Cell selection
description: What odds candidates select replays for a pre-registered grid, the development blocks, the eligibility rule, the bootstrap lower bound and tie-breaks that pick a cell, the power-devig score it reports for information, the selection file it writes and its JSON output.
sources:
  - odds_analytics.cell_selection
  - odds_cli.commands.candidates.select_command
  - odds_cli.commands.candidates._selection_table
  - odds_cli.commands.candidates._selection_json
mentions:
  - odds_analytics.holdout.load_preregistration
  - odds_analytics.holdout_run.pooled_draft
  - odds_analytics.candidate_backtest.collect
  - odds_analytics.retail_ev_metrics.event_clustered_ratio
  - odds_analytics.candidate_report.power_devig
status: stable
---

# Cell selection

`odds candidates select <preregistration>` picks the one cell of a pre-registered grid that runs on the [sealed holdout](candidate-backtest.md#sealed-holdout), by round 1's selection rule ([#821](https://github.com/Acusick1/odds/issues/821#issuecomment-6769611091)), from development data. No eligible cell kills the idea on development data. The step's place in the order of work is on [Take an idea to a verdict](howto-take-an-idea-to-a-verdict.md).

## Command

`odds_cli.commands.candidates.select_command` takes the path of a grid file, `candidates/preregistrations/<issue>-<market>-v<version>.yaml` ([sealed holdout files](candidate-registry.md#sealed-holdout-files)), and `--json <path>`. It opens one database session in a read-only transaction and calls `odds_analytics.cell_selection.select_cell`; it never writes to the database.

Before anything is replayed, `odds_analytics.cell_selection.load_selectable` refuses, printing the reason and exiting 1:

- a file outside `candidates/preregistrations/`, or one not named by its own issue, market and version (`odds_analytics.holdout.load_preregistration`);
- a file whose `selection_rule` is not `odds_analytics.cell_selection.ROUND_ONE_RULE`;
- a grid whose selection file already exists, since a selection is never changed;
- any cell that does not resolve as a pooled draft: a rule that is not registered (the error lists the registered rules), a market a league has no rule for, or params the rule rejects (`odds_lambda.candidate_rules.registry.resolve_rule`).

## Replay

`odds_analytics.cell_selection.replay_cells` takes the cells in grid order (`odds_analytics.holdout.Preregistration.candidate_cells`: the grid's cartesian product in the file's key order, or the listed `cells`). Each cell becomes one pooled draft over the file's leagues with the cell as params (`odds_analytics.holdout_run.pooled_draft`, [ADR-0022](../../docs/adr/0022-pooled-league-candidates.md)), replayed on each development block with `odds_analytics.candidate_backtest.collect`, the replay the [backtest](candidate-backtest.md#replay) uses, so the seal refuses any sealed day.

The blocks are `odds_analytics.cell_selection.DEVELOPMENT_BLOCKS`, each limited to the file's leagues, and a block with none of them is left out (`odds_analytics.cell_selection.development_blocks`):

| Block | Days, inclusive | Leagues |
|---|---|---|
| EPL 2022-24 | 2022-08-01 to 2024-07-31 | EPL |
| EPL 2024-26 | 2024-08-01 to 2026-07-31 | EPL |
| five leagues 2024-25 | 2024-08-01 to 2025-07-31 | `odds_analytics.holdout.FIVE_LEAGUES`: the Championship, La Liga, Serie A, the Bundesliga, Ligue 1 |

These are the development seasons of [ADR-0021](../../docs/adr/0021-sealed-holdout-and-one-shot-gate.md) and [ADR-0024](../../docs/adr/0024-epl-2022-24-is-development-data.md). EPL's Odds API data starts on 2023-04-01, and an EPL bet before 2024-08-01 has no close until the Odds API substitute close of [#730](https://github.com/Acusick1/odds/issues/730) ([substitute-close comparison](substitute-close-comparison.md)) is accepted and built, so until then the EPL 2022-24 block holds no bets with a close. A replay over six leagues takes a few minutes per cell.

## Score and pick

`odds_analytics.cell_selection.score_cell` scores a cell over its bets with a close, each bet's EV that of the [replay backtest](candidate-backtest.md#replay) (`odds_analytics.candidate_report.ScoredBet.ev`): the bet count, the distinct events, each block's bets and mean EV, the pooled mean EV (total EV over the bet count) and the lower bound.

- **Eligible** (`odds_analytics.cell_selection.is_eligible`): at least `odds_analytics.cell_selection.MIN_BETS` (50) bets in at least `odds_analytics.cell_selection.MIN_EVENTS` (30) events, and a mean EV above 0 in every block with at least `odds_analytics.cell_selection.MIN_BLOCK_BETS` (20) bets.
- **Lower bound** (`odds_analytics.cell_selection.lower_bound`): the lower end of the 90% (`odds_analytics.cell_selection.INTERVAL_LEVEL`) event-clustered percentile bootstrap interval of the mean EV, from `odds_analytics.retail_ev_metrics.event_clustered_ratio` with `odds_analytics.cell_selection.RESAMPLES` (10,000) resamples and seed `odds_analytics.cell_selection.SEED` (821), its generator created afresh for each cell. Each resample draws the events, in id order, with replacement, and its mean is its bets' total EV over their count. The bound is the 5% quantile.
- **Pick** (`odds_analytics.cell_selection.pick`): the eligible cell with the highest lower bound; ties go to the higher mean EV, then more events, then the earlier cell in grid order.
- **Power devig.** When any cell placed a bet above `odds_analytics.cell_selection.LONGSHOT_PRICE` (4.0), every cell also reports its mean EV against the close devigged by the power method (`odds_analytics.candidate_report.ScoredBet.ev_power`, [#770](https://github.com/Acusick1/odds/issues/770)), for information. It never changes eligibility or the pick.

## Output

The command prints one table: each cell's number, the parameters that vary across the grid (the title names the fixed ones), bets, events, each block's bets and mean EV, the mean EV, the lower bound, the power-devig mean EV when reported, and whether it is eligible, the picked row marked (`odds_cli.commands.candidates._selection_table`).

With an eligible cell, `odds_analytics.cell_selection.write_selection` writes `candidates/selections/<issue>-<market>-v<version>.yaml`, an `odds_analytics.holdout.Selection` of the issue, market, version, the grid file's sha256 and the picked cell, which `odds_analytics.holdout.load_plan` loads, and the command prints the next step: commit it in its own commit and run the holdout. It refuses to replace an existing file. Without one it prints that the idea is killed on development data and writes nothing; the exit code is 0 either way.

`--json` writes `odds_cli.commands.candidates._selection_json`: the grid's identity, sha256 and selection rule, the blocks, whether the power-devig score is reported, the picked cell's index (0-based, or `null`) and each cell's score with every bet it placed, labelled with its block, in the per-bet form of [the backtest's JSON](candidate-backtest.md#json-output).
