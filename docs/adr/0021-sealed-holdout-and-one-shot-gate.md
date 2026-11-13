---
status: accepted
---

# Seal a holdout and gate registration on one run of it

## Context

Development data has been examined many times, so a result on it cannot promote an idea. A selection made on it overfits it `(inferred)`. The five-league backfill leaves part of the data unexamined, and an idea needs one test on that part before it is registered. The test is worth nothing if its result can be seen before the idea's grid and cell are fixed, or if it can be rerun until it passes (#820, #821, #828).

## Decision

- **Sealed data.** The 2025-26 season (1 August 2025 to 31 July 2026) of La Liga, Serie A, Bundesliga, Ligue 1 and the Championship, and EPL 2022-23 and 2023-24. The EPL seasons have a Betfair Exchange close only if the substitute close of #730 is accepted; one switch, `odds_analytics.holdout.EPL_2022_24_SEALED`, defaults to sealed. Development data is the five leagues' 2024-25, EPL 2024-26 and EPL 2020-22. EPL 2024-26 stays development because every current idea's parameters came from it.
- **The seal is in the replay library.** `odds_analytics.candidate_backtest.final_events`, which every replay (the development backtest and its `--compare`, the rehash replay script, a direct `replay()`) selects its events through, refuses any range that intersects a sealed window of the candidate's sport. The one exception is a `HoldoutAccess` capability, which only a holdout reservation issues. The development backtest has no exception, and a range that mixes development and sealed days is refused. The rehash replay clips its ranges to each sport's development days.
- **A separate command runs the holdout.** `odds candidates holdout <preregistration>` reads the pre-registration, not a draft. It replays the selected cell on each pre-registered league's sealed window, each league as a single-sport draft, pools every bet into one sample and computes one gate on that sample alone. The per-league drafts are the seam a pooled draft (#827) would replace.
- **The grid is frozen first, the cell chosen later.** `candidates/preregistrations/<issue>-<market>-v<version>.yaml` holds the issue, name, version, rule, market, league set, the full grid (every parameter's values and decision times) and the selection rule (a link to its text, [#821](https://github.com/Acusick1/odds/issues/821#issuecomment-6769611091)), and no cell. It must be committed, unchanged and touched by exactly one commit (renames followed), and that commit must be older than the earliest Odds API backfill snapshot (`source_stamped`) the database holds, if any. `candidates/selections/<issue>-<market>-v<version>.yaml` holds the cell the selection rule picked on development data, the grid file's sha256 and nothing else; it must be a member of the grid, committed and unchanged with one commit, and no deadline applies to it. A change to the rule, league set, grid or cell is a new version with new files; one issue can hold a file pair per market and per version. The hook fails any edit, rename or deletion of either folder's files.
- **One shot, durable and atomic.** Before any sealed data is read the command takes an exclusive lock shared by every worktree, requires a clean tree and a committed ledger `candidates/holdout.jsonl` that extends `origin/main`'s and holds no row for the issue and version, then spends the identity (issue, market, version) in `odds-holdout-spent.jsonl` in the repository's git common directory, which every worktree shares, and appends and syncs a `reserved` row. A `result` row follows on success. A failure leaves the reservation, and the run is spent. A pre-commit check (`scripts/check_candidate_versions.py`) fails a ledger whose committed rows change or go, a version reserved twice, or a pre-registration that changes.
- **Gate.** The pooled sample passes when its mean EV over every bet with a close is above 0 and it is not killed (the 95% event-clustered interval's upper bound is above 0). Under 30 bets, not killed alone passes and the result is flagged on the idea issue. No bets fails. A holdout result never promotes; the forward test decides.
- **Order of work.** Screen on development data and select one cell by the pre-registered rule, run the holdout once, then register.

## Consequences

- The guarantee is that the grid was committed before the backfill landed and before the holdout run, and that the run happens once per version. Whether development screening happened before the cell was selected is not provable, only auditable from the pre-registration and the idea issue.
- The seal binds the library, not the database: anyone can query the holdout events with SQL or edit the code. The ledger, the single-commit pre-registration and the append-only check make a peek visible in review, not impossible.
- Sibling worktrees of one clone share the spent file, so they cannot spend a version twice, and a crash leaves it spent. Two clones on different machines share nothing; the merge-time duplicate-reservation check in the hook is the backstop once both rows reach one branch.
- The `Reservation` and `HoldoutAccess` constructors refuse a call outside the module, a guard against accidents and not enforcement: code that wants to read the holdout can read the database.
- With `EPL_2022_24_SEALED` true while #730 is open, a development range over those seasons is refused although it has no close to score. If #730 rejects the substitute close, the switch is set false.
- An idea that has been through the holdout cannot rerun on it. A new version needs a new pre-registration, a new grid and a version no earlier run used.
- The family size k at the gate is 1, which the 95% interval does not depend on.

## Evidence

- Superseded in part: [ADR-0024](0024-epl-2022-24-is-development-data.md) makes EPL 2022-23 and 2023-24 development data, so the sealed holdout is the five leagues' 2025-26 only.
- Issues: #820 (decision), #821 (round rules, cell selection), #828 (implementation), #730 (EPL substitute close), #827 (pooled drafts).
- Code: `odds_analytics.holdout`, `odds_analytics.holdout_run`, `odds_analytics.candidate_backtest.final_events`, `odds_cli.commands.candidates.holdout_command`, `scripts/check_candidate_versions.py`.
