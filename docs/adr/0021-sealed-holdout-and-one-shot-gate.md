---
status: proposed
---

# Seal a holdout and gate registration on one run of it

## Context

Development data has been examined many times, so a result on it cannot promote an idea. A selection made on it overfits it `(inferred)`. The five-league backfill leaves part of the data unexamined, and an idea needs one test on that part before it is registered. The test is worth nothing if its result can be seen before the idea's grid and cell are fixed, or if it can be rerun until it passes (#820, #821, #828).

## Decision

- **Sealed data.** The 2025-26 season (1 August 2025 to 31 July 2026) of La Liga, Serie A, Bundesliga, Ligue 1 and the Championship, and EPL 2022-23 and 2023-24. The EPL seasons have a Betfair Exchange close only if the substitute close of #730 is accepted; one switch, `odds_analytics.holdout.EPL_2022_24_SEALED`, defaults to sealed. Development data is the five leagues' 2024-25, EPL 2024-26 and EPL 2020-22. EPL 2024-26 stays development because every current idea's parameters came from it.
- **Refusal.** `odds candidates backtest` refuses any range of a sealed sport that intersects its sealed days unless the draft's issue has a committed pre-registration, `candidates/preregistrations/<issue>.yaml`, holding the issue, the selected cell's params and the git SHA at which the grid was frozen. The draft's params must equal the cell.
- **One shot.** Each holdout run appends a row to the committed ledger `candidates/holdout.jsonl`. A second run of the same issue and version is refused. A new version needs a new grid.
- **Gate.** The selected cell passes when its mean EV over every bet with a close is above 0 and it is not killed (the 95% interval's upper bound is above 0). Under 30 bets, not killed alone passes and the result is flagged on the idea issue. A holdout result never promotes; the forward test decides.
- **Order of work.** Screen on development data and select one cell by the pre-registered rule, run the holdout once, then register.

## Consequences

- The seal is mechanical in the CLI only. Anyone can query the database or edit the code; the ledger and the committed pre-registration make a peek visible in review, not impossible.
- A pre-registration is checked as committed, unmodified and with its grid commit in HEAD's history. That the grid was truly fixed before screening is the author's claim, and review's to check.
- With `EPL_2022_24_SEALED` true while #730 is open, a range over those seasons is refused although it has no close to score. If #730 rejects the substitute close, the switch is set false.
- An idea that has been through the holdout cannot be rerun on it. A rule change that keeps the version is a version-check failure; one that bumps it needs a new pre-registration and a run that no earlier version used.
- Zero bets in the holdout fails the gate, flagged: nothing was scored.

## Evidence

- Issues: #820 (decision), #821 (round rules, cell selection), #828 (implementation), #730 (EPL substitute close).
- Code: `odds_analytics.holdout`, `odds_cli.commands.candidates.backtest`.
