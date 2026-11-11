---
status: accepted
---

# Pool several leagues into one candidate and count its family by intersecting league sets

## Context

A mechanical candidate bet in one sport, and its family, which sets the Bonferroni size k of its promote bar, was the versions with its rule and sport. Round 1's ideas are cross-league rules over EPL, the Championship, La Liga, Serie A, the Bundesliga and Ligue 1 (#821). One league alone takes a long time to reach the first 100-bet checkpoint, and running one candidate per league would split each idea's bets into six slow verdicts and six families (#817, #819).

## Decision

- **Schema.** `sports`, a list of league keys, replaces `sport` on every candidate, mechanical and agent. A single-league candidate lists one. There is no dual field. `market` stays single.
- **Verdict.** A candidate's bets in all its leagues form one series in placement order, judged at the same checkpoints with the event-clustered bootstrap unchanged. A per-league breakdown is reported beside the verdict for information and never changes it.
- **Family.** Two versions share a family when they have the same rule and market and their league sets intersect. k is counted per version and is not transitive: {EPL, Serie A} shares a family with {EPL} and with {Serie A}, but {EPL} and {Serie A} do not share one with each other.
- **Versioning.** Adding or dropping a league is a new version; the version check compares the league list like any other key. The rename of a one-league `sport: X` to `sports: [X]` is not a change, so the live candidates kept their versions and forward tests through the migration.
- **Closes.** Every listed league must have a close for the candidate's market in `odds_core.close.CLOSES` at registration. A bet with no close after the close grace is skipped, per league as before.
- **Holdout.** The sealed holdout run replays one pooled draft over the pre-registered leagues, each on its own sealed window ([ADR-0021](0021-sealed-holdout-and-one-shot-gate.md)).

## Consequences

- A pooled candidate reaches a checkpoint as fast as its leagues together place bets, and its verdict says nothing about any one league: a rule that wins only in some leagues is promoted or killed as a whole. Narrowing it to the leagues that look good is a new version with a new forward test.
- A pooled version raises the k of every single-league version of the same rule and market in any of its leagues, and those raise its k. Two single-league versions in different leagues stay independent.
- Each per-league `run-candidates-<league>` job runs every active candidate that lists its league, on that league's events. A pooled candidate's decisions in different leagues are taken by different jobs.
- A league without a close (MLB) cannot be registered in a mechanical candidate, though a draft backtest of it still runs.

## Evidence

- Issues: #819 (decision, made with the user), #827 (implementation), #817 (pooling across the six leagues), #821 (round 1).
