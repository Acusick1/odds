---
type: reference
title: Candidate registry
description: The candidate file format, the odds candidates list command, and the version check that stops a candidate changing without a new version.
sources:
  - odds_core.candidates
  - odds_cli.commands.candidates
  - scripts/check_candidate_versions.py
status: stable
---

# Candidate registry

The registry is the `candidates/` directory: one `<name>.yaml` file per candidate, and the only definition of it. No database table copies it. What a candidate is and how it is judged is on [Goal and strategy](../foundations/goal-and-strategy.md#candidates).

## File format

`odds_core.candidates.load_candidates` reads every `*.yaml` file in a directory, validates it against `odds_core.candidates.Candidate` and returns the candidates sorted by name. A missing directory gives an empty list. Every model rejects unknown fields.

| Field | Model | Meaning |
|---|---|---|
| `name`, `description`, `sport`, `version`, `issue`, `start_date`, status | `odds_core.candidates._CandidateBase` | Shared by both kinds. `name` matches the file name and the pattern `odds_core.candidates.NAME_PATTERN`. `sport` is a [sport key](../collection/sport-and-market-keys.md). `issue` is the idea's GitHub issue. `start_date` is the first day it may place bets. The status is an `odds_core.candidates.CandidateStatus` |
| `kind: mechanical`, `rule` and params | `odds_core.candidates.MechanicalCandidate` | `rule` names a rule function. The parameters are a free JSON mapping here, and the rule validates them (#609) |
| `kind: agent` | `odds_core.candidates.AgentCandidate` | No further fields; the agent's prompts and model are its definition ([agent run](../agent/agent-run-job.md)) |

`odds_core.candidates.load_candidates` raises a `ValueError` naming the file when a file fails validation or its `name` differs from its file stem, and when a sport has more than one agent candidate.

A change of status goes through a pull request. The version is a positive integer, bumped whenever the rule, parameters or model change ([versioning](../foundations/goal-and-strategy.md#candidates)).

## `odds candidates list`

`odds_cli.commands.candidates.list_candidates` prints each candidate's name, kind, sport, version, status, start date and issue, optionally filtered by status. It reads the files in `odds_core.candidates.candidates_dir` and does not touch the database.

## Version check

`scripts/check_candidate_versions.py` compares the working tree's `candidates/*.yaml` with the same files at a base commit. It runs as the `candidate-versions` pre-commit hook ([code conventions](../foundations/code-conventions.md#mechanical-gates)), locally and in the `Lint` workflow. The base is `--base`, else the merge-base with `CANDIDATE_BASE_REF` or `origin/$GITHUB_BASE_REF`, else `HEAD`. The workflow sets `CANDIDATE_BASE_REF` to `origin/main`.

It fails when:

- a file that exists at the base is deleted. A candidate is killed by setting its status to killed;
- a file's `version` decreased;
- any key other than the status and `description` changed and the version did not increase (`UNVERSIONED_KEYS` in the script).

A new file always passes. The check does not look at agent prompt files, so an agent candidate's version bump for a prompt or model change is not enforced.
