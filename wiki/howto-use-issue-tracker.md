---
type: howto
title: Use the issue tracker
summary: "gh conventions for issues, the Personal board and draft-PR linking, sub-issues, triage labels and wayfinding maps."
sources:
  - docs/agents/issue-tracker.md#conventions
  - docs/agents/issue-tracker.md#pull-requests-as-a-triage-surface
  - docs/agents/issue-tracker.md#wayfinding-operations
  - docs/agents/triage-labels.md
  - docs/DEVELOPMENT.md#issue-workflow
verified_at: 7887039
---

# Use the issue tracker

Issues and specs live in GitHub Issues on `Acusick1/odds`, added to the `Personal` project board. Use the `gh` CLI; it infers the repo from `git remote -v`. Source of truth: `docs/agents/issue-tracker.md` and `docs/agents/triage-labels.md`, linked from [AGENTS.md](../AGENTS.md#agent-skills).

## Everyday operations

| Task | Command |
|---|---|
| Create | `gh issue create --project "Personal" --title "..." --body "..."` (never omit `--project`; use a heredoc for multi-line bodies) |
| Read | `gh issue view <n> --json number,title,body,labels,comments` |
| List | `gh issue list --state open --json number,title,body,labels,comments` with `--label` and `--state` filters |
| Comment | `gh issue comment <n> --body "..."` |
| Label | `gh issue edit <n> --add-label "..."` or `--remove-label` |
| Close | `gh issue close <n> --comment "..."` |
| Sub-issue | `gh issue create --parent <parent> ...`, or `gh issue edit <parent> --add-sub-issue <child>` (gh 2.94+) |

Without sub-issues, put `Part of #<parent>` at the top of the child body. A bare `#42` may be an issue or a PR: try `gh pr view 42`, then `gh issue view 42`.

## Starting work

Open a draft PR whose body contains `Closes #<number>` as soon as the branch has its first commit. Linking moves the issue to In Progress on the board; merging moves it to Done.

## Triage labels

Five roles, each used as a label with the same name:

| Label | Meaning |
|---|---|
| `needs-triage` | Maintainer needs to evaluate |
| `needs-info` | Waiting on the reporter |
| `ready-for-agent` | Fully specified, ready for an unattended agent |
| `ready-for-human` | Requires human implementation |
| `wontfix` | Will not be actioned |

When a skill names a role, apply the matching label string.

## Wayfinding maps

Used by the wayfinder skill. The map is one issue labelled `wayfinder:map` (Notes, Decisions-so-far, Fog); children are sub-issues labelled `wayfinder:<type>` (`research`, `prototype`, `grilling`, `task`).

- **Blocking** uses GitHub's native issue dependencies: `gh api --method POST repos/<owner>/<repo>/issues/<child>/dependencies/blocked_by -F issue_id=<blocker-db-id>`. The id is the blocker's numeric database id (`gh api repos/<owner>/<repo>/issues/<n> --jq .id`), not `#number` or `node_id`. Without dependencies, put a `Blocked by: #<n>` line at the top of the child body.
- **Frontier**: open children of the map with no open blocker and no assignee; first in map order wins.
- **Claim**: `gh issue edit <n> --add-assignee @me`, the session's first write.
- **Resolve**: comment the answer, close the issue, then append a context pointer (gist and link) to the map's Decisions-so-far.

## Hazards

- External PRs are not a triage surface: the doc sets "PRs as a request surface" to `no`, and `/triage` reads that flag.

## Related

- [Documentation map](documentation-map.md) — where agent skill config sits
- [Code conventions](code-conventions.md)
