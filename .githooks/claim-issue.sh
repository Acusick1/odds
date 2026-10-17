#!/usr/bin/env bash
# post-checkout hook: claim issue <n> when branch <n>-<slug> or issue-<n>-<slug> is created.
# Advisory only; every failure path exits 0.
set -u

# git passes 1 as the third argument for a branch checkout, 0 for a file checkout.
[[ "${3:-}" == 1 ]] || exit 0

branch=$(git symbolic-ref --quiet --short HEAD) || exit 0
[[ "$branch" =~ ^(issue-)?([0-9]+)- ]] || exit 0
n=${BASH_REMATCH[2]}

# A just-created branch has a single reflog entry; later switches to it add none
# until its first commit, so a branch config key stops the claim re-running.
mapfile -t log < <(git reflog show --format=%gs "refs/heads/$branch")
[[ ${#log[@]} -eq 1 && "${log[0]}" == "branch: Created from"* ]] || exit 0
[[ -z "$(git config "branch.$branch.claimChecked")" ]] || exit 0

command -v gh >/dev/null || { echo "claim-issue: gh not found, skipping" >&2; exit 0; }
info=$(gh issue view "$n" --json state,assignees \
  --jq '[.state, ([.assignees[].login] | join(","))] | @tsv' 2>/dev/null) \
  || { echo "claim-issue: cannot read issue #$n, skipping" >&2; exit 0; }
git config "branch.$branch.claimChecked" true

IFS=$'\t' read -r state assignees <<<"$info"
if [[ "$state" == CLOSED ]]; then
  echo "claim-issue: issue #$n is closed" >&2
elif [[ -n "$assignees" ]]; then
  echo "claim-issue: issue #$n is already assigned to ${assignees//,/, }" >&2
elif gh issue edit "$n" --add-assignee @me >/dev/null 2>&1; then
  echo "claim-issue: claimed issue #$n"
else
  echo "claim-issue: could not assign issue #$n, skipping" >&2
fi
exit 0
