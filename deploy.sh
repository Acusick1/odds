#!/usr/bin/env bash
#
# Deploy-with-confidence for the always-on local scheduler.
#
# Pulls the latest code, syncs dependencies, applies migrations, smoke-tests
# the bootstrapped jobs, and only then restarts the systemd-managed scheduler.
# If migration or smoke fails, the script aborts BEFORE the restart, leaving the
# running scheduler untouched on the old code.
#
# After the restart it checks that the service is active, started after this
# deploy began, and is not crash-looping, and exits non-zero otherwise.
#
# Run from the repo root on the host running odds-scheduler.service:
#   ./deploy.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# A non-interactive shell (`ssh host ./deploy.sh`) does not read the rc file
# that puts the uv installer's directory on PATH. Resolve every tool before the
# pull so a missing one cannot leave the checkout ahead of the running scheduler.
export PATH="$HOME/.local/bin:$PATH"
for tool in git uv sudo systemctl; do
    if ! command -v "$tool" >/dev/null; then
        echo "deploy.sh: $tool not found on PATH ($PATH); nothing was changed" >&2
        exit 1
    fi
done

SERVICE=odds-scheduler
SYSTEMCTL="$(command -v systemctl)"

# The restart is the step that makes new code serve. Without a terminal sudo
# cannot prompt, so require the NOPASSWD rule before changing anything.
if [[ ! -t 0 ]] && ! sudo -n -l | grep -q "NOPASSWD:.*$SYSTEMCTL restart $SERVICE"; then
    echo "deploy.sh: no NOPASSWD sudo rule for '$SYSTEMCTL restart $SERVICE' and no terminal to prompt; nothing was changed" >&2
    exit 1
fi

deploy_started_at="$(date +%s)"

echo "==> git pull"
git pull --ff-only

echo "==> uv sync"
uv sync

# Migration ordering: alembic runs before the restart, so the still-running old
# scheduler briefly sees the new schema. This is safe for additive migrations
# (the norm). If a migration is destructive, take the scheduler down first.
echo "==> alembic upgrade head"
uv run alembic upgrade head

# Smoke runs with SCHEDULER_DRY_RUN=true so self-scheduling is a no-op: jobs do
# their real work (gated by their cadence) but write nothing to the live
# schedule store. The running scheduler keeps serving old code until smoke passes.
echo "==> scheduler smoke (SCHEDULER_DRY_RUN=true)"
SCHEDULER_DRY_RUN=true uv run odds scheduler smoke

# Only reached when migrate + smoke both succeed (set -e aborts otherwise).
echo "==> restarting $SERVICE"
sudo "$SYSTEMCTL" restart "$SERVICE"

echo "==> verifying restart"
restarts_before="$(systemctl show "$SERVICE" -p NRestarts --value)"
sleep 15
fail() {
    echo "deploy.sh: $1" >&2
    systemctl status "$SERVICE" --no-pager >&2 || true
    exit 1
}
systemctl is-active --quiet "$SERVICE" || fail "$SERVICE is not active after the restart"
active_since="$(systemctl show "$SERVICE" --timestamp=unix -p ActiveEnterTimestamp --value)"
active_since="${active_since#@}"
[[ "$active_since" =~ ^[0-9]+$ ]] || fail "could not read $SERVICE's ActiveEnterTimestamp ('$active_since')"
(( active_since >= deploy_started_at )) \
    || fail "$SERVICE has been active since $active_since (unix), before this deploy began at $deploy_started_at; it is still on the old code"
restarts_after="$(systemctl show "$SERVICE" -p NRestarts --value)"
(( restarts_after == restarts_before )) \
    || fail "$SERVICE restarted $((restarts_after - restarts_before)) time(s) within 15 s of the deploy; it is crash-looping"
systemctl status "$SERVICE" --no-pager || true

echo "==> scheduled jobs"
uv run odds scheduler list-jobs

echo "==> deploy complete"
