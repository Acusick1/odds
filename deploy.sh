#!/usr/bin/env bash
#
# Deploy-with-confidence for the always-on local scheduler.
#
# Pulls the latest code, syncs dependencies, applies migrations, smoke-tests
# the bootstrapped jobs, and only then restarts the systemd-managed scheduler.
# If migration or smoke fails, the script aborts BEFORE the restart; the running
# scheduler is untouched on the old code unless a migration was pending, in which
# case it was stopped first and stays down.
#
# When the database is not at the migration head, the scheduler is stopped before
# the migration runs, because a migration may drop or rewrite what the running code
# reads. Every failure after that stop says the scheduler is down.
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

# The restart makes new code serve, and the stop precedes a pending migration.
# Without a terminal sudo cannot prompt, so require both NOPASSWD rules before
# changing anything.
if [[ ! -t 0 ]]; then
    sudo_rules="$(sudo -n -l)"
    for action in restart stop; do
        if ! grep -q "NOPASSWD:.*$SYSTEMCTL $action $SERVICE" <<<"$sudo_rules"; then
            echo "deploy.sh: no NOPASSWD sudo rule for '$SYSTEMCTL $action $SERVICE' and no terminal to prompt; nothing was changed" >&2
            exit 1
        fi
    done
fi

scheduler_down=0
report_scheduler_down() {
    local status=$?
    if (( status != 0 && scheduler_down == 1 )); then
        echo "deploy.sh: FAILED with $SERVICE STOPPED. The scheduler is down: fix the failure, rerun ./deploy.sh, or start it with 'sudo $SYSTEMCTL start $SERVICE'." >&2
    fi
}
trap report_scheduler_down EXIT

deploy_started_at="$(date +%s)"

echo "==> git pull"
git pull --ff-only

echo "==> uv sync"
uv sync

# Migration ordering: a pending migration runs with the scheduler stopped, so the
# old code never reads a schema it does not know. The hand-run agent's MCP server
# is not managed here: do not deploy during an agent session.
if ! uv run alembic current | grep -q "(head)"; then
    echo "==> migration pending: stopping $SERVICE"
    sudo "$SYSTEMCTL" stop "$SERVICE"
    scheduler_down=1
fi
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
scheduler_down=0

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
