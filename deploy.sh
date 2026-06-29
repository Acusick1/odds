#!/usr/bin/env bash
#
# Deploy-with-confidence for the always-on local scheduler.
#
# Pulls the latest code, syncs dependencies, applies migrations, smoke-tests
# the bootstrapped jobs, and only then restarts the systemd-managed scheduler.
# If migration or smoke fails, the script aborts BEFORE the restart, leaving the
# running scheduler untouched on the old code.
#
# Run from the repo root on the host running odds-scheduler.service:
#   ./deploy.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

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
echo "==> restarting odds-scheduler"
sudo systemctl restart odds-scheduler

echo "==> systemctl status"
systemctl status odds-scheduler --no-pager || true

echo "==> scheduled jobs"
uv run odds scheduler list-jobs

echo "==> deploy complete"
