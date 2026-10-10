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

# Smoke runs every bootstrapped job's full body end-to-end under a no-side-effect
# policy: the cadence gate is bypassed (so the real fetch+ingest body runs even
# when not "due"), nothing is written to the live schedule store, and every
# outward call (Discord posts, agent paper bets) is suppressed at its sink. The
# running scheduler keeps serving old code until smoke passes — so "smoke passed"
# means "the changed code runs green" and the deploy is safe.
echo "==> scheduler smoke (no side effects)"
uv run odds scheduler smoke

# Only reached when migrate + smoke both succeed (set -e aborts otherwise).
echo "==> restarting odds-scheduler"
sudo systemctl restart odds-scheduler

echo "==> systemctl status"
systemctl status odds-scheduler --no-pager || true

echo "==> scheduled jobs"
uv run odds scheduler list-jobs

echo "==> deploy complete"
