---
type: howto
title: Run the scheduler on an always-on host
summary: Install the systemd unit, redeploy with deploy.sh (pull, sync, migrate, smoke, restart), and stop the service first for destructive migrations.
sources:
  - docs/DEPLOYMENT.md#always-on-host-systemd--deploysh
  - docs/DEPLOYMENT.md#smoke-check
  - docs/CLI.md#smoke-check
verified_at: 7887039
---

# Run the scheduler on an always-on host

The local scheduler ([local scheduler](local-scheduler.md)) can run as a service on a mini-PC under systemd and be redeployed by a script that proves changed jobs run before restarting. Which jobs this host should own is in [job deployment matrix](job-deployment-matrix.md).

## Install the unit

1. Edit `odds-scheduler.service` (repo root): set `User`, `WorkingDirectory`, `EnvironmentFile` (the repo `.env`) and the absolute `uv` path in `ExecStart` (`uv run odds scheduler start`).
   - systemd needs an absolute executable path; `which uv` shows it.
2. Install and start:
   ```bash
   sudo cp odds-scheduler.service /etc/systemd/system/odds-scheduler.service
   sudo systemctl daemon-reload
   sudo systemctl enable --now odds-scheduler
   ```
   - `Restart=always` with `RestartSec=5` restarts it after a crash or reboot.
3. Check: `systemctl status odds-scheduler`, `journalctl -u odds-scheduler -f`.

## Redeploy

Run `./deploy.sh` from the repo root on the host. In order it runs `git pull --ff-only`, `uv sync`, `uv run alembic upgrade head`, `SCHEDULER_DRY_RUN=true uv run odds scheduler smoke`, `sudo systemctl restart odds-scheduler`, then status and `odds scheduler list-jobs`.

- `set -e` aborts before the restart if migration or smoke fails, so the old code keeps serving.
- Dry-run makes self-scheduling a no-op, so the live schedule store is untouched.

## Smoke check options

`odds scheduler smoke` runs every job in `scheduler.bootstrap_jobs` once, expanded per sport, and exits with the failure count.

- `agent-run` is skipped unless `--include-agent`, so a deploy never places paper bets.
- `--no-side-effects` drops `daily-digest` (it posts to Discord).
- `--only` and `--exclude` take base or compound names and may repeat.
- A job's fetch body runs only if its gate says to execute; smoke otherwise exercises import, config, decision and schema paths ([decision engine](scheduling-decision-engine.md)).

## Destructive migrations

`alembic upgrade head` runs before the restart, so the old scheduler briefly sees the new schema. That is safe for additive migrations. For a destructive one, `sudo systemctl stop odds-scheduler` first, migrate, then redeploy ([database migration](howto-database-migration.md)).

## Gotchas

- Do not run this host's scheduler and an AWS scraper for the same jobs unless intended; check `odds scheduler list-jobs --backend aws`.
- `deploy.sh` is not idempotent with local edits: `--ff-only` fails on a diverged checkout.
