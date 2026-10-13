---
type: howto
title: Run the scheduler as a systemd service
description: Install the scheduler's systemd unit on an always-on host, start it, and redeploy it with deploy.sh.
sources:
  - odds-scheduler.service
  - deploy.sh
status: stable
---

# Run the scheduler as a systemd service

Run the [local scheduler](local-scheduler.md) on a mini-PC under systemd and redeploy it with a script that proves changed jobs run before restarting.

Before you start, check that no AWS scraper runs the same jobs (`odds scheduler list-jobs --backend aws`), or both will scrape ([Deployed state](job-deployment-matrix.md#deployed-state)).

## Install the unit

1. **Edit `odds-scheduler.service`** (repo root): set `User`, `WorkingDirectory`, `EnvironmentFile` (the repo `.env`) and the absolute `uv` path in `ExecStart` (`uv run odds scheduler start`). systemd needs an absolute executable path; `which uv` shows it.
2. **Install and start.**
   ```bash
   sudo cp odds-scheduler.service /etc/systemd/system/odds-scheduler.service
   sudo systemctl daemon-reload
   sudo systemctl enable --now odds-scheduler
   ```
   - `Restart=always` with `RestartSec=5` restarts it after a crash or reboot.
3. **Check.** `systemctl status odds-scheduler`, `journalctl -u odds-scheduler -f`.

## Redeploy

Run `./deploy.sh` from the repo root on the host. In order it runs `git pull --ff-only`, `uv sync`, `uv run alembic upgrade head`, `SCHEDULER_DRY_RUN=true uv run odds scheduler smoke`, `sudo systemctl restart odds-scheduler`, then status and `odds scheduler list-jobs`.

- `set -e` aborts before the restart if migration or smoke fails, so the old code keeps serving.
- Dry-run makes self-scheduling a no-op, so the live schedule store is untouched.
- `--ff-only` fails on a diverged checkout, so discard or commit local edits on the host first.
- `alembic upgrade head` runs before the restart, so the old scheduler briefly sees the new schema. That is safe for additive migrations. For a destructive one, `sudo systemctl stop odds-scheduler` first, migrate, then redeploy ([Create and apply a database migration](../foundations/howto-database-migration.md)).

## Related

- [Local scheduler](local-scheduler.md)
- [Scheduler backends](scheduler-backends.md)
