---
type: howto
title: Run the scheduler as a systemd service
description: Install the scheduler's systemd unit on an always-on host, start it at Windows boot on a WSL host, and redeploy it with deploy.sh.
sources:
  - odds-scheduler.service
  - deploy.sh
status: stable
---

# Run the scheduler as a systemd service

Run the [local scheduler](local-scheduler.md) on a mini-PC under systemd and redeploy it with a script that proves changed jobs run before restarting.

Before you start, check that no AWS scraper runs the same jobs (`odds scheduler list-jobs --backend aws`), or both will scrape ([Deployed state](job-deployment-matrix.md#deployed-state)).

## Install the unit

1. **Edit `odds-scheduler.service`** (repo root): set `User`, `WorkingDirectory` (the repo root) and the absolute `uv` path in `ExecStart` (`uv run odds scheduler start`). systemd needs an absolute executable path; `which uv` shows it.
   - Settings read `.env` from `WorkingDirectory`, so the unit sets no `EnvironmentFile=`. systemd does not expand `${VAR}` in one, and its values would override `.env`.
2. **Install and start.**
   ```bash
   sudo cp odds-scheduler.service /etc/systemd/system/odds-scheduler.service
   sudo systemctl daemon-reload
   sudo systemctl enable --now odds-scheduler
   ```
   - `Restart=always` with `RestartSec=5` restarts it after a crash or reboot.
3. **Check.** `systemctl status odds-scheduler`, `journalctl -u odds-scheduler -f`.

## Start it at Windows boot (WSL hosts)

On a Windows host the unit runs inside WSL, which starts only when something calls `wsl.exe` and stops once no `wsl.exe` client remains.

1. **Enable systemd in WSL.** `/etc/wsl.conf` needs `[boot]` `systemd=true`; run `wsl --shutdown` from Windows after changing it.
2. **Add a Windows Task Scheduler task.**
   - Trigger: At startup. Security: run as your Windows user, "Run whether user is logged on or not".
   - Action: `wsl.exe` with arguments `-d <distro> -- sleep infinity` (`wsl -l` lists distros). The `sleep` keeps a client attached so WSL stays up.
   - Settings: clear "Stop the task if it runs longer than", and restart it on failure.
3. **Check after a reboot.** Leave the machine at the login screen for a few minutes, then sign in and compare `systemctl show odds-scheduler -p ActiveEnterTimestamp` with the boot time.

## Redeploy

Run `./deploy.sh` from the repo root on the host. In order it runs `git pull --ff-only`, `uv sync`, `uv run alembic upgrade head`, `SCHEDULER_DRY_RUN=true uv run odds scheduler smoke`, `sudo systemctl restart odds-scheduler`, then status and `odds scheduler list-jobs`.

- `set -e` aborts before the restart if migration or smoke fails, so the old code keeps serving.
- Dry-run makes self-scheduling a no-op, so the live schedule store is untouched.
- `--ff-only` fails on a diverged checkout, so discard or commit local edits on the host first.
- `alembic upgrade head` runs before the restart, so the old scheduler briefly sees the new schema. That is safe for additive migrations. For a destructive one, `sudo systemctl stop odds-scheduler` first, migrate, then redeploy ([Create and apply a database migration](../foundations/howto-database-migration.md)).

## Related

- [Local scheduler](local-scheduler.md)
- [Scheduler backends](scheduler-backends.md)
