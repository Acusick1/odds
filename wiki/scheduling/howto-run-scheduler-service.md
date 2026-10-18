---
type: howto
title: Run the scheduler as a systemd service
description: Install the scheduler's systemd unit on the production mini-PC, start it at Windows boot under WSL, and redeploy production with deploy.sh.
sources:
  - odds-scheduler.service
  - deploy.sh
status: stable
---

# Run the scheduler as a systemd service

Run the [local scheduler](local-scheduler.md) on the mini-PC under systemd and redeploy it with a script that proves changed jobs run before restarting. This host is production: its scheduler runs every scheduled job and its Postgres is the production database ([Where each job runs](job-deployment-matrix.md), [ADR-0008](../../docs/adr/0008-mini-pc-is-production.md)).

The AWS rules must stay disabled, or both hosts collect. `odds scheduler list-jobs --backend aws` shows their state ([Deployed state](job-deployment-matrix.md#deployed-state)).

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
3. **Let the deploying user restart the service without a password.** `deploy.sh` restarts the unit through `sudo` and, with no terminal, refuses to start unless this rule exists. Create it with `sudo visudo -f /etc/sudoers.d/odds-scheduler` (replace `andrew` with the unit's `User`):
   ```
   andrew ALL=(root) NOPASSWD: /usr/bin/systemctl start odds-scheduler, /usr/bin/systemctl stop odds-scheduler, /usr/bin/systemctl restart odds-scheduler, /usr/bin/systemctl enable --now odds-scheduler, /usr/bin/systemctl daemon-reload
   ```
   - The paths must match `command -v systemctl`; `sudo -n -l` lists the rule.
4. **Check.** `systemctl status odds-scheduler`, `journalctl -u odds-scheduler -f`.

## Start it at Windows boot (WSL hosts)

On a Windows host the unit runs inside WSL, which starts only when something calls `wsl.exe` and stops once no `wsl.exe` client remains.

1. **Enable systemd in WSL.** `/etc/wsl.conf` needs `[boot]` `systemd=true`; run `wsl --shutdown` from Windows after changing it.
2. **Add a Windows Task Scheduler task.**
   - Trigger: At startup. Security: run as your Windows user, "Run whether user is logged on or not".
   - Action: `wsl.exe` with arguments `-d <distro> -- sleep infinity` (`wsl -l` lists distros). The `sleep` keeps a client attached so WSL stays up.
   - Settings: clear "Stop the task if it runs longer than", and restart it on failure.
3. **Check after a reboot.** Leave the machine at the login screen for a few minutes, then sign in and compare `systemctl show odds-scheduler -p ActiveEnterTimestamp` with the boot time.

## Redeploy

Run `./deploy.sh` from the repo root on the mini-PC; it is the only path to production. It works from a terminal or from a non-interactive `ssh <host> 'cd ~/odds && ./deploy.sh'`.

1. **Preflight.** It puts `$HOME/.local/bin` (the uv installer's directory) on `PATH` and checks that `git`, `uv`, `sudo` and `systemctl` resolve. Without a terminal it also requires the NOPASSWD rule for `systemctl restart odds-scheduler` ([Install the unit](#install-the-unit)). A failed check exits before anything changes.
2. **Update and validate.** `git pull --ff-only`, `uv sync`, `uv run alembic upgrade head`, `SCHEDULER_DRY_RUN=true uv run odds scheduler smoke`.
3. **Restart.** `sudo systemctl restart odds-scheduler`. New code serves only from here: until the restart the running scheduler keeps the modules it loaded at start, whatever the checkout holds.
4. **Verify.** After 15 s it exits non-zero, printing the unit status, unless the unit is active, its `ActiveEnterTimestamp` is later than the start of the deploy, and its `NRestarts` has not grown (no crash loop). Then it prints `odds scheduler list-jobs`.

- `set -e` aborts before the restart if migration or smoke fails, so the old code keeps serving.
- Dry-run makes self-scheduling a no-op, so the live schedule store is untouched. It does not stop the jobs' own work: smoke runs each job's real `main()`, so a due `fetch-odds` or `fetch-scores` spends Odds API quota ([CLI](local-scheduler.md#cli)).
- `--ff-only` fails on a diverged checkout, so discard or commit local edits on the host first.
- `alembic upgrade head` runs before the restart, so the old scheduler briefly sees the new schema. That is safe for additive migrations. For a destructive one, `sudo systemctl stop odds-scheduler` first, migrate, then redeploy ([Create and apply a database migration](../foundations/howto-database-migration.md)).

## Related

- [Local scheduler](local-scheduler.md)
- [Scheduler backends](scheduler-backends.md)
