---
type: howto
title: Run the scheduler as a systemd service
description: Install the scheduler's systemd unit on the production host, optionally keep WSL running from Windows boot, and redeploy production (the scheduler and the web UI) with deploy.sh.
sources:
  - odds-scheduler.service
  - deploy.sh
  - odds_web.grants
  - odds_web.role_check
  - docker-compose.yml
status: stable
---

# Run the scheduler as a systemd service

Run the [local scheduler](local-scheduler.md) on the production host under systemd and redeploy it with a script that proves changed jobs run before restarting. The scheduler runs every scheduled job and its Postgres is the production database ([Where each job runs](job-deployment-matrix.md), [ADR-0008](../../docs/adr/0008-mini-pc-is-production.md)).

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
3. **Let the deploying user restart the service without a password.** `deploy.sh` restarts the unit (and stops it before a pending migration) through `sudo` and, with no terminal, refuses to start unless both rules exist. Create it with `sudo visudo -f /etc/sudoers.d/odds-scheduler` (replace `<deploy-user>` with the unit's `User`):
   ```
   <deploy-user> ALL=(root) NOPASSWD: /usr/bin/systemctl start odds-scheduler, /usr/bin/systemctl stop odds-scheduler, /usr/bin/systemctl restart odds-scheduler, /usr/bin/systemctl enable --now odds-scheduler, /usr/bin/systemctl daemon-reload
   ```
   - The paths must match `command -v systemctl`; `sudo -n -l` lists the rule.
4. **Check.** `systemctl status odds-scheduler`, `journalctl -u odds-scheduler -f`.

## Start it at Windows boot (WSL hosts only)

If the production host runs Linux under WSL on Windows, the unit runs inside WSL, which starts only when something calls `wsl.exe` and stops once no `wsl.exe` client remains.

1. **Enable systemd in WSL.** `/etc/wsl.conf` needs `[boot]` `systemd=true`; run `wsl --shutdown` from Windows after changing it.
2. **Add a Windows Task Scheduler task.**
   - Trigger: At startup. Security: run as your Windows user, "Run whether user is logged on or not".
   - Action: `wsl.exe` with arguments `-d <distro> -- sleep infinity` (`wsl -l` lists distros). The `sleep` keeps a client attached so WSL stays up.
   - Settings: clear "Stop the task if it runs longer than", and restart it on failure.
3. **Check after a reboot.** Leave the machine at the login screen for a few minutes, then sign in and compare `systemctl show odds-scheduler -p ActiveEnterTimestamp` with the boot time.

## Redeploy

Run `./deploy.sh` from the repo root on the production host; it is the only path to production. It works from a terminal or from a non-interactive `ssh <production-host> 'cd <repo> && ./deploy.sh'`.

1. **Preflight.** It puts `$HOME/.local/bin` (the uv installer's directory) on `PATH` and checks that `git`, `uv`, `sudo` and `systemctl` resolve. It requires `odds-web.service` to be installed, and without a terminal it also requires the NOPASSWD rules, each matching the complete command, for `systemctl restart` and `systemctl stop` of `odds-scheduler` ([Install the unit](#install-the-unit)) and of `odds-web` ([Set up the odds-web host](../operations/howto-set-up-odds-web-host.md)). A failed check exits before anything changes; the unit check runs with or without a terminal. From a terminal without the rule, sudo prompts for a password at the restart, after the smoke run; until it is answered the checkout and environment are ahead of the running scheduler, which keeps serving the old code.
2. **Update and validate.** `git pull --ff-only`, `uv sync`. If `uv run alembic current` is not at head, it stops the scheduler (`sudo systemctl stop odds-scheduler`) and the web UI (`sudo systemctl stop odds-web`), dumps the database with `scripts/backup-prod-db.sh pre-deploy-<short sha>` ([Back up and restore](../foundations/howto-back-up-production-database.md)), then runs `uv run alembic upgrade head`; any later failure prints which of the two are down with the commands that restore the old code, unless only the web UI is down (below) (`git checkout` the previous commit, `uv sync`, `uv run python -m odds_web.grants` on it when the grants had already been applied, start the services; `alembic downgrade` first only if the migration committed, and the pre-deploy dump for rows the migration deleted). If `alembic current` itself fails, it exits without stopping either. Then `SCHEDULER_DRY_RUN=true uv run odds scheduler smoke`; dry run suppresses schedule writes, Discord alerts, alert rate-limit rows and heartbeats, so the new code's `check-health` cannot report a job the running scheduler has not booked yet as a dead chain, and smoke leaves `alert_history` untouched. Then `uv run python -m odds_web.grants` applies the `odds_web` role's grants over `DATABASE_URL`, so a table the migration added to `odds_web.role_check.READ_TABLES` is readable with no manual step, and `uv run python -m odds_web.role_check` connects as the role over `WEB_DATABASE_URL` and fails unless it can read every table in the list and cannot write ([odds-web](../operations/odds-web.md#database-access)).
3. **Restart.** `sudo systemctl restart odds-scheduler`, verified as in step 4, then `sudo systemctl restart odds-web` and the same verification. New code serves only from here: until the restart the running scheduler keeps the modules it loaded at start, whatever the checkout holds.
4. **Verify.** After each restart, after 15 s, it exits non-zero, printing the unit status and both services' states, unless the unit is active, its `ActiveEnterTimestamp` (read as a unix timestamp) is no earlier than the start of the deploy, and its `NRestarts` has not grown (no crash loop). A web failure fails the deploy but never stops or restarts the scheduler, which is already on the new code; the message names the web UI as stopped when it is, and tells you to leave the scheduler alone, read `journalctl -u odds-web`, restart `odds-web` or rerun `./deploy.sh`, never to downgrade or check out the old code. Then it prints `odds scheduler list-jobs`.

- `set -e` aborts before the restart if migration, smoke, the grants or the role check fails. The old code keeps serving unless a migration was pending, in which case the scheduler and the web UI were stopped first and stay down until a rerun or `sudo systemctl start odds-scheduler odds-web`.
- Dry-run makes self-scheduling a no-op, so the live schedule store is untouched. It does not stop the jobs' own work: smoke runs each job's real `main()`. It skips `fetch-odds` and `fetch-scores` unless passed `--include-paid-api`, so a deploy spends no Odds API quota ([CLI](local-scheduler.md#cli)).
- `--ff-only` fails on a diverged checkout, so discard or commit local edits on the host first.
- A pending migration always runs with the scheduler and the web UI stopped, so neither old process reads a schema it does not know. The hand-run agent's MCP server is not managed by the script and reads the database directly: do not deploy during an agent session ([Create and apply a database migration](../foundations/howto-database-migration.md)).

## Apply a compose change to the production database

`deploy.sh` never touches the `postgres` container, so a change to `docker-compose.yml` (for example `shm_size`) takes effect only when the container is recreated by hand on the production host. This restarts the database, so stop the scheduler first.

1. **Stop the scheduler.** `sudo systemctl stop odds-scheduler`; and do not run this during an agent session, whose MCP server reads the database directly.
2. **Recreate Postgres.** `docker compose up -d postgres` from the repo root.
   - It recreates the container from the new definition; the `postgres_data` volume persists, so no data is lost.
3. **Wait for healthy.** `docker compose ps postgres` shows `healthy` (the `pg_isready` healthcheck).
   - Check the change took effect, e.g. `docker inspect <container> --format '{{.HostConfig.ShmSize}}'` prints `1073741824` after `shm_size: 1g`.
4. **Start the scheduler.** `sudo systemctl start odds-scheduler`, then `odds scheduler list-jobs`.

## Related

- [Local scheduler](local-scheduler.md)
- [Scheduler backends](scheduler-backends.md)
