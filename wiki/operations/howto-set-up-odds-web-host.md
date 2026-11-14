---
type: howto
title: Set up the odds-web host
description: Run the one-time, sudo-only wizard that creates the odds_web role, writes WEB_DATABASE_URL, installs the unit and its sudo rules, and exposes the UI on the tailnet, before the first deploy.
sources:
  - scripts/setup-odds-web-host.sh
  - odds_web.grants
  - odds_web.role_check
  - odds-web.service
  - deploy.sh
status: stable
---

# Set up the odds-web host

Do this once on the production host, after the PR that adds [odds-web](odds-web.md) has merged and before the first `./deploy.sh`. Until it is done, `deploy.sh` exits at its preflight with nothing changed, because the `odds-web` unit and its sudo rules are missing ([Redeploy](../scheduling/howto-run-scheduler-service.md#redeploy)).

1. **Run the wizard** from the repo root of the production host, in a terminal: `./scripts/setup-odds-web-host.sh`. It asks before each change, and `Ctrl-C` stops it between stages. Its stages, in order:
   1. **Back up the database** with `./scripts/backup-prod-db.sh odds-web-role`, because the next stage writes to production ([Back up and restore](../foundations/howto-back-up-production-database.md)).
   2. **Create the role and apply the grants.** It creates the login role `odds_web` with a generated password, or resets the password of an existing one, and applies its grants with `uv run python -m odds_web.grants` over `DATABASE_URL`. The grants name the APScheduler tables (`schedules` and `jobs`), so the scheduler must have booted once on this database, and the tables of every migration (the candidate decisions and stored backtests among them), so the host must be migrated to the current head. The grants are safe to re-run, and `deploy.sh` re-applies them after every migration, so a table added to `odds_web.role_check.READ_TABLES` needs no manual step ([Apply the grants by hand](#apply-the-grants-by-hand)).
   3. **Write `WEB_DATABASE_URL` into `.env`** (`odds_core.config.DatabaseConfig.web_url`), first setting the file to mode `600`, then run `odds_web.role_check` as the role. It checks that the role is not elevated, can `SELECT` from every table in `odds_web.role_check.READ_TABLES` (the ones the readers use) and holds no write privilege on any public table, and that an `INSERT` is rejected. The role's read-only default makes PostgreSQL reject the `INSERT` as a read-only transaction before it checks privileges, so either rejection passes; the privilege check is what proves the grants.
   4. **Install and enable `odds-web.service`.** It copies the unit to `/etc/systemd/system`, reloads systemd and enables the unit without starting it. Edit `User` and `WorkingDirectory` in the unit first if this host differs from the defaults.
   5. **Install the sudo rules** `deploy.sh` needs as `/etc/sudoers.d/odds-web`, checked with `visudo -cf` first: `systemctl stop odds-web` and `systemctl restart odds-web` without a password. The paths must match `command -v systemctl`.
   6. **Expose it on the tailnet.** The tailnet access policy is the only authentication, so check it admits only your own devices to this host, and that MagicDNS and HTTPS Certificates are enabled on the DNS page (the wizard opens both admin console pages), then `sudo tailscale serve --bg 8765`. The first `tailscale serve` on a tailnet prints a link to enable Serve; open it and re-run the stage.
2. **Deploy.** `./deploy.sh` starts `odds-web` on the merged code.
3. **Check.** `systemctl status odds-web`, then open the host's tailnet address from the laptop. The portfolio page is at `/`.

Re-running the wizard is safe: each run generates a new password, resets the role to it and rewrites `WEB_DATABASE_URL` to match. Restart `odds-web` afterwards to pick it up.

## Apply the grants by hand

`deploy.sh` applies the grants and checks them as the role after each migration ([Redeploy](../scheduling/howto-run-scheduler-service.md#redeploy)), so this is only for repairing a role between deploys:

1. **Back up the database** with `./scripts/backup-prod-db.sh odds-web-grants`.
2. **Apply them** from the repo root: `uv run python -m odds_web.grants`, then `uv run python -m odds_web.role_check`.

No restart is needed: the next query on the role sees the new grant. Do not re-run the wizard's role stage for this, because it sets a new password that only the next stage writes into `.env`.

## Related

- [odds-web, the read-only web UI](odds-web.md)
- [Run the scheduler as a systemd service](../scheduling/howto-run-scheduler-service.md)
