---
type: howto
title: Set up the odds-web host
description: Run the one-time, sudo-only wizard that creates the odds_web role, writes WEB_DATABASE_URL, installs the unit and its sudo rules, and exposes the UI on the tailnet, before the first deploy.
sources:
  - scripts/setup-odds-web-host.sh
  - scripts/odds_web_grants.sql
  - odds_web.role_check
  - odds-web.service
  - deploy.sh
status: stable
---

# Set up the odds-web host

Do this once on the production host, after the PR that adds [odds-web](odds-web.md) has merged and before the first `./deploy.sh`. Until it is done, `deploy.sh` exits at its preflight with nothing changed, because the `odds-web` unit and its sudo rules are missing ([Redeploy](../scheduling/howto-run-scheduler-service.md#redeploy)).

1. **Run the wizard** from the repo root of the production host, in a terminal: `./scripts/setup-odds-web-host.sh`. It asks before each change, and `Ctrl-C` stops it between stages. Its stages, in order:
   1. **Back up the database** with `./scripts/backup-prod-db.sh odds-web-role`, because the next stage writes to production ([Back up and restore](../foundations/howto-back-up-production-database.md)).
   2. **Create the role and apply the grants.** It creates the login role `odds_web` with a generated password, or resets the password of an existing one, and applies `scripts/odds_web_grants.sql` through the compose `postgres` service. The grants name the APScheduler tables, so the scheduler must have booted once on this database.
   3. **Write `WEB_DATABASE_URL` into `.env`** (`odds_core.config.DatabaseConfig.web_url`), first setting the file to mode `600`, then run `odds_web.role_check` as the role. It checks that the role is not elevated, can `SELECT` from `paper_trades` and holds no write privilege on any public table, and that an `INSERT` is rejected. The role's read-only default makes PostgreSQL reject the `INSERT` as a read-only transaction before it checks privileges, so either rejection passes; the privilege check is what proves the grants.
   4. **Install and enable `odds-web.service`.** It copies the unit to `/etc/systemd/system`, reloads systemd and enables the unit without starting it. Edit `User` and `WorkingDirectory` in the unit first if this host differs from the defaults.
   5. **Install the sudo rules** `deploy.sh` needs as `/etc/sudoers.d/odds-web`, checked with `visudo -cf` first: `systemctl stop odds-web` and `systemctl restart odds-web` without a password. The paths must match `command -v systemctl`.
   6. **Expose it on the tailnet.** The tailnet access policy is the only authentication, so check it admits only your own devices to this host (the wizard opens the admin console), then `sudo tailscale serve --bg 8765`. Serve and HTTPS Certificates must both be enabled for the tailnet in the admin console; without HTTPS Certificates the page fails its TLS handshake.
2. **Deploy.** `./deploy.sh` starts `odds-web` on the merged code.
3. **Check.** `systemctl status odds-web`, then open the host's tailnet address from the laptop. The portfolio page is at `/`.

Re-running the wizard is safe: each run generates a new password, resets the role to it and rewrites `WEB_DATABASE_URL` to match. Restart `odds-web` afterwards to pick it up.

## Related

- [odds-web, the read-only web UI](odds-web.md)
- [Run the scheduler as a systemd service](../scheduling/howto-run-scheduler-service.md)
