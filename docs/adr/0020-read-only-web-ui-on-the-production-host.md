---
status: proposed
---

# Serve a read-only web UI from the production host

## Context

The forward test of each candidate is read today through `odds candidates report`, the MCP `get_portfolio` tool and SQL ([Reporting surfaces](../../wiki/operations/reporting-surfaces.md)). None of them shows a candidate's running estimate, its bets or the coverage of its inputs, and the owner reads them from a laptop and the production host (#807).

The production host runs the scheduler under systemd and holds the production database ([ADR-0008](0008-mini-pc-is-production.md)). Anything that serves the owner a page from that host is a second long-running process next to the scheduler, on the machine whose failure stops collection (#489).

## Decision

`odds web serve` (package `odds-web`) is a long-running, single-user web service on the production host.

- **Read-only role with explicit grants.** It connects as a new Postgres role `odds_web` through `WEB_DATABASE_URL` (`odds_core.config.DatabaseConfig.web_url`) and refuses to start without it, so it cannot fall back to the write-capable `DATABASE_URL`. `odds_web.grants` grants `SELECT` on the tables in `odds_web.role_check.READ_TABLES` only: no `pg_read_all_data`, no default privileges, so a table added later is unreadable until listed. The app's own transactions are also read-only and carry a statement timeout. The `odds_agent` role is the precedent ([ADR-0008](0008-mini-pc-is-production.md)).
- **Tailnet-only exposure.** The process listens on `127.0.0.1` with no host option. `tailscale serve --bg 8765` publishes it to the tailnet, whose access policy is the only authentication.
- **Coupled to `deploy.sh`, not to the scheduler.** `odds-web.service` has no systemd dependency on `odds-scheduler.service` in either direction and carries `CPUQuota` and `MemoryMax`. `deploy.sh` checks the unit and its sudo rules before changing anything, stops the web service with the scheduler before a pending migration, restarts and verifies the scheduler first and the web service second, and a web failure fails the deploy without touching the scheduler.
- **Bounded load.** Report data is cached for about a minute with a single-flight refresh, the engine holds two connections, and bootstrap work runs on one worker thread. A failing or slow query renders its section as unavailable rather than failing the page.

## Consequences

- The production host runs a second service. A bug in it can consume CPU, memory or database connections; the quotas, pool size and statement timeout bound that, and do not remove it.
- A deploy now has two services to restart. A web failure leaves the deploy red while the scheduler runs the new code.
- `deploy.sh` applies the grants after every migration and checks them as the role. A migration that drops or renames a table in `READ_TABLES` fails the grants, and so the deploy before any restart, until the list is updated.
- Reachability and authentication depend on the tailnet policy. Anyone the policy admits to the host's port sees every candidate's bets.
- A one-time, sudo-only host setup precedes the first deploy ([Set up the odds-web host](../../wiki/operations/howto-set-up-odds-web-host.md)).

## Evidence

- Issue: #807 (design and scope).
- Precedents: [ADR-0008](0008-mini-pc-is-production.md) (one production host, the `odds_agent` role), `odds-scheduler.service`, `deploy.sh`.
- Code: `odds_web.db`, `odds_web.app`, `odds_web.grants`, `odds_web.role_check`, `odds-web.service`, `deploy.sh`.
