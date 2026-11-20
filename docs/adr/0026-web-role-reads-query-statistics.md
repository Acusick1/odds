---
status: proposed
---

# Let the read-only web role read query statistics through owner-run functions

The `odds_web` role of the read-only web UI may also read the database's query statistics, through two functions that run as their owner, and nothing else beyond its table allowlist. This supersedes [ADR-0020](0020-read-only-web-ui-on-the-production-host.md) once accepted; everything ADR-0020 decided stands except the role's read scope, and is restated here so this record stands alone.

## Context

ADR-0020 serves a single-user web UI from the production host as the `odds_web` role, which may `SELECT` from an explicit list of tables and nothing else. #925 adds a page that shows which statements cost the database the most time, across every caller (the scheduler, the agent, the web UI and the MCP server), so a slow query is visible before it becomes a bottleneck. Those figures live in the `pg_stat_statements` extension, which Postgres must preload, and are outside the table allowlist.

## Decision

`odds web serve` (package `odds-web`) is a long-running, single-user web service on the production host.

- **Read-only role with explicit grants.** It connects as the Postgres role `odds_web` through `WEB_DATABASE_URL` (`odds_core.config.DatabaseConfig.web_url`) and refuses to start without it, so it cannot fall back to the write-capable `DATABASE_URL`. `odds_web.grants` grants `SELECT` on the tables in `odds_web.role_check.READ_TABLES` only: no `pg_read_all_data`, no default privileges, so a table added later is unreadable until listed. The app's own transactions are also read-only and carry a statement timeout. The `odds_agent` role is the precedent ([ADR-0008](0008-mini-pc-is-production.md)).
- **Query statistics through owner-run functions.** The role may also execute the functions in `odds_web.role_check.READ_FUNCTIONS`, which `odds_web.grants` defines as `SECURITY DEFINER`, owned by the role that applies the grants. They return each statement's normalised text, role, calls, execution times and rows for the current database, and the time the statistics were last reset. The `SELECT` the extension grants to `PUBLIC` on its views is revoked, so the role cannot read them directly. The role is not granted `pg_read_all_stats`, and the deploy-time role check fails if it is a member of that or any other predefined role in `odds_web.role_check.WIDENING_ROLES`, as it does if the role can write or is elevated. Utility statements are not tracked (`pg_stat_statements.track_utility=off` in `docker-compose.yml`), because their text keeps its literals.
- **Tailnet-only exposure.** The process listens on `127.0.0.1` with no host option. `tailscale serve --bg 8765` publishes it to the tailnet, whose access policy is the only authentication.
- **Coupled to `deploy.sh`, not to the scheduler.** `odds-web.service` has no systemd dependency on `odds-scheduler.service` in either direction and carries `CPUQuota` and `MemoryMax`. `deploy.sh` checks the unit and its sudo rules before changing anything, stops the web service with the scheduler before it recreates Postgres or runs a pending migration, restarts and verifies the scheduler first and the web service second, and a web failure fails the deploy without touching the scheduler.
- **Bounded load.** Report data is cached for about a minute with a single-flight refresh, the engine holds two connections, and bootstrap work runs on one worker thread. A failing or slow query renders its section as unavailable rather than failing the page.

## Considered options

- **Grant `pg_read_all_stats`.** One grant, and the view shows every role's statement text. It also opens every other statistics view and the statements of every database on the server, which the page does not need.
- **Read the `pg_stat_statements` view directly.** The extension grants it to `PUBLIC`, but it hides the text of other roles' statements, which is most of what the page is for.
- **Owner-run functions (chosen).** The role reads exactly the columns the page shows, of the current database only, and the functions' definitions sit with the grants that expose them.

## Consequences

- Anyone the tailnet policy admits to the page sees the normalised text of every statement any role runs in the production database, constants replaced by placeholders.
- Preloading the extension needs a Postgres restart. `deploy.sh` recreates the container when its `docker-compose.yml` definition changes, with both services stopped, so the scheduler misses about a minute.
- The functions read the extension's views, so `odds_web.grants`, and with it the deploy, fails before any restart when the extension is missing; the role check calls the functions, so it fails when the extension is not preloaded.
- The production host runs a second service. A bug in it can consume CPU, memory or database connections; the quotas, pool size and statement timeout bound that, and do not remove it.
- A deploy has two services to restart. A web failure leaves the deploy red while the scheduler runs the new code.
- `deploy.sh` applies the grants after every migration and checks them as the role. A migration that drops or renames a table in `READ_TABLES` fails the grants, and so the deploy before any restart, until the list is updated.
- A one-time, sudo-only host setup precedes the first deploy ([Set up the odds-web host](../../wiki/operations/howto-set-up-odds-web-host.md)).

## Evidence

- Issues: #807 (the web UI's design and scope), #925 (the performance page).
- Precedents: [ADR-0008](0008-mini-pc-is-production.md) (one production host, the `odds_agent` role), [ADR-0020](0020-read-only-web-ui-on-the-production-host.md).
- Code: `odds_web.db`, `odds_web.app`, `odds_web.grants`, `odds_web.role_check`, `odds_web.perf`, `odds-web.service`, `deploy.sh`, `docker-compose.yml`.
