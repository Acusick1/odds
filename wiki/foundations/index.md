---
title: Foundations
description: What the system is, the five packages and the CLI, configuration, databases and migrations, code and doc conventions, odds arithmetic and devigging, local development and the test suite.
---

# Foundations

Repo-wide material that every other area assumes: what the system is and which sports are live, how the workspace packages fit together, how settings and databases are wired, the conventions the code follows, and how to set up and test locally.

<!-- pages:start -->
### Reference

- [Code conventions and lint gates](code-conventions.md) — The pre-commit hooks and lint workflow, review-only conventions (type hints, async DB, model types, job entry points), and logging setup.
- [Configuration (Settings)](configuration.md) — The Settings sections and env names, what is read at import, and the variables read with os.getenv outside Settings.
- [Databases and environments](databases-and-environments.md) — The session factory and engine (NullPool in Lambda, pool sizes), the test, local, dev and production databases and the Neon archive, which variable points where, and the --db override.
- [Odds arithmetic helpers](odds-math.md) — The odds_core.odds_math functions, their edge cases, their consumers, and and the two `raw_data` price encodings.
- [Test suite](test-suite.md) — The test directories, pytest configuration, fixtures (py-pglite), markers, and which suites CI and the deploy workflows run.
- [The odds CLI](odds-cli.md) — The Typer app, its command groups and owning pages, shared flags (--sport defaults per command, --db), and help-text facts.
- [Timezones and time helpers](timezones.md) — The UTC rule, the odds_core.time helpers, which ruff DTZ rules apply, and the naive-datetime sites in current code.
- [Workspace packages and repository layout](workspace-packages.md) — The five uv workspace packages, their modules and declared dependencies, the repo layout and stack.

### Explanation

- [About the system and its pivots](project-overview.md) — What the system is, which sports are live, where the edge is meant to come from, what runs where in outline, the three pivots and the NBA-era defaults they left.
- [Devigging and vig-inclusive prices](devigging.md) — What vig and proportional devigging are here, which consumers devig and which stay vig-inclusive, and what that means for comparing prices.
- [Documentation and issue tracking](documentation-and-issue-tracking.md) — Where documentation lives (wiki, ADRs, docs/agents), what is authoritative for what, how the wiki lint and Vale keep pages current, and the issue-tracker conventions.

### How-to guides

- [Add a setting](howto-add-setting.md) — Steps to add a configuration setting so it loads from .env and reaches the production host.
- [Copy production data locally](howto-copy-production-data.md) — Steps to copy one sport and date range from production into the local database, with the command's options and copy behaviour.
- [Create and apply a database migration](howto-database-migration.md) — How to change a model, autogenerate and review an Alembic migration, and promote it local, then CI dev, then production without breaking the running release.
- [Set up and run locally](howto-develop-locally.md) — Install, write a working .env, start Postgres, migrate, make a first fetch, enable the pre-commit hooks and issue claiming, and run the offline tests.
<!-- pages:end -->
