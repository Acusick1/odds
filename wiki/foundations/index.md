---
title: Foundations
description: What the system is, the five packages and the CLI, configuration, databases and migrations, code and doc conventions, odds arithmetic, and local development.
---

# Foundations

Repo-wide material that every other area assumes: what the system is and which sports are live, how the workspace packages fit together, how settings and databases are wired, the conventions the code follows, and how to set up and test locally.

<!-- pages:start -->
### Concepts

- [Code conventions, timezones and documentation](conventions.md) — The UTC rule and its remaining naive-datetime sites, async-only DB access, the mechanical gates (ruff, Vale, lychee, wiki lint) versus review-only conventions, logging setup, what is authoritative for what, and where the issue-tracker conventions live.
- [Odds arithmetic and devigging](odds-math-and-devigging.md) — The shared price-conversion, implied-probability, proportional-devig and profit helpers, which consumers devig and which stay vig-inclusive, and the diverging converter copies that store prices one point apart.
- [Project overview and pivots](project-overview.md) — What the system is, which sports are active, where the edge is meant to come from, what runs where, the repo layout and stack, and which NBA-era defaults still lurk in live code.

### Components

- [Configuration (Settings)](configuration.md) — The Pydantic Settings sections and env names, what is frozen at import, which variables bypass Settings, how .env.example breaks if copied, and how to add a setting.
- [Databases, environments and copying production data](databases-and-environments.md) — How code gets sessions, NullPool in Lambda, the test/local/dev/prod databases and which env var points where, how to copy production data locally, and the fallbacks that can silently hit the wrong database.
- [Packages and the odds CLI](packages-and-cli.md) — Which of the five workspace packages holds what, the dependency rule and the undeclared odds-lambda to odds-analytics cycle, misleading module names, and the odds CLI's command groups, owning pages and shared flags.

### How-tos

- [Create and apply a database migration](howto-database-migration.md) — How to change a model, autogenerate and review an Alembic migration, and promote it local, then CI dev, then prod without breaking the running release.
- [Set up, run and test locally](howto-develop-locally.md) — How to install, write a working .env, start Docker Postgres, migrate, make a first fetch, and run unit and integration tests without touching the dev or production database.
<!-- pages:end -->
