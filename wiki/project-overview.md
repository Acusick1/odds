---
type: concept
title: Project overview
summary: "What the system is, which sports are active, where the edge is meant to come from, what runs where, the top-level repo layout and the tech stack."
sources:
  - packages/odds-core/odds_core/sports.py::SportKey
  - packages/odds-lambda/odds_lambda/scheduling/jobs.py::_JOB_CRON_MAP
  - packages/odds-lambda/odds_lambda/scheduling/jobs.py::_JOB_MODULE_MAP
  - packages/odds-cli/odds_cli/main.py::app
  - deployment/aws/terraform/eventbridge.tf
  - docs/ARCHITECTURE.md#system-overview
  - docs/ARCHITECTURE.md#technical-stack
  - docs/DEPLOYMENT.md#job-deployment-matrix
verified_at: 7887039
---

# Project overview

A single-user, sport-agnostic betting data pipeline with sport-specific LLM betting agents ([README](../README.md), [AGENTS.md](../AGENTS.md#project-overview)). Shared infrastructure (odds collection, storage, paper trading, scheduling, alerting) is reused per sport; each sport adds its own data sources, feature extractors, agent prompts and MCP tools.

## Sports

EPL and MLB are active, with agents in interactive evaluation, not autonomous production. NBA code (injuries, game logs, Polymarket targeting) exists but is deprioritized. `packages/odds-core/odds_core/sports.py::SportKey` allows all three; see [Sport and market keys](sport-and-market-keys.md) and [Project pivots](project-pivots.md) for NBA assumptions still in code.

## Where the edge is meant to come from

Breadth and speed of information synthesis by an LLM agent, not the XGBoost CLV model, which is a supplementary signal. See [Pivot to information edge](decision-pivot-to-information-edge.md), [Betting agent](betting-agent.md) and [CLV model findings](clv-model-findings.md).

## What runs where

Two AWS Lambdas (`odds-scheduler`, `odds-scheduler-scraper`) plus a local scheduler. The agent, daily digest, ESPN fixtures and MLB probables jobs run locally only: Terraform declares no fixed-schedule EventBridge rules (`deployment/aws/terraform/eventbridge.tf`), and the local cron table is `packages/odds-lambda/odds_lambda/scheduling/jobs.py::_JOB_CRON_MAP`. The repo does not record the deployed state; the check is `odds scheduler list-jobs --backend aws`. Design: [Job deployment matrix](job-deployment-matrix.md), [Local scheduler](local-scheduler.md).

## Repository layout

| Path | Contents |
|---|---|
| `packages/` | Five uv workspace members: [odds-core](package-odds-core.md), [odds-lambda](package-odds-lambda.md), [odds-analytics](package-odds-analytics.md), [odds-mcp](mcp-server.md), [odds-cli](cli-overview.md) |
| `docs/` | Human docs, see [Documentation map](documentation-map.md) |
| `experiments/` | `configs/`, `scripts/`, `results/` for modelling runs |
| `scripts/` | Operational scripts (ingestion, tier checks, `ci/`, `wiki_lint.py`) |
| `migrations/`, `alembic.ini` | Alembic, see [migrations](howto-database-migration.md) |
| `tests/` | `unit/`, `integration/`, `migration/`, see [tests](howto-run-tests.md) |
| `deployment/aws/` | Terraform, Dockerfiles, bootstrap |
| `.github/workflows/` | [CI/CD](ci-cd-workflows.md) |
| `wiki/` | This wiki |

`.claude/`, `worktrees/` and `logs/` are gitignored.

## Stack

Python 3.12+, SQLModel on PostgreSQL 15 with asyncpg, APScheduler 4 (local backend), Typer and Rich (CLI), FastMCP, tenacity, structlog, Playwright, XGBoost, uv workspace ([ARCHITECTURE](../docs/ARCHITECTURE.md#technical-stack)). Conventions: [Code conventions](code-conventions.md), [Configuration](configuration.md).

## Hazards

- Root `pyproject.toml` still describes the project as "NBA betting odds data collection and analysis system", and the CLI banner reads "NBA odds collection and analysis" (`packages/odds-cli/odds_cli/main.py::app`). Neither reflects the active sports.

## Related

- [Data sources](data-sources.md) — what feeds the pipeline
- [Reporting surfaces](reporting-surfaces.md) — how the system reports on itself
