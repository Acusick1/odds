---
type: concept
title: Project overview and pivots
description: What the system is, which sports are active, where the edge is meant to come from, what runs where, the repo layout and stack, and which NBA-era defaults still lurk in live code.
sources:
  - packages/odds-core/odds_core/sports.py::SportKey
  - packages/odds-lambda/odds_lambda/scheduling/jobs.py::_JOB_CRON_MAP
  - packages/odds-lambda/odds_lambda/scheduling/jobs.py::_JOB_MODULE_MAP
  - packages/odds-core/odds_core/config.py::SchedulerConfig
  - packages/odds-core/odds_core/config.py::PolymarketConfig
  - packages/odds-cli/odds_cli/main.py::app
  - packages/odds-cli/odds_cli/commands/backfill.py
  - packages/odds-analytics/odds_analytics/backfill_executor.py::BackfillExecutor
  - packages/odds-lambda/odds_lambda/storage/tier_validator.py::TierCoverageValidator
  - packages/odds-lambda/odds_lambda/storage/readers.py::OddsReader
  - packages/odds-analytics/odds_analytics/strategies.py::BasicEVStrategy
  - packages/odds-lambda/odds_lambda/polymarket_fetcher.py::PolymarketClient
  - deployment/aws/terraform/eventbridge.tf
  - docs/ARCHITECTURE.md#system-overview
  - docs/ARCHITECTURE.md#technical-stack
  - docs/DEPLOYMENT.md#job-deployment-matrix
  - docs/audits/comprehension-audit-2026-06-09.md#t1-stratified-doc-rot-from-three-pivots
  - docs/audits/comprehension-audit-2026-06-09.md#t2-pivot-fossils-in-live-code-paths
  - docs/BETTING_AGENT.md
  - docs/INJURIES.md
verified: "7887039"
status: stable
---

# Project overview and pivots

A single-user, sport-agnostic betting data pipeline with sport-specific LLM betting agents ([README](../../README.md), [AGENTS.md](../../AGENTS.md#project-overview)). Shared infrastructure (odds collection, storage, paper trading, scheduling, alerting) is reused per sport; each sport adds its own data sources, feature extractors, agent prompts and MCP tools.

## Sports

EPL and MLB are active, with agents in interactive evaluation, not autonomous production. NBA code (injuries, game logs, Polymarket targeting) exists but is deprioritized. `packages/odds-core/odds_core/sports.py::SportKey` allows all three. EPL odds are stored under market key `1x2` and MLB under `h2h`; see [Data sources, sport keys and market keys](../collection/sources-and-keys.md).

## Where the edge is meant to come from

Breadth and speed of information synthesis by an LLM agent, not the XGBoost CLV model. The deployed model is `epl-clv-home` `v308058c` with a CV R² of about 0.058: too weak to bet on alone, so it is a supplementary signal the agent treats as a sanity check. See [Pivot to information edge](../agent/decision-information-edge.md), [Betting agent](../agent/betting-agent.md) and [What the CLV model can and cannot do](../evaluation/clv-model-findings.md).

## What runs where

Two AWS Lambdas (`odds-scheduler`, `odds-scheduler-scraper`) plus a local scheduler. The repo does not record the deployed state; the only check is `odds scheduler list-jobs --backend aws`.

- The agent, daily digest, ESPN fixtures and MLB probables jobs never run on AWS: Terraform declares no fixed-schedule EventBridge rules for them (`deployment/aws/terraform/eventbridge.tf`), and their local cron table is `packages/odds-lambda/odds_lambda/scheduling/jobs.py::_JOB_CRON_MAP`.
- `agent-run` and `daily-digest` are also opt-in locally: both are commented out of the default `packages/odds-core/odds_core/config.py::SchedulerConfig` `bootstrap_jobs`.
- The remaining fetch jobs may run on AWS, locally, or both.

Design and per-job table: [Jobs: registry, context and where each runs](../scheduling/jobs-and-deployment-matrix.md#where-each-job-runs), [Local scheduler](../scheduling/local-scheduler.md).

## Repository layout

| Path | Contents |
|---|---|
| `packages/` | Five uv workspace members, see [Packages and the odds CLI](packages-and-cli.md) |
| `docs/` | Human docs, see [Conventions](conventions.md#documentation-authority) |
| `experiments/` | `configs/`, `scripts/`, `results/` for modelling runs |
| `scripts/` | Operational scripts (ingestion, tier checks, `ci/`, `wiki_lint.py`) |
| `migrations/`, `alembic.ini` | Alembic, see [migrations](howto-database-migration.md) |
| `tests/` | `unit/`, `integration/`, `migration/`, see [local development](howto-develop-locally.md#run-the-tests) |
| `deployment/aws/` | Terraform, Dockerfiles, bootstrap, see [AWS infrastructure](../operations/aws-infrastructure.md) |
| `.github/workflows/` | [CI/CD](../operations/ci-cd-workflows.md) |
| `wiki/` | This wiki |

`.claude/`, `worktrees/` and `logs/` are gitignored.

## Stack

Python 3.12+, SQLModel on PostgreSQL 15 with asyncpg, APScheduler 4 (local backend), Typer and Rich (CLI), FastMCP, tenacity, structlog, Playwright, XGBoost, uv workspace ([ARCHITECTURE](../../docs/ARCHITECTURE.md#technical-stack)). Conventions: [Conventions](conventions.md), [Configuration](configuration.md).

## The three pivots

The repo carries the residue of three pivots ([audit T1](../../docs/audits/comprehension-audit-2026-06-09.md#t1-stratified-doc-rot-from-three-pivots)). Knowing which era a module comes from tells you whether its defaults match the active sports.

1. **NBA to EPL and MLB.** The first experiments are NBA (February 2026); EPL runs start in March ([MODELING.md](../../docs/MODELING.md#experiment-log)). NBA code remains (injuries, game logs, Polymarket targeting).
2. **Model edge to information edge**, recorded as 2026-04-12 in the audit. The agent's synthesis of many sources is the hypothesised edge ([BETTING_AGENT.md](../../docs/BETTING_AGENT.md)). The NBA injury pipeline's own header says it collects data the deployed model does not use ([INJURIES.md](../../docs/INJURIES.md)); see [NBA injury reports, game logs and player stats](../collection/nba-injuries-and-stats.md).
3. **Structural consolidations**: alerts moved from odds-cli to odds-core, self-scheduling unified into one decision module, scoring inlined into the OddsPortal job. See [Alerts, heartbeats and the health monitor](../operations/alerts-and-health.md), [Self-scheduling](../scheduling/self-scheduling.md), [Publishing, loading and scoring the CLV model](../modelling/model-publishing-and-scoring.md).

## NBA assumptions still in live code

Each fails quietly on EPL and MLB data:

- `fetch current`, `fetch scores`, `discover` and `quality` default `--sport` to `basketball_nba` ([odds CLI](packages-and-cli.md#hazards)).
- `backfill plan` and `backfill scores` hardcode `sport_key="basketball_nba"` (`packages/odds-cli/odds_cli/commands/backfill.py`), as does `BackfillExecutor` (`packages/odds-analytics/odds_analytics/backfill_executor.py`). See [Load historical odds and scores](../collection/howto-load-historical-odds.md).
- `TierCoverageValidator` checks the API fixture list with `sport="basketball_nba"` only (`packages/odds-lambda/odds_lambda/storage/tier_validator.py`); `OddsReader.get_games_by_date` buckets days noon-to-noon UTC "to capture a full NBA game day", and the validator uses it for every sport ([Odds validation and coverage checks](../storage/data-validation.md), [OddsReader](../storage/odds-reader-and-storage-layer.md)).
- `BasicEVStrategy` defaults to markets `h2h`, `spreads`, `totals` and US retail books, which match nothing in OddsPortal EPL data (`1x2`, UK books) ([Backtest strategies](../evaluation/backtest-strategies.md)).
- Polymarket code targets NBA (`PolymarketClient.get_nba_events`, `PolymarketConfig.nba_series_id`), and `PolymarketConfig.enabled` is a separate gate ([Polymarket](../collection/polymarket.md)).
- `.env.example` sets `SPORTS=basketball_nba` and writes list values as comma-separated text, which fails JSON parsing ([Configuration](configuration.md#hazards)).

## Hazards

- The mismatches above surface as empty results (no events, zero opportunities), not errors, so an NBA default and an empty EPL window look alike.
- Root `pyproject.toml` still describes the project as "NBA betting odds data collection and analysis system", and the CLI banner reads "NBA odds collection and analysis" (`packages/odds-cli/odds_cli/main.py::app`). Neither reflects the active sports.
- Verify any doc claim about defaults against code ([Conventions](conventions.md#documentation-authority)).

## Related

- [Packages and the odds CLI](packages-and-cli.md)
- [Betting agent](../agent/betting-agent.md)
- [Reporting surfaces and the daily digest](../operations/reporting-and-daily-digest.md) — how the system reports on itself
