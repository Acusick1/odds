---
type: explanation
title: About the system and its pivots
description: What the system is, which sports are live, where the edge is meant to come from, what runs where in outline, the three pivots and the NBA-era defaults they left.
sources:
  - odds_core.sports.SportKey
  - odds_lambda.scheduling.jobs._JOB_CRON_MAP
  - odds_core.config.SchedulerConfig
  - odds_core.config.PolymarketConfig
  - odds_cli.main.app
  - odds_cli.commands.backfill
  - odds_analytics.backfill_executor.BackfillExecutor
  - odds_lambda.storage.tier_validator.TierCoverageValidator
  - odds_lambda.storage.readers.OddsReader
  - odds_analytics.strategies.BasicEVStrategy
  - odds_lambda.polymarket_fetcher.PolymarketClient
  - deployment/aws/terraform/eventbridge.tf
status: stable
---

# About the system and its pivots

A single-user, sport-agnostic betting data pipeline with sport-specific LLM betting agents ([README](../../README.md), [AGENTS.md](../../AGENTS.md)). Shared infrastructure (odds collection, storage, paper trading, scheduling, alerting) is reused per sport; each sport adds its own data sources, feature extractors, agent prompts and MCP tools. The repo layout and stack are in [Workspace packages and repository layout](workspace-packages.md).

Design principles `(inferred)`: robust collection and storage before analysis; a single-user system where simplicity beats scale; sports and markets as extension points; validation throughout the pipeline; all data preserved for backtesting.

## Sports

EPL and MLB are active, with agents in interactive evaluation, not autonomous production. NBA code (injuries, game logs, Polymarket targeting) exists but is deprioritized. `odds_core.sports.SportKey` allows all three. EPL odds are stored under market key `1x2` and MLB under `h2h`; see [Sport, market and bookmaker keys](../collection/sport-and-market-keys.md).

## Where the edge is meant to come from

Breadth and speed of information synthesis by an LLM agent, not the XGBoost CLV model ([ADR-0005](../../docs/adr/0005-information-edge.md)). The deployed model is `epl-clv-home` `v308058c` with a CV R² of about 0.058: too weak to bet on alone, so it is a supplementary signal the agent treats as a sanity check. See [The betting agent](../agent/betting-agent.md) and [What the CLV model can and cannot do](../evaluation/clv-model-findings.md).

## What runs where

Two AWS Lambdas (`odds-scheduler`, `odds-scheduler-scraper`) plus a local scheduler. The repo does not record the deployed state; the only check is `odds scheduler list-jobs --backend aws`.

- The agent, daily digest, ESPN fixtures and MLB probables jobs never run on AWS: Terraform declares no fixed-schedule EventBridge rules for them (`deployment/aws/terraform/eventbridge.tf`), and their local cron table is `odds_lambda.scheduling.jobs._JOB_CRON_MAP`.
- `agent-run` and `daily-digest` are also opt-in locally: both are commented out of the default `odds_core.config.SchedulerConfig` `bootstrap_jobs`.
- The remaining fetch jobs may run on AWS, locally, or both.

Per-job table: [Where each job runs](../scheduling/job-deployment-matrix.md#where-each-job-runs); local backend: [Local scheduler](../scheduling/local-scheduler.md).

## The three pivots

The repo carries the residue of three pivots (PRs `#214`, `#217`, `#267`, `#387`). Knowing which era a module comes from tells you whether its defaults match the active sports.

1. **NBA to EPL and MLB.** The first experiments are NBA (February 2026); EPL work starts in March (`experiments/results/exp1_feature_correlations/FINDINGS.md`; PRs `#214`, `#217`). NBA code remains (injuries, game logs, Polymarket targeting).
2. **Model edge to information edge**, in April 2026 (`#290`, [ADR-0005](../../docs/adr/0005-information-edge.md)). Injury features add no signal (`experiments/results/exp6b_injury_closing_tier/`); see [NBA injury reports, game logs and player stats](../collection/nba-injuries-and-stats.md).
3. **Structural consolidations**: alerts moved from odds-cli to odds-core (`#267`), self-scheduling unified into one decision module (`#387`), scoring inlined into the OddsPortal job. See [Alerts and the alert_history table](../operations/alerts.md), [Self-scheduling](../scheduling/self-scheduling.md), [Prediction scoring](../modelling/prediction-scoring.md).

## NBA-era defaults still in live code

Each fails quietly on EPL and MLB data: the symptom is an empty result (no events, zero opportunities), not an error, so an NBA default and an empty EPL window look alike. The reference page for each states the default:

- CLI `--sport` defaults to `basketball_nba` in several commands ([Sport defaults](odds-cli.md#sport-defaults)); the root banner describes the tool as NBA (`odds_cli.main.app`).
- `backfill plan` and `backfill scores` hardcode `sport_key="basketball_nba"` (`odds_cli.commands.backfill`), as does `odds_analytics.backfill_executor.BackfillExecutor` ([Odds API historical backfill](../collection/odds-api-backfill.md)).
- Completeness checks discover NBA games only, and `odds_lambda.storage.readers.OddsReader.get_games_by_date` buckets days noon-to-noon UTC "to capture a full NBA game day" for every sport (`odds_lambda.storage.tier_validator.TierCoverageValidator`; [Coverage and completeness checks](../storage/coverage-checks.md), [OddsReader](../storage/odds-reader.md)).
- `odds_analytics.strategies.BasicEVStrategy` defaults to NBA-style markets and US retail books, which match nothing in OddsPortal EPL data (`1x2`, UK books) ([Backtest strategies](../evaluation/backtest-strategies.md)).
- Polymarket code targets NBA (`odds_lambda.polymarket_fetcher.PolymarketClient.get_nba_events`, `odds_core.config.PolymarketConfig.nba_series_id`) ([Polymarket pipeline and tables](../collection/polymarket.md)).
- `.env.example` sets `SPORTS=basketball_nba` ([Configuration](configuration.md)).

## Related

- [Workspace packages and repository layout](workspace-packages.md)
- [The betting agent](../agent/betting-agent.md)
- [Reporting surfaces](../operations/reporting-surfaces.md) — how the system reports on itself
