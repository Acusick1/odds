---
type: explanation
title: About the system and its pivots
description: What the system is, which sports are live, the goal in outline, what runs where in outline, the three pivots and the NBA-era defaults they left.
sources:
  - odds_core.sports.SportKey
  - odds_core.config.SchedulerConfig.bootstrap_jobs
  - odds_core.config.PolymarketConfig
  - odds_cli.commands.backfill
  - odds_analytics.backfill_executor.BackfillExecutor
  - odds_lambda.polymarket_fetcher.PolymarketClient
status: stable
---

# About the system and its pivots

A single-user, sport-agnostic betting data pipeline with sport-specific LLM betting agents ([README](../../README.md), [AGENTS.md](../../AGENTS.md)). Shared infrastructure (odds collection, storage, paper trading, scheduling, alerting) is reused per sport; each sport adds its own data sources, feature extractors, agent prompts and MCP tools. The repo layout and stack are in [Workspace packages and repository layout](workspace-packages.md).

Design principles `(inferred)`: robust collection and storage before analysis; a single-user system where simplicity beats scale; sports and markets as extension points; validation throughout the pipeline; all data preserved for backtesting.

## Sports

EPL and MLB price collection is active. NBA OddsPortal and Betfair collection is configured and starts once the ESPN NBA schedule supplies fixtures (#725); the Odds API collects EPL (key 0) and NBA (key 1) ([keys](../collection/odds-api-client.md#retry-and-keys)). The agents run by hand on the production host, not autonomously, and are paused. NBA injury, game-log and player-stat code exists but runs only by manual command. `odds_core.sports.SportKey` allows all three. EPL odds are stored under market key `1x2` and MLB under `h2h`; see [Sport, market and bookmaker keys](../collection/sport-and-market-keys.md).

## Goal

Profit. Any approach can be a candidate (mechanical rules, models, agents), run in parallel and judged on live paper trades by EV against the Betfair Exchange close ([Goal and strategy](goal-and-strategy.md)). See [The betting agent](../agent/betting-agent.md).

## What runs where

One host, the production host, runs every scheduled job under its local scheduler and holds the production database ([ADR-0008](../../docs/adr/0008-mini-pc-is-production.md)). The default job set is `odds_core.config.SchedulerConfig.bootstrap_jobs`, with `agent-run` commented out; it includes the EPL Polymarket price job, the results jobs (OddsPortal for EPL, the MLB Stats API for MLB), the NBA injury-report job and the `run-candidates` job that places mechanical candidates' paper trades. The `odds-scheduler` Lambda and its Terraform remain but are dormant (a disabled `odds-scheduler-scraper` function may still exist in the account, with no Terraform), and the Neon database they wrote to is a frozen archive. The repo does not record the deployed AWS state; the only check is `odds scheduler list-jobs --backend aws`.

Per-job table: [Where each job runs](../scheduling/job-deployment-matrix.md#where-each-job-runs); local backend: [Local scheduler](../scheduling/local-scheduler.md).

## The three pivots

The repo carries the residue of three pivots (PRs `#214`, `#217`, `#267`, `#387`). Knowing which era a module comes from tells you whether its defaults match the active sports.

1. **NBA to EPL and MLB, then NBA prices back.** The first experiments are NBA (February 2026); EPL work starts in March (PRs `#214`, `#217`). NBA code remains (injuries, game logs, team-name normalization), and OddsPortal and Betfair NBA prices are collected again.
2. **Model edge to information edge**, in April 2026 (`#290`, [ADR-0005](../../docs/adr/0005-information-edge.md)).
3. **Structural consolidations**: alerts moved from odds-cli to odds-core (`#267`), self-scheduling unified into one decision module (`#387`). See [Alerts and the alert_history table](../operations/alerts.md), [Self-scheduling](../scheduling/self-scheduling.md).

## NBA-era defaults still in live code

Each fails quietly on EPL and MLB data: the symptom is an empty result (no events, zero opportunities), not an error, so an NBA default and an empty EPL window look alike. The reference page for each states the default:

- `backfill scores` hardcodes `sport_key="basketball_nba"` (`odds_cli.commands.backfill`) and spends no Odds API units. `odds_analytics.backfill_executor.BackfillExecutor` takes its sport from the acquisition manifest and stores every game a call returns on the event the game resolves to through the [event resolver](../storage/event-resolver.md), reporting a game the resolver drops ([Odds API historical backfill](../collection/odds-api-backfill.md)).
- `odds polymarket discover` lists NBA events (`odds_cli.commands.polymarket.discover`); EPL and NBA match prices (fetched from Polymarket's public Gamma and CLOB APIs with `odds_lambda.polymarket_fetcher.PolymarketClient`) are stored as `polymarket` snapshots ([Polymarket odds snapshots](../collection/polymarket-odds-snapshots.md)).
- Alert intervals and per-job heartbeat hours are `odds_core.config.AlertConfig` fields with no template variable ([Alerts](../operations/alerts.md#alert-history)).

## Related

- [Workspace packages and repository layout](workspace-packages.md)
- [The betting agent](../agent/betting-agent.md)
- [Reporting surfaces](../operations/reporting-surfaces.md) — how the system reports on itself
