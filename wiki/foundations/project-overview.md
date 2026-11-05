---
type: explanation
title: About the system and its pivots
description: What the system is, which sports are live, the goal in outline, what runs where in outline, the three pivots and the NBA-era defaults they left.
sources:
  - odds_core.sports.SportKey
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

`odds_core.sports.SportKey` allows the EPL, MLB and NBA, and five more football leagues (the Championship, La Liga, Serie A, the Bundesliga and Ligue 1) registered for historical loading only. The [deployment matrix](../scheduling/job-deployment-matrix.md#where-each-job-runs) is authoritative for current collection and agent operation. NBA injury reports are scheduled; game logs and player stats use manual commands ([NBA collection](../collection/nba-injuries-and-stats.md#injury-reports)). Sport and market keys are listed in [Sport, market and bookmaker keys](../collection/sport-and-market-keys.md#market-key-dialects).

## Goal

Profit. Any approach can be a candidate (mechanical rules, models, agents), run in parallel and judged on live paper trades by EV against the Betfair Exchange close ([Goal and strategy](goal-and-strategy.md)). See [The betting agent](../agent/betting-agent.md).

## What runs where

The production host runs the local scheduler and production database ([ADR-0008](../../docs/adr/0008-mini-pc-is-production.md)). The [deployment matrix](../scheduling/job-deployment-matrix.md#where-each-job-runs) owns the operational job list and [dormant AWS rules](../scheduling/job-deployment-matrix.md#dormant-aws-rules); the [local scheduler](../scheduling/local-scheduler.md#scheduling-primitives) describes execution.

## The three pivots

The repo carries the residue of three pivots (PRs `#214`, `#217`, `#267`, `#387`). Knowing which era a module comes from tells you whether its defaults match the active sports.

1. **NBA to EPL and MLB, then NBA prices back.** The first experiments are NBA (February 2026); EPL work starts in March (PRs `#214`, `#217`). NBA code remains (injuries, game logs, team-name normalization), and OddsPortal and Betfair NBA prices are collected again.
2. **Model edge to information edge**, in April 2026 (`#290`, [ADR-0005](../../docs/adr/0005-information-edge.md)).
3. **Structural consolidations**: alerts moved from odds-cli to odds-core (`#267`), self-scheduling unified into one decision module (`#387`). See [Alerts and the alert_history table](../operations/alerts.md), [Self-scheduling](../scheduling/self-scheduling.md).

## NBA-era defaults still in live code

Each fails quietly on EPL and MLB data: the symptom is an empty result (no events, zero opportunities), not an error, so an NBA default and an empty EPL window look alike. The reference page for each states the default:

- `backfill scores` hardcodes `sport_key="basketball_nba"` (`odds_cli.commands.backfill`) and spends no Odds API units. `odds_analytics.backfill_executor.BackfillExecutor` takes its sport from the acquisition manifest and stores every game a call returns on the stored event it attaches to through the [event resolver](../storage/event-resolver.md), creating none, and reports a game with no stored event ([Odds API historical backfill](../collection/odds-api-backfill.md)).
- `odds polymarket discover` lists NBA events (`odds_cli.commands.polymarket.discover`); EPL and NBA match prices (fetched from Polymarket's public Gamma and CLOB APIs with `odds_lambda.polymarket_fetcher.PolymarketClient`) are stored as `polymarket` snapshots ([Polymarket odds snapshots](../collection/polymarket-odds-snapshots.md)).

## Related

- [Workspace packages and repository layout](workspace-packages.md)
- [The betting agent](../agent/betting-agent.md)
- [Reporting surfaces](../operations/reporting-surfaces.md) — how the system reports on itself
