# Wiki

This repo is a single-user, sport-agnostic betting data pipeline (odds collection, storage, scheduling, alerting, paper trading) with per-sport LLM betting agents for EPL and MLB. The goal is profit; any approach (today the betting agents) is a candidate judged on live paper trades ([goal and strategy](foundations/goal-and-strategy.md)).

Read this page, then one area index, then the page, then the cited code. Code is authoritative. Page rules: [SCHEMA.md](SCHEMA.md).

## Areas

- [Foundations](foundations/index.md) — read when starting work, setting up, touching config, `.env`, databases or migrations, or checking a repo-wide convention.
- [Storage](storage/index.md) — read when reading or writing odds or events, matching a source to an event, or judging whether stored data is complete.
- [Data collection](collection/index.md) — read when changing how odds, results or sport context data is fetched, or loading history.
- [Scheduling](scheduling/index.md) — read when adding, changing, deploying or debugging a scheduled job, or asking where a job runs.
- [Operations](operations/index.md) — read when deploying, touching Terraform or workflows, or changing alerts, health checks or reports.
- [CLV modelling](modelling/index.md) — read when changing features, the target, training config or the deployed model and its scoring.
- [Backtests and model research](evaluation/index.md) — read when running a backtest or an experiment, or asking what the model can and cannot do.
- [Betting agent](agent/index.md) — read when changing the agent, its prompts or MCP tools, or its briefs and paper trades.

## Constraints

The hard constraints every change must respect are in [AGENTS.md](../AGENTS.md#constraints).
