# Wiki

This repo is a single-user, sport-agnostic betting data pipeline (odds collection, storage, scheduling, alerting, paper trading) with per-sport LLM betting agents for EPL and MLB. The hypothesised edge is the agent's information synthesis; an XGBoost CLV model is a supplementary signal.

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

- **UTC everywhere.** Use `datetime.now(UTC)`; ruff catches only `utcnow()` and bare `now()`. See [timezones](foundations/conventions.md#timezones).
- **Never write to production.** No migrations, fetches or test runs against the production database from a laptop; `.env.example` points at the shared dev database, and `odds copy from-prod` falls back to `DATABASE_URL` as "production". See [databases](foundations/databases-and-environments.md#hazards).
- **The Odds API costs units.** 500 per free key per month; `/odds` costs 1 per region per market, historical 10. See [quota](collection/odds-api-client-and-quota.md).
- **Deployed state is not in the repo.** Which jobs run on AWS, and with what schedule, is only knowable via `odds scheduler list-jobs --backend aws`. See [jobs](scheduling/jobs-and-deployment-matrix.md).
- **Pass `--sport` and the right market key.** Several CLI defaults are `basketball_nba`; EPL odds are `1x2`, MLB `h2h`. See [keys](collection/sources-and-keys.md).
- **No look-ahead.** Historical analysis may use only data at or before decision time; `get_odds_at_time` is symmetric. See [look-ahead](modelling/decision-time-and-look-ahead.md).
- **Training configs must set the target.** Code defaults are `target_type: raw` and `target_bookmaker: pinnacle`, not the production bet365 devigged target. See [CLV target](modelling/clv-target.md).
