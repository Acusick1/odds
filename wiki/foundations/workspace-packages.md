---
type: reference
title: Workspace packages and repository layout
description: The five uv workspace packages, their modules and declared dependencies, the repo layout and stack.
sources:
  - odds_core
  - odds_core.database
  - odds_core.odds_math
  - odds_core.snapshot_utils.extract_odds_from_snapshot
  - odds_core.utils.raw_data_has_market
  - odds_core.paper_trade_models.PaperTrade
  - packages/odds-core/pyproject.toml
  - odds_lambda
  - odds_lambda.scheduling.jobs._JOB_MODULE_MAP
  - odds_lambda.paper_trading.place_trade
  - odds_lambda.copy_from_production.copy_from_prod
  - packages/odds-lambda/pyproject.toml
  - odds_analytics
  - odds_analytics.targets.calculate_regression_target
  - odds_analytics.quality_metrics
  - odds_analytics.candidate_report
  - odds_analytics.candidate_backtest
  - odds_analytics.candidate_parity
  - odds_analytics.utils.per_book_market_holds
  - packages/odds-analytics/pyproject.toml
  - packages/odds-cli/pyproject.toml
  - packages/odds-mcp/pyproject.toml
  - pyproject.toml
  - docker-compose.yml
mentions:
  - odds_core.models.Event
  - odds_core.alerts.job_alert_context
status: stable
---

# Workspace packages and repository layout

The repo is a uv workspace of five packages under `packages/` ([AGENTS.md](../../AGENTS.md)): `odds-core` at the bottom, then `odds-lambda`, `odds-analytics`, and the two front ends `odds-mcp` and `odds-cli`. Dependencies point downward: odds-lambda depends on odds-core, odds-analytics on both, and odds-mcp and odds-cli on all three. The stated rule is that core stays light; the exception is under [Dependency exceptions](#dependency-exceptions).

## Repository layout

| Path | Contents |
|---|---|
| `packages/` | The five uv workspace members |
| `docs/adr/`, `docs/agents/` | Decision records and agent tooling conventions, see [Documentation and issue tracking](documentation-and-issue-tracking.md) |
| `experiments/` | `configs/`, `scripts/`, `results/` for modelling runs |
| `scripts/` | Operational scripts (ingestion, tier checks, `ci/`, `wiki_lint.py`) |
| `migrations/`, `alembic.ini` | Alembic, see [migrations](howto-database-migration.md) |
| `tests/` | `unit/`, `integration/`, `migration/`, see [Test suite](test-suite.md) |
| `deployment/aws/` | Terraform, Dockerfiles, bootstrap, see [AWS infrastructure](../operations/aws-infrastructure.md) |
| `.github/workflows/` | [CI/CD workflows](../operations/ci-cd-workflows.md) |
| `wiki/` | The wiki |

`.claude/`, `worktrees/` and `logs/` are gitignored. The root `pyproject.toml` description reads "NBA betting odds data collection and analysis system" (#465).

## Stack

Python 3.12+ (`pyproject.toml`), a uv workspace. Per package dependencies are in each `packages/<name>/pyproject.toml`:

| Concern | Library | Declared in |
|---|---|---|
| ORM and driver | SQLModel (odds-analytics declares it too, for `sqlmodel.col`), asyncpg, on PostgreSQL 15, run by Docker Compose (`docker-compose.yml`) | `packages/odds-core/pyproject.toml`, `packages/odds-analytics/pyproject.toml` |
| Settings and logging | Pydantic Settings, structlog | `packages/odds-core/pyproject.toml` |
| Local scheduler | APScheduler 4 (pinned alpha), tenacity retries | `packages/odds-lambda/pyproject.toml` |
| Migrations | Alembic | `pyproject.toml` |
| NBA game-log browser scraping | Playwright | `packages/odds-lambda/pyproject.toml` |
| CLI | Typer, Rich | `packages/odds-cli/pyproject.toml` |
| MCP server | FastMCP | `packages/odds-mcp/pyproject.toml` |
| Modelling | XGBoost | `packages/odds-analytics/pyproject.toml` |
| Tests | pytest, pytest-asyncio, py-pglite | `pyproject.toml` |

## odds-core

`packages/odds-core/odds_core/` declares only sqlmodel, sqlalchemy, asyncpg, pydantic, pydantic-settings, structlog, aiohttp and pyyaml (`packages/odds-core/pyproject.toml`).

| Module | Holds |
|---|---|
| `odds_core.models` | `odds_core.models.Event`, `odds_core.models.EventExternalId`, `odds_core.models.HealthCheckAcknowledgement`, `odds_core.models.OddsSnapshot`, `odds_core.models.SnapshotPayload`, `odds_core.models.Price`, `odds_core.models.Odds`, `odds_core.models.FetchLog`, `odds_core.models.DataQualityLog`, `odds_core.models.AlertHistory` and `odds_core.models.utc_now` (tables: [events](../storage/events.md), [odds tables](../storage/odds-tables.md#odds_snapshots), [prices](../storage/price-tables.md)); the closed sets their text columns take (source, market, time basis, market status, outcome, side, health check) are in `odds_core.vocabulary` |
| `odds_core.match_brief_models`, `odds_core.paper_trade_models`, `odds_core.agent_wakeup_models` | Agent tables ([match briefs](../agent/match-briefs.md), [paper trading](../agent/paper-trading.md)); `odds_core.paper_trade_models` also holds the candidate decision table, one row per candidate version, event and entry of the rule's decision times, with each decision's input snapshot ids |
| `odds_core.epl_data_models`, `odds_core.mlb_data_models`, `odds_core.injury_models`, `odds_core.game_log_models`, `odds_core.player_stats_models`, `odds_core.polymarket_models` | Sport-context and Polymarket tables (Polymarket price snapshots are unique per market and time; the MLB schedule table also feeds the MLB fixture calendar) |
| `config.py` | Pydantic Settings, see [Configuration](configuration.md) |
| `odds_core.database` | Engine and `odds_core.database.async_session_maker`, see [Databases and environments](databases-and-environments.md) |
| `odds_core.alerts` | `odds_core.alerts.AlertManager` (silent when `odds_core.config.Settings.alerts_deliverable` is false; `odds_core.alerts.record_to_alert_history` is likewise a no-op in dry run), `odds_core.alerts.DiscordAlert`, `odds_core.alerts.send_job_failure` (one alert per outage, reminding every `stale_reminder_minutes`), [`odds_core.alerts.job_alert_context`](../operations/alerts.md#delivery) (alerts on a job's exception, records a heartbeat on success) `odds_core.alerts.send_outage_warning` and `odds_core.alerts.send_scheduling_failure` (all through `odds_core.alerts.send_recorded_alert`), see [Alerts](../operations/alerts.md#delivery) |
| `odds_core.odds_math` | Price conversion in both directions (decimal-first), implied probability, devig, profit from a decimal or American price, see [Odds arithmetic helpers](odds-math.md) |
| `odds_core.betfair_close` | `odds_core.betfair_close.event_close` (the close of an event, as an `odds_core.betfair_close.ExchangePrice`) over `odds_core.betfair_close.LIVE_CLOSE_WINDOW`, `odds_core.betfair_close.is_in_play` and `odds_core.betfair_close.complete_exchange_prices`, with outcome names and per-book prices in `odds_core.market_prices`: the live Betfair close that the retail-EV builder, the candidate report and the [Betfair close check](../operations/health-monitor.md#betfair-close-check) share |
| `odds_core.as_of` | `odds_core.as_of.RETAIL_KINDS`, `odds_core.as_of.EXCHANGE_AT_T_KINDS` and `odds_core.as_of.MAX_EXCHANGE_AGE`, with `odds_core.as_of.latest_at_or_before`: which stored prices count as known at a decision time, shared by the retail-EV builder and the candidate rules |
| `odds_core.snapshot_utils`, `utils.py` | `odds_core.snapshot_utils.extract_odds_from_snapshot` (parse `raw_data` into `odds_core.models.Odds` carrying the snapshot's id), `odds_core.utils.raw_data_has_market` |
| `odds_core.sports`, `odds_core.team`, `odds_core.time`, `odds_core.logging_setup` | [Sport keys](../collection/sport-and-market-keys.md), [team-name normalization](../storage/event-matching.md), [timezone helpers](timezones.md), structlog setup |
| `odds_core.api_models` | The Odds API response models and event factories |

`odds_core.snapshot_utils.extract_odds_from_snapshot` matches market keys exactly, so `h2h` and `1x2` snapshots are different markets to every reader that uses it.

Importing any `odds_core` submodule runs `odds_core.database`, which builds the engine from `odds_core.config.get_settings()`. Every `odds` command therefore needs `ODDS_API_KEY` and `DATABASE_URL` to resolve, even commands that never touch the API or database ([Engine creation](databases-and-environments.md#engine-creation), [Configuration](configuration.md)).

## odds-lambda

`packages/odds-lambda/odds_lambda/` holds everything that collects, stores and schedules data. The name is historical: the same code runs under the local scheduler.

| Area | Modules | Page |
|---|---|---|
| Odds API | `odds_lambda.data_fetcher` (also relabels soccer three-way `h2h` to `1x2` and renames bookmaker keys to pipeline keys through `odds_core.bookmakers.ODDS_API_BOOKMAKER_KEY_MAP`), `odds_lambda.api_key_manager` (key rotation), `odds_lambda.event_sync` (event discovery and twin merge) | [Odds API client](../collection/odds-api-client.md) |
| Ingestion (decimal price validation, `fetch_logs` rows) | `odds_lambda.ingestion`, `odds_lambda.storage.validators` | [Odds API ingestion service](../collection/odds-ingestion-service.md) |
| OddsPortal scrapers (decimal prices) and historical record helpers (bookmaker keys, tiers, match-link ids) | `odds_lambda.oddsportal_common`, `odds_lambda.oddsportal_adapter`, `odds_lambda.oddsportal_client` (listings, odds feed and the results archive with its finished and cancelled matches; retries transient HTTP and proxy-wrapped upstream errors) | [OddsPortal scraper](../collection/oddsportal-scraper.md) |
| Exchange | `betfair/` | [Betfair Exchange](../collection/betfair-exchange.md) |
| Sport context (ESPN team-schedule fixtures, monthly upcoming scoreboards and the postponed status, MLB, NBA), Polymarket fetchers (match prices stored as decimal reciprocals of midpoints strictly inside (0, 1)) and attaching matches to events without creating them | `odds_lambda.espn_fixture_fetcher`, `odds_lambda.mlb_stats_fetcher`, `odds_lambda.mlb_game_resolver`, `odds_lambda.injury_fetcher`, `odds_lambda.game_log_fetcher`, `odds_lambda.pbpstats_fetcher`, `odds_lambda.polymarket_fetcher`, `odds_lambda.polymarket_odds` | [EPL](../collection/epl-context-data.md), [MLB](../collection/mlb-probable-pitchers.md) (probables, game scores with game numbers, and the `gamePk` resolution and played-slot reading the live jobs, the `gamePk` backfill and MLB settlement use, in [event matching](../storage/event-matching.md#mlb-gamepk-resolver)), [NBA](../collection/nba-injuries-and-stats.md), [Polymarket](../collection/polymarket.md) |
| Storage | `storage/` (one reader/writer pair per source), `odds_lambda.live_events` (event resolution and listing writes for the live jobs and, through `odds_lambda.odds_api_events`, the Odds API writers, MLB by `gamePk`, and the calendar key of a game a job saw but did not price), `odds_lambda.price_rows` (a snapshot envelope read as market snapshots and price rows) | [Storage layer](../storage/storage-layer.md), [live MLB events](../storage/event-matching.md#live-mlb-events) |
| Scheduling | `scheduling/` (`odds_lambda.scheduling.jobs`, the job registry and the tags smoke reads (agent, candidate and paid-API), `odds_lambda.scheduling.decision` (cadence, dormancy and lead runs), `odds_lambda.scheduling.helpers`, `odds_lambda.scheduling.backends`, `odds_lambda.scheduling.bootstrap`, `odds_lambda.scheduling.runner`, the local run wrapper with its timeout and chain re-book, warning once per outage), `odds_lambda.lambda_handler`, `odds_lambda.fetch_tier`, `odds_lambda.tier_utils` | [Job registry](../scheduling/job-registry.md), [Self-scheduling](../scheduling/self-scheduling.md), [Fetch tier](../storage/fetch-tier.md) |
| Jobs | `jobs/*.py`, one `main(ctx)` per file; a per-sport job raises without `ctx.sport`; `odds_lambda.results_run` holds what the two results jobs share | `odds_lambda.scheduling.jobs._JOB_MODULE_MAP` lists all 15; [results jobs](../collection/results-jobs.md#run-alerts-and-booking) |
| Candidate rules | `odds_lambda.candidate_rules` (the market state at a decision time and the one snapshot it was built from, mechanical betting rules, decision times and leads, the once-per-outcome filter, params validation and rule hashing) | [Candidate rules](../evaluation/candidate-rules.md) |
| Services (including the check-health monitor) | `odds_lambda.paper_trading` (placement with price, market, selection and candidate checks (registry entry, version, sport and market) and an optional bankroll, settlement and voiding), `odds_lambda.health_monitor` (the hourly health checks and metrics: stale sources (skipped for a job already failing), Betfair closes, quota, data quality, heartbeats, event identity (including MLB events without a `gamePk` not marked as having none), unsettled EPL and MLB events and schedule chains; silent under `SCHEDULER_DRY_RUN`), `odds_lambda.collection_calendar` (fixture and event calendars, dormancy decision with an optional kickoff lookup and lead runs that take the deciding run's start, and empty-collection guard (which leaves out fixtures a job accounted for by key)), `odds_lambda.copy_from_production` (events, snapshots and odds rows only, each odds row relinked to its snapshot's local id, each event keeping its local surrogate key) | [Paper trading](../agent/paper-trading.md), [health monitor](../operations/health-monitor.md), [copy production data](howto-copy-production-data.md) |

`odds_lambda` re-exports `odds_lambda.data_fetcher.TheOddsAPIClient` ([key handling](../collection/odds-api-client.md#retry-and-rotation)), `odds_lambda.ingestion.OddsIngestionService`, `odds_lambda.fetch_tier.FetchTier`, `odds_lambda.tier_utils.calculate_tier` and `odds_lambda.tier_utils.calculate_hours_until_commence`. Job modules are imported lazily by name from `odds_lambda.scheduling.jobs._JOB_MODULE_MAP`, so a scraper-only Lambda never imports the analytics stack. `odds_lambda.paper_trading` and `odds_lambda.copy_from_production` are service code for the CLI and MCP server rather than Lambda code; the copier is a Rich-console tool that opens a production session per event, and whose `odds_lambda.copy_from_production.copy_from_prod` returns an `odds_lambda.copy_from_production.CopyStats`, whose `events_failed` drives the CLI exit code.

## odds-analytics

`packages/odds-analytics/odds_analytics/` holds candidate scoring, replay and parity, feature extraction, model training and the backfill planner. It depends on core and odds-lambda (storage readers, `odds_lambda.fetch_tier.FetchTier`). Heavy dependencies (mlflow, scikit-learn, optuna) sit behind the `training` extra; `odds_analytics` lazy-loads the modules that need them via `odds_analytics.__getattr__`.

| Area | Modules | Page |
|---|---|---|
| Candidate evaluation | `odds_analytics.candidate_report` (scoring, the family size k at each checkpoint, from recorded decisions, and the checkpoint verdict), `odds_analytics.candidate_backtest` (replay of a rule over completed events at each decision time), `odds_analytics.candidate_parity` (replay of recorded decisions, in the order they were taken, from their recorded input snapshots) | [Candidate registry](../evaluation/candidate-registry.md#odds-candidates-report), [Replay backtest](../evaluation/candidate-backtest.md), [Parity check](../evaluation/candidate-parity.md) |
| Features | `odds_analytics.feature_extraction`, `odds_analytics.feature_groups` (snapshot samplers and the training-array orchestrator), `odds_analytics.retail_ev` (retail-EV event selection, among events that have kicked off, by close, decision rows, their horizon and season labels, loss weights (scaled by the fixed `odds_analytics.retail_ev.LOSS_WEIGHT_SCALE`), kick-off times, drop counts and feature matrix), `odds_analytics.retail_ev_metrics` (keep-bar and candidate-report bootstrap metrics, from rows or from per-event totals), `odds_analytics.retail_ev_evaluation` (the one-look retail-EV evaluation: spec, window-end and readiness guards, refit blocks, verdict, coverage, accidental-rerun check and manifest), `odds_analytics.as_of_team_features` (schedule and standings from the matches played by a retail-EV decision time, optionally only up to a given kick-off), `odds_analytics.standings_features`, `odds_analytics.epl_schedule_features`, `odds_analytics.schedule_features`, `odds_analytics.injury_features`, `odds_analytics.polymarket_features` (Polymarket order-book features from the home side's view) | [Feature pipeline](../modelling/feature-pipeline.md) |
| Training | `training/` (`odds_analytics.training.config`, `odds_analytics.training.data_preparation` (config-driven event selection by status and id-prefix source, and training-data preparation), `odds_analytics.training.cross_validation`, `odds_analytics.training.tuner` (Optuna wrapper with a seedable sampler; its objective takes sample weights), `odds_analytics.training.feature_selection` (variance filter), `odds_analytics.training.tracking`) | [Training pipeline](../modelling/training-pipeline.md) |
| Line-movement models | `odds_analytics.xgboost_line_movement` (accepts sample weights and a base-margin feature column; extracts no features) | [Line-movement model](../modelling/line-movement-models.md) |
| Backfill (the plan carries each game's Odds API id, null when none; the executor skips such a game) | `odds_analytics.gap_backfill_planner`, `odds_analytics.game_selector`, `odds_analytics.backfill_executor` | [Odds API historical backfill](../collection/odds-api-backfill.md) |
| Targets | `odds_analytics.targets` (the `raw` closing-line movement target function) | [Target functions](../modelling/clv-target-functions.md) |
| Coverage reporting | `odds_analytics.quality_metrics` | [Coverage checks](../storage/coverage-checks.md) |
| Helpers | `utils.py` (EV, holds, arbitrage), `odds_analytics.quality_metrics` | |

### Names that mislead

- `odds_analytics.quality_metrics` never touches `odds_core.models.DataQualityLog`; it counts game and odds coverage, with snapshots counted as distinct snapshot times. Data-quality rows are a separate system ([Odds validation](../storage/odds-validator.md), [Coverage checks](../storage/coverage-checks.md)).
- `odds_analytics.utils.per_book_market_holds` is the hold maths the MCP screener calls, over decimal prices ([retail edges](../agent/mcp-market-tools.md#retail-edges)); it is the only reason `odds-mcp` imports analytics ([odds-mcp server](../agent/mcp-server.md)).

## Dependency exceptions

- Every package declares each third-party library its code imports, even when a workspace dependency already brings it in; `deptry` checks this per package in CI ([Code conventions](code-conventions.md#dependency-checks)). The ignores that remain in each package's `[tool.deptry.per_rule_ignores]` are `odds_analytics` (odds-core and odds-lambda, the two undeclared imports above), `botocore` (odds-lambda; installed with boto3), `optuna` (odds-cli; behind analytics' `training` extra), `asyncpg` (odds-core; the driver named in `DATABASE_URL`) and `optuna-integration` (odds-analytics; loaded by Optuna as a plugin).

## Related

- [About the system and its pivots](project-overview.md)
- [The odds CLI](odds-cli.md)
- [Configuration](configuration.md)
