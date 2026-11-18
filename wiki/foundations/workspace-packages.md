---
type: reference
title: Workspace packages and repository layout
description: The six uv workspace packages, their modules and declared dependencies, the repo layout and stack.
sources:
  - odds_core
  - odds_core.database
  - odds_core.odds_math
  - odds_core.quotes.quotes
  - odds_core.bet_models.Bet
  - packages/odds-core/pyproject.toml
  - odds_lambda
  - odds_core.job_specs.JOB_SPECS
  - odds_lambda.betting.place_bet
  - odds_lambda.copy_from_production.copy_from_prod
  - packages/odds-lambda/pyproject.toml
  - odds_analytics
  - odds_analytics.targets.calculate_regression_target
  - odds_analytics.quality_metrics
  - odds_analytics.candidate_report
  - odds_analytics.candidate_backtest
  - odds_analytics.candidate_model
  - odds_analytics.candidate_parity
  - odds_analytics.book_replays
  - odds_analytics.utils.per_book_market_holds
  - packages/odds-analytics/pyproject.toml
  - packages/odds-cli/pyproject.toml
  - packages/odds-mcp/pyproject.toml
  - packages/odds-web/pyproject.toml
  - odds_web
  - odds_web.app
  - pyproject.toml
  - docker-compose.yml
mentions:
  - odds_core.models.Event
  - odds_core.alerts.job_alert_context
status: stable
---

# Workspace packages and repository layout

The repo is a uv workspace of six packages under `packages/` ([AGENTS.md](../../AGENTS.md)): `odds-core` at the bottom, then `odds-lambda`, `odds-analytics`, and the three front ends `odds-mcp`, `odds-cli` and `odds-web`. Dependencies point downward: odds-lambda depends on odds-core, odds-analytics on both, odds-mcp and odds-web on all three, and odds-cli on those four. The stated rule is that core stays light; the exception is under [Dependency exceptions](#dependency-exceptions).

## Repository layout

| Path | Contents |
|---|---|
| `packages/` | The six uv workspace members |
| `docs/adr/`, `docs/agents/` | Decision records and agent tooling conventions, see [Documentation and issue tracking](documentation-and-issue-tracking.md) |
| `experiments/` | `configs/`, `scripts/`, `results/` for modelling runs |
| `scripts/` | Operational scripts (ingestion, tier checks, `ci/`, `wiki_lint.py`) |
| `migrations/`, `alembic.ini` | Alembic, see [migrations](howto-database-migration.md) |
| `tests/` | `unit/`, `integration/`, `migration/`, see [Test suite](test-suite.md) |
| `deployment/aws/` | Terraform, Dockerfiles, bootstrap, see [AWS infrastructure](../operations/aws-infrastructure.md) |
| `.github/workflows/` | [CI/CD workflows](../operations/ci-cd-workflows.md) |
| `wiki/` | The wiki |

`worktrees/`, `logs/` and `.claude/` are gitignored, except the betting agent's skill `.claude/skills/agent/` ([agent runtime](../agent/agent-run-job.md#runtime)).

## Stack

Python 3.12+ (`pyproject.toml`), a uv workspace linted by one root ruff configuration that selects the `DTZ` datetime rules. Per package dependencies are in each `packages/<name>/pyproject.toml`:

| Concern | Library | Declared in |
|---|---|---|
| ORM and driver | SQLModel (odds-analytics declares it too, for `sqlmodel.col`), asyncpg, on PostgreSQL 15, run by Docker Compose (`docker-compose.yml`) | `packages/odds-core/pyproject.toml`, `packages/odds-analytics/pyproject.toml` |
| Settings and logging | Pydantic Settings, structlog | `packages/odds-core/pyproject.toml` |
| Local scheduler | APScheduler 4 (pinned alpha), tenacity retries | `packages/odds-lambda/pyproject.toml` |
| Migrations | Alembic | `pyproject.toml` |
| NBA game-log browser scraping | Playwright | `packages/odds-lambda/pyproject.toml` |
| NBA injury-report PDFs | `nbainjuries` (tabula over JPype) and `jdk4py`, which bundles the JRE it needs | `packages/odds-lambda/pyproject.toml` |
| CLI | Typer, Rich | `packages/odds-cli/pyproject.toml` |
| MCP server | FastMCP | `packages/odds-mcp/pyproject.toml` |
| Web UI | FastAPI, Jinja2, uvicorn (declared by odds-cli, which serves it), htmx and Observable Plot as vendored files, server-rendered SVG for the portfolio and data-health charts, Plot charts for the candidate page's forward test and stored backtests, each mean EV chart with its dashed estimate line | `packages/odds-web/pyproject.toml`, `packages/odds-web/odds_web/static/charts.js`, `packages/odds-cli/pyproject.toml` |
| Modelling | XGBoost | `packages/odds-analytics/pyproject.toml` |
| Tests | pytest, pytest-asyncio, pytest-env, py-pglite ([Test suite](test-suite.md)) | `pyproject.toml` |

## odds-core

`packages/odds-core/odds_core/` declares only sqlmodel, sqlalchemy, asyncpg, pydantic, pydantic-settings, structlog, aiohttp and pyyaml (`packages/odds-core/pyproject.toml`).

| Module | Holds |
|---|---|
| `odds_core.models` | `odds_core.models.Event`, `odds_core.models.EventExternalId`, `odds_core.models.HealthCheckAcknowledgement`, `odds_core.models.OddsSnapshot`, `odds_core.models.SnapshotPayload`, `odds_core.models.Price`, `odds_core.models.FetchLog`, `odds_core.models.DataQualityLog`, `odds_core.models.AlertHistory` and `odds_core.models.utc_now` (`odds_core.models.EventPostponement` records the kick-off an event was postponed from; tables: [events](../storage/events.md), [snapshots](../storage/odds-tables.md#odds_snapshots), [prices](../storage/price-tables.md)); the closed sets their text columns take (source, market, time basis, market status, outcome, side, health check) are in `odds_core.vocabulary` |
| `odds_core.match_brief_models`, `odds_core.bet_models`, `odds_core.agent_wakeup_models` | Agent tables ([briefs](../agent/match-briefs.md), [bets](../agent/bets.md)), the sharp-price types the MCP tools return, the bets (`odds_core.bet_models.Bet`) and the candidate decision table ([decision record](../evaluation/run-candidates-job.md#the-decision-record)) |
| `odds_core.backfill_models` | `odds_core.backfill_models.OddsApiSnapshotGap`, a planned historical Odds API snapshot the provider lacks ([provider gaps](../collection/odds-api-backfill.md#provider-gaps)) |
| `odds_core.epl_data_models`, `odds_core.mlb_data_models`, `odds_core.injury_models`, `odds_core.game_log_models`, `odds_core.player_stats_models` | Sport-context tables (the MLB schedule table also feeds the MLB event calendar) |
| `config.py` | Pydantic Settings, see [Configuration](configuration.md) |
| `odds_core.database` | `odds_core.database.async_session_maker` and `odds_core.database.get_engine`, built lazily from settings, see [Databases and environments](databases-and-environments.md) |
| `odds_core.alerts` | `odds_core.alerts.AlertManager`, `odds_core.alerts.DiscordAlert`, [`odds_core.alerts.job_alert_context`](../operations/alerts.md#delivery) and the other alert senders, and `odds_core.alerts.ping_dead_mans_switch`, see [Alerts](../operations/alerts.md#delivery) |
| `odds_core.odds_math` | Price conversion in both directions (decimal-first), the decimal-price validity rule, implied probability, devig, grading a selection on the final score, profit from a decimal or American price, see [Odds arithmetic helpers](odds-math.md) |
| `odds_core.quotes` | `odds_core.quotes.quotes`, the shared reader of price rows with their snapshot's fields ([reading prices](../storage/price-tables.md#reading-prices)) |
| `odds_core.exchange_price` | `odds_core.exchange_price.exchange_price`: a snapshot's devigged Betfair Exchange price, an `odds_core.exchange_price.ExchangePrice` (its probabilities a read-only mapping), with `odds_core.exchange_price.MAX_EXCHANGE_OVERROUND` |
| `odds_core.book_price` | `odds_core.book_price.book_price`, the devigged price of one book's side in a snapshot, and `odds_core.book_price.polymarket_price`, the one Polymarket pricer ([Polymarket snapshots](../collection/polymarket-odds-snapshots.md#snapshot-shape)) |
| `odds_core.close` | The [close](../modelling/clv-target.md#the-retail-ev-target) of an event (`odds_core.close.event_close`, per (league, market) in `odds_core.close.CLOSES`), a backtest's close with its [substitute](../evaluation/candidate-backtest.md#replay), and `odds_core.close.close_status`, shared by the retail-EV builder, the candidate report and the [Betfair close check](../operations/health-monitor.md#betfair-close-check). The outcomes each market prices are in `odds_core.market_prices` |
| `odds_core.rehash_ledger` | The rehash ledger's row models and reader (`odds_core.rehash_ledger.Rehash`), importing only pydantic so the version hook can load them ([rehash ledger](../evaluation/candidate-registry.md#rehash-ledger)) |
| `odds_core.as_of` | `odds_core.as_of.latest_snapshot`: which stored prices count as known at a decision time ([candidate rules](../evaluation/candidate-rules.md#market-state)) |
| `utils.py` | `odds_core.utils.relabel_three_way_markets` |
| `odds_core.sports`, `odds_core.team`, `odds_core.time`, `odds_core.logging_setup` | [Sport keys](../collection/sport-and-market-keys.md) and each league's timezone, which sets its game day ([coverage checks](../storage/coverage-checks.md#tiercoveragevalidator)), [team-name normalization](../storage/event-matching.md) (the football leagues, MLB and NBA, one alias map), [timezone helpers](timezones.md) (including `odds_core.time.parse_date`), structlog setup |
| `odds_core.api_models` | The Odds API response models and event factories ([Odds API client](../collection/odds-api-client.md)) |

Importing `odds_core.alerts` builds `odds_core.config.get_settings()` ([Read at import](configuration.md#read-at-import)), and the CLI, the jobs and the MCP server all import it. Every `odds` command therefore needs `ODDS_API_KEY` and `DATABASE_URL` to resolve, even commands that never touch the API or database ([Configuration](configuration.md)).

## odds-lambda

`packages/odds-lambda/odds_lambda/` holds everything that collects, stores and schedules data. The name is historical: the same code runs under the local scheduler.

| Area | Modules | Page |
|---|---|---|
| Odds API | `odds_lambda.data_fetcher` (the API client), `odds_lambda.event_sync` (event discovery, resolved through `odds_lambda.odds_api_events`) | [Odds API client](../collection/odds-api-client.md) |
| Ingestion | `odds_lambda.ingestion`, `odds_lambda.storage.validators` | [Odds API ingestion service](../collection/odds-ingestion-service.md) |
| OddsPortal | `odds_lambda.oddsportal_common`, `odds_lambda.oddsportal_adapter`, `odds_lambda.oddsportal_client` | [OddsPortal scraper](../collection/oddsportal-scraper.md) |
| Exchange | `betfair/` | [Betfair Exchange](../collection/betfair-exchange.md) |
| Sport context, schedules and Polymarket | `odds_lambda.espn_fixture_fetcher`, `odds_lambda.mlb_stats_fetcher`, `odds_lambda.mlb_game_resolver`, `odds_lambda.mlb_season_load`, `odds_lambda.injury_fetcher`, `odds_lambda.game_log_fetcher`, `odds_lambda.pbpstats_fetcher`, `odds_lambda.polymarket_fetcher`, `odds_lambda.polymarket_odds`, `odds_lambda.espn_nba_schedule`, `odds_lambda.espn_football_schedule` | [EPL](../collection/epl-context-data.md), [MLB](../collection/mlb-probable-pitchers.md), [MLB gamePk resolver](../storage/event-matching.md#mlb-gamepk-resolver), [MLB season load](../collection/mlb-season-load.md), [NBA](../collection/nba-injuries-and-stats.md), [ESPN football schedule](../collection/espn-football-schedule.md), [Polymarket](../collection/polymarket-odds-snapshots.md) |
| Storage | `storage/` (one reader/writer pair per source), `odds_lambda.live_events` (event resolution for the live jobs and the Odds API writers), `odds_lambda.price_rows` (an envelope read as market snapshots and price rows) | [Storage layer](../storage/storage-layer.md), [live MLB events](../storage/event-matching.md#live-mlb-events) |
| Scheduling | `scheduling/` (`odds_lambda.scheduling.jobs`, `odds_lambda.scheduling.decision`, `odds_lambda.scheduling.helpers`, `odds_lambda.scheduling.backends`, `odds_lambda.scheduling.bootstrap`, `odds_lambda.scheduling.runner`), `odds_lambda.lambda_handler`, `odds_lambda.fetch_tier`, `odds_lambda.tier_utils` | [Job registry](../scheduling/job-registry.md), [Self-scheduling](../scheduling/self-scheduling.md), [Decision engine](../scheduling/decision-engine.md), [Fetch tier](../storage/fetch-tier.md) |
| Jobs | `jobs/*.py`, one `main(ctx)` per file; `odds_lambda.results_run` holds what the two results jobs share | `odds_core.job_specs.JOB_SPECS` lists all 15; [results jobs](../collection/results-jobs.md#run-alerts-and-booking) |
| Candidate rules | `odds_lambda.candidate_rules`: the market state and its query, the fixture context, the rule framework (`odds_lambda.candidate_rules.rules`), one module per rule a rule candidate runs, the rule registry (`odds_lambda.candidate_rules.registry`) and rule pinning (`odds_lambda.candidate_rules.pinning`, over `odds_lambda.candidate_rules.code_hash`) | [Candidate rules](../evaluation/candidate-rules.md) |
| Services (including the check-health monitor) | `odds_lambda.betting` (bet placement through `odds_lambda.betting.place_bet`, settlement and voiding), `odds_lambda.health_readers`, `odds_lambda.health_monitor`, `odds_lambda.collection_calendar` (collection horizons, calendars, dormancy and the empty-collection guard), `odds_lambda.copy_from_production` | [Bets](../agent/bets.md), [health monitor](../operations/health-monitor.md), [empty collection](../operations/alerting-and-heartbeats.md#empty-collection), [copy production data](howto-copy-production-data.md) |

`odds_lambda` re-exports `odds_lambda.data_fetcher.TheOddsAPIClient` ([key handling](../collection/odds-api-client.md#retry-and-keys)), `odds_lambda.ingestion.OddsIngestionService`, `odds_lambda.fetch_tier.FetchTier` and `odds_lambda.tier_utils.calculate_tier`. Job modules are imported lazily by name from `odds_core.job_specs.JOB_SPECS`, so a scraper-only Lambda never imports the analytics stack. `odds_lambda.betting` and `odds_lambda.copy_from_production` (`odds_lambda.copy_from_production.copy_from_prod`, run only through `odds copy from-prod`) are service code for the CLI and MCP server rather than Lambda code.

## odds-analytics

`packages/odds-analytics/odds_analytics/` holds candidate scoring, replay and parity, feature extraction, model training and the backfill planner and executor. It depends on core and odds-lambda (storage readers and writers, `odds_lambda.data_fetcher.TheOddsAPIClient`, `odds_lambda.fetch_tier.FetchTier` and `odds_lambda.tier_utils`). Heavy dependencies (mlflow, scikit-learn, optuna) sit behind the `training` extra; `odds_analytics.training` lazy-loads the modules that need them on first access, through static imports in its `__getattr__` so the rule hash's import closure can follow them ([rule pinning](../evaluation/candidate-rules.md#rule-pinning)).

| Area | Modules | Page |
|---|---|---|
| Candidate evaluation | `odds_analytics.candidate_report` (the report and verdict), `odds_analytics.backtest_store` (stored runs), `odds_analytics.candidate_backtest` (the replay backtest), `odds_analytics.candidate_model` (the retail-EV model rule's model), `odds_analytics.candidate_parity` (the parity check of replayed against recorded bets), `odds_analytics.estimate_backfill` (the decision-time estimates of recorded rule-candidate bets), `odds_analytics.book_replays`, `odds_analytics.candidate_rehash`, `odds_analytics.holdout` and `odds_analytics.holdout_run` (the sealed holdout), `odds_analytics.cell_selection` | [Candidate registry](../evaluation/candidate-registry.md#odds-candidates-report), [Replay backtest](../evaluation/candidate-backtest.md#sealed-holdout), [Cell selection](../evaluation/cell-selection.md), [Parity check](../evaluation/candidate-parity.md), [Book replays](../evaluation/book-replays.md) |
| Features | `odds_analytics.feature_extraction` (tabular features), `odds_analytics.feature_groups` (samplers and the training-array orchestrator), `odds_analytics.retail_ev` (retail-EV rows, weights and features), `odds_analytics.retail_ev_metrics` (keep-bar and candidate-report bootstrap metrics), `odds_analytics.as_of_team_features`, `odds_analytics.standings_features`, `odds_analytics.epl_schedule_features`, `odds_analytics.schedule_features`, `odds_analytics.injury_features` | [Feature pipeline](../modelling/feature-pipeline.md), [Target functions](../modelling/clv-target-functions.md) |
| Training | `training/` (`odds_analytics.training.config`, `odds_analytics.training.data_preparation`, `odds_analytics.training.cross_validation`, `odds_analytics.training.tuner`, `odds_analytics.training.feature_selection`, `odds_analytics.training.tracking`) | [Training pipeline](../modelling/training-pipeline.md) |
| Line-movement models | `odds_analytics.xgboost_line_movement` | [Line-movement model](../modelling/line-movement-models.md) |
| Backfill | `odds_analytics.backfill_plan`, `odds_analytics.backfill_checks`, `odds_analytics.backfill_executor` | [Odds API historical backfill](../collection/odds-api-backfill.md) |
| Targets | `odds_analytics.targets` (the `raw` target, `odds_analytics.targets.calculate_regression_target`) | [Target functions](../modelling/clv-target-functions.md) |
| Coverage reporting | `odds_analytics.quality_metrics` | [Coverage checks](../storage/coverage-checks.md) |
| Helpers | `utils.py` (market vigs and the per-tier coverage table), `odds_analytics.quality_metrics` | |

### Names that mislead

- `odds_analytics.quality_metrics` never touches `odds_core.models.DataQualityLog`; it counts event and odds coverage per fetch tier and per book, with snapshots counted as distinct snapshot times and per-book coverage read from price rows. Data-quality rows are a separate system ([Odds validation](../storage/odds-validator.md), [Coverage checks](../storage/coverage-checks.md)).
- `odds_analytics.utils.per_book_market_holds` is the vig maths the MCP screener calls, over decimal prices ([retail edges](../agent/mcp-market-tools.md#retail-edges)); MCP also imports `odds_analytics.candidate_report` for candidate summaries and `odds_analytics.standings_features` for EPL table ranking ([odds-mcp server](../agent/mcp-server.md)).

## odds-web

`packages/odds-web/odds_web/` is the read-only web UI ([odds-web](../operations/odds-web.md)): `odds_web.app` (the FastAPI app and its routes), `odds_web.db`, `odds_web.cache`, one module per page or section (`odds_web.portfolio`, `odds_web.candidate_page`, `odds_web.backtests`, `odds_web.ledger`, `odds_web.decisions`, `odds_web.status`, `odds_web.coverage`, `odds_web.closes`). It declares odds-core, odds-analytics, odds-lambda, FastAPI, Jinja2, sqlalchemy, sqlmodel and structlog (`packages/odds-web/pyproject.toml`) and has no entry point of its own: `odds web serve` in odds-cli runs it.

## Dependency exceptions

- Every package declares each third-party library its code imports, even when a workspace dependency already brings it in; `deptry` checks this per package in CI ([Code conventions](code-conventions.md#dependency-checks)). The ignores that remain in each package's `[tool.deptry.per_rule_ignores]` are `odds_analytics` (odds-core and odds-lambda, the two undeclared imports above), `botocore` (odds-lambda; installed with boto3), `optuna` (odds-cli; behind analytics' `training` extra), `asyncpg` (odds-core; the driver named in `DATABASE_URL`), `jinja2` and `odds-lambda` (odds-web; Starlette loads the first, and the second is declared ahead of the pages that read job and quota state) and `optuna-integration` (odds-analytics; loaded by Optuna as a plugin).

## Related

- [About the system and its pivots](project-overview.md)
- [The odds CLI](odds-cli.md)
- [Configuration](configuration.md)
