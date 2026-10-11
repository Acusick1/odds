# Architecture Reference

> Reference documentation for system architecture, scheduling, and data pipeline.

## System Overview

A single-user betting odds data collection and analysis system supporting multiple sports
(EPL football and MLB baseball active; NBA code exists but is deprioritized). It collects
odds from sportsbooks, OddsPortal and Betfair Exchange, runs a CLV prediction model, and
serves the data to the betting agent through MCP tools. The package layout is in the
[README](../README.md#repository-layout).

**Design principles (inferred):**
- Data pipeline first: robust collection and storage come before analysis
- Single-user system: latency is not critical, so simplicity wins over scale
- Extensibility: sports and markets can be added
- Data quality: validation throughout the pipeline
- Historical analysis: all data is preserved for backtesting

## Data Sources

### The Odds API

- Base URL: `https://api.the-odds-api.com/v4`
- Sport-parameterized: `/sports/{sport_key}/odds`, `/sports/{sport_key}/scores`, etc.
- Supported sport keys: `soccer_epl`, `baseball_mlb`, `basketball_nba`

See [API Cost](../AGENTS.md#api-cost) for the endpoint pricing breakdown and
[DEPLOYMENT.md](DEPLOYMENT.md#the-odds-api) for tiers.

### OddsPortal (via OddsHarvester scraper)

Headless browser scraping of OddsPortal.com. Provides historical and live odds from UK
bookmakers not available through The Odds API.

- **Scraper**: [OddsHarvester](https://github.com/Acusick1/OddsHarvester) fork, Playwright + Chromium
- **Jobs**: `fetch-oddsportal` (upcoming match odds) and `fetch-oddsportal-results`
  (results + closing odds); see [OddsPortal jobs](#oddsportal-scrape-job)
- **Leagues**: `LEAGUE_SPECS` in `packages/odds-lambda/odds_lambda/jobs/fetch_oddsportal.py`
  (EPL and MLB)
- **Shared utilities**: `packages/odds-lambda/odds_lambda/oddsportal_common.py` (bookmaker
  mapping, odds conversion, tier classification)

### Betfair Exchange

`packages/odds-lambda/odds_lambda/betfair/`, polled by the `fetch-betfair-exchange` job.
Stores best back/lay Match Odds as `betfair_exchange` snapshots.

### Bookmaker Coverage

**US bookmakers** (via Odds API):

| Bookmaker | Type | API Key |
|-----------|------|---------|
| Pinnacle | Sharp, low-margin | pinnacle |
| Circa Sports | Sharp, Vegas-based | circasports |
| DraftKings | Major retail | draftkings |
| FanDuel | Major retail | fanduel |
| BetMGM | Major retail | betmgm |
| Caesars | Major retail | williamhill_us |
| BetRivers | Regional retail | betrivers |
| Bovada | Offshore | bovada |

**UK bookmakers** (via OddsPortal):

| Bookmaker | Type | API Key |
|-----------|------|---------|
| bet365 | Primary target | bet365 |
| Betway | Retail | betway |
| Betfred | Retail | betfred |
| bwin | Retail | bwin |

### Markets Collected

- **h2h**: Moneyline (2-way for NBA and MLB, 3-way for football)
- **spreads**: Point spread / handicap
- **totals**: Over/under total points/goals

### Polymarket

Prediction market data via Gamma API (discovery) and CLOB API (prices/order books). The
pipeline is deprioritized; see [POLYMARKET.md](POLYMARKET.md).

## Multi-Backend Scheduler

The system supports multiple deployment backends through an abstraction layer.

### Backend Interface

All backends implement the `SchedulerBackend` ABC
(`packages/odds-lambda/odds_lambda/scheduling/backends/base.py`). The core methods are:

```python
class SchedulerBackend(ABC):
    async def schedule_next_execution(self, job_name, next_time, payload=None) -> None: ...
    async def cancel_scheduled_execution(self, job_name) -> None: ...
    async def get_scheduled_jobs(self) -> list[ScheduledJob]: ...
    async def get_job_status(self, job_name) -> ScheduledJob | None: ...
    async def health_check(self) -> BackendHealth: ...
    def validate_configuration(self) -> ValidationResult: ...
    def get_backend_name(self) -> str: ...
```

The self-scheduling pattern is built on `schedule_next_execution`: each invocation
schedules its own next wake-up rather than relying on a fixed external cron.

### AWS Lambda Backend

`packages/odds-lambda/odds_lambda/scheduling/backends/aws.py`

- Self-scheduling via EventBridge rules: each invocation updates its rule's schedule
- NullPool database connections (applied automatically inside Lambda)
- Configuration: `SCHEDULER_BACKEND=aws`; see
  [DEPLOYMENT.md](DEPLOYMENT.md#configuration) for the environment variables

### Railway Backend

`packages/odds-lambda/odds_lambda/scheduling/backends/railway.py`

- A no-op shim, not a deployment target. `schedule_next_execution` /
  `cancel_scheduled_execution` only log and return; `get_scheduled_jobs` /
  `get_job_status` raise `BackendUnavailableError`.
- Configuration: `SCHEDULER_BACKEND=railway`

### Local Backend

`packages/odds-lambda/odds_lambda/scheduling/backends/local.py`

- APScheduler 4 with a PostgreSQL data store
- Runs until the process is terminated (Ctrl+C)
- Configuration: `SCHEDULER_BACKEND=local` (the default)

### Job Registry

`packages/odds-lambda/odds_lambda/scheduling/jobs.py`

`_JOB_MODULE_MAP` maps job names to async entry points. Modules are lazy-imported on first
access to avoid pulling unused dependencies (for example, the scraper Lambda does not load
xgboost).

**`jobs.py` is the source of truth for the registered jobs;
[DEPLOYMENT.md](DEPLOYMENT.md#job-deployment-matrix) is authoritative for where each job
runs (AWS / local / both) and on what schedule.**

## Intelligent Scheduling System

`packages/odds-lambda/odds_lambda/scheduling/decision.py`

Game-aware scheduling adapts fetch frequency to game proximity. A single, sport-scoped
decision engine owns the mapping from proximity to polling cadence plus the run/skip gate.
`decide_forward` keys off the next upcoming game (`decide_forward_resilient` adds a
DB-failure fallback cadence so the self-scheduling chain survives a kickoff-query
failure); `decide_backward` keys off recent games for scores and status-update jobs.

Each job passes its own `CadenceConfig`, which sets an interval per fetch tier plus
`no_game` and `db_fallback` intervals. With no upcoming games, a job only wakes at its
`no_game` interval to check again.

### Fetch Tier System

`packages/odds-lambda/odds_lambda/fetch_tier.py`

| Tier | Time Before Game | `interval_hours` | Purpose |
|------|------------------|------------------|---------|
| Opening | 3+ days | 48 hours | Initial line release |
| Early | 1–3 days | 24 hours | Line establishment |
| Sharp | 12–24 hours | 12 hours | Professional betting |
| Pregame | 3–12 hours | 3 hours | Active betting |
| Closing | 0–3 hours | 30 minutes | Critical line movement |
| In-play | After kickoff | 30 minutes | Incidental collection |

The tier is stored in `OddsSnapshot.fetch_tier` for validation and ML features. The
intervals above are `FetchTier.interval_hours`, which `fetch-odds` uses; other jobs define
their own cadence per tier (for example `fetch-oddsportal` polls every 30 minutes to
2 hours).

## Scheduled Jobs

Where and when each job runs is in the
[Job Deployment Matrix](DEPLOYMENT.md#job-deployment-matrix).

### Event Sync (embedded in fetch-odds)

`packages/odds-lambda/odds_lambda/event_sync.py` — `EventSyncService`

- Runs as the first step of every `fetch-odds` invocation
- Syncs upcoming events from the free `/events` endpoint (0 quota cost)
- Returns `EventSyncResult` with inserted/updated counts per sport

### Fetch Odds Job

`packages/odds-lambda/odds_lambda/jobs/fetch_odds.py`

- Game-aware execution (only runs when games are upcoming)
- Fetches current odds for all scheduled games
- Stores hybrid data (raw JSONB + normalized records)
- Calculates next execution from the closest game's tier
- Logs API quota usage

### Fetch Scores Job

`packages/odds-lambda/odds_lambda/jobs/fetch_scores.py`

- Updates completed game results
- Marks events as FINAL with scores
- Self-schedules based on live game activity

### Update Status Job

`packages/odds-lambda/odds_lambda/jobs/update_status.py`

- Marks scheduled games that have started as LIVE (FINAL is set by the scores and results
  jobs)
- Prevents fetching pre-game odds for started games

### OddsPortal Scrape Job

`packages/odds-lambda/odds_lambda/jobs/fetch_oddsportal.py`

- Scrapes upcoming match odds from OddsPortal via OddsHarvester for each league in
  `LEAGUE_SPECS`
- Self-schedules by kickoff proximity: 30 minutes inside 3 hours, 1 hour at 3–12 hours,
  2 hours further out or with no games (`CADENCE`)
- Stores snapshots with tier classification based on time to kickoff
- Runs CLV scoring inline after ingest (see [Score Predictions](#score-predictions-inline))

### OddsPortal Results Job

`packages/odds-lambda/odds_lambda/jobs/fetch_oddsportal_results.py`

- Self-schedules its next run for ~08:00 UTC the next day, in a `finally` block so the
  chain survives a failed scrape
- Scrapes results and closing odds from OddsPortal for past SCHEDULED/LIVE events
- Updates event status to FINAL with scores

### Score Predictions (inline)

`packages/odds-lambda/odds_lambda/jobs/score_predictions.py`

- **Not a standalone scheduled job.** `score_events()` runs inline at the end of the
  `fetch-oddsportal` job, after ingest, when the model supports the sport.
  (`score-predictions-<sport>` can still be invoked manually.)
- Loads the CLV model from S3 with ETag-based caching (see
  [S3 model store](DEPLOYMENT.md#s3-model-store))
- Extracts features for upcoming SCHEDULED events with `XGBoostAdapter` from `odds-analytics`
- Stores predictions in the `Prediction` table with idempotency (unique constraint on
  event_id, snapshot_id, model_name)

### Betfair Exchange Job

`packages/odds-lambda/odds_lambda/jobs/fetch_betfair_exchange.py`

- Polls the Betfair Exchange API for upcoming events and stores best back/lay Match Odds as
  `betfair_exchange` snapshots
- Self-schedules by kickoff proximity: 15 minutes inside 3 hours, 30 minutes at 3–12 hours,
  2 hours further out or with no games (`CADENCE`)

### Fixture and Probables Jobs

- `packages/odds-lambda/odds_lambda/jobs/fetch_espn_fixtures.py`: refreshes current-season
  all-competition EPL fixtures from the ESPN Site API into `espn_fixtures` (feeds the
  `epl_schedule` features)
- `packages/odds-lambda/odds_lambda/jobs/fetch_mlb_probables.py`: snapshots MLB probable
  pitchers from the MLB Stats API into `mlb_probable_pitchers`

### Health Check Job

`packages/odds-lambda/odds_lambda/jobs/check_health.py` runs `HealthMonitor` checks and
alerts on breaches; see [REPORTING.md](REPORTING.md#3-health-monitoring--packagesodds-lambdaodds_lambdahealth_monitorpy).

### Agent Run Job

`packages/odds-lambda/odds_lambda/jobs/agent_run.py` runs one betting agent wake-up and
self-schedules by fixture proximity; see [BETTING_AGENT.md](BETTING_AGENT.md).

### Daily Digest Job

`packages/odds-lambda/odds_lambda/jobs/daily_digest.py` posts a Discord embed of recent
results and upcoming predictions; see
[REPORTING.md](REPORTING.md#4-daily-digest--packagesodds-lambdaodds_lambdajobsdaily_digestpy).

## Event Lifecycle

| Status | Description |
|--------|-------------|
| SCHEDULED | Odds actively being fetched |
| LIVE | Game in progress (stop fetching pre-game odds) |
| FINAL | Game completed with results stored |
| CANCELLED | No further data collection |
| POSTPONED | No further data collection |

## Data Collection Pipeline

```
Scheduler -> API Client -> Validator -> Writer -> Logger
                |              |           |         |
                v              v           v         v
           Retry/keys    Log issues    Raw+Norm   Quota
```

### API Client

`packages/odds-lambda/odds_lambda/data_fetcher.py` - `TheOddsAPIClient`

- Retry logic with tenacity (exponential backoff)
- Quota tracking with API key rotation on 401/429 (multiple keys via `ODDS_API_KEYS`,
  active index tracked in SSM Parameter Store)
- Methods: `get_odds()`, `get_events()`, `get_scores()`, `get_historical_odds()`,
  `get_historical_events()`

### Game Log Pipeline (NBA)

`packages/odds-lambda/odds_lambda/game_log_fetcher.py` - `fetch_game_logs()`

- Fetches NBA team game logs from stats.nba.com via Playwright (headless Firefox)
- Uses Playwright because raw HTTP requests are blocked by Akamai bot detection
- Calls the LeagueGameFinder API from the browser context after establishing session cookies
- Returns ~2,460 rows per season (30 teams × 82 games)
- Storage: `GameLogWriter` (upsert with event matching), `GameLogReader` (pipeline stats)
- CLI: see [CLI.md](CLI.md#game-logs)
- Score backfill (`odds backfill scores`) reads from the game log table

### Validator

`packages/odds-lambda/odds_lambda/storage/validators.py` - `OddsValidator`

**Validation checks:**
- Price within valid range (-10000 to +10000)
- Bookmaker `last_update` not in the future
- Vig between 2% and 15% (non-exchange bookmakers)
- No missing required fields
- Line movement not excessive (>10 point spread)
- Outcomes match expected format

**Behavior:** logs warnings by default and does not reject data (configurable via
`REJECT_INVALID_ODDS`). Issues are written to the `DataQualityLog` table with a severity
(warning, error, critical), an issue type and full context, for manual review.

### Storage Modules

`packages/odds-lambda/odds_lambda/storage/`

| Module | Purpose |
|--------|---------|
| `writers.py` | `OddsWriter` — hybrid raw + normalized odds storage |
| `readers.py` | `OddsReader` — time-aware queries, line movement, line shopping |
| `polymarket_writer.py` | Polymarket event/market/snapshot upserts |
| `polymarket_reader.py` | Polymarket active events, pipeline stats |
| `injury_writer.py` | Injury report upserts with auto event matching |
| `injury_reader.py` | Injury queries by event |
| `game_log_writer.py` | NBA game log upserts |
| `game_log_reader.py` | Game log queries |
| `pbpstats_writer.py` | PBPStats player season stats |
| `pbpstats_reader.py` | Player stats queries |
| `espn_fixture_writer.py` / `espn_fixture_reader.py` | ESPN all-competition fixtures |
| `espn_lineup_writer.py` / `espn_lineup_reader.py` | ESPN starting XIs |
| `fpl_availability_writer.py` / `fpl_availability_reader.py` | FPL player availability |
| `mlb_pitcher_writer.py` / `mlb_pitcher_reader.py` | MLB probable pitchers |
| `validators.py` | `OddsValidator` — data quality checks |
| `tier_validator.py` | Tier assignment validation |

## Discord Alerts

`packages/odds-core/odds_core/alerts.py` — `AlertManager` + `DiscordAlert` (webhook
delivery), configured via `DISCORD_WEBHOOK_URL`. See [REPORTING.md](REPORTING.md) for the
authoritative description of alerting surfaces and backing tables.

## Technical Stack

### Core Technologies

- Python 3.12+
- SQLModel (SQLAlchemy 2.0 + Pydantic)
- PostgreSQL 15+ with JSON/JSONB
- asyncio + aiohttp

### Key Libraries

- **uv**: Package management
- **APScheduler 4**: Job scheduling (local backend)
- **Typer + Rich**: CLI
- **Pydantic Settings**: Configuration
- **Alembic**: Database migrations
- **tenacity**: Retry logic
- **pytest + pytest-asyncio**: Testing
- **structlog**: Structured logging
- **Playwright**: Headless browser (OddsPortal scraping, NBA game logs)
- **XGBoost**: CLV prediction model
- **nba_api**: NBA.com API wrapper
