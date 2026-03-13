# Architecture Reference

> Reference documentation for system architecture, scheduling, and data pipeline.

## System Overview

A single-user betting odds data collection and analysis system supporting multiple sports (NBA, EPL football). Collects odds from sportsbooks and OddsPortal, runs CLV prediction models, and delivers daily prediction digests via Discord. Prioritizes robust data pipeline architecture with comprehensive historical data collection, storage, and validation.

**Core Principles:**
- Data pipeline first - robust collection and storage is paramount
- Single-user system - latency not critical, simplicity over scale
- Extensibility - support additional sports and markets
- Data quality - validation throughout pipeline
- Historical analysis - all data preserved for backtesting

## Package Structure

```
packages/
├── odds-core/      # Models, config, database (odds_core/)
├── odds-lambda/    # Data fetching, storage, scheduling (odds_lambda/)
├── odds-analytics/ # Backtesting, strategies, ML (odds_analytics/)
└── odds-cli/       # CLI commands (odds_cli/)
```

## Data Sources

### The Odds API

- Base URL: `https://api.the-odds-api.com/v4`
- Sport-parameterized: `/sports/{sport_key}/odds`, `/sports/{sport_key}/scores`, etc.
- Supported sport keys: `basketball_nba`, `soccer_epl`

See CLAUDE.md for endpoint pricing breakdown and DEPLOYMENT.md for tier details.

### OddsPortal (via OddsHarvester scraper)

Headless browser scraping of OddsPortal.com via a separate scraper Lambda. Provides historical and live odds from UK bookmakers not available through The Odds API.

- **Scraper**: [OddsHarvester](https://github.com/Acusick1/OddsHarvester) fork, Playwright + Chromium
- **Jobs**: `fetch-oddsportal` (hourly, upcoming match odds), `fetch-oddsportal-results` (daily, results + closing odds)
- **Shared utilities**: `packages/odds-lambda/odds_lambda/oddsportal_common.py` (bookmaker mapping, odds conversion, tier classification)

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

- **h2h**: Moneyline (2-way for NBA, 3-way for football)
- **spreads**: Point spread / handicap
- **totals**: Over/under total points/goals

### Secondary API: NBA API

Library: `nba_api` (Python wrapper for NBA.com)

**Purpose:** Backfill historical scores when The Odds API incomplete

**Key Methods:**
- `get_live_scores()` - Current game scores
- `get_historical_scores(start_date, end_date)` - Historical results
- `match_game_by_teams_and_date()` - Fuzzy matching

### Polymarket (deprioritized)

Prediction market data via Gamma API (discovery) and CLOB API (prices/order books). Pipeline exists but is inactive — not accessible from UK, data likely collinear with sportsbook odds. See [POLYMARKET.md](POLYMARKET.md) for technical details.

## Multi-Backend Scheduler

The system supports multiple deployment backends through an abstraction layer.

### Backend Interface

All backends implement `SchedulerBackend` (`packages/odds-lambda/odds_lambda/scheduling/backends/base.py`):

```python
class SchedulerBackend(ABC):
    @abstractmethod
    async def schedule_job(self, job_name: str, execution_time: datetime) -> None: ...
    @abstractmethod
    def get_backend_info(self) -> dict: ...
```

### AWS Lambda Backend

`packages/odds-lambda/odds_lambda/scheduling/backends/aws.py`

- Self-scheduling pattern using EventBridge rules
- Each invocation schedules its next execution
- NullPool database connections (required)
- Configuration: `SCHEDULER_BACKEND=aws`, `AWS_LAMBDA_ARN`, `AWS_REGION`

### Railway Backend

`packages/odds-lambda/odds_lambda/scheduling/backends/railway.py`

- Uses APScheduler with persistent scheduling
- Long-running process with managed restarts
- Standard connection pooling
- Configuration: `SCHEDULER_BACKEND=railway`

### Local Backend

`packages/odds-lambda/odds_lambda/scheduling/backends/local.py`

- APScheduler for development and testing
- Runs until process terminated (Ctrl+C)
- Configuration: `SCHEDULER_BACKEND=local`

### Job Registry

`packages/odds-lambda/odds_lambda/scheduling/jobs.py`

Centralized mapping of job names to async entry points. Modules are lazy-imported on first access to avoid pulling unused dependencies (e.g. scraper Lambda doesn't load xgboost).

| Job | Module | Scheduling |
|-----|--------|------------|
| `fetch-odds` | `odds_lambda.jobs.fetch_odds` | Self-scheduling (tier-based) |
| `fetch-scores` | `odds_lambda.jobs.fetch_scores` | Self-scheduling |
| `update-status` | `odds_lambda.jobs.update_status` | Self-scheduling |
| `check-health` | `odds_lambda.jobs.check_health` | Self-scheduling |
| `fetch-polymarket` | `odds_lambda.jobs.fetch_polymarket` | Self-scheduling (deprioritized) |
| `backfill-polymarket` | `odds_lambda.jobs.backfill_polymarket` | Fixed: every 3 days (deprioritized) |
| `fetch-oddsportal` | `odds_lambda.jobs.fetch_oddsportal` | Fixed: hourly |
| `fetch-oddsportal-results` | `odds_lambda.jobs.fetch_oddsportal_results` | Fixed: 08:00 UTC daily |
| `score-predictions` | `odds_lambda.jobs.score_predictions` | Fixed: hourly at :15 |
| `daily-digest` | `odds_lambda.jobs.daily_digest` | Fixed: 08:00 UTC daily |

## Intelligent Scheduling System

`packages/odds-lambda/odds_lambda/scheduling/intelligence.py`

Game-aware scheduling that adapts fetch frequency based on game proximity.

**Key Features:**
- Automatically discovers upcoming games from API
- Adjusts frequency as games approach using tiered intervals
- No fetching during off-season
- Self-scheduling pattern for serverless deployment

### Fetch Tier System

`packages/odds-lambda/odds_lambda/fetch_tier.py`

| Tier | Time Before Game | Interval | Purpose |
|------|------------------|----------|---------|
| Opening | 3+ days | 48 hours | Initial line release |
| Early | 1-3 days | 24 hours | Line establishment |
| Sharp | 12-24 hours | 12 hours | Professional betting |
| Pregame | 3-12 hours | 3 hours | Active betting |
| Closing | 0-3 hours | 30 minutes | Critical line movement |

Tier tracking stored in `OddsSnapshot.fetch_tier` for validation and ML features.

## Scheduled Jobs

### Event Sync (embedded in fetch-odds)

`packages/odds-lambda/odds_lambda/event_sync.py` — `EventSyncService`

- Runs as the first step of every `fetch-odds` invocation
- Syncs upcoming events from the free `/events` endpoint (0 quota cost)
- Returns `EventSyncResult` with inserted/updated counts per sport

### Fetch Odds Job

`packages/odds-lambda/odds_lambda/jobs/fetch_odds.py`

- Game-aware execution (only runs when games upcoming)
- Fetches current odds for all scheduled games
- Stores hybrid data (raw JSONB + normalized records)
- Calculates next execution based on closest game's tier
- Logs API quota usage

### Fetch Scores Job

`packages/odds-lambda/odds_lambda/jobs/fetch_scores.py`

- Updates completed game results
- Marks events as FINAL with scores
- Self-schedules based on live game activity

### Update Status Job

`packages/odds-lambda/odds_lambda/jobs/update_status.py`

- Transitions event statuses (scheduled -> live -> final)
- Prevents fetching odds for completed games

### OddsPortal Scrape Job

`packages/odds-lambda/odds_lambda/jobs/fetch_oddsportal.py`

- Runs hourly on the scraper Lambda
- Scrapes upcoming EPL match odds from OddsPortal via OddsHarvester
- Stores snapshots with tier classification based on time to kickoff
- Shared utilities in `oddsportal_common.py` (bookmaker key mapping, decimal→American conversion)

### OddsPortal Results Job

`packages/odds-lambda/odds_lambda/jobs/fetch_oddsportal_results.py`

- Runs daily at 08:00 UTC on the scraper Lambda
- Scrapes EPL results and closing odds from OddsPortal
- Updates event status to FINAL with scores

### Score Predictions Job

`packages/odds-lambda/odds_lambda/jobs/score_predictions.py`

- Runs hourly at :15 (offset to allow scraper to finish)
- Loads CLV model from S3 (`odds-models` bucket) with ETag-based caching
- Extracts tabular features for upcoming SCHEDULED events
- Stores predictions in `Prediction` table with idempotency (unique constraint on event_id, snapshot_id, model_name)
- Uses `TabularFeatureExtractor` from `odds-analytics`

### Daily Digest Job

`packages/odds-lambda/odds_lambda/jobs/daily_digest.py`

- Runs daily at 08:00 UTC
- Queries completed events (last 24h) and upcoming events (next 48h) from `Prediction` table
- Formats Discord embed with post-match results and upcoming predictions
- Sends via `AlertManager.send_embed()` (Discord webhook)

## S3 Model Store

Models are published to S3 and loaded by the score-predictions Lambda job.

- **Bucket**: `odds-models-{account_id}` (versioned, encrypted, no public access)
- **Path convention**: `s3://{bucket}/{model_name}/latest/` + `{version}/`
- **Artifacts**: `model.pkl`, `config.yaml`, `metadata.json`
- **Publishing**: `odds model publish` CLI command
- **Loading**: `packages/odds-lambda/odds_lambda/model_loader.py` — HEAD + ETag check per invocation, re-downloads only if changed

## Event Lifecycle

| Status | Description |
|--------|-------------|
| SCHEDULED | Odds actively being fetched |
| LIVE | Game in progress (stop fetching odds) |
| FINAL | Game completed with results stored |
| CANCELLED | No further data collection |
| POSTPONED | No further data collection |

## Data Collection Pipeline

```
Scheduler -> API Client -> Validator -> Writer -> Logger
                |              |           |         |
                v              v           v         v
           Rate limit    Log issues    Raw+Norm   Quota
```

### API Client

`packages/odds-lambda/odds_lambda/data_fetcher.py` - `TheOddsAPIClient`

- Retry logic with tenacity (exponential backoff)
- Rate limiting
- Quota tracking with API key rotation (multiple keys via `ODDS_API_KEYS`, active index tracked in SSM Parameter Store)
- Methods: `get_odds()`, `get_scores()`, `get_historical_odds()`, `get_historical_events()`

### Game Log Pipeline

`packages/odds-lambda/odds_lambda/game_log_fetcher.py` - `fetch_game_logs()`

- Fetches NBA team game logs from stats.nba.com via Playwright (headless Firefox)
- Uses Playwright to bypass Akamai bot detection (raw HTTP requests are blocked)
- Calls LeagueGameFinder API from browser context after establishing session cookies
- Returns ~2,460 rows per season (30 teams × 82 games)
- Storage: `GameLogWriter` (upsert with event matching), `GameLogReader` (pipeline stats)
- CLI: `odds nba-stats fetch --season 2024-25`, `odds nba-stats status`
- Score backfill (`odds backfill scores`) reads from game log table

### Validator

`packages/odds-lambda/odds_lambda/storage/validators.py` - `OddsValidator`

**Validation Checks:**
- Price within valid range (-10000 to +10000)
- Timestamp not in future
- Vig reasonable (2-15%)
- No missing required fields
- Line movement not excessive (>10 point spread)
- Outcomes match expected format

**Behavior:** Logs warnings by default, does not reject data (configurable via `REJECT_INVALID_ODDS`)

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
| `validators.py` | `OddsValidator` — data quality checks |
| `tier_validator.py` | Tier assignment validation |

## Data Quality Monitoring

All issues logged to `DataQualityLog` table:
- Severity level (warning, error, critical)
- Issue type classification
- Full context for debugging
- Timestamp for analysis

**Action:** Log and flag suspicious data but do not reject. Allow manual review.

## Discord Alerts

`packages/odds-cli/odds_cli/alerts/base.py`

- `DiscordAlert` class with webhook support
- `AlertManager` routing class
- Used by daily-digest job and quota/health alerts
- Configured via `DISCORD_WEBHOOK_URL` environment variable

## Technical Stack

### Core Technologies

- Python 3.12
- SQLModel (SQLAlchemy 2.0 + Pydantic)
- PostgreSQL 15+ with JSON/JSONB
- asyncio + aiohttp

### Key Libraries

- **uv**: Package management
- **APScheduler**: Job scheduling (local/Railway backends)
- **Typer + Rich**: CLI
- **Pydantic Settings**: Configuration
- **Alembic**: Database migrations
- **tenacity**: Retry logic
- **pytest + pytest-asyncio**: Testing
- **structlog**: Structured logging
- **Playwright**: Headless browser (OddsPortal scraping, NBA game logs)
- **XGBoost**: CLV prediction model
- **nba_api**: NBA.com API wrapper
