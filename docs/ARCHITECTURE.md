# Architecture Reference

> Reference documentation for system architecture, scheduling, and data pipeline.

## System Overview

A single-user betting odds data collection and analysis system for NBA games. Prioritizes robust data pipeline architecture with comprehensive historical data collection, storage, and validation.

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

### Primary API: The Odds API

- Base URL: `https://api.the-odds-api.com/v4`
- NBA Endpoint: `/sports/basketball_nba/odds`
- Scores Endpoint: `/sports/basketball_nba/scores`
- Historical Endpoint: `/sports/basketball_nba/odds/history`

See CLAUDE.md for endpoint pricing breakdown and DEPLOYMENT.md for tier details.

### Bookmaker Coverage (8 books)

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

### Markets Collected

- **h2h**: Moneyline (win/loss)
- **spreads**: Point spread
- **totals**: Over/under total points

### Secondary API: NBA API

Library: `nba_api` (Python wrapper for NBA.com)

**Purpose:** Backfill historical scores when The Odds API incomplete

**Key Methods:**
- `get_live_scores()` - Current game scores
- `get_historical_scores(start_date, end_date)` - Historical results
- `match_game_by_teams_and_date()` - Fuzzy matching

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

Centralized mapping of job names to functions:
- `fetch-odds`
- `fetch-scores`
- `update-status`

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

### Fetch Odds Job

`packages/odds-lambda/odds_lambda/jobs/fetch_odds.py`

- Game-aware execution (only runs when games upcoming)
- Fetches current odds for all scheduled NBA games
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
- Quota tracking
- Methods: `get_odds()`, `get_scores()`, `get_historical_odds()`, `get_historical_events()`

### NBA Score Fetcher

`packages/odds-lambda/odds_lambda/nba_score_fetcher.py` - `NBAScoreFetcher`

- Fetches NBA scores using nba_api library
- Used by `odds backfill scores` command

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

### Writer

`packages/odds-lambda/odds_lambda/storage/writers.py` - `OddsWriter`

- `store_odds_snapshot()` - Hybrid raw + normalized storage
- `upsert_event()` - Create/update events
- `bulk_insert_historical()` - Backfill operations
- `log_data_quality_issue()` - Record quality issues

### Reader

`packages/odds-lambda/odds_lambda/storage/readers.py` - `OddsReader`

- `get_odds_at_time()` - Critical for backtesting (prevents look-ahead bias)
- `get_line_movement()` - Time series of odds changes
- `get_events_by_date_range()` - Query events
- `get_best_odds()` - Line shopping

## Data Quality Monitoring

All issues logged to `DataQualityLog` table:
- Severity level (warning, error, critical)
- Issue type classification
- Full context for debugging
- Timestamp for analysis

**Action:** Log and flag suspicious data but do not reject. Allow manual review.

## Technical Stack

### Core Technologies

- Python 3.11+
- SQLModel (SQLAlchemy 2.0 + Pydantic)
- PostgreSQL 15+ with JSON/JSONB
- asyncio + aiohttp

### Key Libraries

- **uv**: Package management
- **APScheduler**: Job scheduling
- **Typer + Rich**: CLI
- **Pydantic Settings**: Configuration
- **Alembic**: Database migrations
- **tenacity**: Retry logic
- **pytest + pytest-asyncio**: Testing
- **structlog**: Structured logging
- **nba_api**: NBA.com API wrapper
