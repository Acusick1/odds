# Betting Odds Data Pipeline - Technical Specification

IMPORTANT (LLM agents): This document is read-only and should only be modified by the user. Do not create additional documentation files (outside of README.md) without confirming with the user first, only make a case for additional documentation files when you believe they are absolutely necessary.

**Document Status**: Updated October 2024 to reflect production architecture with multi-backend scheduler and intelligent data collection. Original design specification available in git history.

## System Overview

A single-user betting odds data collection and analysis system focused on NBA games with extensibility to other sports. The system prioritizes robust data pipeline architecture over immediate analytics capabilities, with a focus on comprehensive historical data collection, storage, and validation.

### Core Principles

- **Data pipeline first**: Robust data collection and storage is paramount
- **Single-user system**: Latency is not critical, simplicity over scale
- **Extensibility**: Built to easily support additional sports and markets
- **Data quality**: Validation and quality checks throughout the pipeline
- **Historical analysis**: All data preserved for backtesting and research

### Primary Use Cases

- Pre-match betting decisions and research
- Historical odds analysis over multiple seasons
- Backtesting betting strategies against historical data
- Cross-bookmaker odds comparison
- Line movement tracking and analysis
- Arbitrage and expected value opportunity detection

---

## Technical Stack

### Core Technologies

- **Python 3.11+**: Primary language
- **SQLModel**: ORM and data validation (combines SQLAlchemy 2.0 + Pydantic)
- **PostgreSQL 15+**: Database with JSONB support for hybrid storage
- **asyncio + aiohttp**: Asynchronous API calls and operations

### Key Libraries

- **uv**: Fast Python package installer and dependency management
- **APScheduler**: Job scheduling for periodic data collection
- **Typer + Rich**: CLI interface with formatted output
- **Pydantic Settings**: Configuration management via environment variables
- **Alembic**: Database migrations
- **tenacity**: Retry logic with exponential backoff
- **pytest + pytest-asyncio**: Testing framework
- **structlog**: Structured logging
- **ruff**: Fast Python linter and formatter (used in pre-commit hooks)

### Deployment

- **Multi-Backend Scheduler**: AWS Lambda (production), Railway, or Local APScheduler
- **AWS Lambda**: Primary production deployment (~$0.20/month) with self-scheduling via EventBridge
- **Docker + docker-compose**: Local development and portable deployment
- **PostgreSQL**: Managed database service (Neon/Railway)

---

## Data Sources

### Primary API: The Odds API

- Base URL: `https://api.the-odds-api.com/v4`
- NBA Endpoint: `/sports/basketball_nba/odds`
- Scores Endpoint: `/sports/basketball_nba/scores`
- Historical Endpoint: `/sports/basketball_nba/odds/history`

### API Pricing Model

- Cost per "object" where 1 object = 1 game/event
- Number of bookmakers and markets requested does not affect cost
- 20,000 requests/month tier: $25/month
- Expected usage with 30-minute sampling: ~14,400 requests/month

### Bookmaker Coverage (8 books)

1. **Pinnacle** - Sharp, low-margin bookmaker (key for EV calculations)
2. **Circa Sports** - Sharp, Vegas-based
3. **DraftKings** - Major US retail sportsbook
4. **FanDuel** - Major US retail sportsbook
5. **BetMGM** - Major US retail sportsbook
6. **Caesars** - Major US retail sportsbook (williamhill_us in API)
7. **BetRivers** - Regional retail sportsbook
8. **Bovada** - Offshore sportsbook

### Markets Collected

- **Moneyline (h2h)**: Win/loss bets
- **Spreads**: Point spread bets
- **Totals**: Over/under total points bets

### API Configuration

```python
regions: ["us"]
markets: ["h2h", "spreads", "totals"]
odds_format: "american"
```

---

## Data Models

### Database Schema

All models use SQLModel for combined ORM and validation capabilities.

#### Event Model

```python
class EventStatus(str, Enum):
    SCHEDULED = "scheduled"
    LIVE = "live"
    FINAL = "final"
    CANCELLED = "cancelled"
    POSTPONED = "postponed"

class Event(SQLModel, table=True):
    __tablename__ = "events"
    
    # Primary identification
    id: str = Field(primary_key=True)  # API event ID
    sport_key: str = Field(index=True)
    sport_title: str
    
    # Event details
    commence_time: datetime = Field(index=True)
    home_team: str = Field(index=True)
    away_team: str = Field(index=True)
    status: EventStatus = Field(default=EventStatus.SCHEDULED)
    
    # Results (populated after game completion)
    home_score: int | None = None
    away_score: int | None = None
    completed_at: datetime | None = None
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=datetime.now(timezone.utc))
```

#### OddsSnapshot Model (Raw Data Preservation)

```python
class OddsSnapshot(SQLModel, table=True):
    __tablename__ = "odds_snapshots"

    id: int | None = Field(default=None, primary_key=True)
    event_id: str = Field(foreign_key="events.id", index=True)
    snapshot_time: datetime = Field(index=True)

    # Full API response stored as JSON
    raw_data: dict = Field(sa_column=Column(JSON))

    # Quick statistics
    bookmaker_count: int
    api_request_id: str | None = None  # For debugging

    # Fetch tier tracking (for adaptive sampling validation and ML features)
    fetch_tier: str | None = None  # opening, early, sharp, pregame, closing
    hours_until_commence: float | None = None

    created_at: datetime = Field(default_factory=datetime.now(timezone.utc))

    __table_args__ = (
        Index('ix_event_snapshot_time', 'event_id', 'snapshot_time'),
        Index('ix_event_tier', 'event_id', 'fetch_tier'),
    )
```

**Fetch Tier Tracking**: Each snapshot records which tier it belongs to (`fetch_tier`) and timing (`hours_until_commence`). This enables:

- Validation that intelligent scheduling is working correctly
- ML feature engineering (e.g., "closing line" vs "opening line" features)
- Analysis of line movement patterns by tier
- See `core/fetch_tier.py` for FetchTier enum and `core/tier_utils.py` for calculations

#### Odds Model (Normalized for Querying)

```python
class Odds(SQLModel, table=True):
    __tablename__ = "odds"
    
    id: int | None = Field(default=None, primary_key=True)
    event_id: str = Field(foreign_key="events.id", index=True)
    
    # Bookmaker information
    bookmaker_key: str = Field(index=True)
    bookmaker_title: str
    market_key: str = Field(index=True)  # h2h, spreads, totals
    
    # Outcome data
    outcome_name: str  # Team name or Over/Under
    price: int  # American odds (e.g., -110, +150)
    point: float | None = None  # Spread/total line (e.g., -2.5, 218.5)
    
    # Timestamps
    odds_timestamp: datetime = Field(index=True)  # When odds were valid
    last_update: datetime  # Bookmaker's last update time
    created_at: datetime = Field(default_factory=datetime.now(timezone.utc))
    
    # Data quality
    is_valid: bool = Field(default=True)
    validation_notes: str | None = None
    
    __table_args__ = (
        Index('ix_event_bookmaker_market', 'event_id', 'bookmaker_key', 'market_key'),
        Index('ix_bookmaker_time', 'bookmaker_key', 'odds_timestamp'),
    )
```

#### DataQualityLog Model

```python
class DataQualityLog(SQLModel, table=True):
    __tablename__ = "data_quality_logs"
    
    id: int | None = Field(default=None, primary_key=True)
    event_id: str | None = Field(foreign_key="events.id")
    
    severity: str  # warning, error, critical
    issue_type: str  # missing_data, suspicious_odds, stale_timestamp, etc.
    description: str
    raw_data: dict | None = Field(sa_column=Column(JSON))
    
    created_at: datetime = Field(default_factory=datetime.now(timezone.utc), index=True)
```

#### FetchLog Model

```python
class FetchLog(SQLModel, table=True):
    __tablename__ = "fetch_logs"
    
    id: int | None = Field(default=None, primary_key=True)
    fetch_time: datetime = Field(default_factory=datetime.now(timezone.utc), index=True)
    
    sport_key: str
    events_count: int
    bookmakers_count: int
    
    success: bool
    error_message: str | None = None
    
    # API quota tracking
    api_quota_remaining: int | None = None
    response_time_ms: int | None = None
```

### Storage Strategy: Hybrid Approach

**Raw Data (OddsSnapshot)**: Complete API responses stored as JSON for:

- Debugging and auditing
- Schema flexibility
- Exact data preservation
- Note: PostgreSQL JSON type is used; consider migrating to JSONB for better query performance

**Normalized Data (Odds)**: Individual odds records for:

- Efficient querying and filtering
- Time-series analysis
- Standard SQL operations
- Performance optimization

---

## System Architecture

### Component Structure

High-level directory organization:

```bash
betting-odds-system/
├── core/
│   ├── models.py              # SQLModel schema definitions
│   ├── database.py            # Database session management
│   ├── data_fetcher.py        # The Odds API client
│   ├── config.py              # Pydantic Settings configuration
│   ├── ingestion.py           # Odds ingestion service
│   ├── backfill_executor.py   # Historical data backfill
│   ├── game_selector.py       # Game selection for backfill
│   ├── fetch_tier.py          # FetchTier enum definition
│   ├── tier_utils.py          # Tier calculation utilities
│   └── scheduling/            # Multi-backend scheduler
│       ├── intelligence.py    # Game-aware scheduling logic
│       ├── jobs.py            # Job registry
│       ├── health_check.py    # System health monitoring
│       └── backends/
│           ├── base.py        # Abstract backend interface
│           ├── aws.py         # AWS Lambda + EventBridge
│           ├── railway.py     # Railway deployment
│           └── local.py       # APScheduler for dev
│
├── jobs/                      # Standalone job executables
│   ├── fetch_odds.py          # Odds collection job
│   ├── fetch_scores.py        # Scores update job
│   └── update_status.py       # Status transition job
│
├── storage/
│   ├── writers.py             # Write operations
│   ├── readers.py             # Read operations & queries
│   └── validators.py          # Data quality validation
│
├── analytics/
│   ├── backtesting/           # Backtesting system
│   │   ├── models.py          # Data models & results
│   │   ├── services.py        # BacktestEngine
│   │   └── config.py          # Configuration
│   ├── strategies.py          # Strategy implementations
│   ├── utils.py               # Odds calculations & metrics
│   └── ml_strategy_example.py # ML integration example
│
├── cli/
│   └── commands/              # CLI command modules
│       ├── fetch.py, status.py, backfill.py
│       ├── backtest.py, scheduler.py, validate.py
│
├── deployment/                # Deployment configurations
│   ├── aws/                   # Lambda + Terraform
│   └── railway/               # Railway configs
│
├── tests/                     # 82 test files, 5000+ lines
├── migrations/                # Alembic migrations
└── docker-compose.yml, Dockerfile, pyproject.toml
```

See actual filesystem for complete structure.

### Multi-Backend Scheduler Architecture

The system uses an abstraction layer to support multiple deployment backends, allowing the same codebase to run on AWS Lambda, Railway, or locally for development.

**Backend Interface** (`core/scheduling/backends/base.py`):
All backends implement `SchedulerBackend` abstract base class with methods:

- `schedule_job(job_name, execution_time)`: Schedule a future job execution
- `get_backend_info()`: Return backend-specific metadata

**AWS Lambda Backend** (`core/scheduling/backends/aws.py`):

- Self-scheduling pattern using EventBridge rules
- Each job invocation schedules its next execution
- NullPool database connections (no connection reuse across invocations)
- Serverless with automatic scaling
- Configuration: `SCHEDULER_BACKEND=aws`, `AWS_LAMBDA_ARN`, `AWS_REGION`

**Railway Backend** (`core/scheduling/backends/railway.py`):

- Uses APScheduler with persistent scheduling
- Long-running process with managed restarts
- Standard connection pooling
- Configuration: `SCHEDULER_BACKEND=railway`

**Local Backend** (`core/scheduling/backends/local.py`):

- APScheduler for development and testing
- Runs until process terminated (Ctrl+C)
- Full logging for debugging
- Configuration: `SCHEDULER_BACKEND=local`

**Job Registry** (`core/scheduling/jobs.py`):

- Centralized mapping of job names to functions
- Lazy-loaded to avoid circular imports
- Jobs: `fetch-odds`, `fetch-scores`, `update-status`

**Scheduling Intelligence** (`core/scheduling/intelligence.py`):

- Game-aware execution logic (described in Scheduler Architecture section)
- Calculates optimal next execution time based on game state
- Used by all backends to determine job timing

### Data Collection Pipeline

**Flow**:

1. **Scheduler** triggers fetch job at configured interval
2. **API Client** fetches odds data with rate limiting
3. **Validator** checks data quality and logs issues
4. **Writer** stores both raw (JSONB) and normalized data
5. **Logger** records fetch success/failure and quota usage

### API Client & Data Pipeline

**TheOddsAPIClient** (`core/data_fetcher.py`):
Handles all API interactions with retry logic (tenacity), rate limiting, and quota tracking. Methods: `get_odds()`, `get_scores()`, `get_historical_odds()`, `get_historical_events()`.

**OddsValidator** (`storage/validators.py`):
Data quality validation with checks for odds range, timestamps, vig (2-15%), line movement, and required fields. Logs warnings but does not reject data (configurable).

**OddsWriter** (`storage/writers.py`):
Write operations including `store_odds_snapshot()` (hybrid raw + normalized storage), `upsert_event()`, `bulk_insert_historical()` for backfill, and `log_data_quality_issue()`.

**OddsReader** (`storage/readers.py`):
Query operations including `get_odds_at_time()` (critical for backtesting to prevent look-ahead bias), `get_line_movement()`, `get_events_by_date_range()`, and `get_best_odds()`.

---

## Scheduler Architecture

### Intelligent Scheduling System

The system uses **game-aware intelligent scheduling** that adapts data collection frequency based on game proximity, eliminating fixed sampling intervals. Implemented in `core/scheduling/intelligence.py`.

**Key Features**:

- Automatically discovers upcoming games from API
- Adjusts fetch frequency as games approach using tiered intervals
- No fetching during off-season (waits for new games)
- Self-scheduling pattern for serverless deployment (AWS Lambda)

**Fetch Tier System** (`core/fetch_tier.py`):

- **Opening** (3+ days before): Every 48 hours - initial line release
- **Early** (1-3 days before): Every 24 hours - line establishment
- **Sharp** (12-24 hours before): Every 12 hours - professional betting period
- **Pregame** (3-12 hours before): Every 3 hours - active betting
- **Closing** (0-3 hours before): Every 30 minutes - critical line movement period

Tier tracking is stored in `OddsSnapshot.fetch_tier` for validation and ML feature engineering.

### Scheduled Jobs

The system executes three autonomous jobs that self-schedule based on game state:

**Fetch Odds Job** (`jobs/fetch_odds.py`):

- Game-aware execution (only runs when games upcoming)
- Fetches current odds for all scheduled NBA games
- Stores hybrid data (raw JSONB + normalized records)
- Calculates next execution based on closest game's tier
- Logs API quota usage

**Fetch Scores Job** (`jobs/fetch_scores.py`):

- Updates completed game results
- Marks events as FINAL with scores
- Self-schedules based on live game activity

**Update Status Job** (`jobs/update_status.py`):

- Transitions event statuses (scheduled → live → final)
- Prevents fetching odds for completed games

### Event Lifecycle

- **SCHEDULED**: Odds actively being fetched
- **LIVE**: Game in progress (stop fetching odds)
- **FINAL**: Game completed with results stored
- **CANCELLED/POSTPONED**: No further data collection

---

## Configuration Management

Configuration uses Pydantic Settings with composition pattern for logical grouping. All settings loaded from environment variables. See `core/config.py` for complete implementation.

### Configuration Structure

**API Settings** (`APIConfig`):

- `ODDS_API_KEY`: The Odds API authentication key (required)
- `ODDS_API_BASE_URL`: API base URL (default: https://api.the-odds-api.com/v4)
- `ODDS_API_QUOTA`: Monthly request quota (default: 20,000)

**Database Settings** (`DatabaseConfig`):

- `DATABASE_URL`: PostgreSQL connection string (required)
- `DATABASE_POOL_SIZE`: Connection pool size (default: 5, NullPool for Lambda)

**Data Collection** (`DataCollectionConfig`):

- `SPORTS`: Sports to track (default: ["basketball_nba"])
- `BOOKMAKERS`: List of 8 bookmakers (Pinnacle, Circa, DraftKings, FanDuel, etc.)
- `MARKETS`: Bet types to collect (default: ["h2h", "spreads", "totals"])
- `REGIONS`: Odds regions (default: ["us"])

**Scheduler Settings** (`SchedulerConfig`):

- `SCHEDULER_BACKEND`: Backend type - "aws", "railway", or "local" (default: "local")
- `SCHEDULER_DRY_RUN`: Enable dry-run mode for testing (default: false)
- `SCHEDULER_LOOKAHEAD_DAYS`: Days ahead to check for games (default: 7)

**AWS Settings** (`AWSConfig`) - Only needed for Lambda deployment:

- `AWS_REGION`: AWS region for Lambda (optional)
- `AWS_LAMBDA_ARN`: Lambda function ARN for self-scheduling (optional)

**Data Quality** (`DataQualityConfig`):

- `ENABLE_VALIDATION`: Run data quality checks (default: true)
- `REJECT_INVALID_ODDS`: Reject vs. log invalid data (default: false)

**Alert Settings** (`AlertConfig`) - Infrastructure exists, not actively used:

- `DISCORD_WEBHOOK_URL`: Discord webhook for alerts (optional)
- `ALERT_ENABLED`: Enable alert system (default: false)

**Logging** (`LoggingConfig`):

- `LOG_LEVEL`: Logging verbosity (default: INFO)
- `LOG_FILE`: Log file path (default: logs/odds_pipeline.log)

See `.env.example` for complete environment variable template.

---

## CLI Interface

### Command Structure

**Data Collection** (Implemented):

```bash
odds fetch current                      # Manual fetch current odds
odds fetch current --sport basketball_nba
odds fetch scores                       # Fetch scores
odds fetch scores --sport basketball_nba --days 3
```

**Backfill Commands** (Implemented):

```bash
odds backfill plan --start YYYY-MM-DD --end YYYY-MM-DD --games N
odds backfill execute --plan backfill_plan.json
odds backfill status                    # Show backfill progress
```

**Backtesting Commands** (Implemented):

```bash
odds backtest run --strategy STRATEGY --start YYYY-MM-DD --end YYYY-MM-DD
odds backtest run --strategy basic_ev --start 2024-10-01 --end 2024-12-31 --output-json results.json
odds backtest show results.json         # Display saved results
odds backtest show results.json --verbose  # Detailed breakdowns
odds backtest compare result1.json result2.json result3.json  # Compare strategies
odds backtest export results.json --output bets.csv  # Convert to CSV
odds backtest list-strategies           # List available strategies
```

**Status & Monitoring** (Implemented):

```bash
odds status show                        # System health overview
odds status show --verbose              # Detailed statistics
odds status quota                       # Check API usage remaining
odds status events --days 7             # List recent events
odds status events --team "Lakers"      # Filter by team
```

### CLI Output Formatting

The CLI uses the **Rich** library for formatted terminal output:

- Colored, formatted tables for status and results
- Progress bars for long-running operations (backfill, backtesting)
- Status indicators (✓, ✗, ⚠) for system health
- Syntax highlighting for JSON output

---

## Historical Data Backfill

### Strategy

**Initial Backfill Target**: Sample of last season's regular season games

**Sampling Approach**:

- Select representative games distributed across season
- Include all 30 teams proportionally
- Mix of different matchup types
- ~20% sample rate (every 5th game)

**Selection Criteria** (Implemented in core/game_selector.py):

```python
class GameSelector:
    """
    Strategic game selection for backfill operations

    Criteria:
    - Even distribution across season dates
    - Proportional team representation
    - Variety in matchups and game importance
    - Multiple selection strategies (uniform, random, weighted)
    """
```

**Data Collection**:

- Use historical odds API endpoint
- Fetch odds snapshots for selected games
- Fetch final scores for validation
- Store using same schema as live data

**Purpose**:

- Validate schema design with real data
- Test query patterns and performance
- Enable immediate backtesting capability
- Verify data quality checks work correctly

---

## Alert System Infrastructure

### Design Philosophy

Build alert infrastructure now for easy activation later, but do not implement active alerting yet.

### Base Alert System

```python
class AlertBase(ABC):
    """Base class for all alert types"""
    
    @abstractmethod
    async def send(self, message: str, severity: str):
        """Send alert via specific channel"""

class DiscordAlert(AlertBase):
    """Discord webhook implementation"""
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    async def send(self, message: str, severity: str):
        # Format and send to Discord

class AlertManager:
    """Route alerts to appropriate channels"""
    def __init__(self, config: Settings):
        self.enabled = config.alert_enabled
        self.channels = []
        
        if config.discord_webhook_url:
            self.channels.append(DiscordAlert(config.discord_webhook_url))
    
    async def alert(self, message: str, severity: str = "info"):
        if not self.enabled:
            return
        for channel in self.channels:
            await channel.send(message, severity)
```

### Future Alert Triggers (Not Implemented Yet)

When alerts are enabled, they could trigger on:

- Arbitrage opportunities above threshold
- Expected value bets above threshold
- Large line movements
- Data quality issues
- System errors
- Daily summary reports

---

## Backtesting Infrastructure

**Status**: ✓ **FULLY IMPLEMENTED AND OPERATIONAL** - The backtesting system is complete with multiple strategies, comprehensive metrics, CLI interface, and full test coverage.

### Strategy Pattern

```python
class BacktestEvent(BaseModel):
    """Event validated for backtesting - guaranteed to have final scores."""
    id: str
    commence_time: datetime
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    status: EventStatus

class BettingStrategy(ABC):
    """Base class for all betting strategies"""

    @abstractmethod
    async def evaluate_opportunity(
        self,
        event: BacktestEvent,
        odds_snapshot: list[Odds],
        config: BacktestConfig
    ) -> list[BetOpportunity]:
        """Evaluate event and return betting opportunities"""

    def get_name(self) -> str:
        """Return strategy name"""

    def get_params(self) -> dict:
        """Return strategy parameters"""
```

### Implemented Strategies

**1. FlatBettingStrategy** (`analytics/strategies.py`):

```python
class FlatBettingStrategy(BettingStrategy):
    """Baseline strategy - bet on every game matching pattern"""
    def __init__(
        self,
        market: str = "h2h",  # h2h, spreads, totals
        outcome_pattern: str = "home",  # home, away, favorite
        bookmaker: str = "fanduel"
    ):
        ...
```

**2. BasicEVStrategy** (`analytics/strategies.py`):

```python
class BasicEVStrategy(BettingStrategy):
    """Expected value betting using sharp vs retail odds comparison"""
    def __init__(
        self,
        sharp_book: str = "pinnacle",
        retail_books: list[str] = ["fanduel", "draftkings", "betmgm"],
        min_ev_threshold: float = 0.03,  # 3% minimum EV
        markets: list[str] = ["h2h", "spreads", "totals"]
    ):
        ...
```

**3. ArbitrageStrategy** (`analytics/strategies.py`):

```python
class ArbitrageStrategy(BettingStrategy):
    """Risk-free arbitrage betting across bookmakers"""
    def __init__(
        self,
        min_profit_margin: float = 0.01,  # 1% minimum profit
        max_hold: float = 0.10,  # 10% max market hold
        bookmakers: list[str] = None  # All major books by default
    ):
        ...
```

### Machine Learning Integration

**Status**: ✓ **FULLY SUPPORTED** - The backtesting infrastructure fully supports ML-based strategies including XGBoost, neural networks, and time series models.

**Key Features Enabling ML**:

1. **Event + Odds as Input**: Strategy receives complete `BacktestEvent` object and `odds_snapshot` list, perfect for feature engineering
2. **Type Safety**: BacktestEvent guarantees final scores exist, eliminating None checks in model code
3. **Confidence Field**: `BetOpportunity.confidence` (0-1) maps directly to model probability predictions
4. **Kelly Criterion Integration**: Model confidence is automatically used for optimal bet sizing
5. **Async Support**: Allows batch predictions and external API calls
6. **Look-Ahead Bias Prevention**: Built-in timestamp controls ensure only historical data is used

**Example Implementation** (`analytics/ml_strategy_example.py`):

```python
class XGBoostStrategy(BettingStrategy):
    """ML-based betting strategy using XGBoost for game outcome prediction."""

    def __init__(
        self,
        model_path: str | None = None,
        min_ev_threshold: float = 0.05,  # 5% minimum EV
        min_model_confidence: float = 0.55,  # Only bet when >55% confident
        sharp_book: str = "pinnacle",
        retail_books: list[str] | None = None,
    ):
        # Load trained XGBoost model from pickle file
        if model_path and Path(model_path).exists():
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)

    async def evaluate_opportunity(
        self,
        event: BacktestEvent,  # Type-safe with guaranteed scores
        odds_snapshot: list[Odds],
        config: BacktestConfig,
    ) -> list[BetOpportunity]:
        """
        Process:
            1. Engineer features from event and odds data
            2. Run XGBoost model to predict win probabilities
            3. Compare predictions to available odds
            4. Return opportunities where EV exceeds threshold
        """
        # Feature engineering
        features = self._engineer_features(event, odds_snapshot)

        # Get model predictions
        home_win_prob = self._predict_home_win_probability(features)

        # Find +EV opportunities
        for retail_book in retail_books:
            retail_odds = self._find_odds(odds_snapshot, retail_book, "h2h", event.home_team)

            if retail_odds and home_win_prob >= min_confidence:
                ev = calculate_ev(home_win_prob, retail_odds.price, 100.0)

                if ev_percentage >= min_ev:
                    opportunities.append(
                        BetOpportunity(
                            confidence=home_win_prob,  # Used for Kelly sizing
                            rationale=f"XGBoost: {home_win_prob:.1%} win prob"
                        )
                    )

        return opportunities
```

**Feature Engineering Examples**:

- Sharp bookmaker implied probabilities
- Retail consensus probabilities
- Sharp vs retail differentials
- Day of week / game timing
- Team stats (offensive/defensive rating)
- Recent form (last N games)
- Rest days between games
- Head-to-head history
- Home court advantage metrics

**Supported ML Architectures**:

- **XGBoost / LightGBM**: Gradient boosting for classification
- **Neural Networks**: Deep learning with PyTorch/TensorFlow
- **Time Series Models**: LSTM, GRU for sequential prediction
- **Ensemble Models**: Combining multiple model predictions
- **Online Learning**: Update models with new results

**Integration Pattern**:

1. Train model offline using historical event + odds data
2. Save model to pickle file
3. Load in strategy constructor
4. Engineer features in `_engineer_features()`
5. Generate predictions in `evaluate_opportunity()`
6. Use model confidence for EV calculation and Kelly sizing
7. Backtest strategy against historical data to validate

See [analytics/ml_strategy_example.py](analytics/ml_strategy_example.py) for complete working example with training code.

### Backtesting Engine

```python
class BacktestEngine:
    """
    Simulate betting strategies against historical data

    Implemented Features:
    - Look-ahead bias prevention (configurable decision time before game)
    - Multiple bet sizing methods (Kelly, flat, percentage)
    - Bankroll management with min/max bet constraints
    - Comprehensive performance metrics calculation
    - Progress tracking with Rich progress bars
    - Data quality tracking and reporting
    """

    def __init__(
        self,
        strategy: BettingStrategy,
        config: BacktestConfig,
        session: AsyncSession
    ):
        ...

    async def run(self) -> BacktestResult:
        """
        Execute backtest workflow:
        1. Query historical events with results in date range
        2. For each event, get odds at decision time (e.g., 1 hour before game)
        3. Apply strategy to identify betting opportunities
        4. Calculate appropriate stake using configured sizing method
        5. Evaluate bet result using actual game outcome
        6. Track bankroll evolution and equity curve
        7. Calculate comprehensive performance metrics
        8. Return complete BacktestResult with all data
        """
```

### Data Models

**BacktestConfig** (`analytics/backtesting/config.py`):
Configuration for backtest execution with settings for initial bankroll, date range, bet sizing method (fractional_kelly/flat/percentage), Kelly fraction, bet limits, and decision timing (hours before game to prevent look-ahead bias).

**BetRecord** (`analytics/backtesting/models.py`):
Complete record of individual bet with event details, market/outcome, bookmaker, odds, stake, result (win/loss/push), profit, bankroll tracking, and strategy rationale. Used for detailed analysis and CSV export.

**BacktestResult** (`analytics/backtesting/models.py`):
Comprehensive results container with 30+ fields including:

- Performance: ROI, win rate, total profit, bet counts
- Risk metrics: Sharpe ratio, Sortino ratio, max drawdown, profit factor, Calmar ratio
- Breakdowns: By market, bookmaker, month
- Time series: Equity curve, all bet records
- Export methods: `to_json()`, `to_csv()`, `from_json()` for persistence

See `analytics/backtesting/models.py` for complete definitions and `BACKTESTING_GUIDE.md` for usage documentation.

### Bet Sizing & Utilities

**Bet Sizing Methods** (configured in BacktestConfig):

1. **Fractional Kelly** (default, recommended): Quarter-Kelly (0.25) for optimal risk/reward balance
2. **Flat Betting**: Fixed dollar amount per bet, good for baseline comparison
3. **Percentage Betting**: Fixed percentage of current bankroll (default 2%)

All methods enforce min/max bet constraints and prevent betting with insufficient bankroll.

**Utility Functions** (`analytics/utils.py`):
Comprehensive toolkit for odds conversion (American ↔ decimal), probability calculations, expected value, Kelly Criterion, profit calculations, and risk metrics (Sharpe, Sortino, max drawdown, profit factor). Also includes arbitrage detection. See `BACKTESTING_GUIDE.md` for details.

---

## Docker Configuration

Docker configuration files exist in repo root for local development:

- `docker-compose.yml`: PostgreSQL container for local testing
- `Dockerfile`: Application container definition

Docker-based architecture ensures portability across deployment platforms.

---

## Development Workflow

### Dependency Management

The project uses **uv** for fast and reliable Python package management. Use a `toml` file for development and export to a `requirements.txt` for containerization.

### Code Quality

**Ruff** linting and formatting enforced via pre-commit hooks (`.pre-commit-config.yaml`). Run `pre-commit install` to enable automatic checks on commit. Aim for zero type errors in the repository.

**IMPORTANT FOR LLM AGENTS**: Do NOT manually run `ruff` or `pre-commit` commands. These are automated via git hooks and CI. Running them wastes time and tokens.

---

## Development Guide

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/unit/test_models.py

# Run tests matching pattern
uv run pytest -k "backtest"

# Run with verbose output
uv run pytest -v

# Run integration tests only
uv run pytest tests/integration/
```

### Database Migrations

**CRITICAL**: Always create migrations when modifying `core/models.py`.

```bash
# Create new migration after model changes
uv run alembic revision --autogenerate -m "description"

# Review the generated migration file in migrations/versions/
# Alembic may miss certain changes (e.g., column renames, custom types)

# Apply migrations
uv run alembic upgrade head

# Rollback one migration
uv run alembic downgrade -1

# View migration history
uv run alembic history

# Check current database version
uv run alembic current
```

### Local Development Setup

```bash
# Start local PostgreSQL (via Docker)
docker-compose up -d

# Run scheduler locally (for testing scheduler changes)
uv run python -m cli scheduler start

# Manual data fetch (for testing without scheduler)
uv run python -m cli fetch current

# Check system status
uv run python -m cli status show

# Interactive Python shell with context
uv run python
>>> from core.database import get_async_session
>>> from core.models import Event, Odds
>>> import asyncio
```

### Database Environment Guidelines

**Four database environments** exist, each with specific purposes:

**Test Database** (`odds_test`):

- **Purpose**: Automated testing only (pytest)
- **Setup**: Automatically managed by pytest fixtures in `tests/conftest.py`
- **Connection**: `postgresql+asyncpg://postgres:dev_password@localhost:5432/odds_test`
- **Behavior**: Tables created/dropped per test run, completely isolated from development data
- **When to use**: Never set manually - pytest uses this automatically

**Local Database** (`odds`):

- **Purpose**: Local development and experimentation
- **Setup**: Docker Compose PostgreSQL container
- **Connection**: Set `DATABASE_URL=${LOCAL_DATABASE_URL}` in `.env`
- **When to use**: Daily development, testing migrations, manual data exploration
- **Start**: `docker-compose up -d`

**Dev Database** (remote, managed):

- **Purpose**: CI/CD integration testing and validation
- **Setup**: Managed PostgreSQL (e.g., Neon dev branch)
- **Behavior**: Used by GitHub Actions dev workflow, **destroyed after each test run** (ephemeral)
- **When to use**: Only via CI/CD pipeline (GitHub Actions), not for manual development
- **Connection**: Set via GitHub secrets (`DATABASE_URL`)

**Production Database** (remote, managed):

- **Purpose**: Live production data collection
- **Setup**: Managed PostgreSQL (Neon/Railway production branch)
- **When to use**: Only via production Lambda functions
- **⚠️ CRITICAL**: **NEVER point DATABASE_URL at production during local development**

**Environment Variable Setup**:

```bash
# Local development (.env file)
DATABASE_URL=${LOCAL_DATABASE_URL}
LOCAL_DATABASE_URL=postgresql+asyncpg://postgres:dev_password@localhost:5432/odds

# Running tests (automatic - no config needed)
# pytest automatically uses odds_test database via conftest.py

# CI/CD (GitHub Actions)
# Uses secrets.DATABASE_URL (dev or prod depending on workflow)

# Production (AWS Lambda)
# Set via Terraform environment variables
```

**Migration Testing Workflow** (CRITICAL - prevents production breakage):

1. Create migration locally: `uv run alembic revision --autogenerate -m "description"`
2. Test against **local database**: `uv run alembic upgrade head`
3. Run migration tests against **test database**: `uv run pytest tests/migration/ -v`
4. Push to GitHub → triggers **dev CI deployment** (ephemeral environment)
5. If dev tests pass → manual approval → **production deployment** via GitHub Actions

**Safety Rules**:

- **NEVER manually run migrations against production database**
- **NEVER set DATABASE_URL to production** during local development
- Test database (`odds_test`) is isolated - safe to drop/recreate anytime
- Local database (`odds`) is for development - can be reset if needed
- Always test migrations in order: local → test → CI dev → prod

### Code Style Guidelines

**Type Hints** (REQUIRED):

- All function signatures must include type hints for parameters and return values
- Use modern syntax: `str | None` (not `Optional[str]`)
- Use `list[T]`, `dict[K, V]` (not `List[T]`, `Dict[K, V]`)

**Async/Await** (REQUIRED):

- All database operations must use `async`/`await`
- Use `AsyncSession` for database sessions (never sync `Session`)
- CLI commands use `asyncio.run()` to execute async functions
- Never mix sync and async database code

**Data Models**:

- Use **SQLModel** for database models (ORM + validation)
- Use **Pydantic BaseModel** for non-database data structures (API responses, configs)
- Use **dataclasses** for simple data containers without validation

### File Modification Workflows

**When modifying `core/models.py` (database schema)**:

1. Make model changes
2. Create Alembic migration: `uv run alembic revision --autogenerate -m "description"`
3. **Review generated migration** (Alembic misses some changes like renames)
4. Test migration: `uv run alembic upgrade head`
5. Update affected queries in `storage/readers.py` or `storage/writers.py`
6. Run tests to ensure nothing broke

**When adding betting strategies**:

1. Create new class in `analytics/strategies.py` inheriting from `BettingStrategy`
2. Implement `evaluate_opportunity()` method
3. Add to `AVAILABLE_STRATEGIES` dict at bottom of `analytics/strategies.py`
4. Register in CLI: Add to `cli/commands/backtest.py` strategy choices
5. Write unit test in `tests/unit/test_strategies.py`
6. Document in backtesting section if complex

**When modifying `core/data_fetcher.py` (API client)**:

1. Preserve retry logic (tenacity decorators on all API calls)
2. Update quota tracking if request patterns change
3. Add new fixtures to `tests/fixtures/` for new endpoints
4. Update integration tests in `tests/integration/`
5. Never remove rate limiting or error handling

**When adding scheduler jobs** (`jobs/*.py`):

1. Create job function (must be async)
2. Add to `core/scheduling/jobs.py` registry (JOB_REGISTRY dict)
3. Implement self-scheduling pattern (calculate and return next execution time)
4. Add error handling (log errors, never crash)
5. Register in CLI: `cli/commands/scheduler.py`
6. Test locally with `odds scheduler start`

**When modifying configuration** (`core/config.py`):

1. Add field to appropriate Config class (with type hint and default)
2. Update `.env.example` with new variable and description
3. Ensure backward compatibility (provide sensible default)
4. Document in Configuration Management section if user-facing
5. Never remove existing config fields (deprecate instead)

### Critical Constraints & Gotchas

**AWS Lambda Limitations**:

- **15-minute max execution time** - Design jobs to complete in <5 minutes
- **Must use NullPool** for database connections (no connection reuse between invocations)
- **No persistent state** - Each invocation is stateless
- Cold start latency ~1-2 seconds (acceptable for this use case)
- EventBridge rules limited to 300 per region

**Timezone Handling** (CRITICAL):

- **Always store UTC** in database: `datetime.now(timezone.utc)`
- **Never use `datetime.now(timezone.utc)()`** - deprecated in Python 3.12+
- **Never use `datetime.now()`** without timezone parameter
- API returns UTC timestamps
- Convert to local time **only for display** (CLI output)

**API Cost Awareness**:

- Each game fetched = **1 quota unit** (regardless of bookmakers/markets requested)
- Historical endpoint costs **10x more** (10 units per game)
- Backfill operations can consume significant quota
- Always check remaining quota: `odds status quota`
- Intelligent scheduling optimizes request usage automatically

**Database Connection Patterns**:

- Always use async context manager: `async with get_async_session() as session`
- **Never share sessions across async tasks** (creates connection pool issues)
- Lambda uses **NullPool** (set automatically when `SCHEDULER_BACKEND=aws`)
- Local/Railway use standard connection pool (default 5 connections)
- Always close sessions properly (context managers handle this)

**Look-Ahead Bias Prevention** (Backtesting):

- **Always use `get_odds_at_time()`** for historical analysis
- Never use latest/current odds when backtesting
- Backtesting enforces `decision_time` (hours before game) to prevent bias
- This is a critical data integrity requirement

**Data Quality & Validation**:

- Validation **logs issues by default**, does not reject data
- Set `REJECT_INVALID_ODDS=true` to enable strict validation (rejects bad data)
- Review `DataQualityLog` table regularly for patterns
- Missing bookmakers/markets are logged but not treated as errors

---

## Deployment

The system supports three scheduler backends for different deployment scenarios:

### AWS Lambda (Primary Production)

- **Cost**: ~$0.20/month for compute (within free tier)
- **Pattern**: Self-scheduling via EventBridge rules
- **Benefits**: Serverless, auto-scaling, minimal maintenance
- **Configuration**: `SCHEDULER_BACKEND=aws`
- **Database**: NullPool connection mode to avoid event loop issues
- **Deployment**: See `deployment/aws/` for Terraform and Lambda configuration

### Railway (Alternative Cloud)

- **Cost**: $5-10/month (app + managed PostgreSQL)
- **Pattern**: Continuous deployment from GitHub
- **Benefits**: Simple setup, managed database, auto-restarts
- **Configuration**: `SCHEDULER_BACKEND=railway`
- **Deployment**: Docker-based, auto-deploys on git push

### Local (Development)

- **Pattern**: APScheduler with long-running process
- **Benefits**: Easy debugging, no cloud dependencies
- **Configuration**: `SCHEDULER_BACKEND=local`
- **Usage**: `odds scheduler start` (runs until Ctrl+C)

**Portability**: Docker-based architecture allows deployment on any platform (DigitalOcean, Azure, GCP, etc.)

### Expected Costs

**API Usage** (The Odds API):

- $25/month for 20,000 requests tier
- Intelligent scheduling optimizes usage based on game schedule
- Typical usage: ~10-15k requests/month during NBA season

**Compute**:

- AWS Lambda: ~$0.20/month (within free tier)
- Railway: ~$5-10/month (includes PostgreSQL)

---

## Testing Strategy

### Test Structure

```bash
tests/
├── unit/
│   ├── test_models.py           # SQLModel validation (implemented)
│   ├── test_validators.py       # Data quality checks (implemented)
│   ├── test_config.py           # Configuration loading (implemented)
│   └── test_backfill_executor.py # Backfill executor tests (implemented)
├── integration/
│   ├── test_database.py         # DB operations (implemented)
│   └── test_backfill_integration.py # Backfill integration tests (implemented)
├── fixtures/
│   ├── sample_odds_response.json     # Implemented
│   └── sample_scores_response.json   # Implemented
└── conftest.py                  # Pytest configuration (implemented)
```

**Backtesting Tests** (Implemented):

- `tests/unit/test_backtesting_models.py` - Data model tests (325 lines)
- `tests/integration/test_backtest_integration.py` - Full backtest workflow (342 lines)
- `tests/unit/test_utils.py` - Utility function tests

**Not Yet Implemented**:

- test_parsers.py
- test_api_client.py
- test_scheduler.py
- historical_data.json fixture

### Key Test Areas

**Data Validation**:

- Schema validation with real API responses
- Edge cases (missing data, invalid odds)
- Data quality checks trigger correctly

**Database Operations**:

- Hybrid storage (raw + normalized) works correctly
- Indexes perform as expected
- Query patterns are efficient

**API Client**:

- Rate limiting prevents quota overruns
- Retry logic handles transient failures
- Error handling logs appropriately

**Backtesting** (Implemented):

- ✓ Look-ahead bias prevention validated
- ✓ Bet sizing calculations verified (Kelly, flat, percentage)
- ✓ Performance metrics accurate (13+ metrics)
- ✓ Strategy execution tested (all 3 strategies)
- ✓ JSON/CSV export and reconstruction tested
- ✓ Empty result handling tested
- ✓ Multi-event backtests with equity curves tested

---

## Data Quality Monitoring

### Validation Checks

**Odds Validation**:

- Price within valid range (-10000 to +10000)
- Timestamp not in future
- Juice/vig reasonable (2-15%)
- No missing required fields
- Outcomes match expected format

**Line Movement Validation**:

- Changes not excessive (e.g., >10 point spread move)
- Timestamps sequential
- Bookmaker consistency

**Completeness Validation**:

- All configured bookmakers present
- All markets present for each bookmaker
- Outcomes complete (both sides of market)

### Quality Logging

All validation issues logged to `DataQualityLog` table with:

- Severity level (warning, error, critical)
- Issue type classification
- Full context for debugging
- Timestamp for analysis

**Action**: Log and flag suspicious data but do not reject. Allow manual review of patterns.

---

## Query Patterns

### Common Analytics Queries

**Note**: Basic query functions are implemented in `storage/readers.py`. Advanced analytics with pandas DataFrame support planned for future `analytics/queries.py` module.

**Implemented in OddsReader**:

```python
async def get_line_movement(event_id: str, bookmaker: str, market: str) -> list[Odds]:
    """
    Return time series of odds changes
    Currently returns list of Odds objects (not DataFrame)
    """

async def get_best_odds(event_id: str, market: str, outcome: str) -> Odds:
    """
    Find highest odds across all bookmakers for specific outcome
    Useful for line shopping
    """

async def get_odds_at_time(
    event_id: str,
    timestamp: datetime,
    tolerance_minutes: int = 5
) -> list[Odds]:
    """
    Get odds snapshot closest to specified time
    Critical for backtesting to prevent look-ahead bias
    """
```

**Planned for analytics/queries.py**:

```python
async def compare_bookmakers(
    event_id: str,
    market: str,
    timestamp: datetime = None
) -> pd.DataFrame:
    """
    Compare odds across all bookmakers for a market
    Columns: bookmaker, outcome, price, point, implied_probability
    """
```

### Backtesting Analytics

**Current Implementation**: Full backtesting system with comprehensive analytics

**Available Metrics** (calculated automatically):

- ROI (Return on Investment)
- Win Rate
- Sharpe Ratio (risk-adjusted returns)
- Sortino Ratio (downside risk)
- Max Drawdown (peak-to-trough decline)
- Profit Factor (gross profit / gross loss)
- Calmar Ratio (annual return / max drawdown)
- Longest winning/losing streaks
- Market-by-market breakdown
- Bookmaker-by-bookmaker breakdown
- Monthly performance breakdown
- Daily equity curve

**Export Formats**:

- JSON (full reconstruction capability)
- CSV (spreadsheet-ready bet records)
- Rich console output (formatted tables)

---

## System Monitoring

System health monitoring via `odds status show` command provides:

- Data pipeline health: last fetch time, success rate, API quota remaining
- Data quality: validation warning rate, missing data percentage
- Storage metrics: event counts, odds records, database size
- Database connection status

Run `odds status show --verbose` for detailed statistics.

---

## Logging Configuration

System uses **structlog** for structured, machine-readable logging. Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL. Logs written to console (JSON format) and rotating files with 30-day retention. Configuration via `LOG_LEVEL` and `LOG_FILE` environment variables.

---

## Performance Considerations

**Database**: Composite indexes on event_id/timestamp/bookmaker for efficient queries. Connection pooling (5 connections, or NullPool for Lambda). Batch inserts for backfill operations.

**API**: Exponential backoff with tenacity library (max 3 retries). Quota tracking prevents overrun. Intelligent scheduling optimizes request usage.

---

## Operational Notes

**Backups**: Managed database providers (Neon/Railway) handle automated backups with point-in-time recovery.

**Recovery**: Data loss addressed via backfill. API quota managed by intelligent scheduling. Docker containers auto-restart on failure.

**Security**: API keys in environment variables only. SQLModel provides parameterized queries. Regular database backups.

---

## Extensibility

The system is designed for easy extension:

**Adding Sports**: Add sport key to `SPORTS` config list. No code changes required - same schema supports all sports.

**Adding Bookmakers**: Add bookmaker key to `BOOKMAKERS` config list. Automatic inclusion in data collection.

**Adding Markets**: Add market key to `MARKETS` config list. Schema supports arbitrary bet types.

**Adding Strategies**: Inherit from `BettingStrategy` base class and implement `evaluate_opportunity()` method. See `analytics/strategies.py` and ML example in `analytics/ml_strategy_example.py`.

Monorepo structure supports future microservices extraction, API layer addition (FastAPI), and integration with external tools (Jupyter, BI platforms).
