# Betting Odds Data Pipeline - Technical Specification

IMPORTANT (LLM agents): This document is read-only and should only be modified by the user.

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
- **APScheduler**: Job scheduling for periodic data collection
- **Typer + Rich**: CLI interface with formatted output
- **Pydantic Settings**: Configuration management via environment variables
- **Alembic**: Database migrations
- **pytest + pytest-asyncio**: Testing framework
- **structlog**: Structured logging

### Deployment
- **Docker + docker-compose**: Local development containerization
- **Railway**: Production hosting (alternative: any Docker-compatible platform)
- **PostgreSQL**: Managed database service

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
```
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
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
```

#### OddsSnapshot Model (Raw Data Preservation)
```python
class OddsSnapshot(SQLModel, table=True):
    __tablename__ = "odds_snapshots"
    
    id: int | None = Field(default=None, primary_key=True)
    event_id: str = Field(foreign_key="events.id", index=True)
    snapshot_time: datetime = Field(index=True)
    
    # Full API response stored as JSONB
    raw_data: dict = Field(sa_column=Column(JSONB))
    
    # Quick statistics
    bookmaker_count: int
    api_request_id: str | None = None  # For debugging
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    __table_args__ = (
        Index('ix_event_snapshot_time', 'event_id', 'snapshot_time'),
    )
```

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
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
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
    raw_data: dict | None = Field(sa_column=Column(JSONB))
    
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
```

#### FetchLog Model
```python
class FetchLog(SQLModel, table=True):
    __tablename__ = "fetch_logs"
    
    id: int | None = Field(default=None, primary_key=True)
    fetch_time: datetime = Field(default_factory=datetime.utcnow, index=True)
    
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

**Raw Data (OddsSnapshot)**: Complete API responses stored as JSONB for:
- Debugging and auditing
- Schema flexibility
- Exact data preservation

**Normalized Data (Odds)**: Individual odds records for:
- Efficient querying and filtering
- Time-series analysis
- Standard SQL operations
- Performance optimization

---

## System Architecture

### Component Structure

```
betting-odds-system/
├── core/
│   ├── models.py              # SQLModel schema definitions
│   ├── database.py            # Database connection & session management
│   ├── data_fetcher.py        # The Odds API client
│   └── config.py              # Configuration management
│
├── storage/
│   ├── writers.py             # Write operations (inserts, updates)
│   ├── readers.py             # Read operations (queries)
│   └── validators.py          # Data quality validation
│
├── scheduler/
│   ├── main.py                # APScheduler orchestration
│   └── jobs.py                # Job definitions
│
├── analytics/
│   └── queries.py             # Common query patterns
│
├── alerts/
│   └── base.py                # Alert infrastructure (for future use)
│
├── cli/
│   ├── main.py                # Typer CLI entry point
│   └── commands/
│       ├── fetch.py           # Data collection commands
│       ├── status.py          # System health monitoring
│       ├── backfill.py        # Historical data collection
│       └── validate.py        # Data quality checks
│
├── migrations/                # Alembic database migrations
├── tests/                     # Test suite
├── docker-compose.yml         # Local development setup
├── Dockerfile                 # Container definition
└── requirements.txt           # Python dependencies
```

### Data Collection Pipeline

**Flow**:
1. **Scheduler** triggers fetch job at configured interval
2. **API Client** fetches odds data with rate limiting
3. **Validator** checks data quality and logs issues
4. **Writer** stores both raw (JSONB) and normalized data
5. **Logger** records fetch success/failure and quota usage

### API Client Design

```python
class TheOddsAPIClient:
    """
    Handles all interactions with The Odds API
    
    Features:
    - Rate limiting to respect quota
    - Retry logic with exponential backoff
    - Error handling and logging
    - Quota tracking
    """
    
    async def get_odds(sport, regions, markets, odds_format) -> dict
    async def get_scores(sport, days_from) -> dict
    async def get_historical_odds(sport, date, regions, markets) -> dict
```

### Data Validation

```python
class OddsValidator:
    """
    Data quality checks applied to all incoming data
    
    Checks:
    - Odds within valid range (-10000 to +10000)
    - Timestamps not in future
    - Reasonable juice/vig (2-15%)
    - Line movement not excessive
    - Required fields present
    - No duplicate outcomes
    
    Behavior: Log warnings and flag suspicious data, do not reject
    """
    
    @staticmethod
    def validate_odds_snapshot(data: dict) -> tuple[bool, list[str]]
```

### Storage Operations

```python
class OddsWriter:
    """
    All write operations to database
    
    Operations:
    - store_odds_snapshot(): Hybrid storage of raw + normalized
    - upsert_event(): Insert or update event details
    - bulk_insert_historical(): Efficient backfill operations
    - log_data_quality_issue(): Record validation problems
    """

class OddsReader:
    """
    Common query patterns
    
    Operations:
    - get_odds_at_time(): Snapshot at specific timestamp
    - get_line_movement(): Time series for event
    - get_events_by_date_range(): Event queries
    - get_best_odds(): Best price across bookmakers
    """
```

---

## Scheduler Configuration

### Sampling Modes

The system supports two sampling modes with easy switching:

#### Fixed Sampling (Default)
- Fetch odds at regular intervals regardless of game timing
- **Default interval**: 30 minutes
- Simpler logic, predictable API usage
- Configuration:
  ```python
  SAMPLING_MODE = "fixed"
  FIXED_INTERVAL_MINUTES = 30
  ```

#### Adaptive Sampling (Built-in, Disabled by Default)
- Adjust frequency based on proximity to game time
- More frequent as games approach
- Configuration:
  ```python
  SAMPLING_MODE = "adaptive"
  ADAPTIVE_INTERVALS = {
      "far": 30,      # >24h from game: every 30 minutes
      "medium": 15,   # 3-24h from game: every 15 minutes
      "near": 5,      # 1-3h from game: every 5 minutes
      "imminent": 1   # <1h from game: every 1 minute
  }
  ```

### Scheduled Jobs

**Primary Job - Fetch Odds**:
- Frequency: Based on sampling mode configuration
- Fetches all upcoming NBA games
- Stores snapshots with validation
- Logs API usage

**Secondary Job - Fetch Scores**:
- Frequency: Every 6 hours
- Updates completed game results
- Updates event status

**Tertiary Job - Update Status**:
- Frequency: Every hour
- Updates event statuses (scheduled → live → final)
- Stops fetching odds for completed games

### Event Lifecycle

- **SCHEDULED**: Odds actively being fetched
- **LIVE**: Game in progress (stop fetching odds)
- **FINAL**: Game completed with results stored
- **CANCELLED/POSTPONED**: No further data collection

---

## Configuration Management

### Settings Schema

```python
class Settings(BaseSettings):
    # API Configuration
    odds_api_key: str
    odds_api_base_url: str = "https://api.the-odds-api.com/v4"
    odds_api_quota: int = 20_000
    
    # Database
    database_url: str
    database_pool_size: int = 5
    
    # Data Collection
    sports: list[str] = ["basketball_nba"]
    bookmakers: list[str] = [
        "pinnacle",
        "circasports", 
        "draftkings",
        "fanduel",
        "betmgm",
        "williamhill_us",  # Caesars
        "betrivers",
        "bovada"
    ]
    markets: list[str] = ["h2h", "spreads", "totals"]
    regions: list[str] = ["us"]
    
    # Sampling Configuration
    sampling_mode: str = "fixed"  # "fixed" or "adaptive"
    fixed_interval_minutes: int = 30
    adaptive_intervals: dict = {
        "far": 30,
        "medium": 15,
        "near": 5
    }
    
    # Data Quality
    enable_validation: bool = True
    reject_invalid_odds: bool = False  # Log but don't reject
    
    # Alerts (infrastructure for future use)
    discord_webhook_url: str | None = None
    alert_enabled: bool = False
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/odds_pipeline.log"
    
    class Config:
        env_file = ".env"
```

### Environment Variables

Required `.env` file structure:
```
# API
ODDS_API_KEY=your_key_here
ODDS_API_BASE_URL=https://api.the-odds-api.com/v4

# Database
DATABASE_URL=postgresql://user:password@host:port/database

# Sampling
SAMPLING_MODE=fixed
FIXED_INTERVAL_MINUTES=30

# Logging
LOG_LEVEL=INFO

# Optional - Alerts (future use)
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
ALERT_ENABLED=false
```

---

## CLI Interface

### Command Structure

**Data Collection**:
```bash
odds fetch                              # Manual fetch current odds
odds fetch --sport basketball_nba       # Specific sport
odds backfill --start YYYY-MM-DD --end YYYY-MM-DD --sample N
```

**Status & Monitoring**:
```bash
odds status                             # System health overview
odds status --verbose                   # Detailed statistics
odds validate                           # Run data quality checks
odds quota                              # Check API usage remaining
```

**Data Inspection**:
```bash
odds events --days 7                    # List recent events
odds events --team "Lakers"             # Filter by team
odds show-odds --event-id abc123        # View odds for specific event
odds line-movement --event-id abc123 --bookmaker fanduel
```

**Configuration**:
```bash
odds config show                        # Display current config
odds config set-mode adaptive           # Switch sampling mode
odds config test-alerts                 # Test alert delivery
```

**Database Management**:
```bash
odds db migrate                         # Run pending migrations
odds db stats                           # Database statistics
odds db clean --older-than 90           # Archive old data
```

**Scheduler Control**:
```bash
odds scheduler start                    # Start background jobs
odds scheduler stop                     # Stop background jobs  
odds scheduler status                   # Check if running
```

### CLI Output Formatting

Use **Rich** library for:
- Colored, formatted tables
- Progress bars for backfill operations
- Status indicators (✓, ✗, ⚠)
- Syntax highlighting for JSON/config display

---

## Historical Data Backfill

### Strategy

**Initial Backfill Target**: Sample of last season's regular season games

**Sampling Approach**:
- Select representative games distributed across season
- Include all 30 teams proportionally
- Mix of different matchup types
- ~20% sample rate (every 5th game)

**Selection Criteria**:
```python
def select_sample_games(season_games: list, sample_rate: float = 0.2):
    """
    Criteria:
    - Even distribution across season dates
    - Proportional team representation
    - Variety in matchups and game importance
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

### Strategy Pattern

```python
class BettingStrategy(ABC):
    """Base class for all betting strategies"""
    
    @abstractmethod
    def should_bet(self, event: Event, odds_snapshot: dict) -> bool:
        """Determine if strategy triggers a bet"""
    
    @abstractmethod
    def calculate_stake(self, bankroll: float, opportunity: dict) -> float:
        """Calculate bet size"""

# Example implementations
class ArbitrageStrategy(BettingStrategy):
    """Risk-free arbitrage betting"""
    def __init__(self, min_profit_margin: float = 0.01):
        self.min_profit_margin = min_profit_margin

class EVStrategy(BettingStrategy):
    """Positive expected value vs sharp lines"""
    def __init__(self, min_ev: float = 0.03, sharp_book: str = "pinnacle"):
        self.min_ev = min_ev
        self.sharp_book = sharp_book
```

### Backtesting Engine

```python
class BacktestEngine:
    """
    Simulate betting strategies against historical data
    
    Features:
    - Look-ahead bias prevention (use odds from before game time)
    - Realistic bet sizing (Kelly, flat, percentage)
    - Transaction cost modeling (optional)
    - Bankroll management
    - Performance metrics calculation
    """
    
    def __init__(
        self,
        strategy: BettingStrategy,
        initial_bankroll: float = 10000,
        start_date: datetime = None,
        end_date: datetime = None
    ):
        pass
    
    async def run(self) -> BacktestResult:
        """
        Execute backtest:
        1. Query historical odds at decision times
        2. Apply strategy to determine bets
        3. Query actual results
        4. Calculate profit/loss
        5. Track bankroll evolution
        6. Compute performance metrics
        """

@dataclass
class BacktestResult:
    """Complete backtest performance metrics"""
    strategy_name: str
    total_bets: int
    winning_bets: int
    losing_bets: int
    total_wagered: float
    total_profit: float
    roi: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    average_odds: float
```

### Bet Sizing Approaches

**Standard Method: Fractional Kelly**
- Industry standard for professional betting
- Formula: stake = (kelly_fraction × edge × bankroll) / odds
- Recommended fraction: 0.25 (quarter-Kelly)
- Provides ~75% of full Kelly returns with lower volatility

**Alternative Methods**:
- Flat betting: Fixed dollar amount per bet
- Percentage: Fixed percentage of current bankroll
- Full Kelly: Optimal but high variance

---

## Docker Configuration

### Local Development

```yaml
# docker-compose.yml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: odds
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: dev_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  app:
    build: .
    depends_on:
      - postgres
    environment:
      DATABASE_URL: postgresql://postgres:dev_password@postgres:5432/odds
      ODDS_API_KEY: ${ODDS_API_KEY}
      SAMPLING_MODE: fixed
      FIXED_INTERVAL_MINUTES: 30
    volumes:
      - .:/app
    command: python scheduler/main.py

volumes:
  postgres_data:
```

### Container Definition

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Default command (can be overridden)
CMD ["python", "scheduler/main.py"]
```

---

## Deployment

### Railway Configuration

Railway deployment is Docker-based, making migration to any platform trivial.

**Setup**:
1. Connect GitHub repository to Railway
2. Add PostgreSQL addon
3. Configure environment variables
4. Auto-deploy on git push

**Environment Variables** (Railway dashboard):
- `ODDS_API_KEY`
- `DATABASE_URL` (auto-populated by Railway)
- `SAMPLING_MODE`
- `FIXED_INTERVAL_MINUTES`
- All other configuration from Settings class

**Migration Path**: Since deployment is Docker-based, the same image runs on:
- Railway (current choice)
- DigitalOcean / AWS / Azure
- Any Docker-compatible hosting
- Local machine

---

## Cost Analysis

### Expected Monthly Costs

**30-minute Fixed Sampling**:
- NBA games per day: ~10 average
- Fetches per day: 48 (every 30 minutes)
- Objects per day: ~480
- Objects per month: ~14,400
- **API Cost**: $25/month (20k request tier)

**Railway Hosting**:
- App + PostgreSQL: $7-10/month
- **Hosting Cost**: $7-10/month

**Total Monthly**: $32-35/month

### One-time Costs

**Historical Backfill**:
- Sample 20% of last season's games: ~250 events
- Estimated cost: ~$25 one-time

**Total Initial Setup**: $57-60

### Cost Scaling

If sampling frequency changes:
- 15-minute intervals: ~29k requests/month → $50/month API
- 10-minute intervals: ~43k requests/month → $75/month API
- 5-minute intervals: ~86k requests/month → $150/month API

Storage scales linearly but remains negligible (~50-100 MB/month).

---

## Testing Strategy

### Test Structure

```
tests/
├── unit/
│   ├── test_models.py           # SQLModel validation
│   ├── test_validators.py       # Data quality checks
│   ├── test_parsers.py          # API response parsing
│   └── test_config.py           # Configuration loading
├── integration/
│   ├── test_database.py         # DB operations
│   ├── test_api_client.py       # API client with mocking
│   ├── test_scheduler.py        # Job execution
│   └── test_backtest.py         # Backtesting engine
└── fixtures/
    ├── sample_odds_response.json
    ├── sample_scores_response.json
    └── historical_data.json
```

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

**Backtesting**:
- Look-ahead bias prevention works
- Bet sizing calculations correct
- Performance metrics accurate

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

**Line Movement**:
```python
async def get_line_movement(event_id: str, bookmaker: str, market: str) -> pd.DataFrame:
    """
    Return time series of odds changes for visualization
    Columns: timestamp, outcome, price, point
    """
```

**Best Odds**:
```python
async def get_best_odds(event_id: str, market: str, outcome: str) -> Odds:
    """
    Find highest odds across all bookmakers for specific outcome
    Useful for line shopping
    """
```

**Odds at Decision Time**:
```python
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

**Market Comparison**:
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

---

## Security Considerations

### API Key Management
- Store in environment variables only
- Never commit to version control
- Use separate keys for development/production

### Database Access
- Use connection pooling
- Parameterized queries (SQLModel handles this)
- Regular backups

### Logging
- Sanitize sensitive data from logs
- Rotate log files
- Monitor for unusual patterns

---

## Extensibility Design

### Adding New Sports
1. Add sport key to configuration
2. No code changes required
3. Same schema supports all sports

### Adding New Bookmakers
1. Add bookmaker key to configuration
2. No code changes required
3. Automatic inclusion in data collection

### Adding New Markets
1. Add market key to configuration
2. Schema supports arbitrary markets
3. Validation adapts automatically

### Adding New Strategies
1. Inherit from `BettingStrategy` base class
2. Implement `should_bet()` and `calculate_stake()`
3. Register with backtest engine

---

## System Monitoring

### Health Checks

**Data Pipeline Health**:
- Last successful fetch time
- Fetch success rate (last 24 hours)
- API quota remaining
- Database connection status

**Data Quality Metrics**:
- Validation warning rate
- Missing data percentage
- Stale bookmaker data detection

**Storage Metrics**:
- Total events stored
- Total odds records
- Database size
- Growth rate

### CLI Status Output

```
╔══════════════════════════════════════════════╗
║        Odds Pipeline Status                  ║
╠══════════════════════════════════════════════╣
║ Last Fetch:         2025-10-16 14:30:00 UTC ║
║ Status:             ✓ Healthy                ║
║                                              ║
║ Events (Scheduled): 12                       ║
║ Events (Total):     8,429                    ║
║ Odds Records:       2,847,392                ║
║                                              ║
║ API Quota Used:     12,847 / 20,000         ║
║ Quota Remaining:    7,153 (35.8%)           ║
║                                              ║
║ DB Size:            1.2 GB                   ║
║ Fetch Success Rate: 99.2% (24h)             ║
║ Validation Warnings: 23 (0.1%)              ║
╚══════════════════════════════════════════════╝
```

---

## Logging Configuration

### Structured Logging

Use `structlog` for machine-readable logs:

```python
import structlog

logger = structlog.get_logger()

# Log with context
logger.info(
    "odds_fetched",
    event_id="abc123",
    bookmakers=8,
    markets=3,
    elapsed_ms=234
)
```

### Log Levels

- **DEBUG**: Detailed technical information
- **INFO**: Normal operations (fetches, processing)
- **WARNING**: Data quality issues, retries
- **ERROR**: Failed operations, invalid data
- **CRITICAL**: System failures, quota exceeded

### Log Output

- Console: Structured JSON for production
- File: Rotating daily logs
- Retention: 30 days

---

## Performance Considerations

### Database Optimization

**Indexes**:
- Composite indexes on frequently joined columns
- Time-based indexes for range queries
- Bookmaker/market indexes for filtering

**Connection Pooling**:
- Pool size: 5 connections (single-user system)
- Async operations prevent blocking

**Query Optimization**:
- Use indexes for all time-range queries
- Batch insert for backfill operations
- EXPLAIN ANALYZE for slow queries

### API Rate Limiting

**Rate Limiter Implementation**:
- Track requests per month
- Prevent quota overrun
- Warn at 80% usage
- Automatic throttling near limit

**Backoff Strategy**:
- Exponential backoff on API errors
- Max retry attempts: 3
- Jitter to prevent thundering herd

---

## Disaster Recovery

### Backup Strategy

**Database Backups**:
- Automated daily backups (Railway provides this)
- Point-in-time recovery capability
- Export critical data periodically

**Configuration Backups**:
- Store `.env.example` in repository
- Document all settings
- Version control migrations

### Recovery Procedures

**Data Loss**:
- Re-run backfill for missing historical data
- Resume normal collection immediately
- Validate data consistency

**API Quota Exhaustion**:
- Automatic throttling prevents overrun
- If exceeded, wait for monthly reset
- Reduce sampling frequency temporarily

**System Failure**:
- Docker containers restart automatically
- Railway has auto-restart
- Monitor health checks

---

## Future Considerations

### Potential Enhancements

**Data Collection**:
- Player props markets
- Alternate lines
- Live betting odds
- Additional sports

**Analytics**:
- Advanced arbitrage detection
- Closing line value tracking
- Expected value calculators
- Statistical models

**Interface**:
- Web dashboard for visualization
- Mobile notifications
- Real-time alerting system

**Performance**:
- Redis caching layer
- WebSocket connections for live data
- Query result caching

### Architecture Evolution

Current monorepo structure supports:
- Extraction of modules to microservices if needed
- Addition of API layer (FastAPI) if web interface desired
- Horizontal scaling of workers for multiple sports
- Integration with external tools (Jupyter, BI platforms)

---

## References

### API Documentation
- The Odds API: https://the-odds-api.com/liveapi/guides/v4/
- Historical Odds: https://the-odds-api.com/historical-odds-data/

### Technology Documentation
- SQLModel: https://sqlmodel.tiangolo.com/
- PostgreSQL JSONB: https://www.postgresql.org/docs/current/datatype-json.html
- APScheduler: https://apscheduler.readthedocs.io/
- Typer: https://typer.tiangolo.com/
- Rich: https://rich.readthedocs.io/

### Deployment
- Railway: https://docs.railway.app/
- Docker: https://docs.docker.com/