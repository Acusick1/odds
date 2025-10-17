# Betting Odds Pipeline - Implementation Plan

## Phase 1: Foundation (Days 1-2)

### 1.1 Project Setup
- [ ] Initialize Python project structure with uv
- [ ] Create `pyproject.toml` for dependency management
- [ ] Export `requirements.txt` for Docker containerization
- [ ] Set up `.env.example` and `.gitignore`
- [ ] Create `docker-compose.yml` for local PostgreSQL
- [ ] Set up ruff configuration for linting/formatting
- [ ] Configure pre-commit hooks with ruff
- [ ] Install pre-commit hooks (`pre-commit install`)

### 1.2 Configuration & Database
- [ ] Implement `core/config.py` with Pydantic Settings
- [ ] Implement `core/database.py` with async SQLAlchemy engine
- [ ] Define all SQLModel schemas in `core/models.py`
- [ ] Initialize Alembic and create initial migration
- [ ] Test database connection and model creation

## Phase 2: Data Collection (Days 3-4)

### 2.1 API Client
- [ ] Implement `core/data_fetcher.py` (TheOddsAPIClient)
  - `get_odds()` method with rate limiting
  - `get_scores()` method
  - Error handling and retry logic using tenacity (exponential backoff)
  - Quota tracking from response headers
- [ ] Test API client with real API calls

### 2.2 Data Validation
- [ ] Implement `storage/validators.py` (OddsValidator)
  - Odds range validation (-10000 to +10000)
  - Timestamp validation (not in future)
  - Juice/vig checks (2-15%)
  - Completeness checks (all markets/outcomes present)
- [ ] Test validation with edge cases

### 2.3 Storage Operations
- [ ] Implement `storage/writers.py`
  - `store_odds_snapshot()`: Hybrid raw + normalized storage
  - `upsert_event()`: Event insert/update
  - `log_fetch()`: FetchLog creation
  - `log_data_quality_issue()`: DataQualityLog creation
- [ ] Implement `storage/readers.py`
  - `get_events_by_date_range()`
  - `get_odds_at_time()`: Critical for backtesting
  - `get_line_movement()`: Time series queries
- [ ] Test storage operations with sample data

## Phase 3: Scheduler & CLI (Days 5-6)

### 3.1 Scheduler
- [ ] Implement `scheduler/jobs.py`
  - Fetch odds job (respects sampling mode)
  - Fetch scores job (every 6 hours)
  - Update event status job (hourly)
- [ ] Implement `scheduler/main.py` with APScheduler
- [ ] Test scheduler with short intervals

### 3.2 CLI - Core Commands
- [ ] Implement `cli/main.py` with Typer app
- [ ] Implement `cli/commands/fetch.py`
  - `odds fetch`: Manual fetch current odds
  - `odds fetch --sport basketball_nba`
- [ ] Implement `cli/commands/status.py`
  - `odds status`: System health overview with Rich tables
  - `odds quota`: API usage remaining
- [ ] Implement `cli/commands/validate.py`
  - `odds validate`: Run data quality checks
- [ ] Test CLI commands

## Phase 4: Historical Backfill (Days 7-8)

### 4.1 Backfill Implementation
- [ ] Implement `cli/commands/backfill.py`
  - Date range backfill with progress bars
  - Sample rate selection (default 20%)
  - Batch processing for efficiency
- [ ] Add historical endpoint support to API client
- [ ] Test backfill with small date range

### 4.2 Data Inspection Commands
- [ ] Implement `odds events --days 7`
- [ ] Implement `odds show-odds --event-id <id>`
- [ ] Implement `odds line-movement --event-id <id> --bookmaker <key>`

## Phase 5: Alert Infrastructure (Day 9)

### 5.1 Alert System (Disabled by Default)
- [ ] Implement `alerts/base.py`
  - `AlertBase` abstract class
  - `DiscordAlert` implementation
  - `AlertManager` for routing
- [ ] Add alert configuration to Settings
- [ ] Test alert delivery (when enabled)

## Phase 6: Backtesting Foundation (Days 10-11)

### 6.1 Strategy Pattern
- [ ] Implement `analytics/backtest.py`
  - `BettingStrategy` abstract base class
  - `BacktestEngine` with look-ahead bias prevention
  - `BacktestResult` dataclass
- [ ] Implement example strategies
  - `ArbitrageStrategy`
  - `EVStrategy` (vs Pinnacle)
- [ ] Implement bet sizing methods (Kelly, flat, percentage)

### 6.2 Analytics Queries
- [ ] Implement `analytics/queries.py`
  - `get_best_odds()`: Line shopping
  - `compare_bookmakers()`: Market comparison
  - Additional utility queries

## Phase 7: Testing & Documentation (Days 12-13)

### 7.1 Testing
- [ ] Unit tests for models, validators, parsers
- [ ] Integration tests for database operations
- [ ] Integration tests for API client (with mocking)
- [ ] Create test fixtures from real API responses

### 7.2 Documentation
- [ ] Add docstrings to all public functions
- [ ] Create README.md with setup instructions
- [ ] Document environment variables
- [ ] Add usage examples

## Phase 8: Deployment (Day 14)

### 8.1 Docker & Deployment
- [ ] Create `Dockerfile`
- [ ] Test local docker-compose setup
- [ ] Set up Railway project
- [ ] Add PostgreSQL addon on Railway
- [ ] Configure environment variables
- [ ] Deploy and verify scheduler runs
- [ ] Monitor first 24 hours of data collection

## Phase 9: Initial Operation (Days 15-30)

### 9.1 Data Collection & Validation
- [ ] Run scheduler continuously for NBA games
- [ ] Monitor data quality logs
- [ ] Review API quota usage
- [ ] Verify database growth is as expected

### 9.2 Historical Backfill
- [ ] Execute backfill for sample of last season (~250 games)
- [ ] Validate historical data quality
- [ ] Test backtest engine with historical data

## Future Enhancements (Post-MVP)

### Not in Initial Scope
- [ ] Web dashboard (FastAPI + frontend)
- [ ] Real-time alerting (arbitrage, EV opportunities)
- [ ] Player props and alternate lines
- [ ] Additional sports beyond NBA
- [ ] Redis caching layer
- [ ] Advanced statistical models

---

## Critical Path Dependencies

```
Phase 1 (Setup)
    ↓
Phase 2 (Data Collection)
    ↓
Phase 3 (Scheduler & CLI)
    ↓
Phase 4 (Backfill) → Phase 5 (Alerts) → Phase 6 (Backtesting)
    ↓
Phase 7 (Testing)
    ↓
Phase 8 (Deployment)
    ↓
Phase 9 (Operation)
```

## Quick Start Commands (After Implementation)

```bash
# Setup with uv
uv venv                                       # Create virtual environment
source .venv/bin/activate                     # Activate (Linux/Mac)
uv pip install -e .                           # Install in editable mode
pre-commit install                            # Install pre-commit hooks

# Local development
docker-compose up -d postgres                 # Start PostgreSQL
alembic upgrade head                          # Run migrations

# Development commands
odds fetch                                    # Test manual fetch
odds status                                   # Check system health
odds scheduler start                          # Start background jobs

# Code quality (automatically via pre-commit)
pre-commit run --all-files                    # Manual check all files
# Note: ruff runs automatically on git commit

# Backfill historical data
odds backfill --start 2024-10-01 --end 2024-10-15 --sample 0.2

# Production
docker-compose up -d                          # Run full stack
```

## Key Files to Create (35 files)

```
Core (4):        config.py, database.py, models.py, data_fetcher.py
Storage (3):     writers.py, readers.py, validators.py
Scheduler (2):   main.py, jobs.py
CLI (5):         main.py, fetch.py, status.py, backfill.py, validate.py
Analytics (2):   queries.py, backtest.py
Alerts (1):      base.py
Config (9):      pyproject.toml, requirements.txt, .env.example, .gitignore,
                 docker-compose.yml, Dockerfile, alembic.ini,
                 .pre-commit-config.yaml, ruff.toml (or pyproject.toml section)
Tests (8):       test_models.py, test_validators.py, test_database.py, test_api_client.py, + fixtures
Docs (1):        README.md
```

## Time Estimate
- **MVP (Phases 1-8)**: 14 days
- **Initial Operation (Phase 9)**: 15-30 days
- **Total to Stable System**: ~30 days

## Success Criteria
1. ✓ Scheduler collects odds every 30 minutes
2. ✓ Data validation catches and logs quality issues
3. ✓ Database contains both raw and normalized data
4. ✓ CLI provides easy system monitoring
5. ✓ Backtest engine works with historical data
6. ✓ System runs autonomously on Railway
7. ✓ Monthly costs under $40
8. ✓ Pre-commit hooks enforce code quality (ruff)
9. ✓ Zero type errors in repository
