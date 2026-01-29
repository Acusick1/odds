# Database Reference

> Reference documentation for database models, environments, migrations, and query patterns.

## Database Schema

All models defined in `packages/odds-core/odds_core/models.py` using SQLModel.

### Core Tables

**Event** (`events`):
- Stores NBA games with teams, commence time, status, final scores
- Statuses: SCHEDULED → LIVE → FINAL (or CANCELLED/POSTPONED)
- Primary key: API event ID (string)

**OddsSnapshot** (`odds_snapshots`):
- Stores complete API responses as JSON for each fetch
- Tracks fetch tier (opening/early/sharp/pregame/closing) and timing
- Used for debugging, auditing, and ML feature engineering

**Odds** (`odds`):
- Normalized odds records for efficient querying
- One row per bookmaker/market/outcome combination
- Tracks price (American odds), point (spread/total), timestamps

**DataQualityLog** (`data_quality_logs`):
- Validation issues and warnings
- Severities: warning, error, critical
- Issue types: missing_data, suspicious_odds, stale_timestamp, etc.

**FetchLog** (`fetch_logs`):
- Records each API fetch operation
- Tracks success/failure, quota usage, response times

## Storage Strategy: Hybrid Approach

**Raw Data (OddsSnapshot):** Complete API responses as JSON for:
- Debugging and auditing
- Schema flexibility
- Exact data preservation

**Normalized Data (Odds):** Individual records for:
- Efficient querying and filtering
- Time-series analysis
- Standard SQL operations

## Database Environments

### Test Database (`odds_test`)

- **Purpose:** Automated testing only (pytest)
- **Setup:** Managed by pytest fixtures in `tests/conftest.py`
- **Connection:** `postgresql+asyncpg://postgres:postgres@localhost:5432/odds_test`
- **Behavior:** Tables created/dropped per test run
- **Usage:** Never set manually - pytest uses automatically

### Local Database (`odds`)

- **Purpose:** Local development and experimentation
- **Setup:** Docker Compose PostgreSQL container (postgres:15)
- **Host:** localhost:5433 (not 5432 - avoids conflicts with system postgres)
- **Credentials:** postgres/postgres
- **Database:** odds
- **Start:** `docker compose up -d`

### Dev Database (remote)

- **Purpose:** CI/CD integration testing
- **Setup:** Managed PostgreSQL (e.g., Neon dev branch)
- **Behavior:** Destroyed after each test run (ephemeral)
- **Usage:** Only via CI/CD pipeline (GitHub Actions)

### Production Database (remote)

- **Purpose:** Live production data collection
- **Setup:** Managed PostgreSQL (Neon/Railway production branch)
- **Usage:** Only via production Lambda functions
- **CRITICAL:** NEVER point DATABASE_URL at production during local development

### Environment Variable Setup

**Local development** (`.env` file):
- `DATABASE_URL=${LOCAL_DATABASE_URL}`
- `LOCAL_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/odds`

**Tests:** Automatic - pytest uses `odds_test` via `conftest.py`

**CI/CD:** Uses `secrets.DATABASE_URL` in GitHub Actions

**Production:** Set via Terraform for AWS Lambda

## Database Migrations

**CRITICAL:** Always create migrations when modifying `packages/odds-core/odds_core/models.py`.

```bash
# Create migration after model changes
uv run alembic revision --autogenerate -m "description"

# Review generated migration (Alembic may miss renames, custom types)

# Apply migrations
uv run alembic upgrade head

# Rollback one migration
uv run alembic downgrade -1

# View history
uv run alembic history

# Check current version
uv run alembic current
```

### Migration Testing Workflow

1. Create migration locally: `uv run alembic revision --autogenerate -m "description"`
2. Test against local database: `uv run alembic upgrade head`
3. Run migration tests: `uv run pytest tests/migration/ -v`
4. Push to GitHub -> triggers dev CI deployment
5. If dev tests pass -> manual approval -> production deployment

### Safety Rules

- NEVER manually run migrations against production
- NEVER set DATABASE_URL to production during local development
- Test database (`odds_test`) is isolated - safe to drop/recreate
- Local database (`odds`) can be reset if needed
- Test migrations in order: local -> test -> CI dev -> prod

## Query Patterns

### OddsReader Methods

See `packages/odds-lambda/odds_lambda/storage/readers.py` for complete implementations.

**Key methods:**
- `get_odds_at_time()` - Get odds at specific timestamp (CRITICAL for backtesting to prevent look-ahead bias)
- `get_line_movement()` - Time series of odds changes for a bookmaker/market
- `get_best_odds()` - Highest odds across all bookmakers (line shopping)
- `get_events_by_date_range()` - Query events within date range, optionally filter by status

## Performance Considerations

### Indexes

Composite indexes on frequently queried combinations:
- `(event_id, snapshot_time)` - Time-series queries
- `(event_id, fetch_tier)` - Tier-based analysis
- `(event_id, bookmaker_key, market_key)` - Odds lookups
- `(bookmaker_key, odds_timestamp)` - Bookmaker analysis

### Connection Pooling

- Default: 5 connections
- Lambda: NullPool (no connection reuse)
- Always use async context manager: `async with get_async_session() as session`
- Never share sessions across async tasks

### Batch Operations

Use `bulk_insert_historical()` for backfill operations to reduce round trips.
