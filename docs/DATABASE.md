# Database Reference

> Reference documentation for database models, environments, migrations, and query patterns.

## Database Schema

All models live in `packages/odds-core/odds_core/`.

### Core Tables

Defined in `models.py`:

**Event** (`events`):
- Stores games with teams, commence time, status, final scores
- Supports multiple sports via `sport_key` (`soccer_epl`, `baseball_mlb`, `basketball_nba`)
- Statuses: SCHEDULED → LIVE → FINAL (or CANCELLED/POSTPONED); see
  [Event Lifecycle](ARCHITECTURE.md#event-lifecycle)
- Primary key (`id`), whose form identifies the source:

  | Source | Pattern |
  |--------|---------|
  | The Odds API | hex UUID from the API |
  | OddsPortal (historical ingest) | `op_{season}_{HOME}_{AWAY}_{date}` |
  | OddsPortal (live scrape) | `op_live_{HOME}_{AWAY}_{date}` |
  | football-data.co.uk | `fduk_{season}_{HOME}_{AWAY}_{date}` |

**OddsSnapshot** (`odds_snapshots`):
- Stores complete API responses as JSON for each fetch
- Tracks fetch tier (opening/early/sharp/pregame/closing/in_play) and timing
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

**AlertHistory** (`alert_history`):
- Tracks sent alerts to enforce rate limiting, plus job heartbeats (see
  [REPORTING.md](REPORTING.md#8-storage-backing-reports))

### Storage strategy: hybrid raw + normalized

Each fetch is stored twice:

- **Raw (`OddsSnapshot`)**: the complete API response as JSON, for debugging, auditing,
  schema flexibility and exact data preservation.
- **Normalized (`Odds`)**: individual records for efficient querying, filtering,
  time-series analysis and standard SQL.

### Prediction Table

Defined in `prediction_models.py`:

**Prediction** (`predictions`):
- Stores CLV predictions from the scoring pipeline
- Fields: `event_id` (FK), `snapshot_id` (FK), `model_name`, `model_version`, `predicted_clv`, `created_at`
- Unique constraint: `(event_id, snapshot_id, model_name)` for idempotent upserts

### Agent Tables

`match_briefs` (`match_brief_models.py`), `paper_trades` (`paper_trade_models.py`) and
`agent_wakeups` (`agent_wakeup_models.py`) back the betting agent; see
[REPORTING.md](REPORTING.md#8-storage-backing-reports) for their writers and readers and
[BETTING_AGENT.md](BETTING_AGENT.md#match_briefs-table) for the brief schema.

### Sport Context Tables

| Model | Table | File | Contents |
|-------|-------|------|----------|
| `EspnFixture` | `espn_fixtures` | `epl_data_models.py` | All-competition EPL fixtures (ESPN) |
| `EspnLineup` | `espn_lineups` | `epl_data_models.py` | EPL starting XIs (ESPN) |
| `FplAvailability` | `fpl_availability` | `epl_data_models.py` | FPL player availability |
| `MlbProbablePitchers` | `mlb_probable_pitchers` | `mlb_data_models.py` | MLB probable-pitcher snapshots |
| `InjuryReport` | `nba_injury_reports` | `injury_models.py` | Point-in-time NBA injury reports (see [INJURIES.md](INJURIES.md#schema)) |
| `NbaPlayerSeasonStats` | `nba_player_season_stats` | `player_stats_models.py` | Per-player season stats from PBPStats, used for injury impact weighting |
| `NbaTeamGameLog` | `nba_team_game_logs` | `game_log_models.py` | NBA team game logs, used for rest features and score backfill |

### Polymarket Tables

Five tables in `polymarket_models.py` (`polymarket_events`, `polymarket_markets`,
`polymarket_price_snapshots`, `polymarket_orderbook_snapshots`, `polymarket_fetch_logs`).
See [POLYMARKET.md](POLYMARKET.md#data-model) for the schema.

## Database Environments

### Test database

- **Purpose:** automated testing (pytest)
- **Setup:** `tests/conftest.py` starts an in-process PostgreSQL via py-pglite on TCP port
  5434, creates all tables once per session, and truncates every table before each test
- **Usage:** automatic for `tests/unit` and `tests/integration`, which a plain `uv run pytest`
  runs (`testpaths` and `addopts` in `pyproject.toml`). `tests/migration` connects to
  `DATABASE_URL` instead, so it is excluded from the default run and must be named explicitly:
  `uv run pytest tests/migration -v`. The deploy workflows run it against the dev and production
  databases

### Local database (`odds`)

- **Purpose:** local development and experimentation
- **Setup:** Docker Compose PostgreSQL container (`postgres:15`)
- **Host:** localhost:5433 (not 5432, to avoid conflicts with a system PostgreSQL)
- **Credentials:** postgres/postgres
- **Database:** odds
- **Start:** `docker compose up -d postgres`

### Dev database (remote)

- **Purpose:** CI/CD integration testing
- **Setup:** managed PostgreSQL (e.g. a Neon dev branch)
- **Behavior:** ephemeral
- **Usage:** only via the CI/CD pipeline (GitHub Actions, `secrets.DATABASE_URL`)

### Production database (remote)

- **Purpose:** live production data collection
- **Setup:** managed PostgreSQL (Neon production branch)
- **Usage:** only via production deployments
- **CRITICAL:** NEVER point `DATABASE_URL` at production during local development

### Environment variables

`.env.example` defines `LOCAL_DATABASE_URL`, `DEV_DATABASE_URL` and `PROD_DATABASE_URL`, and
sets `DATABASE_URL=${DEV_DATABASE_URL}` by default. For local development set:

```bash
DATABASE_URL=${LOCAL_DATABASE_URL}
LOCAL_DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/odds
```

CI/CD uses `secrets.DATABASE_URL` in GitHub Actions; AWS Lambda gets `DATABASE_URL` from
Terraform. The betting agent can run under a separate read-mostly role via
`AGENT_DATABASE_URL`.

## Database Migrations

**CRITICAL:** Always create a migration when modifying model files in
`packages/odds-core/odds_core/`.

```bash
# Create migration after model changes
uv run alembic revision --autogenerate -m "description"

# Review the generated migration (Alembic may miss renames and custom types)

# Apply migrations
uv run alembic upgrade head

# Roll back one migration
uv run alembic downgrade -1

# View history
uv run alembic history

# Check current version
uv run alembic current
```

### Migration testing workflow

1. Create the migration locally: `uv run alembic revision --autogenerate -m "description"`
2. Test against the local database: `uv run alembic upgrade head`
3. Run migration tests (excluded from the default run): `uv run pytest tests/migration/ -v`
4. Merge to `main`: CI runs tests, then the dev deployment applies the migration and runs
   `tests/migration` against the dev database
5. Production deployment is triggered manually and requires approval (see
   [DEPLOYMENT.md](DEPLOYMENT.md#deploying))

### Safety rules

- NEVER manually run migrations against production
- NEVER set `DATABASE_URL` to production during local development
- The local database (`odds`) can be reset if needed
- Test migrations in order: local → CI dev → prod

## Query Patterns

The storage modules are listed in [ARCHITECTURE.md](ARCHITECTURE.md#storage-modules).

### OddsReader methods

`packages/odds-lambda/odds_lambda/storage/readers.py`:

- `get_odds_at_time()`: odds at a specific timestamp (CRITICAL for backtesting, to prevent
  look-ahead bias)
- `get_line_movement()`: time series of odds changes for a bookmaker/market
- `get_best_odds()`: highest odds across all bookmakers (line shopping)
- `get_events_by_date_range()`: events within a date range, optionally filtered by status

## Performance Considerations

### Indexes

Composite indexes on frequently queried combinations:
- `(event_id, snapshot_time)` - Time-series queries
- `(event_id, fetch_tier)` - Tier-based analysis
- `(event_id, bookmaker_key, market_key)` - Odds lookups
- `(bookmaker_key, odds_timestamp)` - Bookmaker analysis

### Connection pooling

- Default pool size: 5 (`DATABASE_POOL_SIZE`)
- Lambda: NullPool (no connection reuse), applied when `AWS_LAMBDA_FUNCTION_NAME` is set
- Always use the async context manager: `async with async_session_maker() as session`
- Never share sessions across async tasks

### Batch operations

Use `OddsWriter.bulk_insert_odds()` for backfill operations to reduce round trips.
