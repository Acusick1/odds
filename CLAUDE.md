# Betting Odds Data Pipeline

IMPORTANT (LLM agents): This document is read-only. Do not create additional documentation files without user consent. Do NOT manually run `ruff` or `pre-commit` - automated via git hooks.

## Project Overview

Single-user betting odds data collection and analysis system for NBA games. Integrates sportsbook odds and Polymarket prediction market data for cross-source analysis and closing line value (CLV) prediction. Prioritizes robust data pipeline architecture with comprehensive historical data collection, storage, and validation. Supports backtesting betting strategies against historical data.

## Package Structure

```
packages/
├── odds-core/      # Models, config, database (odds_core/)
│   └── polymarket_models.py  # Polymarket DB schemas
├── odds-lambda/    # Data fetching, storage, scheduling (odds_lambda/)
│   ├── polymarket_fetcher.py    # Gamma + CLOB API client, market classifier
│   ├── polymarket_ingestion.py  # Orchestrates event discovery + snapshot collection
│   ├── polymarket_matching.py   # Matches Polymarket events to sportsbook Events
│   ├── storage/polymarket_writer.py  # Upserts events/markets, stores snapshots
│   ├── storage/polymarket_reader.py  # Queries active events, pipeline stats
│   └── jobs/
│       ├── fetch_polymarket.py     # Live polling job (scheduled)
│       └── backfill_polymarket.py  # Historical price backfill from CLOB API
├── odds-analytics/ # Backtesting, strategies, ML (odds_analytics/)
└── odds-cli/       # CLI commands (odds_cli/)
    └── commands/polymarket.py  # discover, status, backfill, link, book
```

## Critical Constraints

### Timezone Handling (CRITICAL)

- ALWAYS use `datetime.now(UTC)` (from `datetime import UTC`)
- NEVER use `datetime.utcnow()` (deprecated in Python 3.12+)
- NEVER use `datetime.now()` without timezone
- Store UTC in database, convert only for display

### AWS Lambda

- 15-minute max execution - design jobs for <5 min
- Must use NullPool for DB connections (automatic when `SCHEDULER_BACKEND=aws`)
- Stateless between invocations

### API Cost

- 1 game = 1 quota unit
- Historical endpoint = 10x cost (10 units per game)
- Check quota: `odds status quota`

### Database Connections

- ALWAYS use async context manager: `async with async_session_maker() as session`
- NEVER share sessions across async tasks
- Lambda uses NullPool automatically. Local/Railway use pool of 5

### Look-Ahead Bias (Backtesting)

- ALWAYS use `get_odds_at_time()` for historical analysis
- NEVER use current odds when backtesting
- Backtesting enforces `decision_time` (hours before game)

### Data Quality

- Validation logs issues by default, does not reject data
- Set `REJECT_INVALID_ODDS=true` for strict validation
- Review `DataQualityLog` table for patterns

## Polymarket Integration

**Strategic goal:** Predict closing line values using all available market data, execute on whichever venue (sportsbook or Polymarket) offers best price relative to predicted close. Line movement prediction, not outcome prediction.

### Data Sources

- **Gamma API** — market/event discovery (metadata, status, volume). No auth required.
- **CLOB API** — prices, order books, price history. No auth required.
- NBA: `series_id=10345`, game `tag_id=100639`

### 30-Day Data Retention (CRITICAL)

CLOB `/prices-history` data expires on a ~30-day rolling basis. The backfill job (`odds polymarket backfill`) must run every 3–5 days to avoid data loss. This is the most time-sensitive aspect of the project.

### Polymarket-Specific Patterns

- Token IDs are strings, not integers
- Order books need client-side sorting (CLOB returns unsorted)
- Prices are implied probabilities (0.0–1.0), not American odds
- Event matching parses ticker format (`nba-{away}-{home}-{yyyy}-{mm}-{dd}`) against sportsbook Events with ±24h date window
- `PolymarketEvent.event_id` FK starts `NULL`, linked lazily via `match_polymarket_event()`
- Market type classified from question text via regex in `classify_market()`
- Polling uses fixed `price_poll_interval` (default 300s); `FetchTier` filters which events to collect, not frequency

## Code Style

### Type Hints (Required)

- Use modern syntax: `str | None` not `Optional[str]`
- Use `list[T]`, `dict[K, V]` not `List[T]`, `Dict[K, V]`
- All functions must have type hints for parameters and return values

### Async/Await (Required)

- All DB operations must be async
- Use `AsyncSession`, never sync `Session`
- CLI uses `asyncio.run()` for async functions

### Model Types

| Type | Use Case |
|------|----------|
| SQLModel | Database models |
| Pydantic BaseModel | API responses, configs |
| dataclass | Simple containers |

## Key Commands

**Before running commands:** Check docs/CLI.md for full command reference and options.

```bash
# Fetch data
uv run odds fetch current
uv run odds fetch scores

# Check status
uv run odds status show
uv run odds status quota

# Run backtest
uv run odds backtest run --strategy basic_ev --start 2024-10-01 --end 2024-12-31

# Polymarket
uv run odds polymarket discover
uv run odds polymarket status
uv run odds polymarket backfill
uv run odds polymarket link

# Run tests
uv run pytest

# Database migrations
uv run alembic revision --autogenerate -m "description"
uv run alembic upgrade head

# Local development
docker-compose up -d
uv run odds scheduler start
```

## Database Environments

| Environment | Purpose | Safety |
|-------------|---------|--------|
| `odds_test` | Automated testing | Safe - isolated |
| `odds` (local) | Development | Safe - can reset |
| Dev (remote) | CI/CD | Ephemeral |
| Production | Live data | NEVER connect locally |

**CRITICAL:** NEVER set `DATABASE_URL` to production during local development.

## Documentation

| Document | Content |
|----------|---------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture, scheduler, backends, pipeline |
| [docs/DATABASE.md](docs/DATABASE.md) | Schemas, environments, migrations, queries |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | File workflows, testing, code style, config |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | AWS/Railway/Local setup, costs |
| [docs/CLI.md](docs/CLI.md) | Full command reference |
| [BACKTESTING_GUIDE.md](docs/BACKTESTING_GUIDE.md) | Comprehensive backtesting reference |
| [docs/POLYMARKET.md](docs/POLYMARKET.md) | Polymarket data model, pipeline, API reference |
| [.env.example](.env.example) | Environment variable template |
