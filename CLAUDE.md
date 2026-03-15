# Betting Odds Data Pipeline

IMPORTANT (LLM agents): This document is read-only. Do not create additional documentation files without user consent. Run `uv run ruff check --fix` and `uv run ruff format` immediately before committing — not during development while changes are still in progress.

## Strategic Goal

Predict line movement (closing line value) using cross-source market data. The model targets the delta between current and closing fair prices, identifying when current prices are mispriced relative to where they'll close. Execute on whichever venue (sportsbook or Betfair exchange) offers the best price relative to the predicted close. Line movement prediction, not outcome prediction.

## Project Overview

Single-user betting odds data collection and analysis system. **Active focus is EPL football** — NBA support exists but is deprioritised (NBA CLV ~3.6% R² is insufficient to overcome vig, and cross-source execution isn't viable: Polymarket inaccessible from UK, Betfair has no NBA match odds liquidity. Football has deep Betfair liquidity, more data, and more bookmaker competition).

Two primary data sources: The Odds API (US bookmakers, live polling — currently disabled) and OddsPortal (UK bookmakers, headless scraper — active, hourly EPL collection). A scoring pipeline produces CLV predictions per snapshot, delivered via daily Discord digest. Supports backtesting strategies against historical data.

## Package Structure

```
packages/
├── odds-core/      # Models, config, database (odds_core/)
│   ├── models.py              # Event, OddsSnapshot, Odds, FetchLog, DataQualityLog
│   ├── prediction_models.py   # Prediction table (CLV scoring output)
│   └── config.py              # Pydantic Settings (API, DB, scheduler, alerts)
├── odds-lambda/    # Data fetching, storage, scheduling (odds_lambda/)
│   ├── data_fetcher.py        # TheOddsAPIClient (with key rotation)
│   ├── event_sync.py          # EventSyncService (free /events endpoint)
│   ├── oddsportal_common.py   # Shared: bookmaker mapping, odds conversion, tier classification
│   ├── model_loader.py        # S3 model loading with ETag caching
│   ├── storage/               # readers + writers (odds, polymarket, injury, game_log, pbpstats)
│   ├── scheduling/            # Multi-backend scheduler (AWS/Railway/local)
│   └── jobs/                  # All scheduled job entry points (see ARCHITECTURE.md)
├── odds-analytics/ # Backtesting, strategies, ML (odds_analytics/)
└── odds-cli/       # CLI commands (odds_cli/)
    ├── commands/              # 16 command groups (see CLI.md)
    └── alerts/base.py         # AlertManager + DiscordAlert (webhook delivery)
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full system architecture, all 10 scheduled jobs, and storage module index.

## Critical Constraints

### Timezone Handling (CRITICAL)

- ALWAYS use `datetime.now(UTC)` (from `datetime import UTC`)
- NEVER use `datetime.utcnow()` (deprecated in Python 3.12+)
- NEVER use `datetime.now()` without timezone
- Store UTC in database, convert only for display

### AWS Lambda

- Two Lambda functions: scheduler (512 MB, 5 min) and scraper (2 GB, 10 min)
- Must use NullPool for DB connections (automatic when `SCHEDULER_BACKEND=aws`)
- Stateless between invocations; API key rotation state in SSM Parameter Store

### API Cost

- `/odds` endpoint: 1 unit per region × per market per call (returns all games in one call — cost does not scale with number of games). Current config: 1 region × 3 markets = 3 units per fetch.
- `/events` endpoint: free (no quota cost) — returns game metadata (IDs, teams, times) without odds
- `/scores` endpoint: 1 unit without `daysFrom`, 2 units with `daysFrom`
- `/historical/odds` endpoint: 10 units per region × per market
- Bookmakers do not affect cost (server-side filter only)
- Multiple API keys rotated via `ODDS_API_KEYS` (comma-separated), active index tracked in SSM
- Check quota: `odds status quota`

### Database Connections

- ALWAYS use async context manager: `async with async_session_maker() as session`
- NEVER share sessions across async tasks
- No wrapper functions — use `async_session_maker()` directly everywhere
- Lambda uses NullPool automatically. Local/Railway use pool of 5

### Look-Ahead Bias (Backtesting)

- ALWAYS use `get_odds_at_time()` for historical analysis
- NEVER use current odds when backtesting
- Backtesting enforces `decision_time` (hours before game)

### Data Quality

- Validation logs issues by default, does not reject data
- Set `REJECT_INVALID_ODDS=true` for strict validation
- Review `DataQualityLog` table for patterns

## Polymarket Integration (deprioritized)

Pipeline exists but is inactive. Not accessible from UK, data likely collinear with sportsbook odds, 30-day CLOB retention creates ongoing maintenance burden. See [docs/POLYMARKET.md](docs/POLYMARKET.md) for technical details if needed.

## Code Style

### Type Hints (Required)

- Use modern syntax: `str | None` not `Optional[str]`
- Use `list[T]`, `dict[K, V]` not `List[T]`, `Dict[K, V]`
- All functions must have type hints for parameters and return values

### Async/Await (Required)

- All DB operations must be async
- Use `AsyncSession`, never sync `Session`
- CLI uses `asyncio.run()` for async functions

### Conventions

- Job entry points are `async def main()` functions in `odds_lambda/jobs/`
- `EventSyncService` syncs games from free `/events` endpoint as first step in `fetch_odds` job
- OddsPortal jobs share utilities via `oddsportal_common.py` (bookmaker key mapping, decimal→American conversion, tier classification)
- Scoring pipeline: `score_predictions` job loads model from S3 via `model_loader.load_model()` (ETag-cached), extracts features, stores to `Prediction` table with idempotent upsert (`ON CONFLICT DO NOTHING` on event_id + snapshot_id + model_name)
- Discord alerts: `AlertManager.send_embed()` is generic delivery; each job owns its embed formatting
- Training pipeline: features NaN-fill when optional data unavailable — rows kept, not dropped

### Model Types

| Type | Use Case |
|------|----------|
| SQLModel | Database models |
| Pydantic BaseModel | API responses, configs |
| dataclass | Simple containers |

## Key Commands

**Full reference:** [docs/CLI.md](docs/CLI.md)

```bash
# Fetch data
uv run odds fetch current --sport soccer_epl
uv run odds fetch scores

# Scrape OddsPortal
uv run odds scrape upcoming --league england-premier-league --market 1x2

# Check status
uv run odds status show
uv run odds status quota

# ML training and model publishing
uv run odds train run --config experiments/configs/my_config.yaml
uv run odds train tune --config experiments/configs/my_config.yaml --train-best
uv run odds model publish --name epl-clv-home --path models/model.pkl

# Run backtest
uv run odds backtest run --strategy basic_ev --start 2024-10-01 --end 2024-12-31

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
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture, scheduler, backends, jobs, pipeline |
| [docs/DATABASE.md](docs/DATABASE.md) | Schemas, environments, migrations, queries |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | File workflows, testing, code style, config |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | AWS Lambda (2 functions), Terraform, S3, costs |
| [docs/CLI.md](docs/CLI.md) | Full command reference (16 command groups) |
| [docs/DATA_MODELS.md](docs/DATA_MODELS.md) | Training pipeline, feature groups, target definition |
| [docs/MODELING.md](docs/MODELING.md) | Modeling rationale, experiments, open questions |
| [docs/DEBUGGING_SCHEDULER.md](docs/DEBUGGING_SCHEDULER.md) | Scheduler diagnostics, EventBridge, CloudWatch |
| [docs/POLYMARKET.md](docs/POLYMARKET.md) | Polymarket data model, pipeline (deprioritized) |
| [docs/BACKTESTING_GUIDE.md](docs/BACKTESTING_GUIDE.md) | Backtesting strategies, bet sizing, custom strategies |
| [docs/INJURIES.md](docs/INJURIES.md) | Injury report pipeline (no predictive value for CLV) |
| [docs/BOOKMAKER_LINE_RELEASE.md](docs/BOOKMAKER_LINE_RELEASE.md) | Bookmaker line timing analysis |
| [.env.example](.env.example) | Environment variable template |
