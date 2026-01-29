# Development Reference

> Reference documentation for development workflows, code style, and configuration.

## Configuration Management

Configuration uses Pydantic Settings with environment variables. See `packages/odds-core/odds_core/config.py`.

### API Settings (`APIConfig`)

| Variable | Description | Default |
|----------|-------------|---------|
| `ODDS_API_KEY` | The Odds API authentication key | Required |
| `ODDS_API_BASE_URL` | API base URL | https://api.the-odds-api.com/v4 |
| `ODDS_API_QUOTA` | Monthly request quota | 20,000 |

### Database Settings (`DatabaseConfig`)

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | Required |
| `DATABASE_POOL_SIZE` | Connection pool size | 5 (NullPool for Lambda) |

### Data Collection (`DataCollectionConfig`)

| Variable | Description | Default |
|----------|-------------|---------|
| `SPORTS` | Sports to track | ["basketball_nba"] |
| `BOOKMAKERS` | Bookmaker list | [pinnacle, circa, draftkings, fanduel, betmgm, williamhill_us, betrivers, bovada] |
| `MARKETS` | Bet types | ["h2h", "spreads", "totals"] |
| `REGIONS` | Odds regions | ["us"] |

### Scheduler Settings (`SchedulerConfig`)

| Variable | Description | Default |
|----------|-------------|---------|
| `SCHEDULER_BACKEND` | Backend type | "local" |
| `SCHEDULER_DRY_RUN` | Dry-run mode | false |
| `SCHEDULER_LOOKAHEAD_DAYS` | Days ahead to check | 7 |

### AWS Settings (`AWSConfig`)

| Variable | Description | Default |
|----------|-------------|---------|
| `AWS_REGION` | AWS region for Lambda | Optional |
| `AWS_LAMBDA_ARN` | Lambda function ARN | Optional |

### Data Quality (`DataQualityConfig`)

| Variable | Description | Default |
|----------|-------------|---------|
| `ENABLE_VALIDATION` | Run data quality checks | true |
| `REJECT_INVALID_ODDS` | Reject vs log invalid data | false |

### Logging (`LoggingConfig`)

| Variable | Description | Default |
|----------|-------------|---------|
| `LOG_LEVEL` | Logging verbosity | INFO |
| `LOG_FILE` | Log file path | logs/odds_pipeline.log |

See `.env.example` for complete template.

## Code Style Guidelines

### Type Hints (Required)

```python
# Good
def process_odds(event_id: str, limit: int | None = None) -> list[Odds]: ...

# Bad - don't use Optional or typing module
from typing import Optional, List
def process_odds(event_id: str, limit: Optional[int] = None) -> List[Odds]: ...
```

- All function signatures need type hints
- Use `str | None` not `Optional[str]`
- Use `list[T]`, `dict[K, V]` not `List[T]`, `Dict[K, V]`

### Async/Await (Required)

```python
# Good
async def get_event(session: AsyncSession, event_id: str) -> Event:
    result = await session.execute(select(Event).where(Event.id == event_id))
    return result.scalar_one()

# Bad - never use sync Session
def get_event(session: Session, event_id: str) -> Event: ...
```

- All database operations must use `async`/`await`
- Use `AsyncSession`, never sync `Session`
- CLI commands use `asyncio.run()` for async functions
- Never mix sync and async database code

### Model Types

| Type | Use Case |
|------|----------|
| SQLModel | Database models (ORM + validation) |
| Pydantic BaseModel | API responses, configs |
| dataclass | Simple containers without validation |

## File Modification Workflows

### Modifying Database Schema

`packages/odds-core/odds_core/models.py`

1. Make model changes
2. Create Alembic migration: `uv run alembic revision --autogenerate -m "description"`
3. Review generated migration (Alembic misses renames, custom types)
4. Test migration: `uv run alembic upgrade head`
5. Update affected queries in `readers.py` or `writers.py`
6. Run tests

### Adding Betting Strategies

1. Create class in `packages/odds-analytics/odds_analytics/strategies.py` inheriting from `BettingStrategy`
2. Implement `evaluate_opportunity()` method
3. Add to `AVAILABLE_STRATEGIES` dict
4. Register in CLI: `packages/odds-cli/odds_cli/commands/backtest.py`
5. Write tests
6. Document if complex

### Modifying API Client

`packages/odds-lambda/odds_lambda/data_fetcher.py`

1. Preserve retry logic (tenacity decorators)
2. Update quota tracking if request patterns change
3. Add fixtures to `tests/fixtures/` for new endpoints
4. Update integration tests
5. Never remove rate limiting or error handling

### Adding Scheduler Jobs

`packages/odds-lambda/odds_lambda/jobs/*.py`

1. Create async job function
2. Add to `packages/odds-lambda/odds_lambda/scheduling/jobs.py` registry
3. Implement self-scheduling (calculate next execution time)
4. Add error handling (log errors, never crash)
5. Register in CLI: `packages/odds-cli/odds_cli/commands/scheduler.py`
6. Test locally: `odds scheduler start`

### Modifying Configuration

`packages/odds-core/odds_core/config.py`

1. Add field to appropriate Config class (with type hint and default)
2. Update `.env.example`
3. Ensure backward compatibility (sensible default)
4. Never remove existing config fields (deprecate instead)

## Testing

### Running Tests

```bash
# All tests
uv run pytest

# Unit tests only
uv run pytest tests/unit/

# Integration tests
uv run pytest tests/integration/

# Migration tests
uv run pytest tests/migration/ -v
```

### Test Structure

```
tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
├── migration/      # Migration tests
└── fixtures/       # Test data
```

Test database uses `odds_test` (isolated from dev data).

## Local Development Setup

```bash
# Start PostgreSQL
docker-compose up -d

# Run scheduler locally
uv run odds scheduler start

# Manual data fetch
uv run odds fetch current

# Check status
uv run odds status show
```

## Dependency Management

Project uses **uv** workspace architecture. Dependencies managed via `pyproject.toml` and `uv.lock`.

## Code Quality

**Ruff** enforced via pre-commit hooks (`.pre-commit-config.yaml`).

```bash
# Enable hooks
pre-commit install
```

**LLM Agents:** Do NOT manually run `ruff` or `pre-commit`. Automated via git hooks and CI.

## Issue Creation Workflow

1. Generate markdown following schema in `issues/ISSUE_SCHEMA.md`
2. Show preview for user review
3. After approval, post with `gh issue create`
4. Return issue URL

## Logging

Uses **structlog** for structured logging.

- Levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
- Format: JSON to console, rotating files
- Retention: 30 days
- Config: `LOG_LEVEL`, `LOG_FILE` environment variables

## Extensibility

### Adding Sports/Bookmakers/Markets

Add to respective config lists in `.env`:
- `SPORTS`
- `BOOKMAKERS`
- `MARKETS`

No code changes required.

### Adding Strategies

Inherit from `BettingStrategy` and implement `evaluate_opportunity()`. See `packages/odds-analytics/odds_analytics/strategies.py` for examples.
