# Betting Odds Pipeline & Agent

IMPORTANT (LLM agents): This document is read-only. Do not create additional documentation files without user consent. Run `uv run ruff check --fix` and `uv run ruff format` immediately before committing — not during development while changes are still in progress.

## Strategic Goal

Identify and exploit structural mispricing in betting markets across multiple sports. The edge is **breadth and speed of information synthesis** — an LLM agent that simultaneously monitors bookmaker odds, exchange orderbooks, public sentiment signals, lineup announcements, press conferences, and fixture context can reason across all of it to identify mispricings that no individual bettor can. Structural biases (bookmaker liability shading, accumulator distortion, public money loading) and information gaps are both valid edge types. Execute on whichever venue offers the best price relative to estimated fair value. See [docs/BETTING_AGENT.md](docs/BETTING_AGENT.md) for the agent architecture.

## Project Overview

Sport-agnostic betting data pipeline with sport-specific LLM agents. Infrastructure (odds collection, storage, paper trading, scheduling, alerting) is shared; each sport adds its own data sources, feature extractors, agent prompts, and MCP tools. The agent researches matches via web search and Playwright, consumes pipeline data through MCP tools, and places paper trades with explicit reasoning.

**Target sports:** EPL (active — agent in interactive evaluation), MLB and NBA (planned).

**Shared data sources:** The Odds API (US bookmakers — currently disabled), OddsPortal (UK bookmakers, headless scraper). **Sport-specific sources:** football-data.co.uk (historical EPL with Pinnacle + Betfair Exchange closing odds). An XGBoost CLV model produces supplementary predictions, but the agent's primary edge comes from information synthesis, not the model. See [docs/AGENT_DATA_SOURCES.md](docs/AGENT_DATA_SOURCES.md) for the full data source inventory.

## Package Structure

```
packages/
├── odds-core/      # Shared models, config, database (odds_core/)
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
├── odds-mcp/       # MCP server — agent tool interface (odds_mcp/)
│   └── agents/                # Sport-specific agent prompts (epl/, mlb/)
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

### Model Evaluation

- Walk-forward CV is the primary evaluation metric — it simulates the production retrain cycle (train on past, predict next chunk, roll forward)
- Set `test_split: 0.0` for walk-forward CV experiments — a fixed chronological test split wastes training data and doesn't reflect deployment (we retrain on all available data before predicting new events)
- CV R² from tuning is optimistically biased (best of N trials), but relative comparisons between experiments are fair when both are tuned

### Look-Ahead Bias (Backtesting)

- ALWAYS use `get_odds_at_time()` for historical analysis
- NEVER use current odds when backtesting
- Backtesting enforces `decision_time` (hours before game)

### Data Quality

- Validation logs issues by default, does not reject data
- Set `REJECT_INVALID_ODDS=true` for strict validation
- Review `DataQualityLog` table for patterns

## Polymarket Integration (deprioritized)

Full pipeline built (API client, 5 DB tables, storage, ingestion, feature extractors) but deprioritized. EPL match-level volume is thin ($10K-$100K per match) and AMM-driven — the orderbook reflects automated market maker parameters, not genuine public sentiment. Not accessible from UK for trading. NBA and MLB volume may differ — revisit per sport. Pipeline code exists if liquidity improves. See [docs/POLYMARKET.md](docs/POLYMARKET.md) for technical details.

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
| [docs/BETTING_AGENT.md](docs/BETTING_AGENT.md) | Agent architecture, matchday workflow, phased rollout |
| [docs/AGENT_DATA_SOURCES.md](docs/AGENT_DATA_SOURCES.md) | Agent data source inventory and evaluation |
| [docs/BOOKMAKER_LINE_RELEASE.md](docs/BOOKMAKER_LINE_RELEASE.md) | Bookmaker line timing analysis |
| [.env.example](.env.example) | Environment variable template |
