# Betting Odds Pipeline & Agent

A single-user, sport-agnostic betting odds pipeline with sport-specific LLM betting agents.
Shared infrastructure (odds collection, storage, paper trading, scheduling, alerting) is
reused across sports; each sport adds its own data sources, feature extractors, agent
prompts, and MCP tools.

**Active sports:** EPL football and MLB baseball (agents in interactive evaluation).
NBA infrastructure exists but is deprioritized.

The pipeline collects bookmaker odds (The Odds API) and UK exchange/retail odds (OddsPortal,
football-data.co.uk), stores hybrid raw + normalized data, and runs an XGBoost CLV model as a
supplementary signal. The betting agent researches matches via web search and Playwright,
consumes pipeline data through MCP tools, and places paper trades with explicit reasoning.

## Repository Layout

```
packages/
├── odds-core/      # Shared models, config, database, alerts
├── odds-lambda/    # Data fetching, storage, scheduling, scheduled jobs
├── odds-analytics/ # Backtesting, strategies, ML / feature extraction
├── odds-mcp/       # MCP server — the betting agent's tool interface
└── odds-cli/       # CLI commands
```

## Quick Start

Prerequisites: Python 3.12+, PostgreSQL 15+, [uv](https://github.com/astral-sh/uv),
a [The Odds API](https://the-odds-api.com/) key, and Docker (for local Postgres).

```bash
# Install dependencies (includes the OddsHarvester scraper from git)
uv sync

# Configure environment
cp .env.example .env          # edit with your API key and database URL

# Start local Postgres (exposed on host port 5433) and run migrations
docker compose up -d postgres
uv run alembic upgrade head
```

Basic usage:

```bash
uv run odds fetch current --sport soccer_epl   # fetch current odds
uv run odds status show                         # system health
uv run odds status quota                         # API quota (free tier: 500 units/month per key)
uv run odds scheduler start                      # run the local scheduler
```

For local-clone development of the OddsHarvester scraper, see [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

## Documentation

| Document | Content |
|----------|---------|
| [AGENTS.md](AGENTS.md) | Project overview, conventions, and critical constraints (entry point for technical detail) |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture, scheduler, backends, jobs, pipeline |
| [docs/DATABASE.md](docs/DATABASE.md) | Schemas, environments, migrations, queries |
| [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) | File workflows, testing, code style, config |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | AWS Lambda (2 functions), Terraform, S3, costs, authoritative job/schedule matrix |
| [docs/CLI.md](docs/CLI.md) | Full command reference |
| [docs/DATA_MODELS.md](docs/DATA_MODELS.md) | Training data reference: sources, target, feature groups, config fields |
| [docs/MODELING.md](docs/MODELING.md) | CLV model findings, experimental evidence and experiment log |
| [docs/DEBUGGING_SCHEDULER.md](docs/DEBUGGING_SCHEDULER.md) | How to debug the AWS self-scheduling jobs (EventBridge, CloudWatch) |
| [docs/POLYMARKET.md](docs/POLYMARKET.md) | Polymarket data model, pipeline (deprioritized) |
| [docs/BACKTESTING_GUIDE.md](docs/BACKTESTING_GUIDE.md) | Backtesting strategies, bet sizing, custom strategies |
| [docs/INJURIES.md](docs/INJURIES.md) | Injury report pipeline (no predictive value for CLV) |
| [docs/BETTING_AGENT.md](docs/BETTING_AGENT.md) | Agent architecture, matchday workflow, MCP tools |
| [docs/AGENT_DATA_SOURCES.md](docs/AGENT_DATA_SOURCES.md) | Agent data source inventory and evaluation |
| [docs/BOOKMAKER_LINE_RELEASE.md](docs/BOOKMAKER_LINE_RELEASE.md) | Bookmaker line timing analysis |
| [docs/REPORTING.md](docs/REPORTING.md) | Reporting surfaces (alerts, digests, health, CLI/MCP, traces) and backing tables |
| [docs/agents/](docs/agents/) | Agent skill configuration: issue tracker, triage labels, domain docs |
| [.env.example](.env.example) | Environment variable template |

## License

Private project — all rights reserved.
