# Betting Odds Data Pipeline

A single-user NBA betting odds data collection and analysis system. Built for robust data pipeline architecture with comprehensive historical data collection, storage, and validation.

## Features

- **Automated Data Collection**: Fetches odds from 8 bookmakers every 30 minutes (configurable)
- **Hybrid Storage**: Raw JSONB snapshots + normalized relational data
- **Data Quality**: Validation and quality checks throughout the pipeline
- **Historical Backfill**: Sample historical data for backtesting
- **Backtesting Engine**: Test betting strategies against historical data
- **CLI Interface**: Rich terminal interface for system management

## Tech Stack

- Python 3.11+
- PostgreSQL 15+ with JSONB
- SQLModel (SQLAlchemy 2.0 + Pydantic)
- asyncio + aiohttp
- APScheduler, Typer, Rich, structlog

## Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 15+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- The Odds API key ([get one here](https://the-odds-api.com/))

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd odds
```

2. Set up Python environment with uv:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
```

Or with pip:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

3. Install pre-commit hooks:
```bash
pre-commit install
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API key and database URL
```

5. Start PostgreSQL (using Docker):
```bash
docker-compose up -d postgres
```

6. Run database migrations:
```bash
alembic upgrade head
```

### Usage

#### Manual Data Fetch
```bash
odds fetch                    # Fetch current odds
odds fetch --sport basketball_nba
```

#### System Status
```bash
odds status                   # System health overview
odds status --verbose         # Detailed statistics
odds quota                    # Check API usage
```

#### Start Scheduler
```bash
odds scheduler start          # Start background jobs
```

#### Historical Backfill

Build a backtesting database with historical odds data:

```bash
# Step 1: Create a plan (discovers games, estimates quota)
odds backfill plan --start 2023-10-01 --end 2024-04-30 --games 166 --output backfill.json

# Step 2: Review and test
cat backfill.json | jq '.total_games, .estimated_quota_usage'
odds backfill execute --plan backfill.json --dry-run

# Step 3: Execute (consumes API quota!)
odds backfill execute --plan backfill.json

# Check coverage
odds backfill status
```

See [HISTORICAL_BACKFILL_GUIDE.md](HISTORICAL_BACKFILL_GUIDE.md) for detailed strategy and usage.

#### Data Analysis & Visualization

Explore collected data with Jupyter notebooks:

```bash
# Launch Jupyter Lab
uv run jupyter lab

# Open notebooks/odds_analysis.ipynb
```

The analysis notebook includes:
- Line movement visualization
- Bookmaker comparison
- Market efficiency analysis (vig calculation)
- Data quality metrics

See [notebooks/README.md](notebooks/README.md) for detailed notebook documentation.

#### Data Inspection
```bash
odds events --days 7                        # List recent events
odds show-odds --event-id <id>              # View odds for event
odds line-movement --event-id <id> --bookmaker fanduel
```

## Development

### Project Structure

```
betting-odds-system/
├── core/               # Core models, config, database, API client
├── storage/            # Data writers, readers, validators
├── scheduler/          # APScheduler jobs
├── cli/                # Typer CLI commands
├── analytics/          # Query patterns and backtesting
├── alerts/             # Alert infrastructure
├── tests/              # Test suite
└── migrations/         # Alembic migrations
```

### Code Quality

The project uses **ruff** for linting and formatting, enforced via pre-commit hooks:

```bash
# Manually run checks
pre-commit run --all-files

# Hooks run automatically on git commit
git commit -m "your message"
```

### Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=core --cov=storage --cov=scheduler
```

### Database Migrations

```bash
# Create a new migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

## Configuration

All configuration is managed via environment variables. See [.env.example](.env.example) for all available options.

Key settings:
- `ODDS_API_KEY`: Your The Odds API key
- `DATABASE_URL`: PostgreSQL connection string
- `SAMPLING_MODE`: `fixed` (default) or `adaptive`
- `FIXED_INTERVAL_MINUTES`: Fetch interval (default: 30)

## Docker Deployment

### Local Development
```bash
docker-compose up -d
```

### Production Build
```bash
docker build -t betting-odds-pipeline .
docker run -e ODDS_API_KEY=your_key betting-odds-pipeline
```

## Cost Estimates

- **API**: ~$25/month (20k requests, 30-min sampling)
- **Hosting** (Railway): $7-10/month
- **Total**: ~$32-35/month

## Documentation

- **[CLAUDE.md](CLAUDE.md)** - Complete technical specification
- **[HISTORICAL_BACKFILL_GUIDE.md](HISTORICAL_BACKFILL_GUIDE.md)** - Guide to building a backtesting database
- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Detailed setup instructions
- **[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)** - Development roadmap
- **[DEPLOYMENT_VERIFICATION.md](DEPLOYMENT_VERIFICATION.md)** - System verification report

## License

Private project - all rights reserved.
