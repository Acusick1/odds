# Betting Odds Data Pipeline

A single-user NBA betting odds data collection and analysis system. Built for robust data pipeline architecture with comprehensive historical data collection, storage, and backtesting.

## Features

- **Automated Data Collection**: Multi-backend scheduler (AWS Lambda, Railway, or local)
- **Hybrid Storage**: Raw JSONB snapshots + normalized relational data
- **Data Quality**: Validation and quality checks throughout pipeline
- **Historical Backfill**: Strategic sampling for backtesting database
- **Backtesting Engine**: Test betting strategies against historical data
- **Rich CLI**: Terminal interface for system management

## Tech Stack

- Python 3.11+, PostgreSQL 15+ with JSONB
- SQLModel (SQLAlchemy 2.0 + Pydantic)
- asyncio + aiohttp
- Multi-backend scheduler (AWS Lambda, Railway, APScheduler)

## Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 15+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- [The Odds API](https://the-odds-api.com/) key

### Installation

```bash
# Clone and enter directory
git clone <repository-url>
cd odds

# Set up Python environment
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Configure environment
cp .env.example .env
# Edit .env with your API key and database URL

# Start PostgreSQL
docker-compose up -d postgres

# Run migrations
alembic upgrade head
```

### Basic Usage

```bash
# Manual data fetch
odds fetch current

# System status
odds status show
odds status show --verbose

# Start scheduler (local mode)
odds scheduler start
```

## Configuration

All configuration via environment variables in `.env`:

```bash
# Required
ODDS_API_KEY=your_key_here
DATABASE_URL=postgresql://user:pass@localhost:5432/odds

# Scheduler Backend
SCHEDULER_BACKEND=local  # local, aws, or railway

# Optional
SCHEDULER_LOOKAHEAD_DAYS=7
LOG_LEVEL=INFO
```

### Scheduler Backends

The system supports three scheduler backends:

**Local (Development)**
```bash
SCHEDULER_BACKEND=local
odds scheduler start  # Runs until Ctrl+C
```

**AWS Lambda (Production, ~$0.20/month)**
```bash
SCHEDULER_BACKEND=aws
AWS_REGION=us-east-1
# See AWS Lambda Deployment section below
```

**Railway (Production, ~$5/month)**
```bash
SCHEDULER_BACKEND=railway
# See Railway Deployment section below
```

All backends use the same adaptive scheduling logic:
- 30 minutes when games are within 3 hours
- 3 hours when games are 3-12 hours away
- 12 hours when games are 12-24 hours away
- 24 hours when games are 1-3 days away
- 48 hours when games are 3+ days away
- Skips execution when no games scheduled

## Historical Backfill

Build a backtesting database with historical odds data:

### 1. Create a Plan

```bash
# Plan for 133 games with 5 snapshots each (uses ~20k quota)
odds backfill plan \
  --start 2023-10-01 \
  --end 2024-04-30 \
  --games 133 \
  --output backfill.json

# Review plan
cat backfill.json | jq '.total_games, .estimated_quota_usage'
```

### 2. Test with Dry Run

```bash
odds backfill execute --plan backfill.json --dry-run
```

### 3. Execute Backfill

```bash
# ⚠️ This consumes API quota!
odds backfill execute --plan backfill.json

# Check results
odds backfill status
```

### Backfill Strategy

The system uses a 5-snapshot adaptive approach per game:
- **Opening line** (3 days before): Initial market
- **Early action** (24h before): Public betting begins
- **Sharp action** (12h before): Professional bettors active
- **Pre-game** (3h before): Final adjustments
- **Closing line** (30min before): Market consensus

This captures complete line movement for sophisticated backtesting.

**Quota Math:**
- 5 snapshots × 30 API requests = 150 requests per game
- 133 games × 150 = ~20,000 requests (full monthly quota)

## Backtesting

Test betting strategies against historical data:

```bash
# List available strategies
odds backtest list-strategies

# Run backtest
odds backtest run \
  --strategy basic_ev \
  --start 2024-10-01 \
  --end 2024-12-31 \
  --bankroll 10000 \
  --output-json results.json

# View results
odds backtest show results.json --verbose

# Compare strategies
odds backtest compare strategy1.json strategy2.json strategy3.json

# Export to CSV
odds backtest export results.json bets.csv
```

**Available Strategies:**
- `flat` - Baseline flat betting
- `basic_ev` - Expected value betting (sharp vs retail odds)
- `arbitrage` - Risk-free arbitrage opportunities

See [BACKTESTING_GUIDE.md](BACKTESTING_GUIDE.md) for detailed strategy documentation.

## Deployment

### AWS Lambda (Recommended)

**Cost:** ~$0.20/month | **Benefits:** Auto-scales, off-season pauses

```bash
# 1. Build Lambda package
cd deployment/aws
./build_lambda.sh

# 2. Configure Terraform
cd terraform
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your database URL and API key

# 3. Deploy infrastructure
terraform init
terraform apply

# 4. Bootstrap scheduling
aws lambda invoke \
  --function-name odds-scheduler \
  --payload '{"job": "fetch-odds"}' \
  response.json
```

**Monitoring:**
```bash
# Stream logs
aws logs tail /aws/lambda/odds-scheduler --follow

# Check schedules
aws events list-rules --name-prefix odds-

# Manual trigger
aws lambda invoke \
  --function-name odds-scheduler \
  --payload '{"job": "fetch-odds"}' \
  response.json
```

### Railway

**Cost:** ~$5/month | **Benefits:** Simple setup

```bash
# 1. Install Railway CLI
npm i -g @railway/cli

# 2. Create railway.json
cp deployment/railway/railway.json.example railway.json

# 3. Configure environment in Railway dashboard
SCHEDULER_BACKEND=railway
DATABASE_URL=<auto-populated>
ODDS_API_KEY=<your-key>

# 4. Deploy
railway login
railway link
railway up
```

### Docker (Local/Development)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop services
docker-compose down
```

## Project Structure

```
betting-odds-system/
├── core/                   # Core models, config, database, API client
│   ├── scheduling/         # Multi-backend scheduler
│   │   ├── intelligence.py # Game-aware scheduling logic
│   │   └── backends/       # AWS, Railway, Local
├── storage/                # Data writers, readers, validators
├── jobs/                   # Standalone scheduler jobs
├── cli/                    # Typer CLI commands
├── analytics/              # Backtesting engine and strategies
├── tests/                  # Test suite
├── migrations/             # Alembic migrations
└── deployment/             # AWS Lambda & Railway configs
```

## Development

### Database Migrations

```bash
# Create migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

### Adding Custom Betting Strategies

```python
# analytics/strategies.py
from analytics.backtesting import BettingStrategy, BacktestEvent, BetOpportunity

class MyStrategy(BettingStrategy):
    async def evaluate_opportunity(
        self,
        event: BacktestEvent,  # Type-safe, guaranteed scores
        odds_snapshot: list[Odds],
        config: BacktestConfig,
    ) -> list[BetOpportunity]:
        # Your strategy logic here
        return opportunities
```

Register in `cli/commands/backtest.py`:
```python
STRATEGIES = {
    "my_strategy": MyStrategy,
    # ...
}
```

## CLI Command Reference

### Data Collection

```bash
odds fetch current                  # Fetch current odds
odds fetch scores                   # Update game scores
```

### System Status

```bash
odds status show                    # Health overview
odds status show --verbose          # Detailed stats
odds status quota                   # API usage
odds status events --days 7         # Recent events
odds status events --team "Lakers"  # Filter by team
```

### Scheduler

```bash
odds scheduler start                # Start local scheduler
odds scheduler info                 # Show configuration
odds scheduler test-backend         # Test connectivity
odds scheduler run-once fetch-odds  # Execute single job
```

### Backfill

```bash
odds backfill plan --start YYYY-MM-DD --end YYYY-MM-DD --games N
odds backfill execute --plan backfill.json
odds backfill execute --plan backfill.json --dry-run
odds backfill status
```

### Backtesting

```bash
odds backtest run --strategy STRATEGY --start DATE --end DATE
odds backtest show results.json [--verbose]
odds backtest compare result1.json result2.json
odds backtest export results.json --output bets.csv
odds backtest list-strategies
```

## Data Models

### Events
Game/event information with final scores and status tracking.

### Odds (Normalized)
Individual odds records for efficient querying:
- Bookmaker, market (h2h/spreads/totals), outcome
- American odds format, optional point spread/total
- Timestamps for line movement analysis

### OddsSnapshot (Raw)
Complete API responses stored as JSONB for:
- Exact data preservation
- Debugging and auditing
- Schema flexibility

### Quality & Fetch Logs
Validation warnings and API operation tracking.

## Bookmakers (8)

1. **Pinnacle** - Sharp, low-margin (key for EV calculations)
2. **Circa Sports** - Sharp, Vegas-based
3. **DraftKings** - Major US retail
4. **FanDuel** - Major US retail
5. **BetMGM** - Major US retail
6. **Caesars** - Major US retail
7. **BetRivers** - Regional retail
8. **Bovada** - Offshore

## Troubleshooting

### Database Connection Issues

```bash
# Check PostgreSQL is running
docker ps | grep postgres

# Start if needed
docker-compose up -d postgres

# Test connection
psql $DATABASE_URL
```

### API Quota Issues

```bash
# Check remaining quota
odds status quota

# View recent usage
odds status show --verbose
```

### Scheduler Not Executing

```bash
# Check if games exist
odds status events --days 7

# Test job manually
odds scheduler run-once fetch-odds

# View logs
tail -f logs/odds_pipeline.log
```

### Migration Issues

```bash
# Check migration status
alembic current
alembic history

# Apply pending migrations
alembic upgrade head
```

## Documentation

- **[CLAUDE.md](CLAUDE.md)** - Complete technical specification for LLM agents
- **[BACKTESTING_GUIDE.md](BACKTESTING_GUIDE.md)** - Backtesting strategies and metrics reference

## License

Private project - all rights reserved.
