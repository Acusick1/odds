# CLI Reference

> Complete reference for all CLI commands.

## Data Collection

### Fetch Current Odds

```bash
odds fetch current
odds fetch current --sport basketball_nba
```

Manually fetches current odds for upcoming games.

### Fetch Scores

```bash
odds fetch scores
odds fetch scores --sport basketball_nba --days 3
```

Fetches game scores and updates event status.

## Backfill Commands

### Discover Historical Games

```bash
odds discover games --start YYYY-MM-DD --end YYYY-MM-DD
```

Discovers historical games from The Odds API for potential backfill.

### Detect Gaps and Generate Plan

```bash
odds backfill gaps --start YYYY-MM-DD --end YYYY-MM-DD
odds backfill gaps --start YYYY-MM-DD --end YYYY-MM-DD --max-quota 5000
odds backfill gaps --start YYYY-MM-DD --end YYYY-MM-DD --output gap_plan.json
```

Analyzes tier coverage gaps (OPENING, EARLY, SHARP, PREGAME, CLOSING) and generates a prioritized backfill plan. Prioritizes CLOSING tier (highest value) down to OPENING (lowest). Use `--max-quota` to limit API usage.

### Plan Backfill

```bash
odds backfill plan --start YYYY-MM-DD --end YYYY-MM-DD --games N
```

Creates a backfill plan file with selected games.

### Execute Backfill

```bash
odds backfill execute --plan backfill_plan.json
```

Executes a prepared backfill plan.

### Backfill Status

```bash
odds backfill status
```

Shows progress of ongoing or completed backfill.

### Backfill Scores

```bash
odds backfill scores --start YYYY-MM-DD --end YYYY-MM-DD
odds backfill scores --start YYYY-MM-DD --end YYYY-MM-DD --dry-run
```

Backfills missing scores using NBA API. Use `--dry-run` to preview changes.

## Backtesting Commands

See `BACKTESTING_GUIDE.md` for comprehensive backtesting documentation.

### Run Backtest

```bash
odds backtest run --strategy STRATEGY --start YYYY-MM-DD --end YYYY-MM-DD
odds backtest run --strategy basic_ev --start 2024-10-01 --end 2024-12-31 --output-json results.json
```

**Required:**
- `--strategy, -s`: Strategy name (flat, basic_ev, arbitrage)
- `--start`: Start date
- `--end`: End date

**Optional:**
- `--bankroll, -b`: Initial bankroll (default: 10000)
- `--output-json, -j`: Save to JSON file
- `--output-csv, -c`: Save to CSV file
- `--bet-sizing`: Method (default: fractional_kelly)
- `--kelly-fraction`: Kelly fraction (default: 0.25)
- `--flat-bet-amount`: Flat bet size (default: 100)
- `--percentage-bet`: Percentage of bankroll (default: 0.02)

### Show Results

```bash
odds backtest show results.json
odds backtest show results.json --verbose
```

Displays saved backtest results.

### Compare Results

```bash
odds backtest compare result1.json result2.json result3.json
```

Compares multiple backtest results side-by-side.

### Export to CSV

```bash
odds backtest export results.json --output bets.csv
```

Converts JSON results to CSV format.

### List Strategies

```bash
odds backtest list-strategies
```

Lists all available strategies with parameters.

## Status & Monitoring

### System Status

```bash
odds status show
odds status show --verbose
```

Shows system health overview:
- Last fetch time and success rate
- API quota remaining
- Event counts and database status
- Data quality metrics

### Check API Quota

```bash
odds status quota
```

Displays remaining API requests for current billing period.

### List Events

```bash
odds status events --days 7
odds status events --team "Lakers"
```

Lists recent events, optionally filtered by team.

## Validation & Data Management

### Validate Coverage

```bash
odds validate coverage --start YYYY-MM-DD --end YYYY-MM-DD
```

Validates data completeness for date range.

### Copy from Production

```bash
odds copy from-prod --start YYYY-MM-DD --end YYYY-MM-DD
```

Copies data from production database to local database.

## Scheduler Commands

### Start Scheduler

```bash
odds scheduler start
```

Starts the scheduler (local backend only). Runs until Ctrl+C.

## Historical Backfill Strategy

### Sampling Approach

For initial backfill, target a representative sample:
- Select games distributed across season
- Include all 30 teams proportionally
- Mix of matchup types
- ~20% sample rate recommended

### Selection Criteria

`packages/odds-analytics/odds_analytics/game_selector.py` implements:
- Even distribution across dates
- Proportional team representation
- Variety in matchups
- Multiple selection strategies (uniform, random, weighted)

### Workflow

```bash
# 1. Discover available games
odds discover games --start 2024-10-01 --end 2024-12-31

# 2. Plan backfill (select 250 games)
odds backfill plan --start 2024-10-01 --end 2024-12-31 --games 250

# 3. Execute plan
odds backfill execute --plan backfill_plan.json

# 4. Backfill any missing scores
odds backfill scores --start 2024-10-01 --end 2024-12-31

# 5. Check status
odds backfill status
```

### Purpose

- Validate schema with real data
- Test query patterns and performance
- Enable immediate backtesting
- Verify data quality checks
