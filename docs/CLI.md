# CLI Reference

> Complete reference for all CLI commands.

## Command Groups

| Group | Purpose |
|-------|---------|
| `fetch` | Fetch odds data from APIs |
| `scrape` | Scrape odds from OddsPortal |
| `discover` | Discover upcoming/historical games |
| `backfill` | Historical data backfill |
| `backtest` | Backtest betting strategies |
| `train` | Train and tune ML models |
| `model` | Model artifact management (S3) |
| `status` | System status and monitoring |
| `validate` | Validate data completeness |
| `quality` | Data quality coverage analysis |
| `copy` | Copy data from production |
| `scheduler` | Scheduler management (local) |
| `polymarket` | Polymarket data operations (deprioritized) |
| `injuries` | NBA injury report operations |
| `nba-stats` | NBA game log operations |
| `pbpstats` | PBPStats player season stats |

## Data Collection

### Fetch Current Odds

```bash
odds fetch current
odds fetch current --sport basketball_nba
odds fetch current --sport soccer_epl
```

Manually fetches current odds for upcoming games.

### Fetch Scores

```bash
odds fetch scores
odds fetch scores --sport basketball_nba --days 3
```

Fetches game scores and updates event status.

### Scrape OddsPortal

```bash
odds scrape upcoming
odds scrape upcoming --league england-premier-league --market 1x2 --market over_under_2.5
odds scrape upcoming --dry-run
odds scrape upcoming --from-file matches.json
```

Scrapes upcoming match odds from OddsPortal via OddsHarvester (headless browser).

**Options:**
- `--sport, -s`: OddsHarvester sport name (default: football)
- `--league, -l`: OddsHarvester league name (default: england-premier-league)
- `--market, -m`: Markets to scrape, repeatable (default: 1x2)
- `--dry-run`: Scrape and convert but don't store
- `--from-file`: Load matches from JSON file instead of scraping

## Discovery Commands

### Discover Upcoming Games (free)

```bash
odds discover upcoming
odds discover upcoming --sport basketball_nba
```

Syncs upcoming games from the free `/events` endpoint (0 quota units). Runs automatically at the start of every `fetch-odds` job, but can also be called directly.

### Discover Historical Games

```bash
odds discover games --start YYYY-MM-DD --end YYYY-MM-DD
```

Discovers historical games from The Odds API for potential backfill.

## Backfill Commands

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

## ML Training Commands

### Train a Model

```bash
odds train run --config experiments/configs/my_config.yaml
odds train run --config my_config.yaml --output models/output.pkl --verbose
odds train run --config my_config.yaml --track --sport soccer_epl
```

Trains an ML model using a configuration file.

**Required:**
- `--config, -c`: Path to training configuration file (YAML/JSON)

**Optional:**
- `--output, -o`: Override output path for model
- `--dry-run`: Show what would be done without executing
- `--verbose, -v`: Enable detailed output logging
- `--track`: Enable MLflow experiment tracking
- `--tracking-uri`: Override MLflow tracking URI
- `--sport`: Override sport key filter

### Tune Hyperparameters

```bash
odds train tune --config experiments/configs/my_config.yaml
odds train tune --config my_config.yaml --n-trials 200 --train-best
```

Runs Optuna hyperparameter optimization. Requires config with a `tuning` section.

**Required:**
- `--config, -c`: Path to training configuration file with tuning section

**Optional:**
- `--output, -o`: Override output path for best config
- `--train-best`: Train final model with best parameters after tuning
- `--n-trials`: Override number of trials from config
- `--timeout`: Override timeout in seconds
- `--study-name`: Optuna study name for persistence/resumption
- `--storage`: Optuna storage URL
- `--track`: Enable MLflow experiment tracking
- `--sport`: Override sport key filter

### Validate Config

```bash
odds train validate --config experiments/configs/my_config.yaml
```

Validates a configuration file without executing training.

### List Configs

```bash
odds train list-configs
odds train list-configs --directory experiments/configs
```

Lists available configuration files in a directory.

## Model Management

### Publish Model to S3

```bash
odds model publish --name epl-clv-home --path models/model.pkl
odds model publish --name epl-clv-home --path models/model.pkl --version v1.2
```

Publishes a trained model to S3. Uploads `model.pkl`, `config.yaml`, and `metadata.json` to both `{name}/{version}/` and `{name}/latest/`.

**Required:**
- `--name, -n`: Model name (S3 key prefix)
- `--path, -p`: Path to model .pkl file

**Optional:**
- `--bucket, -b`: S3 bucket name (default: odds-models-{account_id})
- `--version, -v`: Version label (default: git SHA or timestamp)

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

## NBA Data Commands

### Game Logs

```bash
odds nba-stats fetch --season 2024-25
odds nba-stats fetch --all
odds nba-stats status
```

Fetches NBA team game logs from stats.nba.com via Playwright (headless Firefox). `--all` fetches seasons 2021-22 through current.

### Player Stats (PBPStats)

```bash
odds pbpstats fetch --season 2024-25
odds pbpstats fetch --all
odds pbpstats backfill
odds pbpstats status
```

Fetches player season stats (on/off ratings, minutes) from PBPStats API. Used for injury impact scoring.

### Injury Reports

```bash
odds injuries fetch
odds injuries backfill --season 2024-25
odds injuries backfill --season 2025-26 --hours-before 12,8,2 --dry-run
odds injuries status
```

Fetches NBA injury reports from official PDFs. Backfill computes target timestamps from game `commence_time` values.

## Polymarket Commands (deprioritized)

Pipeline exists but is inactive. Not accessible from UK, data likely collinear with sportsbook odds.

### Discover Events

```bash
odds polymarket discover
```

Lists active NBA events currently on Polymarket from the Gamma API.

### Pipeline Status

```bash
odds polymarket status
```

Shows collection health: event counts (linked/unlinked), total snapshots, date coverage, last fetch result.

### Backfill Price History

```bash
odds polymarket backfill
odds polymarket backfill --include-spreads --include-totals
odds polymarket backfill --dry-run
```

Fetches historical price data for closed markets from the CLOB API.

### Link Events

```bash
odds polymarket link
odds polymarket link --dry-run
```

Matches unlinked Polymarket events to sportsbook Event records via ticker parsing.

### View Order Book

```bash
odds polymarket book <ticker>
```

Shows the live order book for a game's moneyline market.

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
