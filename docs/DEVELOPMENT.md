# Development Reference

> Reference documentation for configuration and development workflows. Setup is in the
> [README Quick Start](../README.md#quick-start); code style is in
> [AGENTS.md](../AGENTS.md#code-style).

## Configuration Management

Configuration uses Pydantic Settings with environment variables.
`packages/odds-core/odds_core/config.py` is authoritative for every setting and default;
the tables below cover the common ones. See `.env.example` for a complete template.

### API Settings (`APIConfig`)

| Variable | Description | Default |
|----------|-------------|---------|
| `ODDS_API_KEY` | The Odds API authentication key | Required |
| `ODDS_API_BASE_URL` | API base URL | https://api.the-odds-api.com/v4 |
| `ODDS_API_KEYS` | Comma-separated rotation keys | Optional (overrides single key) |
| `ODDS_API_QUOTA` | Monthly request quota per key | 500 |

### Database Settings (`DatabaseConfig`)

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | Required |
| `DATABASE_POOL_SIZE` | Connection pool size | 5 (NullPool for Lambda) |

### Data Collection (`DataCollectionConfig`)

| Variable | Description | Default |
|----------|-------------|---------|
| `SPORTS` | Sports to track | ["soccer_epl", "baseball_mlb"] (also supports "basketball_nba") |
| `BOOKMAKERS` | Odds API bookmaker list | [pinnacle, circasports, draftkings, fanduel, betmgm, williamhill_us, betrivers, bovada] |
| `MARKETS` | Bet types | ["h2h", "spreads", "totals"] |
| `REGIONS` | Odds regions | ["us"] |

### Scheduler Settings (`SchedulerConfig`)

| Variable | Description | Default |
|----------|-------------|---------|
| `SCHEDULER_BACKEND` | Backend type | "local" |
| `SCHEDULER_DRY_RUN` | Dry-run mode | false |
| `SCHEDULER_LOOKAHEAD_DAYS` | Days ahead to check | 7 |
| `SCHEDULER_BOOTSTRAP_JOBS` | Jobs the local scheduler starts | fetch-oddsportal, fetch-oddsportal-results, fetch-espn-fixtures, fetch-betfair-exchange, fetch-mlb-probables |

### AWS Settings (`AWSConfig`)

| Variable | Description | Default |
|----------|-------------|---------|
| `AWS_REGION` | AWS region for Lambda | Optional |
| `AWS_LAMBDA_ARN` | Lambda function ARN | Optional |
| `AWS_RULE_PREFIX` | EventBridge rule prefix | Optional |

Inside Lambda the AWS backend reads `LAMBDA_ARN` and `RULE_PREFIX` instead; see
[DEPLOYMENT.md](DEPLOYMENT.md#configuration).

### Data Quality (`DataQualityConfig`)

| Variable | Description | Default |
|----------|-------------|---------|
| `ENABLE_VALIDATION` | Run data quality checks | true |
| `REJECT_INVALID_ODDS` | Reject vs log invalid data | false |

### Alerts (`AlertConfig`)

| Variable | Description | Default |
|----------|-------------|---------|
| `DISCORD_WEBHOOK_URL` | Discord webhook for alerts/digests | Optional |
| `ALERT_ENABLED` | Enable alert sending | false |
| `QUOTA_WARNING_THRESHOLD` | Quota warning at this fraction remaining | 0.2 |
| `QUOTA_CRITICAL_THRESHOLD` | Quota critical at this fraction remaining | 0.1 |
| `CONSECUTIVE_FAILURES_THRESHOLD` | Alert after N consecutive failures | 3 |
| `STALE_DATA_HOURS` | Alert when no new snapshots for this long | 2 |
| `ALERT_RATE_LIMIT_MINUTES` | Minimum gap between alerts of one type | 30 |
| `DATA_QUALITY_ERROR_THRESHOLD` | Alert above this many quality errors in 24h | 10 |

Heartbeat expectations and retention are also in `AlertConfig`; see
[REPORTING.md](REPORTING.md#3-health-monitoring--packagesodds-lambdaodds_lambdahealth_monitorpy).

### Logging (`LoggingConfig`)

| Variable | Description | Default |
|----------|-------------|---------|
| `LOG_LEVEL` | Logging verbosity | INFO |
| `LOG_FILE` | Log file path | logs/odds_pipeline.log |

### Other settings

`ModelConfig` (`MODEL_NAME`, `MODEL_BUCKET`), `PolymarketConfig` (`POLYMARKET_*`, see
[POLYMARKET.md](POLYMARKET.md#configuration)) and `BetfairConfig` (`BETFAIR_*`) are defined
in the same file.

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
2. Implement the `evaluate_opportunity()` method
3. Register it in the `STRATEGIES` dict in `packages/odds-cli/odds_cli/commands/backtest.py`
4. Write tests
5. Document if complex

### Modifying API Client

`packages/odds-lambda/odds_lambda/data_fetcher.py`

1. Preserve retry logic (tenacity decorators)
2. Update quota tracking if request patterns change
3. Add fixtures to `tests/fixtures/` for new endpoints
4. Update integration tests
5. Never remove retry/backoff, key rotation or error handling

### Adding Scheduler Jobs

`packages/odds-lambda/odds_lambda/jobs/*.py`

1. Create an `async def main(ctx: JobContext)` entry point
2. Add it to `_JOB_MODULE_MAP` in `packages/odds-lambda/odds_lambda/scheduling/jobs.py`
3. Implement self-scheduling (calculate next execution time)
4. Wrap the job body in `job_alert_context(job_name)` so failures alert and heartbeats are recorded
5. If the job runs per sport, add it to `_PER_SPORT_JOBS`; if it runs on a fixed local
   cron, add it to `_JOB_CRON_MAP` (both in `scheduling/jobs.py`)
6. Test locally: `odds scheduler run-once <job>`, then `odds scheduler start`

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

Tests run against an in-process PostgreSQL (py-pglite, port 5434) set up in
`tests/conftest.py`; they never touch the local or remote databases. See
[DATABASE.md](DATABASE.md#test-database).

## Running Experiments

Experiments live in `experiments/scripts/` as standalone Python scripts. Each script writes
its outputs to `experiments/results/<experiment_name>/`. Findings and the experiment log
are in [MODELING.md](MODELING.md).

### Conventions

- Scripts use the existing training pipeline (`prepare_training_data`, `MLTrainingConfig`,
  etc.) to load data.
- Reuse YAML configs in `experiments/configs/` where possible.
- Scripts must be runnable via `uv run python experiments/scripts/<script>.py`.
- Commit the script before running it, then commit the results, so `FINDINGS.md` can record
  the git SHA of the code that produced them.

### Required outputs

- **`FINDINGS.md`**, the primary record of the experiment, with these sections:
  - **Setup**: date, git SHA, dataset size, sampling method, target definition, exact
    command to reproduce
  - **Key Results**: numbers, tables, significance
  - **Interpretation**: what the results mean, caveats, alternative explanations
  - **Implications**: what to do next, go/no-go recommendation for downstream experiments
- **Plots** saved as PNGs in the same directory.
- **Data artifacts** (CSVs, etc.) for downstream analysis.

### Verification artifacts

Every experiment saves feature verification artifacts to catch silent data bugs (zeroed
features, missing values, wrong data selection). `odds train run` and `odds train tune`
save them automatically. Standalone scripts call `save_verification_artifacts()` after
preparing data:

```python
from odds_analytics.training import save_verification_artifacts

save_verification_artifacts(X, y, feature_names, output_dir, event_ids=event_ids)
```

This writes three files:

- **`feature_stats.csv`**: per-feature count, null_count, null_pct, mean, std, min, max,
  nunique. Check for features that are all-NaN, all-zero, or constant.
- **`feature_sample.csv`**: the first 20 rows of the feature matrix with event IDs and
  target. Spot-check that values look reasonable.
- **`feature_target_corr.csv`**: per-feature Pearson and Spearman correlation with the
  target. Check for flipped signs or lost signal against expectations.

Before reporting results in `FINDINGS.md`, verify that:

1. No features are 100% NaN or constant, unless expected (for example an optional feature
   group that was not loaded).
2. The feature count matches the configured feature groups.
3. Correlations are directionally consistent with prior experiments.

### After running

1. Add a row to the [Experiment log](MODELING.md#experiment-log) with the headline result
   and decision.
2. If the result changes what we know, update
   [Current conclusions](MODELING.md#current-conclusions) in place and add the experiment
   to [Evidence](MODELING.md#evidence), linking its `FINDINGS.md`.

## Dependency Management

Project uses **uv** workspace architecture. Dependencies managed via `pyproject.toml` and `uv.lock`.

## Code Quality

Pre-commit hooks (`.pre-commit-config.yaml`) run on every commit, and the `Lint` workflow runs them on all files for every pushed branch except `main`:

- **Ruff** lints and formats Python.
- **Vale** checks `README.md`, `AGENTS.md` and `docs/` for wording that narrates change instead of describing the current state. Rules live in `.vale/styles/Temporal/`; errors block the commit, warnings are advisory.
- **lychee** checks Markdown links and `#heading` anchors offline. It runs in Docker.

```bash
# Enable hooks
pre-commit install

# Run every hook on every file
pre-commit run --all-files
```

## Issue Workflow

See [docs/agents/issue-tracker.md](agents/issue-tracker.md).

## Logging

Uses **structlog** for structured logging.

- Levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
- Format: human-readable console output and a plain-text log file; JSON when JSON output is
  enabled (Lambda)
- Files rotate at 10 MB, keeping 5 backups (`packages/odds-core/odds_core/logging_setup.py`)
- Config: `LOG_LEVEL`, `LOG_FILE` environment variables

## Extensibility

### Adding Sports/Bookmakers/Markets

`BOOKMAKERS` and `MARKETS` are config lists in `.env` and need no code changes. `SPORTS`
only accepts a `SportKey` (`packages/odds-core/odds_core/sports.py`), so a new sport
needs code as well as config.

Strategies are covered under [Adding Betting Strategies](#adding-betting-strategies).
