---
type: component
title: odds CLI
summary: "The odds Typer app, its 18 command groups and the wiki page that owns each, plus shared conventions (--sport default, --db override, exit styles)."
sources:
  - packages/odds-cli/odds_cli/main.py::app
  - packages/odds-cli/odds_cli/main.py::callback
  - packages/odds-cli/odds_cli/db_override.py::override_database_url
  - packages/odds-cli/odds_cli/db_override.py::swap_dbname
  - packages/odds-cli/odds_cli/commands/fetch.py
  - packages/odds-cli/odds_cli/commands/backtest.py::STRATEGIES
  - packages/odds-cli/odds_cli/commands/scheduler.py::run_once
  - packages/odds-cli/odds_cli/commands/train.py
  - packages/odds-cli/odds_cli/commands/validate.py
  - docs/CLI.md#command-groups
verified_at: 7887039
---

# odds CLI

`packages/odds-cli/odds_cli/main.py::app` is a Typer app installed as the `odds` script. Each group is a module in `packages/odds-cli/odds_cli/commands/` registered with `add_typer`. `docs/CLI.md` is the reference; each command's `--help` is authoritative for flags and defaults ([Documentation map](documentation-map.md)). The `callback` configures human-readable logging ([Code conventions](code-conventions.md#logging)). Commands are documented on the page that owns their domain, so no command reference is repeated here.

## Groups and owning pages

| Group | Page |
|---|---|
| `fetch` | [Ingestion flow](odds-ingestion-flow.md), [fetch-scores job](job-fetch-scores.md) |
| `scrape` | [OddsPortal scraper](oddsportal-scraper.md) |
| `discover`, `backfill` | [Historical backfill](howto-historical-backfill.md) |
| `backtest` | [Run a backtest](howto-run-backtest.md), [Strategies](backtest-strategies.md) |
| `train` | [Training pipeline](training-pipeline.md), [Tuner](hyperparameter-tuner.md), [Training config](training-config.md) |
| `model` | [S3 model store](s3-model-store.md) |
| `status` | [Reporting surfaces](reporting-surfaces.md), [Odds API quota](odds-api-quota.md) |
| `validate`, `quality` | [Completeness validation](completeness-validation.md) |
| `copy` | [Copy from production](copy-from-production.md) |
| `scheduler` | [Local scheduler](local-scheduler.md) |
| `polymarket` | [Polymarket pipeline](polymarket-pipeline.md) |
| `injuries` | [NBA injury reports](nba-injury-reports.md) |
| `nba-stats`, `pbpstats` | [NBA stats data](nba-stats-data.md) |
| `paper` | [Paper trading](paper-trading.md) |
| `agent` | [agent-run job](job-agent-run.md) |

## Shared conventions

- **Sport.** `--sport` takes a [sport key](sport-and-market-keys.md). Defaults differ per group (below).
- **Database override.** `scheduler start` and `agent run` accept `--db NAME`, which calls `override_database_url`: it swaps the dbname in `DATABASE_URL`, writes `os.environ`, and clears the settings cache (`swap_dbname` rejects names containing `/` or `?`). In-process sessions keep the original engine ([Configuration](configuration.md#hazards)).
- **Async.** Commands wrap async code in `asyncio.run()`.

## Hazards

- `--sport` defaults to `basketball_nba` in `fetch current`, `fetch scores`, `discover` and `quality`; `copy from-prod` defaults to `soccer_epl`; `backtest` and `train` take an optional filter; `agent run` requires it. Pass it explicitly ([Project pivots](project-pivots.md)).
- Two exit styles coexist: `typer.Exit(code=...)` in most groups and `sys.exit(1)` in `validate` (10 sites) and `quality` (3).
- The root help text and callback docstring say "NBA". `backtest run --strategy` help lists `lstm`, which is not a key of `STRATEGIES`. `scheduler run-once` help names three of the 14 jobs, and `scheduler info` has no help text because its docstring follows a statement.
- `train validate` warns "Very small test split" for `test_split < 0.1`, although `test_split: 0.0` is the documented protocol ([Walk-forward CV](walk-forward-cv.md)).
- `copy from-prod` takes dates as positional arguments where `backfill` and `quality` use `--start`/`--end`.
- `odds agent run` imports the underscore-private `_run_claude_agent` from odds-lambda.

## Related

- [Project overview](project-overview.md)
- [odds-analytics package](package-odds-analytics.md)
