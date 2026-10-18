---
type: reference
title: The odds CLI
description: The Typer app, its command groups and owning pages, shared flags (--sport defaults per command, --db), and help-text facts.
sources:
  - odds_cli.main.app
  - odds_cli.main.callback
  - odds_cli.db_override.override_database_url
  - odds_cli.db_override.swap_dbname
  - odds_cli.commands.fetch
  - odds_cli.commands.copy_from_prod
  - odds_cli.commands.polymarket
  - odds_cli.commands.quality
  - odds_cli.commands.discover
  - odds_cli.commands.backtest
  - odds_cli.commands.backtest.STRATEGIES
  - odds_cli.commands.scheduler.run_once
  - odds_cli.commands.scheduler.info
  - odds_cli.commands.validate
  - odds_cli.commands.train
  - odds_cli.commands.agent
  - packages/odds-cli/pyproject.toml
status: stable
---

# The odds CLI

`odds_cli.main.app` is a Typer app installed as the `odds` script. Each group is a module in `packages/odds-cli/odds_cli/commands/` registered with `add_typer`. Each command's `--help` is authoritative for flags and defaults. Commands are documented on the page that owns their domain; none is repeated here. `odds_cli.main.callback` configures human-readable logging ([Code conventions](code-conventions.md#logging)). The root help text and banner describe the tool as NBA (#448).

## Command groups

| Group | Page |
|---|---|
| `odds fetch` | [Odds API ingestion service](../collection/odds-ingestion-service.md), [Results and status jobs](../collection/results-jobs.md) |
| `scrape` | [OddsPortal scraper](../collection/oddsportal-scraper.md) |
| `odds_cli.commands.discover`, `odds_cli.commands.backfill` | [Odds API historical backfill](../collection/odds-api-backfill.md) |
| `backtest` | [Backtest engine](../evaluation/backtest-engine.md), [Backtest strategies](../evaluation/backtest-strategies.md) |
| `odds train` | [Training pipeline](../modelling/training-pipeline.md) (`tune` stores a new timestamped study in a local SQLite file unless `--in-memory`), [Hyperparameter tuner](../modelling/hyperparameter-tuner.md) |
| `model` | [Model store](../modelling/model-store.md) |
| `odds status` | [odds status](../operations/health-monitor.md#odds-status), [Odds API quota](../collection/odds-api-quota.md) |
| `validate`, `quality` | [Coverage and completeness checks](../storage/coverage-checks.md) |
| `copy` | [Copy production data locally](howto-copy-production-data.md) |
| `scheduler` | [Local scheduler](../scheduling/local-scheduler.md) |
| `polymarket` | [Polymarket pipeline and tables](../collection/polymarket.md) |
| `injuries`, `nba-stats`, `pbpstats` | [NBA feeds](../collection/nba-injuries-and-stats.md) |
| `paper` | [Paper trading](../agent/paper-trading.md) |
| `agent` | [Agent runtime and the agent-run job](../agent/agent-run-job.md) |

## Sport defaults

`--sport` takes a [sport key](../collection/sport-and-market-keys.md); defaults differ per group (#465):

| Command | Default |
|---|---|
| `odds_cli.commands.fetch` (`current`, `scores`) | `basketball_nba` |
| `odds_cli.commands.discover` | `basketball_nba` |
| `odds_cli.commands.quality` | `basketball_nba` |
| `odds_cli.commands.polymarket.discover` | `basketball_nba` |
| `odds_cli.commands.copy_from_prod` | `soccer_epl` |
| `odds_cli.commands.backtest`, `odds_cli.commands.train` | optional filter |
| `odds_cli.commands.agent.run` | required |

Pass `--sport` explicitly. [About the system and its pivots](project-overview.md#nba-era-defaults-still-in-live-code) lists the other NBA-era defaults.

## Shared conventions

- **Database override.** `scheduler start` and `agent run` accept `--db NAME`, which calls `odds_cli.db_override.override_database_url`: it swaps the dbname in `DATABASE_URL`, writes `os.environ`, and clears the settings cache (`odds_cli.db_override.swap_dbname` rejects names containing `/` or `?`). The engine already built in-process keeps the original URL; only subprocesses see the swap ([Engine creation](databases-and-environments.md#engine-creation), [Read at import](configuration.md#read-at-import)). On `odds agent run` it has no effect at all when `AGENT_DATABASE_URL` is set ([Database role](../agent/agent-run-job.md#database-role)).
- **Async.** Commands wrap async code in `asyncio.run()`.
- **Exit codes.** Most groups exit with `typer.Exit(code=...)`; `odds_cli.commands.validate` (10 sites) and `odds_cli.commands.quality` (3) call `sys.exit(1)`.

## Help text

The help text disagrees with the code in four places (#448, #466):

- The root help, banner and callback docstring say "NBA" (`odds_cli.main.app`, `odds_cli.main.callback`).
- `--strategy` on `odds backtest run` lists `lstm`, which is not a key of `odds_cli.commands.backtest.STRATEGIES`.
- `odds scheduler run-once` help names three jobs (`fetch-odds`, `fetch-scores`, `update-status`) of the 14 in `odds_lambda.scheduling.jobs._JOB_MODULE_MAP` (`odds_cli.commands.scheduler.run_once`).
- `odds scheduler info` has no help text, because its docstring follows a statement (`odds_cli.commands.scheduler.info`).

## Related

- [Workspace packages and repository layout](workspace-packages.md)
- [Configuration](configuration.md)
