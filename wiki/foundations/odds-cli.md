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
  - odds_cli.commands.scheduler.run_once
  - odds_cli.commands.scheduler.info
  - odds_cli.commands.validate
  - odds_cli.commands.train
  - odds_cli.commands.evaluate
  - odds_cli.commands.agent
  - packages/odds-cli/pyproject.toml
status: stable
---

# The odds CLI

`odds_cli.main.app` is a Typer app installed as the `odds` script. Each group is a module in `packages/odds-cli/odds_cli/commands/` registered with `add_typer`. Each command's `--help` is authoritative for flags and defaults. Commands are documented on the page that owns their domain (the `polymarket` group's Polymarket EPL commands included); none is repeated here. `odds_cli.main.callback` configures human-readable logging ([Code conventions](code-conventions.md#logging)). The root help text and banner describe the tool as NBA (#448).

## Command groups

`packages/odds-cli/pyproject.toml` declares Typer, Rich, PyYAML, structlog, sqlalchemy, apscheduler and numpy, each because `odds_cli` imports it directly; `optuna` comes only through analytics' `training` extra.

| Group | Page |
|---|---|
| `odds fetch` | [Odds API ingestion service](../collection/odds-ingestion-service.md), [Results and status jobs](../collection/results-jobs.md) |
| `scrape` | [OddsPortal scraper](../collection/oddsportal-scraper.md) |
| `odds_cli.commands.discover`, `odds_cli.commands.backfill` | [Odds API historical backfill](../collection/odds-api-backfill.md); `discover games` and `backfill execute` store events and snapshots through the [event resolver](../storage/event-resolver.md); `discover games` commits every 100 games and prints dropped counts, and `backfill gaps` and `plan` write `odds_api_id` into plan games |
| `odds train` | [Training pipeline](../modelling/training-pipeline.md) (`tune` stores a new timestamped study in a local SQLite file unless `--in-memory`; both subcommands pass the split's sample weights and kick-off times on, [sample weights](../modelling/training-pipeline.md#sample-weights)), [Hyperparameter tuner](../modelling/hyperparameter-tuner.md) |
| `odds evaluate` | [Retail-EV evaluation](../evaluation/retail-ev-evaluation.md) (`--allow-incomplete <event ID>`, repeatable, and `--force` are recorded in the manifest) |
| `odds status` | [odds status](../operations/health-monitor.md#odds-status), [Odds API quota](../collection/odds-api-quota.md) |
| `validate`, `quality` | [Coverage and completeness checks](../storage/coverage-checks.md) |
| `copy` | [Copy production data locally](howto-copy-production-data.md) |
| `scheduler` | [Local scheduler](../scheduling/local-scheduler.md) |
| `polymarket` | [Polymarket pipeline and tables](../collection/polymarket.md) (`odds_cli.commands.polymarket.link` matches an event from its ticker), [Polymarket odds snapshots](../collection/polymarket-odds-snapshots.md#backfill-command) (`odds_cli.commands.polymarket.backfill_odds` attaches matches to events before storing history, and reports the matches that failed) |
| `injuries`, `nba-stats`, `pbpstats` | [NBA feeds](../collection/nba-injuries-and-stats.md) |
| `paper` | [Paper trading](../agent/paper-trading.md) |
| `candidates` | [Candidate registry](../evaluation/candidate-registry.md#odds-candidates-list), [Replay backtest](../evaluation/candidate-backtest.md), [Parity check](../evaluation/candidate-parity.md) |
| `agent` | [Agent runtime and the agent-run job](../agent/agent-run-job.md) |

## Sport defaults

`--sport` takes a [sport key](../collection/sport-and-market-keys.md); defaults differ per group (#465):

| Command | Default |
|---|---|
| `odds_cli.commands.fetch` (`current`, `scores`; `scores` finds each event by its Odds API id and logs `score_event_not_found` for an unknown one) | `basketball_nba`; `results` defaults to `soccer_epl`, settles `baseball_mlb` from the MLB Stats API and prints the settled, cancelled, postponed, moved and unresolved counts ([results jobs](../collection/results-jobs.md#fetch-oddsportal-results)) |
| `odds_cli.commands.discover` | `basketball_nba` |
| `odds_cli.commands.quality` | `basketball_nba` |
| `odds_cli.commands.polymarket.discover` | `basketball_nba` |
| `odds_cli.commands.copy_from_prod` | `soccer_epl` |
| `odds_cli.commands.train` | optional filter |
| `odds_cli.commands.agent.run` | required |

Pass `--sport` explicitly. [About the system and its pivots](project-overview.md#nba-era-defaults-still-in-live-code) lists the other NBA-era defaults.

## Shared conventions

- **Database override.** `scheduler start` and `agent run` accept `--db NAME`, which calls `odds_cli.db_override.override_database_url`: it swaps the dbname in `DATABASE_URL`, writes `os.environ`, and clears the settings cache (`odds_cli.db_override.swap_dbname` rejects names containing `/` or `?`). The engine already built in-process keeps the original URL; only subprocesses see the swap ([Engine creation](databases-and-environments.md#engine-creation), [Read at import](configuration.md#read-at-import)). On `odds agent run` it has no effect at all when `AGENT_DATABASE_URL` is set ([Database role](../agent/agent-run-job.md#database-role)).
- **Async.** Commands wrap async code in `asyncio.run()`.
- **Exit codes.** Most groups exit with `typer.Exit(code=...)`; `odds_cli.commands.validate` (10 sites) and `odds_cli.commands.quality` (3) call `sys.exit(1)`. `odds_cli.commands.copy_from_prod` exits 1 when `odds_lambda.copy_from_production.CopyStats.events_failed` is non-zero.

## Help text

The help text disagrees with the code in four places (#448, #466):

- The root help, banner and callback docstring say "NBA" (`odds_cli.main.app`, `odds_cli.main.callback`).
- `odds scheduler run-once` help names a sport-suffixed per-sport job and `update-status` of the 15 in `odds_lambda.scheduling.jobs._JOB_MODULE_MAP` (`odds_cli.commands.scheduler.run_once`).
- `odds scheduler info` has no help text, because its docstring follows a statement (`odds_cli.commands.scheduler.info`). It lists each sport's Odds API bookmakers.

## Related

- [Workspace packages and repository layout](workspace-packages.md)
- [Configuration](configuration.md)
