---
type: component
title: odds-core package
summary: "The shared base package (models, config, database, alerts, odds maths, helpers) and the dependency rule that keeps it at the bottom of the graph."
sources:
  - packages/odds-core/odds_core/__init__.py
  - packages/odds-core/odds_core/database.py
  - packages/odds-core/odds_core/odds_math.py
  - packages/odds-core/odds_core/snapshot_utils.py::extract_odds_from_snapshot
  - packages/odds-core/odds_core/utils.py::raw_data_has_market
  - packages/odds-core/odds_core/paper_trade_models.py::PaperTrade
  - packages/odds-core/pyproject.toml
verified_at: 7887039
---

# odds-core package

`packages/odds-core/odds_core/` is the bottom of the workspace graph: [odds-lambda](package-odds-lambda.md) depends on it, [odds-analytics](package-odds-analytics.md) on both, and `odds-mcp` and `odds-cli` on all three. Its declared dependencies are only sqlmodel, asyncpg, pydantic-settings, python-dotenv, structlog and aiohttp (`packages/odds-core/pyproject.toml`). Package layout: [AGENTS.md](../AGENTS.md#package-structure).

## Module map

| Module | Holds |
|---|---|
| `models.py` | `Event`, `OddsSnapshot`, `Odds`, `FetchLog`, `DataQualityLog`, `AlertHistory`, `utc_now` ([events](table-events.md), [snapshots](table-odds-snapshots.md), [odds](table-odds.md)) |
| `prediction_models.py`, `match_brief_models.py`, `paper_trade_models.py`, `agent_wakeup_models.py` | Prediction and agent tables ([briefs](match-briefs.md), [paper trades](paper-trading.md)) |
| `epl_data_models.py`, `mlb_data_models.py`, `injury_models.py`, `game_log_models.py`, `player_stats_models.py`, `polymarket_models.py` | Sport-context and Polymarket tables |
| `config.py` | Pydantic Settings, see [Configuration](configuration.md) |
| `database.py` | Engine and `async_session_maker`, see [Database connections](database-connections.md) |
| `alerts.py` | `AlertManager`, `DiscordAlert`, see [Alerts module](alerts-module.md) |
| `odds_math.py` | Price conversion, implied probability, devig, see [Odds math](odds-math.md) |
| `snapshot_utils.py`, `utils.py` | `extract_odds_from_snapshot` (parse `raw_data` into `Odds`), `raw_data_has_market` |
| `sports.py`, `team.py`, `time.py`, `logging_setup.py` | [Sport keys](sport-and-market-keys.md), team-name normalization, [timezone helpers](timezone-handling.md), structlog setup |
| `api_models.py` | The Odds API response models and event factories |

## Behaviour to know

- `packages/odds-core/odds_core/__init__.py` imports `odds_core.database`, which builds the engine from `get_settings()` at import. Importing any `odds_core` submodule therefore requires `ODDS_API_KEY` and `DATABASE_URL` to resolve (`packages/odds-core/odds_core/database.py`).
- `extract_odds_from_snapshot` matches market keys exactly, so `h2h` and `1x2` snapshots are different markets to every reader that uses it.

## Hazards

- The "light" constraint is not absolute. `packages/odds-core/odds_core/paper_trade_models.py::PaperTrade` imports `odds_analytics.backtesting.models` (type-checking only, plus a lazy import inside `to_bet_record`), so core has a reverse edge to analytics.
- `python-dotenv` is declared but nothing imports it; `.env` is read only by pydantic-settings, see [Configuration](configuration.md#hazards).

## Related

- [Project overview](project-overview.md) — repo layout and stack
