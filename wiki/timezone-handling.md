---
type: concept
title: Timezone handling
summary: "UTC everywhere with datetime.now(UTC), what ruff enforces, the odds_core.time helpers, and the naive-datetime sites that remain."
sources:
  - packages/odds-core/odds_core/time.py::ensure_utc
  - packages/odds-core/odds_core/time.py::parse_api_datetime
  - packages/odds-core/odds_core/time.py::utc_isoformat
  - packages/odds-core/odds_core/time.py::EASTERN
  - packages/odds-core/odds_core/models.py::utc_now
  - packages/odds-lambda/odds_lambda/scheduling/backends/aws.py::AWSEventBridgeBackend
  - packages/odds-cli/odds_cli/commands/backtest.py::parse_date
  - packages/odds-lambda/odds_lambda/oddsportal_common.py::parse_odds_timestamp
  - packages/odds-lambda/odds_lambda/injury_fetcher.py::_to_naive_et
verified_at: 7887039
---

# Timezone handling

Rule ([AGENTS.md](../AGENTS.md#timezone-handling-critical)): use `datetime.now(UTC)` (`from datetime import UTC`), store UTC in the database, convert only for display. Datetime columns are `DateTime(timezone=True)`; the default factory is `packages/odds-core/odds_core/models.py::utc_now`.

## Enforcement

Ruff selects only `DTZ003` (`utcnow()`) and `DTZ005` (`now()` without tz) in the root `pyproject.toml`. Other naive constructors (`datetime(...)`, `strptime`, `date.today()`, `fromtimestamp`) pass lint, which is how the sites below survive. Violations are caught by review, not tooling.

## Helpers (`packages/odds-core/odds_core/time.py`)

| Helper | Behaviour |
|---|---|
| `ensure_utc` | Naive input is assumed to be UTC (`replace`); aware input is converted with `astimezone` |
| `parse_api_datetime` | Parses The Odds API strings, rewriting a trailing `Z`, then `ensure_utc` |
| `to_eastern`, `EASTERN` | America/New_York for NBA game days and injury reports |
| `utc_isoformat` | ISO 8601 with a trailing `Z` |

## Hazards

Naive datetimes still produced by current code:

- `AWSEventBridgeBackend._parse_schedule_expression` builds `datetime(year=..., ...)` with no tzinfo from a cron expression (`packages/odds-lambda/odds_lambda/scheduling/backends/aws.py`); the formatter next to it strips tzinfo deliberately after converting to UTC. See [AWS EventBridge backend](aws-eventbridge-backend.md).
- `packages/odds-cli/odds_cli/commands/backtest.py::parse_date` returns a naive datetime from `strptime`, while `quality.py` and `discover.py` attach `tzinfo=UTC` after the same call. See [Backtest engine](backtest-engine.md).
- `packages/odds-lambda/odds_lambda/oddsportal_common.py::parse_odds_timestamp` returns a naive datetime by design (OddsPortal odds-history stamps carry no zone and a wrong year); `fix_odds_timestamp` repairs the year ([OddsPortal scraper](oddsportal-scraper.md)).
- `packages/odds-lambda/odds_lambda/injury_fetcher.py::_to_naive_et` converts to naive Eastern time on purpose because the `nbainjuries` API requires it ([NBA injuries](nba-injury-reports.md)).
- `packages/odds-cli/odds_cli/commands/pbpstats.py` and `nba_stats.py` derive the NBA season from `date.today()` (local date, not UTC).

Duplicated parsing: `Z`-suffix ISO parsing is inlined in `snapshot_utils`, `espn_fixture_fetcher`, `mlb_stats_fetcher`, `fetch_oddsportal_results`, the MCP EPL tools and `sequence_loader` modules instead of calling `parse_api_datetime`.

## Related

- [Code conventions](code-conventions.md) — other gates ruff enforces
- [Look-ahead bias](look-ahead-bias.md) — time-window correctness
- [Configuration](configuration.md)
