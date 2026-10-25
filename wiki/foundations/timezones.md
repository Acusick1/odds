---
type: reference
title: Timezones and time helpers
description: The UTC rule, the odds_core.time helpers, which ruff DTZ rules apply, and the naive-datetime sites in current code.
sources:
  - odds_core.time.ensure_utc
  - odds_core.time.parse_api_datetime
  - odds_core.time.utc_isoformat
  - odds_core.time.to_eastern
  - odds_core.time.EASTERN
  - odds_core.models.utc_now
  - odds_lambda.scheduling.backends.aws.AWSEventBridgeBackend
  - odds_lambda.oddsportal_common.parse_odds_timestamp
  - odds_lambda.oddsportal_common.fix_odds_timestamp
  - odds_lambda.injury_fetcher._to_naive_et
  - odds_cli.commands.quality
  - odds_cli.commands.discover
  - odds_cli.commands.pbpstats
  - odds_cli.commands.nba_stats
  - odds_core.snapshot_utils
  - odds_lambda.espn_fixture_fetcher
  - odds_lambda.mlb_stats_fetcher
  - pyproject.toml
status: stable
---

# Timezones and time helpers

Rule: use `datetime.now(UTC)` (`from datetime import UTC`), store UTC in the database, convert only for display. Datetime columns are `DateTime(timezone=True)`; the default factory is `odds_core.models.utc_now`.

## Lint coverage

Ruff selects only `DTZ003` (`utcnow()`) and `DTZ005` (`now()` without tz) (`[tool.ruff.lint]` in the root `pyproject.toml`). `datetime(...)`, `strptime`, `date.today()` and `fromtimestamp` without a zone pass lint, which is how the sites below survive; review catches them.

## Helpers

Helpers in `odds_core.time`:

| Helper | Behaviour |
|---|---|
| `odds_core.time.ensure_utc` | Naive input is assumed to be UTC (`replace`); aware input is converted with `astimezone` |
| `odds_core.time.parse_api_datetime` | Parses The Odds API strings, rewriting a trailing `Z`, then `odds_core.time.ensure_utc` |
| `odds_core.time.to_eastern`, `odds_core.time.EASTERN` | America/New_York for NBA game days, injury reports and MLB schedule dates (`odds_lambda.mlb_game_resolver.mlb_date`) |
| `odds_core.time.utc_isoformat` | ISO 8601 with a trailing `Z` |

`Z`-suffix ISO parsing is inlined in `odds_core.snapshot_utils` (which parses `last_update` and each outcome's price in one pass), `odds_lambda.espn_fixture_fetcher` (team-schedule and month-scoreboard events, whose status text is stored as ESPN gives it), `odds_lambda.mlb_stats_fetcher` (MLB game start times, for probables, results and single-game lookups; a game with `startTimeTBD` carries a placeholder time), and the MCP EPL tools instead of calling `odds_core.time.parse_api_datetime`. Time-window correctness for historical analysis is a separate rule: [Decision time and look-ahead bias](../modelling/look-ahead-bias.md).

## Naive datetimes in current code

- `odds_lambda.scheduling.backends.aws.AWSEventBridgeBackend._parse_schedule_expression` builds `datetime(year=..., ...)` with no tzinfo from a cron expression; the formatter next to it strips tzinfo deliberately after converting to UTC ([Scheduler backends](../scheduling/scheduler-backends.md)).
- `odds_lambda.oddsportal_common.parse_odds_timestamp` returns a naive datetime by design (OddsPortal odds-history stamps carry no zone and a wrong year); `odds_lambda.oddsportal_common.fix_odds_timestamp` repairs the year ([OddsPortal scraper](../collection/oddsportal-scraper.md)).
- `odds_lambda.injury_fetcher._to_naive_et` converts to naive Eastern time on purpose because the `nbainjuries` API requires it ([NBA feeds](../collection/nba-injuries-and-stats.md)).
- `odds_cli.commands.pbpstats` and `odds_cli.commands.nba_stats` derive the NBA season from `date.today()` (local date, not UTC) (#461).

## Related

- [Code conventions and lint gates](code-conventions.md)
- [Events table](../storage/events.md)
