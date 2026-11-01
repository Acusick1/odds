---
type: reference
title: Timezones and time helpers
description: The UTC rule, the odds_core.time helpers, the ruff DTZ lint gate, and the naive-datetime sites in current code.
sources:
  - odds_core.time.ensure_utc
  - odds_core.time.parse_api_datetime
  - odds_core.time.utc_isoformat
  - odds_core.time.to_eastern
  - odds_core.time.EASTERN
  - odds_core.time.parse_date
  - odds_core.models.utc_now
  - odds_lambda.scheduling.backends.aws.AWSEventBridgeBackend
  - odds_lambda.scheduling.backends.aws._format_aws_schedule
  - odds_lambda.oddsportal_common.parse_odds_timestamp
  - odds_lambda.oddsportal_common.fix_odds_timestamp
  - odds_lambda.injury_fetcher._to_naive_et
  - odds_cli.commands.quality
  - odds_cli.commands.pbpstats
  - odds_cli.commands.nba_stats
  - odds_analytics.injury_features.date_to_nba_season
  - odds_lambda.espn_fixture_fetcher
  - odds_lambda.mlb_stats_fetcher
  - pyproject.toml
status: stable
---

# Timezones and time helpers

Rule: use `datetime.now(UTC)` (`from datetime import UTC`), store UTC in the database, convert only for display. Datetime columns are `DateTime(timezone=True)`; the default factory is `odds_core.models.utc_now`.

## Lint coverage

Ruff selects the whole `DTZ` (flake8-datetimez) family (`[tool.ruff.lint]` in the root `pyproject.toml`), so `datetime(...)` without `tzinfo`, `strptime` without `%z` or a chained `.replace(tzinfo=...)`, `date.today()`, `utcnow()` and `now()` without a zone all fail lint. Tests that build a naive datetime on purpose, to check how code handles one, carry `# noqa: DTZ001`.

## Helpers

Helpers in `odds_core.time`:

| Helper | Behaviour |
|---|---|
| `odds_core.time.ensure_utc` | Naive input is assumed to be UTC (`replace`); aware input is converted with `astimezone` |
| `odds_core.time.parse_api_datetime` | Parses The Odds API strings, rewriting a trailing `Z`, then `odds_core.time.ensure_utc` |
| `odds_core.time.to_eastern`, `odds_core.time.EASTERN` | America/New_York for NBA game days, injury reports and MLB schedule dates (`odds_lambda.mlb_game_resolver.mlb_date`) |
| `odds_core.time.utc_isoformat` | ISO 8601 with a trailing `Z` |
| `odds_core.time.parse_date` | Parses a calendar date in a `strptime` format into a `date`, without a naive datetime in between |

`Z`-suffix ISO parsing is inlined in `odds_lambda.espn_fixture_fetcher` (team-schedule and month-scoreboard events, whose status text is stored as ESPN gives it), `odds_lambda.mlb_stats_fetcher` (MLB game start times, for probables, results and single-game lookups; a game with `startTimeTBD` carries a placeholder time), and the MCP EPL tools instead of calling `odds_core.time.parse_api_datetime`. Time-window correctness for historical analysis is a separate rule: [Decision time and look-ahead bias](../modelling/look-ahead-bias.md).

`odds_cli.commands.quality` reads its `--start` and `--end` dates (`YYYY-MM-DD`) as midnight UTC, with `strptime` and an explicit `tzinfo=UTC` rather than these helpers. `odds_cli.commands.nba_stats` and `odds_cli.commands.pbpstats` default to the NBA season of the current UTC date, from `odds_analytics.injury_features.date_to_nba_season`, and their `--all` runs through that season.

## Naive datetimes in current code

Two helpers produce naive datetimes on purpose, by stripping the zone with `.replace(tzinfo=None)`, which the lint does not flag:

- `odds_lambda.scheduling.backends.aws._format_aws_schedule` converts to UTC and then drops the zone, because EventBridge cron fields are UTC by definition. `odds_lambda.scheduling.backends.aws.AWSEventBridgeBackend._parse_schedule_expression` reads them back as UTC-aware datetimes ([Scheduler backends](../scheduling/scheduler-backends.md)).
- `odds_lambda.injury_fetcher._to_naive_et` converts to naive Eastern time because the `nbainjuries` API requires it ([NBA feeds](../collection/nba-injuries-and-stats.md)).

OddsPortal odds-history stamps carry no zone and a wrong year. `odds_lambda.oddsportal_common.parse_odds_timestamp` reads them as UTC, and `odds_lambda.oddsportal_common.fix_odds_timestamp` repairs the year against the match time ([OddsPortal scraper](../collection/oddsportal-scraper.md)).

## Related

- [Code conventions and lint gates](code-conventions.md)
- [Events table](../storage/events.md)
