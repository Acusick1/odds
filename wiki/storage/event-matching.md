---
type: reference
title: Event matching across sources
description: The matcher each ingest path uses to find its events row (window, team comparison, ambiguity rule) and the duplicates and NULL event ids that follow.
sources:
  - odds_lambda.oddsportal_common.find_existing_event
  - odds_lambda.storage.writers.OddsWriter.find_or_create_event
  - odds_lambda.storage.injury_writer.InjuryWriter._match_event
  - odds_lambda.storage.game_log_writer.GameLogWriter._match_event
  - odds_lambda.polymarket_matching.match_polymarket_event
  - odds_lambda.polymarket_matching.NBA_ABBREV_MAP
  - odds_lambda.polymarket_matching.normalize_team
  - odds_core.team.normalize_team_name
status: stable
---

# Event matching across sources

There is no shared matcher for deciding which `events` row a piece of incoming data belongs to: each ingest path has its own query, window and ambiguity rule, and exact string equality on team names is the common core. The `events` columns and id prefixes are on [Events table](events.md); this page compares the matchers as implemented (read from the code, not run).

## Matchers

| Matcher | Used by | Window | Teams | Ambiguity |
|---|---|---|---|---|
| `odds_lambda.oddsportal_common.find_existing_event` | historical OddsPortal and FDUK ingest scripts | commence time ±24 h | exact home and away as stored, no sport filter | logs, returns `None` |
| `odds_lambda.storage.writers.OddsWriter.find_or_create_event` | live OddsPortal scrape and Betfair jobs | ±2 h | `odds_core.team.normalize_team_name` applied first; sport filter | logs, returns the first candidate |
| `odds_lambda.storage.injury_writer.InjuryWriter._match_event` | [NBA injury reports](../collection/nba-injuries-and-stats.md) | ET `game_date` 10:00 to next day 06:00, converted to UTC | team is home or away | `None` |
| `odds_lambda.storage.game_log_writer.GameLogWriter._match_event` | [NBA game logs](../collection/nba-injuries-and-stats.md) | same ET window (a verbatim copy) | team is home or away | `None` |
| `odds_lambda.polymarket_matching.match_polymarket_event` | `odds polymarket link` only | ticker date at 12:00 UTC ±24 h | exact, via `odds_lambda.polymarket_matching.NBA_ABBREV_MAP` | `None` |

## Consequences of the differences

- **Create on miss.** If `odds_lambda.storage.writers.OddsWriter.find_or_create_event` finds nothing it creates an event, so a name or window mismatch there yields a duplicate event rather than a miss. A game absent from `events` gets an `op_`/`fduk_` row from the historical scripts, so cross-source joins depend on the matchers agreeing.
- **Different windows.** `odds_lambda.storage.writers.OddsWriter.find_or_create_event` uses ±2 h and `odds_lambda.oddsportal_common.find_existing_event` ±24 h; both reuse an existing event (including a hex-id one) on a team match.
- **Different ambiguity policy.** `odds_lambda.storage.writers.OddsWriter.find_or_create_event` picks the first of several candidates; the others return `None`.
- **No normalization in the historical matcher.** `odds_lambda.oddsportal_common.find_existing_event` compares names as stored, so the historical path depends on names already being canonical.
- **Alias gaps become NULL.** Injury and game-log matching fall back to the raw team name when `odds_lambda.polymarket_matching.normalize_team` returns `None` (the game-log writer tries its abbreviation map first), so an alias gap shows up as a NULL `event_id`. Injury reports are league-wide, so unmatched pairs are expected for teams not playing that day. Unmatched NBA rows keep `event_id` NULL and are invisible to per-event readers.
- **Manual Polymarket linking.** Polymarket linking is manual and the fetch job does not call it ([Polymarket pipeline and tables](../collection/polymarket.md)).

The OddsWriter's own event handling, including how the live id embeds the start minute, is on [OddsWriter](odds-writer.md#find_or_create_event).

## Related

- [Events table](events.md)
- [OddsWriter](odds-writer.md)
- [Historical ingest scripts](../collection/historical-ingest-scripts.md)
- [OddsReader](odds-reader.md)
