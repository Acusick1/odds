---
type: reference
title: NBA injury reports, game logs and player stats
description: "The NBA injury-report job and the manual feeds (stats.nba.com game logs, PBPStats player stats): sources, CLIs, writers and event matching."
sources:
  - odds_core.injury_models.InjuryReport
  - odds_lambda.injury_fetcher.fetch_injury_report
  - odds_lambda.injury_fetcher.fetch_latest_injury_report
  - odds_lambda.injury_fetcher.fetch_latest_injury_report_isolated
  - odds_lambda.injury_fetcher.SUBPROCESS_TIMEOUT_SECONDS
  - odds_lambda.jobs.fetch_injuries
  - odds_lambda.jobs.fetch_injuries.CADENCE
  - odds_lambda.jobs.fetch_injuries.LEADS
  - odds_lambda.jobs.fetch_injuries.MAX_SLOTS_BACK
  - odds_lambda.jobs.fetch_injuries.REPORT_DUE_HORIZON
  - odds_lambda.jobs.fetch_injuries.STALE_REPORT_AGE
  - odds_lambda.injury_fetcher._ensure_java_home
  - odds_lambda.injury_fetcher._to_naive_et
  - odds_lambda.storage.injury_writer.InjuryWriter
  - odds_lambda.storage.injury_writer.UpsertResult
  - odds_lambda.storage.injury_reader.InjuryReader
  - odds_lambda.game_log_fetcher.fetch_game_logs
  - odds_lambda.pbpstats_fetcher.fetch_player_season_stats
  - odds_lambda.storage.game_log_writer.GameLogWriter
  - odds_lambda.storage.game_log_reader.GameLogReader
  - odds_lambda.storage.pbpstats_writer.PbpStatsWriter
  - odds_lambda.storage.pbpstats_reader.PbpStatsReader
  - odds_core.game_log_models.NbaTeamGameLog
  - odds_core.player_stats_models.NbaPlayerSeasonStats
  - odds_cli.commands.injuries
  - odds_cli.commands.nba_stats
  - odds_cli.commands.pbpstats
  - odds_cli.commands.backfill
status: stable
---

# NBA injury reports, game logs and player stats

Three NBA feeds. Only the injury reports are scheduled, by `fetch-injuries-nba` ([the job](#the-fetch-injuries-job)); game logs and player stats are manual CLIs. The rest of scheduled NBA collection is prices ([fetch-oddsportal-job](fetch-oddsportal-job.md#leagues), [betfair-exchange](betfair-exchange.md)) ([project-overview](../foundations/project-overview.md)). The feature groups built on these tables are in [feature-groups](../modelling/feature-groups.md).

| | Injury reports | Game logs | Player stats |
|---|---|---|---|
| Table / model | `nba_injury_reports` / `odds_core.injury_models.InjuryReport` | `nba_team_game_logs` / `odds_core.game_log_models.NbaTeamGameLog` | `nba_player_season_stats` / `odds_core.player_stats_models.NbaPlayerSeasonStats` |
| Source | official NBA PDFs via `nbainjuries` | stats.nba.com LeagueGameFinder | `api.pbpstats.com/get-totals/nba` |
| Writer / key | `odds_lambda.storage.injury_writer.InjuryWriter.upsert_injury_reports`, `uq_injury_report_time_team_player_date` (`report_time, team, player_name, game_date`) | `odds_lambda.storage.game_log_writer.GameLogWriter.upsert_game_logs`, `uq_game_log_game_team` (`nba_game_id, team_id`) | `odds_lambda.storage.pbpstats_writer.PbpStatsWriter.upsert_player_stats`, `uq_player_season_stats_player_season` (`player_id, season`) |
| Collection | `fetch-injuries-nba` job; CLI `odds injuries fetch\|backfill\|status` | CLI `odds nba-stats fetch\|status` | CLI `odds pbpstats fetch\|backfill\|status` |

## Injury reports

League-wide injury report snapshots parsed from the official PDFs by `nbainjuries`; models `odds_core.injury_models.InjuryReport` and `odds_core.injury_models.InjuryStatus` are in `odds_core.injury_models`. `report_time` is the UTC snapshot time; `game_date` is the report's calendar date and `game_time_et` the tip-off string in ET (for example `07:00 PM ET`); `matchup` is `BOS@ORL`-style; `team` is the full name; `player_name` is `Last, First`; `event_id` is a nullable integer FK to `events.id`, set at write time.

- `odds_lambda.injury_fetcher.fetch_injury_report(target_time_utc)` is synchronous. It converts to naive ET (`odds_lambda.injury_fetcher._to_naive_et`, deliberate because the `nbainjuries` API requires it; [timezones](../foundations/timezones.md)), checks `check_reportvalid`, parses via `odds_lambda.injury_fetcher._parse_records`, and drops "NOT YET SUBMITTED" rows and unknown statuses. `nbainjuries` is imported lazily because it starts a JVM (tabula via jpype).
- `odds_lambda.storage.injury_writer.InjuryWriter.upsert_injury_reports` matches each `(team, game_date)` to an `events` row (see [Event matching](#event-matching)) and returns an `odds_lambda.storage.injury_writer.UpsertResult`: the rows written and the canonical `(team, game_date)` pairs that matched no event.
- `odds injuries fetch|backfill|status` (`odds_cli.commands.injuries`). Backfill is event-driven: for NBA events in the season window it subtracts each `--hours-before` offset (default `8,2`), rounds down to the report slot (hourly before 2025-12-22 09:00 ET, quarter-hourly after, `odds_cli.commands.injuries._NEW_ERA_CUTOFF`, an Eastern-aware datetime compared directly with the UTC target time), dedupes, and sleeps `--delay-ms` (default 500).

## The fetch-injuries job

`odds_lambda.jobs.fetch_injuries.main` runs as `fetch-injuries-nba` (per sport, NBA only through its `odds_lambda.scheduling.jobs._JOB_SPORTS` entry, in the default `bootstrap_jobs`). It self-schedules from the stored NBA events, which `fetch-espn-fixtures-nba` creates ([NBA schedule](polymarket-odds-snapshots.md#nba-schedule-from-espn)), through `odds_lambda.collection_calendar.decide_collection_run` with `stored_only` ([dormancy](../scheduling/decision-engine.md#dormancy)).

- **Schedule.** `odds_lambda.jobs.fetch_injuries.CADENCE` is 6 h at every tier, so with the NBA overnight window (07:00 to 15:00 UTC) the first cadence run of a game day is when the window ends, 15:00 UTC, and one runs about every 6 h after. `odds_lambda.jobs.fetch_injuries.LEADS` adds a run 2 h and one 30 min before each kickoff ([lead runs](../scheduling/decision-engine.md#lead-runs)); games with the same tip-off share them, so a game day costs about 10 to 15 runs. The 2 h run catches the report once most starters' status is filed, and the 30 min run the last slot published before tip-off. A run books its next one before it fetches.
- **Fetch.** `odds_lambda.injury_fetcher.fetch_latest_injury_report` takes the newest quarter-hour slot at or before the run and steps back up to `odds_lambda.jobs.fetch_injuries.MAX_SLOTS_BACK` slots (6 h) when a slot is not published (a slot appears a few minutes after its time, and the NBA has stopped publishing for hours, on 2026-10-30 from 06:15 to 15:45 ET) and over a slot whose PDF fails to parse, as one still uploading does. `nbainjuries` reports a network error or a 403 as "not published". `odds_lambda.injury_fetcher.fetch_latest_injury_report_isolated` runs it in a child process (`python -m odds_lambda.injury_fetcher`), because JPype cannot restart a JVM and one started from an executor thread kept the interpreter from exiting in testing; the child is killed on a timeout (`odds_lambda.injury_fetcher.SUBPROCESS_TIMEOUT_SECONDS`) or a cancelled run, and its logs reach the scheduler's, far inside `odds_lambda.scheduling.runner.JOB_TIMEOUT`.
- **JVM.** `odds_lambda.injury_fetcher._ensure_java_home` points JPype at the JRE bundled by the `jdk4py` dependency unless `JAVA_HOME` is set, so the host needs no system Java. A run takes about 8 s on the production host, nearly all of it JVM start-up and the PDF download (October 2026).
- **Failures.** The run raises, so `odds_core.alerts.job_alert_context` alerts and no heartbeat is written, only on a genuine error: the child failing, or every slot in reach failing to parse. A publishing gap never fails the run: with a stored game within `odds_lambda.jobs.fetch_injuries.REPORT_DUE_HORIZON` (3 h) and the newest report older than `odds_lambda.jobs.fetch_injuries.STALE_REPORT_AGE` (1 h) or absent, it sends the warning `injury_report_stale:fetch-injuries-nba` with the newest slot's age. A `(team, game_date)` in the report that matches no event sends `injury_unmatched:fetch-injuries-nba`. The off-season is dormant.
- **Heartbeat.** `fetch-injuries-nba` is expected every 15 h (the last run of a night can be after 01:00 UTC, so up to 14 h pass before the first run after the overnight window, plus slack) ([heartbeats](../operations/health-monitor.md#heartbeats)).

## Game logs

stats.nba.com blocks raw HTTP (Akamai), so `odds_lambda.game_log_fetcher.fetch_game_logs` is synchronous Playwright: headless Firefox loads nba.com for cookies, then calls the API from the page. Two rows per game, about 2,460 per regular season. Because Playwright's sync API cannot run inside an event loop, the CLI fetches all seasons first, then writes asynchronously.

`odds_lambda.storage.game_log_reader.GameLogReader.get_game_logs_for_event` and `odds_lambda.storage.game_log_reader.GameLogReader.get_team_previous_game` serve the `rest` features (the first takes an event's integer id), which are all-NaN for EPL and dropped by the variance filter ([feature-groups](../modelling/feature-groups.md)). `odds backfill scores` fills missing scores from the logs linked to events ([odds-api-backfill](odds-api-backfill.md)).

## Player stats

`odds_lambda.pbpstats_fetcher.PlayerSeasonRecord` carries on/off ratings, minutes, games played and box totals, regular season only. `odds_lambda.pbpstats_fetcher.convert_name` turns "First Last" into "Last, First" (suffix-aware) to match injury-report names. Traded players appear once, under the final team. `odds_lambda.storage.pbpstats_reader.PbpStatsReader.get_players_for_teams` returns a dict keyed by `player_name`, which the injury features use for impact weighting ([feature-groups](../modelling/feature-groups.md)). The NBA CLIs derive the season from `date.today()` (local date, not UTC).

## Event matching

Both writers link rows to `events` with the same ET window rule: events whose `commence_time` is in game_date 10:00 ET to next day 06:00 ET with the team as home or away; exactly one candidate sets `event_id` (its integer id), otherwise NULL (ambiguous logs a warning). Injury teams are canonicalised with `odds_core.team.normalize_team_name` and the candidate events are NBA only; the game-log writer maps `team_abbreviation` through a hardcoded `_ABBREV_TO_CANONICAL`. A report lists only the teams playing on its game dates, so each of its `(team, game_date)` pairs should link: in the reports of 2026-10-21 to 2026-10-30 all 152 pairs, covering all 30 teams, matched an event. Unmatched rows are invisible to per-event readers such as `odds_lambda.storage.injury_reader.InjuryReader.get_injuries_for_event` ([event-matching](../storage/event-matching.md)).

## Scope and known gaps

- NBA-only: `odds backfill scores` (a game-log backfill that spends no Odds API units) and the stats lookup use `Event.sport_key == "basketball_nba"` in the CLI, and `odds_lambda.polymarket_matching.CANONICAL_TO_ABBREV` in `odds_analytics.feature_groups.collect_event_data`, which loads the NBA-only data beside the market's quotes and each source's raw-target close (the `injuries` and `rest` groups are NBA-only but valid names for any sport's delta targets). `odds backfill scores` hardcodes `sport_key="basketball_nba"` with no `--sport` option ([odds-api-backfill](odds-api-backfill.md)).
- `odds_cli.commands.nba_stats.ALL_SEASONS` ends at 2025-26 and `odds_cli.commands.pbpstats.ALL_SEASONS` at 2024-25, and `odds pbpstats backfill` is `fetch --all` over that shorter list. 2025-26 player stats need `--season` (#463). Without `--season` or `--all`, `odds_cli.commands.nba_stats.fetch` and `odds_cli.commands.pbpstats.fetch` use `odds_analytics.injury_features.date_to_nba_season` of the current UTC time. Both `--all` help strings say "2021-22 through current", which is false for PBPStats (#466).
- An alias gap leaves `event_id` NULL, as a team with no event does; the job warns about those pairs ([event-matching](../storage/event-matching.md)).

## Related

- [Event matching across sources](../storage/event-matching.md)
- [Feature groups](../modelling/feature-groups.md), the `injuries` and `rest` groups
- [The odds CLI](../foundations/odds-cli.md)
