---
type: reference
title: Event merge
description: How events for one match are merged into one canonical event, which event is kept, who merges, how twins and rescheduled duplicates are detected, and the odds events merge-twins and repair-names commands.
sources:
  - odds_lambda.storage.event_merge.EventMerger
  - odds_lambda.storage.event_merge.EVENT_MATCH_WINDOW
  - odds_lambda.storage.event_merge.find_match_id
  - odds_lambda.storage.event_merge.find_historical_match_id
  - odds_lambda.storage.event_merge.source_pair
  - odds_lambda.storage.event_name_repair.repair_annotated_names
  - odds_lambda.betfair.adapter.strip_pitcher_annotation
  - odds_core.event_source.EventSource
  - odds_core.event_source.canonical_event_id
  - odds_lambda.health_monitor.HealthMonitor.check_event_identity
  - odds_lambda.copy_from_production.ProductionDataCopier
  - odds_cli.commands.events
status: stable
---

# Event merge

A match must be one `events` row, because rows that join prices from several sources key on `event_id`. Sources mint their own ids ([Events table](events.md#id-prefix-identifies-the-source)), so one match can exist as several events (twins). `odds_lambda.storage.event_merge.EventMerger` is the only code that collapses them.

## Canonical event

`odds_core.event_source.canonical_event_id` keeps the Odds API event when one exists, because score and status jobs match on its id. Otherwise `odds_core.event_source.EventSource` orders the sources: historical OddsPortal (`op_`), live OddsPortal (`op_live_`), then football-data.co.uk (`fduk_`). The smallest id breaks ties, so the choice is deterministic. The kept event keeps its own commence time.

## Finding and merging are separate

The finders only read. `odds_lambda.storage.event_merge.find_match_id` (±2 h, used by `odds_lambda.storage.writers.OddsWriter.find_or_create_event`) returns the canonical candidate. `odds_lambda.storage.event_merge.find_historical_match_id` (±24 h, same sport, used by `odds_lambda.oddsportal_common.find_existing_event`) returns a lone candidate, or among several only the canonical one of those within ±2 h, else `None`. So neighbouring fixtures of the same teams (an NBA back-to-back) are never conflated, and a loader that gets `None` creates its own event. No finder merges or deletes.

Merging happens in three places only:

| Caller | Entry point |
|---|---|
| Odds API event sync ([fetch-odds](../collection/fetch-odds-job.md)) | `odds_lambda.storage.event_merge.EventMerger.adopt_twins`: every non-Odds-API event with the same sport and teams within ±2 h of each synced event |
| `odds events repair-names --apply` | `odds_lambda.storage.event_name_repair.repair_annotated_names`: strips pitcher annotations from stored team names, then merges each resulting twin group into its best-sourced event that was never annotated |
| `odds events merge-twins --apply` | `odds_lambda.storage.event_merge.EventMerger.merge_group` for every twin group |

The production copier does not merge. It replaces an event's rows by event id, so merging a local twin into that event first would delete the twin's own rows (football-data.co.uk, Polymarket, Betfair history) with the replaced ones, and a scraper event could absorb an Odds API event. It leaves local twins in place and reports how many matches have them.

## Merging

`odds_lambda.storage.event_merge.EventMerger.merge` re-points every foreign key to `events.id`, recomputes `hours_until_commence` and `fetch_tier` of a duplicate's snapshots against the kept event's kick-off when the two kick-offs differ (`odds_lambda.storage.event_merge.EventMerger._rebase_snapshot_timing`), copies status, scores and `completed_at` from a duplicate when the kept event has no scores (logging `merge_conflicting_scores` when both have different ones), and deletes the duplicates. The foreign keys are read from the PostgreSQL catalog (`odds_lambda.storage.event_merge.EventMerger._event_reference_columns`), so every referencing table is covered whichever models the process imported. It runs in the caller's transaction and logs `events_merged` with the rows moved per table. The `predictions` unique key `(event_id, snapshot_id, model_name)` cannot collide because a duplicate's snapshots move with its predictions.

## Detection

No database constraint forbids twins. `odds_lambda.health_monitor.HealthMonitor.check_event_identity` runs on every check-health run ([health monitor](../operations/health-monitor.md)) and raises one warning per sport and kind, none merging anything:

- **Twins** (`duplicate_events:<sport>`): `odds_lambda.storage.event_merge.EventMerger.find_twin_groups` clusters events with the same sport and teams whose commence times are within ±2 h of the cluster's first event.
- **Repeated fixtures** (`repeated_fixtures:<sport>`): `odds_lambda.storage.event_merge.EventMerger.find_repeated_fixtures` finds a home and away pair stored more than once in a season (August to July) further apart than ±2 h, for `odds_lambda.health_monitor.LEAGUE_SPORTS`. This is a rescheduled match under a second id.

## Command

`odds events merge-twins [--sport SPORT] [--apply]` (`odds_cli.commands.events`) prints a table of twin groups by source set (`odds_lambda.storage.event_merge.source_pair`) and their count. Without `--apply` it writes nothing. With it, the groups are merged in one transaction and the remaining count is printed. `odds events repair-names [--sport SPORT] [--apply]` (default `baseball_mlb`) repairs events created before Betfair names were stripped correctly: annotated events with a clean twin are merged into it, the rest are renamed in place so the results job can settle them; without `--apply` the plan is rolled back, and a second run changes nothing. It never merges repeated fixtures; `odds events merge KEEP_ID DROP_ID [--apply]` merges one named pair (same sport and teams required, the two ids must differ, dry run without `--apply`), for a rescheduled match that `repeated_fixtures` flags. Keep the event with the real kick-off and result.

## Related

- [Event matching across sources](event-matching.md)
- [Events table](events.md)
- [OddsWriter](odds-writer.md)
