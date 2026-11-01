---
type: reference
title: Storage layer
description: The odds_lambda.storage module map, the commit and flush contract, and the upsert/insert/store verbs.
sources:
  - odds_lambda.storage
  - odds_lambda.storage.event_merge.EventMerger
  - odds_lambda.storage.readers.OddsReader
  - odds_lambda.storage.validators.OddsValidator
  - odds_lambda.storage.tier_validator.TierCoverageValidator
  - odds_lambda.storage.injury_writer.InjuryWriter
  - odds_lambda.storage.mlb_pitcher_writer.MlbPitcherWriter
  - odds_lambda.storage.injury_writer.InjuryWriter.upsert_injury_reports
  - odds_lambda.storage.mlb_pitcher_writer.MlbPitcherWriter.insert_snapshots
  - odds_lambda.ingestion.OddsIngestionService.ingest_sport
status: stable
---

# Storage layer

`packages/odds-lambda/odds_lambda/storage/` holds every DB access class for the pipeline. `__init__.py` is empty, so no contract is written down in the package; this page is the contract. Each module takes an `AsyncSession` in its constructor and issues SQLAlchemy statements against models from `odds-core`.

## Modules

| Module | Class | Notes |
|---|---|---|
| `odds_lambda.storage.writers` / `odds_lambda.storage.readers` | [`odds_lambda.storage.writers.OddsWriter`](odds-writer.md) / [`odds_lambda.storage.readers.OddsReader`](odds-reader.md) | Event status and scores, envelopes stored as snapshots (one per market, counted as distinct snapshot times by the reader) with their source, market, time basis and price rows (reporting whether each was inserted), fetch logs, including their API quota, and data-quality logs; the reader reads events (by date range, sport, status and minimum snapshot count), their snapshots, and logs (the newest recorded quota and plan size of each Odds API key among them), and no prices |
| `odds_lambda.storage.event_unmatched` | functions | [Event resolver](event-resolver.md#no-game-in-the-schedule): marks an MLB event with no game UNMATCHED with an acknowledgement, clears both when it gets a `gamePk`, and reads the events' acknowledged reasons; flushes (clearing does not) |
| `odds_lambda.storage.game_pks` | functions | Reads the MLB `gamePk`s events hold as `mlb_stats_api` external ids, and the events holding given `gamePk`s; read-only |
| `odds_lambda.storage.event_resolver` | functions | [Event resolver](event-resolver.md): finds or creates a listed game's event (or, for `odds_lambda.storage.event_resolver.attach_event`, only attaches to an existing one) under an advisory lock taken in key order (±2 h live window, ±24 h historical, where `odds_lambda.storage.event_resolver.AmbiguousHistoricalMatch` refuses several candidates and none within ±2 h), and records its external ids, the `gamePk` through `odds_lambda.storage.event_resolver.record_game_pk`, which the gamePk backfill also calls; flushes. The live jobs call it through `odds_lambda.live_events` and the Odds API writers (ingestion, event sync, `odds discover games`, the backfill) through `odds_lambda.odds_api_events` |
| `odds_lambda.storage.event_kickoff` | functions | [Kick-off authority](kickoff-authority.md): the one function that moves `commence_time`, for the sources in the sport's authority order (ESPN last for NBA), against a clock the caller passes (the MLB Stats API, and football-data.co.uk for an event no other source observed, may move to a past start); does not flush |
| `odds_lambda.storage.shaped_snapshots` | functions | [Writing price rows](price-tables.md#writing-price-rows): stores an envelope as snapshots with their source, market, time basis and prices (one snapshot per market), and a fetch's payload (removing one whose snapshots were all skipped); flushes |
| `odds_lambda.storage.snapshot_store` | functions | [Storing a snapshot under its key](price-tables.md#storing-a-snapshot-under-its-key): inserts a snapshot and its prices, or keeps the stored one; flushes in a savepoint |
| `odds_lambda.storage.data_quality` | `odds_lambda.storage.data_quality.record_issue` | Logs a warning and adds the matching `data_quality_logs` row for an event's integer id, without flushing; used by the resolver, `odds_lambda.storage.snapshot_store`, `odds_lambda.storage.shaped_snapshots` and the merger |
| `odds_lambda.storage.validators` | `odds_lambda.storage.validators.OddsValidator` | [Odds validation](odds-validator.md) (one envelope per call; prices are checked as decimals; zero-margin books skip the vig floor) |
| `odds_lambda.storage.tier_validator` | `odds_lambda.storage.tier_validator.TierCoverageValidator` | [Coverage and completeness checks](coverage-checks.md) (tiers from each snapshot's time basis, market and time, `odds_lambda.storage.tier_validator.covered_tiers` over `odds_lambda.tier_utils.snapshot_tiers`, which holds the OddsPortal and `close` rules; discovery matches by `odds_api` external id, for dates within 3 days of the current UTC date) |
| `espn_*`, `fpl_availability_*` | per-source pairs | [EPL context](../collection/epl-context-data.md) |
| `mlb_pitcher_*` | `odds_lambda.storage.mlb_pitcher_writer.MlbPitcherWriter` | [MLB pitchers](../collection/mlb-probable-pitchers.md) |
| `injury_*` | `odds_lambda.storage.injury_writer.InjuryWriter` (teams normalised by `odds_core.team.normalize_team_name`, matched to NBA events by `odds_lambda.storage.nba_event_match.match_nba_event`, as game logs are) | [Injuries](../collection/nba-injuries-and-stats.md) |
| `game_log_*`, `pbpstats_*` | per-source pairs | [NBA stats](../collection/nba-injuries-and-stats.md) |

## Commit contract

- `odds_lambda.storage.writers.OddsWriter` never calls `commit()`. It only adds, deletes or executes statements, except that `odds_lambda.storage.writers.OddsWriter.store_envelope` flushes each inserted snapshot, inside a savepoint, so its price rows can carry its id ([OddsWriter](odds-writer.md#store_envelope)). `odds_lambda.storage.event_merge.EventMerger` also leaves committing to its caller for its foreign-key `UPDATE`s and its deletes (duplicate events, and a duplicate's candidate decisions of a candidate version the kept event also holds decisions of, with their trades), flushing only after deleting a duplicate's clashing external ids and acknowledgements; it reads external ids to pick the event a twin group keeps (`odds_lambda.storage.event_merge.kept_event`), and it re-points every reference to `events.id`, external ids and acknowledgements included; the caller commits, as `odds_lambda.ingestion.OddsIngestionService.ingest_sport` does, once per event.
- Every per-source writer except `odds_lambda.storage.writers.OddsWriter` flushes after its bulk upsert (for example `odds_lambda.storage.injury_writer.InjuryWriter.upsert_injury_reports`, which matches events through the shared team normalizer and returns the unmatched pairs) but still does not commit.
- Each unit of work gets its own session, never one shared across tasks ([Databases and environments](../foundations/databases-and-environments.md)).

## Verb conventions

- `upsert_*`: Postgres `ON CONFLICT DO UPDATE`, usually with a dynamically built `set_` excluding the key columns. Each per-source writer has its own copy of this loop. Events are not upserted: the [event resolver](event-resolver.md) finds or creates them.
- `insert_*` / `bulk_insert_*`: `ON CONFLICT DO NOTHING` (`odds_lambda.storage.mlb_pitcher_writer.MlbPitcherWriter.insert_snapshots`).
- `store_*`: plain `session.add()`.
- The rule is not enforced anywhere; check the method body before relying on conflict behaviour.

## Related

- [Odds API ingestion service](../collection/odds-ingestion-service.md), the main caller of the writers
- [OddsWriter](odds-writer.md)
- [OddsReader](odds-reader.md)
- [Code conventions](../foundations/code-conventions.md)
