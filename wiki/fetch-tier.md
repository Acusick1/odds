---
type: concept
title: Fetch tier
summary: The opening/early/sharp/pregame/closing/in-play tiers by hours to kickoff, their intervals, where tiers are stored and used, and the four boundary encodings that disagree.
sources:
  - docs/ARCHITECTURE.md#fetch-tier-system
  - docs/BOOKMAKER_LINE_RELEASE.md#implications
  - docs/DEBUGGING_SCHEDULER.md#4-check-tier-coverage
  - docs/DEBUGGING_SCHEDULER.md#all-events-tagged-with-the-same-tier
  - packages/odds-lambda/odds_lambda/fetch_tier.py::FetchTier
  - packages/odds-lambda/odds_lambda/tier_utils.py::calculate_tier
  - packages/odds-lambda/odds_lambda/oddsportal_common.py::hours_to_tier
  - packages/odds-lambda/odds_lambda/scheduling/decision.py::_classify
  - packages/odds-core/odds_core/models.py::OddsSnapshot
verified_at: 7887039
---

# Fetch tier

A fetch tier labels an odds snapshot by how long before kickoff it was taken. `packages/odds-lambda/odds_lambda/fetch_tier.py::FetchTier` is the enum; the value is stored as a string in `OddsSnapshot.fetch_tier` (nullable, indexed, with `ix_event_tier` on event and tier) next to `hours_until_commence`, and feeds validation and ML features.

| Tier | Hours to kickoff | `interval_hours` |
|---|---|---|
| opening | 72+ | 48 |
| early | 24-72 | 24 |
| sharp | 12-24 | 12 |
| pregame | 3-12 | 3 |
| closing | 0-3 | 0.5 |
| in_play | after kickoff | 0.5 |

`FetchTier.max_hours` holds the upper bounds (3, 12, 24, 72, inf). Pinnacle first appears in the sharp window and opening-tier snapshots hold only soft-book lines (`docs/BOOKMAKER_LINE_RELEASE.md#implications`, partly marked inferred there).

## Who writes the tier

`OddsWriter.store_odds_snapshot` always computes it per event from `snapshot_time` and `event.commence_time` via `calculate_tier_from_timestamps` (`packages/odds-lambda/odds_lambda/storage/writers.py::OddsWriter`), so all ingestion paths through the writer agree. The Polymarket writer and the gap-backfill planner call the same function. See [odds writer](odds-writer.md) and [table odds_snapshots](table-odds-snapshots.md).

## Four boundary encodings

| Encoding | Comparison | Notes |
|---|---|---|
| `packages/odds-lambda/odds_lambda/tier_utils.py::calculate_tier` | `<=` | `<0` is IN_PLAY; used for stored tiers |
| `packages/odds-lambda/odds_lambda/scheduling/decision.py::_classify` | `<` on `max_hours` | used for [scheduling](scheduling-decision-engine.md) |
| `packages/odds-lambda/odds_lambda/oddsportal_common.py::hours_to_tier` | `<` | returns a string; no in-play branch, so negative hours give "closing"; used only by `scripts/ingest_oddsportal.py` and `scripts/ingest_football_data_uk.py` |
| `FetchTier.max_hours` | docstring says `<` | data only |

At exactly 3.0, 12.0, 24.0 or 72.0 hours the stored tier is the nearer one and the scheduling tier the farther one. Tier strings are ML features, so rows from `hours_to_tier` scripts can differ at the instant boundary.

## Hazards

- `docs/ARCHITECTURE.md#fetch-tier-system` says `fetch-odds` uses `FetchTier.interval_hours`. Nothing reads it for scheduling; `fetch-odds` has its own `CADENCE` with equal values ([job-fetch-odds](job-fetch-odds.md)).
- The `OddsSnapshot.fetch_tier` field description lists five tiers and omits `in_play`; `get_priority_order` appends `IN_PLAY` last.
- Missing-tier diagnosis steps are in [debug the AWS scheduler](howto-debug-aws-scheduler.md); a validation view is [completeness validation](completeness-validation.md) (`CRITICAL_TIERS` are pregame and closing).

## Related

- [Self-scheduling](self-scheduling.md), [job-fetch-oddsportal](job-fetch-oddsportal.md).
