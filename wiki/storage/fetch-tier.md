---
type: concept
title: Fetch tier
description: "The opening/early/sharp/pregame/closing/in-play tiers by hours to kickoff, where tiers are written and read, the four boundary encodings that disagree, and how tiers drive cadence, validation and sampling."
sources:
  - packages/odds-lambda/odds_lambda/fetch_tier.py::FetchTier
  - packages/odds-lambda/odds_lambda/tier_utils.py::calculate_tier
  - packages/odds-lambda/odds_lambda/oddsportal_common.py::hours_to_tier
  - packages/odds-lambda/odds_lambda/scheduling/decision.py::_classify
  - packages/odds-lambda/odds_lambda/scheduling/decision.py::CadenceConfig
  - packages/odds-lambda/odds_lambda/storage/writers.py::OddsWriter
  - packages/odds-lambda/odds_lambda/storage/tier_validator.py::TierCoverageValidator
  - packages/odds-analytics/odds_analytics/feature_groups.py::TierSampler
  - packages/odds-core/odds_core/models.py::OddsSnapshot
status: stable
---

# Fetch tier

A fetch tier labels an odds snapshot by how long before kickoff it was taken. `packages/odds-lambda/odds_lambda/fetch_tier.py::FetchTier` is the enum; the value is stored as a string in `OddsSnapshot.fetch_tier` (nullable, indexed, with `ix_event_tier` on event and tier) next to `hours_until_commence`. Tiers feed scheduling cadence, validation and ML sampling.

## Tier table

| Tier | Hours to kickoff | `interval_hours` | Purpose |
|---|---|---|---|
| opening | 72+ | 48 | initial line release |
| early | 24-72 | 24 | line establishment |
| sharp | 12-24 | 12 | professional betting |
| pregame | 3-12 | 3 | active betting |
| closing | 0-3 | 0.5 | critical line movement |
| in_play | after kickoff | 0.5 | incidental collection |

`FetchTier.max_hours` holds the upper bounds (3, 12, 24, 72, inf). Pinnacle first appears in the sharp window (12-24 h) and opening-tier snapshots hold only soft-book lines (inferred; an unrecorded NBA measurement, see [exp-nba-timing-and-injuries](../evaluation/exp-nba-timing-and-injuries.md)).

## Where tiers are written and read

- **Written** by `OddsWriter.store_odds_snapshot`, which always computes the tier per event from `snapshot_time` and `event.commence_time` via `packages/odds-lambda/odds_lambda/tier_utils.py::calculate_tier_from_timestamps` (`packages/odds-lambda/odds_lambda/storage/writers.py::OddsWriter`), so all ingestion paths through the writer agree. The Polymarket writer and the gap-backfill planner call the same function. The tier is NULL when the event row is missing ([odds-storage](odds-storage.md)). Historical scripts compute it with `packages/odds-lambda/odds_lambda/oddsportal_common.py::hours_to_tier` instead; a FDUK synthetic 24 h opening snapshot becomes `early`, not `sharp`, for that reason ([howto-load-historical-odds](../collection/howto-load-historical-odds.md)).
- **Scheduling** uses the tier to pick a cadence. Each job module's `CADENCE` (`packages/odds-lambda/odds_lambda/scheduling/decision.py::CadenceConfig`) replaces `interval_hours`, which nothing reads. Hours between runs:

  | Job | closing | pregame | sharp | early | opening | no game |
  |---|---|---|---|---|---|---|
  | `fetch-odds` | 0.5 | 3 | 12 | 24 | 48 | 24 |
  | `fetch-oddsportal` | 0.5 | 1 | 2 | 2 | 2 | 2 |
  | `fetch-betfair-exchange` | 0.25 | 0.5 | 2 | 2 | 2 | 2 |

  Full decision logic is on [self-scheduling](../scheduling/self-scheduling.md).
- **Validation**: `packages/odds-lambda/odds_lambda/storage/tier_validator.py::TierCoverageValidator` treats pregame and closing (`CRITICAL_TIERS`) as required, depending on tiers stored with `<=` boundaries ([data-validation](data-validation.md#completeness-and-coverage-checks)).
- **ML sampling**: `packages/odds-analytics/odds_analytics/feature_groups.py::TierSampler` picks the latest snapshot whose tier is the `decision_tier` or any tier farther from kickoff, excludes IN_PLAY, and so falls back to farther tiers; about 93% of "pregame" OddsPortal events used the sharp-tier snapshot. The production model samples `decision_tier: sharp` ([decision-time-and-look-ahead](../modelling/decision-time-and-look-ahead.md), [feature-pipeline](../modelling/feature-pipeline.md)).

## Four boundary encodings

| Encoding | Comparison | Notes |
|---|---|---|
| `packages/odds-lambda/odds_lambda/tier_utils.py::calculate_tier` | `<=` | `<0` is IN_PLAY; used for stored tiers |
| `packages/odds-lambda/odds_lambda/scheduling/decision.py::_classify` | `<` on `max_hours` | used for scheduling |
| `packages/odds-lambda/odds_lambda/oddsportal_common.py::hours_to_tier` | `<` | returns a string; no in-play branch, so negative hours give "closing"; used only by `scripts/ingest_oddsportal.py` and `scripts/ingest_football_data_uk.py` |
| `FetchTier.max_hours` | docstring says `<` | data only |

At exactly 3.0, 12.0, 24.0 or 72.0 hours the stored tier is the nearer one and the scheduling tier the farther one. Tier strings are ML features, so rows from `packages/odds-lambda/odds_lambda/oddsportal_common.py::hours_to_tier` scripts can differ at the instant boundary.

## Hazards

- `packages/odds-lambda/odds_lambda/fetch_tier.py::FetchTier.interval_hours` is not read for scheduling; `fetch-odds` has its own `CADENCE` (`packages/odds-lambda/odds_lambda/jobs/fetch_odds.py::CADENCE`) with equal values ([odds-api-ingestion](../collection/odds-api-ingestion.md)).
- The `OddsSnapshot.fetch_tier` field description lists five tiers and omits `in_play`; `get_priority_order` appends `IN_PLAY` last.
- The boundary disagreement above is silent: a snapshot taken exactly at a boundary is tagged differently by the writer and by the scheduler's classifier, and different again by the historical scripts.
- Missing-tier diagnosis steps are in [howto-debug-aws-scheduler](../scheduling/howto-debug-aws-scheduler.md), which also lists the four-encoding disagreement at 3/12/24/72 h as a cause of tier surprises.

## Related

- [Odds snapshots, odds rows and OddsWriter](odds-storage.md)
- [Self-scheduling](../scheduling/self-scheduling.md)
- [OddsPortal scraper and job](../collection/oddsportal.md)
