---
type: reference
title: Fetch tier
description: The FetchTier enum and bounds, where tiers are written and read, and the boundary encodings.
sources:
  - odds_lambda.fetch_tier.FetchTier
  - odds_lambda.fetch_tier.FetchTier.max_hours
  - odds_lambda.fetch_tier.FetchTier.interval_hours
  - odds_lambda.tier_utils.calculate_tier
  - odds_lambda.tier_utils.calculate_tier_from_timestamps
  - odds_lambda.oddsportal_common.hours_to_tier
  - odds_lambda.scheduling.decision._classify
  - odds_lambda.storage.writers.OddsWriter.store_odds_snapshot
  - odds_lambda.storage.tier_validator.TierCoverageValidator
  - odds_lambda.storage.tier_validator.CRITICAL_TIERS
  - odds_analytics.feature_groups.TierSampler
  - odds_analytics.quality_metrics.QualityMetrics
  - odds_core.models.OddsSnapshot
  - migrations/versions/9ff81f073e6b_recalculate_historical_fetch_tiers.py
status: stable
---

# Fetch tier

A fetch tier labels an odds snapshot by how long before kickoff it was taken. `odds_lambda.fetch_tier.FetchTier` is the enum; the value is stored as a string in `odds_core.models.OddsSnapshot.fetch_tier` (nullable, indexed, with `ix_event_tier` on event and tier) next to `hours_until_commence`. Tiers feed scheduling cadence, validation and ML sampling.

## Tier table

| Tier | Hours to kickoff | `interval_hours` | Purpose |
|---|---|---|---|
| opening | 72+ | 48 | initial line release |
| early | 24-72 | 24 | line establishment |
| sharp | 12-24 | 12 | professional betting |
| pregame | 3-12 | 3 | active betting |
| closing | 0-3 | 0.5 | critical line movement |
| in_play | after kickoff | 0.5 | incidental collection |

The boundaries in the table are `odds_lambda.fetch_tier.FetchTier.max_hours` and the intervals are `odds_lambda.fetch_tier.FetchTier.interval_hours`; the interval is read only by `odds_analytics.quality_metrics.QualityMetrics`; schedulers use each job's `odds_lambda.scheduling.decision.CadenceConfig` ([Decision engine](../scheduling/decision-engine.md#cadence)).

## Where tiers are written and read

- **Written** by `odds_lambda.storage.writers.OddsWriter.store_odds_snapshot`, which always computes the tier per event from `snapshot_time` and `event.commence_time` via `odds_lambda.tier_utils.calculate_tier_from_timestamps`, so all ingestion paths through the writer agree. The Polymarket writer and the gap-backfill planner call the same function. The tier is NULL when the event row is missing ([OddsWriter](odds-writer.md#store_odds_snapshot)). Rows stored before per-event tiering were recalculated by `migrations/versions/9ff81f073e6b_recalculate_historical_fetch_tiers.py`. Historical scripts compute it with `odds_lambda.oddsportal_common.hours_to_tier` instead; an FDUK snapshot exactly 24 h out becomes `early`, not `sharp`, for that reason ([Historical ingest scripts](../collection/historical-ingest-scripts.md)).
- **Scheduling** uses the tier to pick a cadence; the per-job cadence table is on [Decision engine](../scheduling/decision-engine.md#cadence).
- **Validation**: `odds_lambda.storage.tier_validator.TierCoverageValidator` treats pregame and closing (`odds_lambda.storage.tier_validator.CRITICAL_TIERS`) as required, depending on tiers stored with `<=` boundaries ([Coverage and completeness checks](coverage-checks.md)).
- **ML sampling**: `odds_analytics.feature_groups.TierSampler` picks the latest snapshot whose tier is the `decision_tier` or any tier farther from kickoff, excludes IN_PLAY, and so falls back to farther tiers; about 93% of "pregame" OddsPortal events used the sharp-tier snapshot. The production model samples `decision_tier: sharp` ([Look-ahead bias](../modelling/look-ahead-bias.md), [Feature pipeline](../modelling/feature-pipeline.md)).

Missing-tier diagnosis steps are in [Debug the AWS self-scheduling jobs](../scheduling/howto-debug-aws-scheduler.md).

## Boundary encodings

| Encoding | Comparison | Notes |
|---|---|---|
| `odds_lambda.tier_utils.calculate_tier` | `<=` | `<0` is IN_PLAY; used for stored tiers |
| `odds_lambda.scheduling.decision._classify` | `<` on `max_hours` | used for scheduling |
| `odds_lambda.oddsportal_common.hours_to_tier` | `<` | returns a string; no in-play branch, so negative hours give "closing"; used only by `scripts/ingest_oddsportal.py` and `scripts/ingest_football_data_uk.py` (which tiers its closing snapshot with it and raises an FDUK opening snapshot from `closing` to `pregame`) |
| `odds_lambda.fetch_tier.FetchTier.max_hours` | docstring says `<` | data only |

At exactly 3, 12, 24 or 72 h the writer stores the nearer tier and the scheduler's classifier picks the farther one; the historical scripts use `<` too (#446). Tier strings are ML features, so rows from `odds_lambda.oddsportal_common.hours_to_tier` scripts can differ at the instant boundary.

## Field description and priority order

The `odds_core.models.OddsSnapshot.fetch_tier` field description lists five tiers and omits `in_play`. `odds_lambda.fetch_tier.FetchTier.get_priority_order` returns the tiers closest to kickoff first and appends `IN_PLAY` last, though its docstring lists only five.

## Related

- [OddsWriter](odds-writer.md)
- [Odds snapshot and odds tables](odds-tables.md)
- [Self-scheduling](../scheduling/self-scheduling.md)
- [OddsPortal scraper](../collection/oddsportal-scraper.md)
