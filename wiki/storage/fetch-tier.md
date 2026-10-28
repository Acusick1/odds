---
type: reference
title: Fetch tier
description: The FetchTier enum and bounds, the one hours-to-tier rule, the tier a snapshot covers by its time basis, and where tiers are written and read.
sources:
  - odds_lambda.fetch_tier.FetchTier
  - odds_lambda.fetch_tier.FetchTier.max_hours
  - odds_lambda.fetch_tier.FetchTier.interval_hours
  - odds_lambda.tier_utils.calculate_tier
  - odds_lambda.tier_utils.snapshot_tier
  - odds_lambda.storage.tier_validator.covered_tiers
  - odds_lambda.tier_utils.snapshot_tiers
  - odds_lambda.storage.event_merge.retime_snapshots
  - odds_lambda.storage.writers.OddsWriter.store_odds_snapshot
  - odds_lambda.storage.writers.OddsWriter.store_envelope
  - odds_lambda.storage.tier_validator.TierCoverageValidator
  - odds_lambda.storage.tier_validator.CRITICAL_TIERS
  - odds_analytics.feature_groups.TierSampler
  - odds_analytics.quality_metrics.QualityMetrics
  - odds_core.models.OddsSnapshot
  - migrations/versions/9ff81f073e6b_recalculate_historical_fetch_tiers.py
status: stable
---

# Fetch tier

A fetch tier labels an odds snapshot by how long before kickoff it was taken. `odds_lambda.fetch_tier.FetchTier` is the enum; the value is stored as a string in `odds_core.models.OddsSnapshot.fetch_tier` ([odds_snapshots columns](odds-tables.md#odds_snapshots)) (nullable, indexed, with `ix_event_tier` on event and tier) next to `hours_until_commence`. Tiers feed scheduling cadence, validation and ML sampling.

## Tier table

| Tier | Hours to kickoff | `interval_hours` | Purpose |
|---|---|---|---|
| opening | 72+ | 48 | initial line release |
| early | 24-72 | 24 | line establishment |
| sharp | 12-24 | 12 | professional betting |
| pregame | 3-12 | 3 | active betting |
| closing | 0-3 | 0.5 | critical line movement |
| in_play | after kickoff | 0.5 | incidental collection |

The boundaries in the table are `odds_lambda.fetch_tier.FetchTier.max_hours` and the intervals are `odds_lambda.fetch_tier.FetchTier.interval_hours`; the interval is read only by `odds_analytics.quality_metrics.QualityMetrics`, which counts snapshots as distinct snapshot times; schedulers use each job's `odds_lambda.scheduling.decision.CadenceConfig`, which keeps these boundaries but sets its own intervals and its own check-in for a sport with no fixture ([Decision engine](../scheduling/decision-engine.md#cadence)); a job's extra runs before kickoffs are [lead runs](../scheduling/decision-engine.md#lead-runs), outside the cadence.

## The one hours-to-tier rule

`odds_lambda.tier_utils.calculate_tier(hours_until)` is the only function that turns hours before kick-off into a tier. Negative hours are IN_PLAY. Otherwise a time belongs to the nearest tier whose `odds_lambda.fetch_tier.FetchTier.max_hours` it does not exceed (`<=`), so a time exactly 3, 12, 24 or 72 h out is CLOSING, PREGAME, SHARP or EARLY. The scheduler's cadence ([Decision engine](../scheduling/decision-engine.md#forward-decisions)) and the stale-source check use it for a kick-off's distance. A snapshot's tier also depends on what its time means, so snapshots are tiered through the next section's functions, which call it.

## The tier a snapshot covers

`odds_lambda.tier_utils.snapshot_tier(snapshot_time, commence_time, time_basis)` reads the snapshot's time by what it means ([time basis](price-tables.md#closed-sets)):

| Time basis | Tier |
|---|---|
| `close` | CLOSING, whatever the stamp (an FDUK closing price is stamped at kick-off) |
| `upper_bound` | `odds_lambda.tier_utils.calculate_tier`, except that a time exactly on a boundary covers the farther tier only, since the price was taken at or before it: an FDUK opening stamped 3.0 h out is PREGAME, and the Odds API backfill's fetches at 72, 24, 12 and 3 h before kick-off cover OPENING, EARLY, SHARP and PREGAME |
| `observed`, `source_stamped` | `odds_lambda.tier_utils.calculate_tier` |

`odds_lambda.tier_utils.snapshot_tiers` applies it to an event's snapshots, given with their source, time basis and market, and adds two rules that need the other snapshots:

- **OddsPortal history closing.** A market's latest OddsPortal `source_stamped` time is its history closing price, so it is CLOSING however near or after kick-off the stamp (the latest change among the books) lies. A market with a single stamp before kick-off may hold only an opening whose closing was never scraped, so that stamp keeps its own tier; an opening is never stamped after kick-off.
- **A source's `close` is its only closing.** When a source has a `close` snapshot for a market, none of the source's other snapshots of that market is CLOSING; one that would be covers PREGAME. An FDUK opening stamped under 3 h before kick-off therefore stays in the pregame pool the [tier sampler](../modelling/feature-pipeline.md) draws decision rows from.
 Every tier in the system comes from `odds_lambda.tier_utils.snapshot_tier` or `odds_lambda.tier_utils.snapshot_tiers`: the stored `fetch_tier`, the coverage check (`odds_lambda.storage.tier_validator.covered_tiers`) and the gap planner, which tiers each planned fetch as the `upper_bound` snapshot the backfill stores, so a filled game is not planned again.

## Where tiers are written and read

`fetch_tier` is a column of `odds_core.models.OddsSnapshot`; the redesigned snapshot columns beside it (source, market, time basis and market metadata) carry no tier ([odds_snapshots](odds-tables.md#odds_snapshots)).

- **Written** through `odds_lambda.tier_utils.snapshot_tier` with the snapshot's own time basis: once per envelope by `odds_lambda.storage.writers.OddsWriter.store_envelope` (behind `odds_lambda.storage.writers.OddsWriter.store_odds_snapshot`; its shape's time basis, none without a shape), on the snapshot only (odds rows carry no tier), against `event.commence_time`; by the Polymarket history backfill and the legacy Polymarket writer (`observed`); and by `scripts/ingest_football_data_uk.py`, which writes `odds_lambda.tier_utils.snapshot_tiers` of its `upper_bound` opening and `close` closing. `scripts/ingest_oddsportal.py` writes `odds_lambda.tier_utils.snapshot_tiers` of the run's snapshots, so each market's history closing is `closing` even when stamped after kick-off. The tier is NULL when the event row is missing, and a snapshot split from a multi-market envelope takes the envelope's tier; a snapshot already stored under its key keeps the tier it was stored with ([OddsWriter](odds-writer.md#store_odds_snapshot)). Rows stored before per-event tiering were recalculated by `migrations/versions/9ff81f073e6b_recalculate_historical_fetch_tiers.py`. When the kick-off moves or events merge, `odds_lambda.storage.event_merge.retime_snapshots` rewrites the event's tiers with `odds_lambda.tier_utils.snapshot_tiers`.
- **Scheduling** uses the tier to pick a cadence; the per-job cadence table is on [Decision engine](../scheduling/decision-engine.md#cadence).
- **Validation**: `odds_lambda.storage.tier_validator.TierCoverageValidator` does not read the stored tier. It takes each snapshot's tier from `odds_lambda.storage.tier_validator.covered_tiers` against the event's current kick-off, and treats pregame and closing (`odds_lambda.storage.tier_validator.CRITICAL_TIERS`) as required ([Coverage and completeness checks](coverage-checks.md)).
- **ML sampling**: `odds_analytics.feature_groups.TierSampler` picks the latest snapshot whose tier is the `decision_tier` or any tier farther from kickoff, excludes IN_PLAY, and so falls back to farther tiers ([Look-ahead bias](../modelling/look-ahead-bias.md), [Feature pipeline](../modelling/feature-pipeline.md)).

Missing-tier diagnosis steps are in [Debug the AWS self-scheduling jobs](../scheduling/howto-debug-aws-scheduler.md).

## Field description and priority order

The `odds_core.models.OddsSnapshot.fetch_tier` field description lists five tiers and omits `in_play`. `odds_lambda.fetch_tier.FetchTier.get_priority_order` returns the tiers closest to kickoff first and appends `IN_PLAY` last, though its docstring lists only five.

## Related

- [OddsWriter](odds-writer.md)
- [Odds snapshot and odds tables](odds-tables.md)
- [Self-scheduling](../scheduling/self-scheduling.md)
- [OddsPortal scraper](../collection/oddsportal-scraper.md)
