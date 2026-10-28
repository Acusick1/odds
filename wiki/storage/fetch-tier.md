---
type: reference
title: Fetch tier
description: The FetchTier enum and bounds, the one hours-to-tier rule, the tier a snapshot covers by its time basis, and where tiers are read.
sources:
  - odds_lambda.fetch_tier.FetchTier
  - odds_lambda.fetch_tier.FetchTier.max_hours
  - odds_lambda.fetch_tier.FetchTier.interval_hours
  - odds_lambda.tier_utils.calculate_tier
  - odds_lambda.tier_utils.snapshot_tier
  - odds_lambda.storage.tier_validator.covered_tiers
  - odds_lambda.tier_utils.snapshot_tiers
  - odds_lambda.storage.tier_validator.TierCoverageValidator
  - odds_lambda.storage.tier_validator.CRITICAL_TIERS
  - odds_analytics.feature_groups.TierSampler
  - odds_analytics.quality_metrics.QualityMetrics
  - odds_core.models.OddsSnapshot
status: stable
---

# Fetch tier

A fetch tier labels an odds snapshot by how long before kickoff it was taken. `odds_lambda.fetch_tier.FetchTier` is the enum. No tier is stored: every reader derives a snapshot's tier from its time, source, time basis and market against the event's current kick-off ([odds_snapshots columns](odds-tables.md#odds_snapshots)), so a moved kick-off re-tiers every snapshot at once. Tiers feed scheduling cadence, validation and ML sampling.

## Tier table

| Tier | Hours to kickoff | `interval_hours` | Purpose |
|---|---|---|---|
| opening | 72+ | 48 | initial line release |
| early | 24-72 | 24 | line establishment |
| sharp | 12-24 | 12 | professional betting |
| pregame | 3-12 | 3 | active betting |
| closing | 0-3 | 0.5 | critical line movement |
| in_play | after kickoff | 0.5 | incidental collection |

The boundaries in the table are `odds_lambda.fetch_tier.FetchTier.max_hours` and the intervals are `odds_lambda.fetch_tier.FetchTier.interval_hours`; the interval is read only by `odds_analytics.quality_metrics.QualityMetrics`, which counts snapshots as distinct snapshot times and gives each the tier `odds_lambda.tier_utils.snapshot_tiers` derives; schedulers use each job's `odds_lambda.scheduling.decision.CadenceConfig`, which keeps these boundaries but sets its own intervals and its own check-in for a sport with no fixture ([Decision engine](../scheduling/decision-engine.md#cadence)); a job's extra runs before kickoffs are [lead runs](../scheduling/decision-engine.md#lead-runs), outside the cadence.

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

Every tier in the system comes from `odds_lambda.tier_utils.snapshot_tier` or `odds_lambda.tier_utils.snapshot_tiers`: the coverage check (`odds_lambda.storage.tier_validator.covered_tiers`), the coverage metrics (`odds_analytics.quality_metrics.QualityMetrics`), the training sampler and `raw` close (`odds_analytics.feature_groups.TierSampler`, `odds_analytics.feature_groups.raw_closes`) and the gap planner, which tiers each planned fetch as the `upper_bound` snapshot the backfill stores, so a filled game is not planned again.

## Where tiers are read

- **Scheduling** uses the tier to pick a cadence; the per-job cadence table is on [Decision engine](../scheduling/decision-engine.md#cadence).
- **Validation**: `odds_lambda.storage.tier_validator.TierCoverageValidator` takes each snapshot's tier from `odds_lambda.storage.tier_validator.covered_tiers` against the event's current kick-off, and treats pregame and closing (`odds_lambda.storage.tier_validator.CRITICAL_TIERS`) as required ([Coverage and completeness checks](coverage-checks.md)).
- **ML sampling**: `odds_analytics.feature_groups.TierSampler` takes each snapshot's tier from `odds_analytics.feature_groups.tiers_of`, which applies `odds_lambda.tier_utils.snapshot_tiers` to the event's snapshots of the market, picks the latest snapshot, only of the sources whose close has a sharp book, whose tier is the `decision_tier` or any tier farther from kickoff, excludes IN_PLAY, and so falls back to farther tiers. The `raw` target's close is the decision snapshot's own source's latest CLOSING snapshot by the same tiers ([closing snapshot](../modelling/clv-target-functions.md#closing-snapshot-of-a-raw-target), [Look-ahead bias](../modelling/look-ahead-bias.md), [Feature pipeline](../modelling/feature-pipeline.md)).

Missing-tier diagnosis steps are in [Debug the AWS self-scheduling jobs](../scheduling/howto-debug-aws-scheduler.md).

## Priority order

`odds_lambda.fetch_tier.FetchTier.get_priority_order` returns the tiers closest to kickoff first and appends `IN_PLAY` last, though its docstring lists only five.

## Related

- [OddsWriter](odds-writer.md)
- [Odds snapshots table](odds-tables.md)
- [Self-scheduling](../scheduling/self-scheduling.md)
- [OddsPortal scraper](../collection/oddsportal-scraper.md)
