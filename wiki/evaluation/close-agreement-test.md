---
type: reference
title: Close agreement test
description: The test that decides whether a candidate close agrees with a reference close (its per-outcome and per-band statistics, the pass thresholds pre-set on issue 730 and the verdicts), and the freshness and price-age figures reported with it.
sources:
  - odds_analytics.close_agreement.Delta
  - odds_analytics.close_agreement.stat
  - odds_analytics.close_agreement.outcome_verdict
  - odds_analytics.close_agreement.band_verdict
  - odds_analytics.close_agreement.overall_verdict
  - odds_analytics.close_agreement.band_of
  - odds_analytics.close_agreement.agreement
  - odds_analytics.close_agreement.freshness
status: stable
---

# Close agreement test

`odds_analytics.close_agreement` decides whether a candidate price can stand in for a reference close. The [substitute-close comparison](substitute-close-comparison.md) runs it with FDUK's Betfair Exchange close as the reference, and the [live-close alternatives comparison](live-close-alternatives-comparison.md) with the Betfair job's live close. Both prices are devigged proportionally ([devigging](../foundations/devigging.md)).

## Rows

One `odds_analytics.close_agreement.Delta` per outcome of each compared event holds the reference and candidate probabilities. Its `dp_pp` is `p_candidate - p_reference` in percentage points.

## Statistics and verdicts

`odds_analytics.close_agreement.agreement` slices the rows by outcome and by band of the reference probability (`odds_analytics.close_agreement.band_of`: below 0.25, 0.25 up to 0.5, and 0.5 and above). Each slice reports n, mean and sample sd in pp (`odds_analytics.close_agreement.stat`) with a verdict. The thresholds were pre-set on [#730](https://github.com/Acusick1/odds/issues/730) and are constants in the module:

- **Per outcome** (`odds_analytics.close_agreement.outcome_verdict`): pass when `|mean dp|` is at most 0.3 pp and the sd at most 1.0 pp. An outcome with under two rows is inconclusive.
- **Per band** (`odds_analytics.close_agreement.band_verdict`): pass when `|mean dp|` is at most 0.5 pp. A band under 50 rows is inconclusive, not a pass.
- **Overall** (`odds_analytics.close_agreement.overall_verdict`): fail if any slice fails, pass only if every slice passes, otherwise inconclusive.

## Freshness and age

`odds_analytics.close_agreement.freshness` reports, with the verdicts:

- The share of events with a reference close whose candidate price is within 5, 10, 30 and 60 minutes before kick-off, with the counts. An event with no usable candidate price counts against the share.
- The age of the candidate prices used, in minutes before kick-off: n, minimum, 25th, 50th, 75th and 90th percentiles, maximum and mean.

Freshness alone does not make a valid substitute ([#730](https://github.com/Acusick1/odds/issues/730)), so read the age distribution with the verdict.

## Tests

`tests/unit/test_close_agreement.py` tests the statistics, verdicts and bands.
