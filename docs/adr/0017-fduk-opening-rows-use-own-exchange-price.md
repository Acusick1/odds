---
status: accepted
---

# Pair FDUK opening rows only with their own snapshot's exchange price

In the retail-EV target, a decision row whose retail price comes from an FDUK opening snapshot takes its baseline exchange price from that same snapshot and no price or feature from any other source. This adds to [ADR-0012](0012-retail-ev-against-betfair-exchange-close.md) and does not supersede it.

## Context

ADR-0012's baseline is the devigged exchange price from the latest live Betfair job or FDUK opening snapshot at or before t. That assumes t is when the retail price was taken. For an FDUK opening snapshot it is not: the stamp is FDUK's latest documented collection time and an upper bound: the price was taken at or before it, by an amount FDUK does not record ([historical ingest scripts](../../wiki/collection/historical-ingest-scripts.md#snapshot-shape)). Any other source's price at or before the stamp can then postdate the retail price.

The as-of-t feature experiment (#545, PR #548) showed the leak in practice. Polymarket prices looked up at the FDUK stamp predicted those rows' residual. Shifting every Polymarket lookup back 1 hour removed 54 to 79% of the gain, and after that shift no gain had a CI that excluded zero ([as-of-t groups](../../wiki/evaluation/exp-epl-retail-ev-as-of-t-groups.md)).

## Decision

- A row whose retail snapshot kind is an upper-bound time (`odds_analytics.retail_ev.UPPER_BOUND_TIME_KINDS`, today the FDUK opening snapshot) takes its baseline from the Betfair Exchange price in its own snapshot, subject to [ADR-0014](0014-ignore-wide-exchange-prices.md)'s overround cap. With none, the row is dropped (`odds_analytics.retail_ev.DropCounts.rows_without_own_exchange`).
- Such a row takes no price or feature from another source: no Polymarket price, lagged exchange price, schedule or standings (`odds_analytics.retail_ev.RetailEvRow.decision_time_is_exact`).
- Rows from live collection, whose times are exact, follow ADR-0012 unchanged.

## Considered options

- **Drop FDUK opening rows.** It removes the leak class, but most 2024-26 events outside live collection have only an FDUK opening snapshot, so the model would train on about 70 live events (69 in the [as-of-t run](../../wiki/evaluation/exp-epl-retail-ev-as-of-t-groups.md#setup)).
- **Shift t back by an assumed lag.** FDUK does not record when each price was taken, so any lag is a guess, and a lag long enough to be safe for most matches would make the row's horizon wrong for the rest.
- **Keep ADR-0012 as written.** It leaks, as #548 showed.

## Consequences

- FDUK opening rows add training data and baseline comparisons, but cannot show the value of any cross-source feature. Those features are judged on live rows only.
- The 2024-26 development data loses 195 rows whose own snapshot has no usable exchange price.
- A new snapshot kind whose time is an upper bound joins `UPPER_BOUND_TIME_KINDS` under this rule, not a new ADR.

## Evidence

- Issue #545 and PR #548; [as-of-t groups](../../wiki/evaluation/exp-epl-retail-ev-as-of-t-groups.md); [CLV target](../../wiki/modelling/clv-target.md#the-retail-ev-target).
