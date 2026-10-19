---
status: proposed
---

# Ignore wide Betfair Exchange books in the retail-EV target

The retail-EV target treats a Betfair Exchange snapshot whose back overround exceeds 5% as no price, both for the baseline's exchange price at t and for the close. This adds to [ADR-0012](0012-retail-ev-against-betfair-exchange-close.md) and does not supersede it.

## Context

ADR-0012 takes the devigged exchange back price at or before t as the baseline and the last pre-off exchange price as the close. It caps the exchange price's age at 2 hours, because an old price weakens the baseline and flatters the model. It says nothing about thin books.

The live Betfair job records the best back price of each runner. Days before kick-off a market can have few back offers, so the three best backs sum to far more than 1; overrounds up to 157% were seen. Proportional devigging of such a book does not give a fair price. `experiments/scripts/retail_ev_exchange_overround.py` builds the decision rows from 2024-08-01 to 2026-10-18 on the development database with no cap. It reports the standard deviation of the exchange price at t minus the close, by horizon and the snapshot's back overround.

Every book over 2% is 3 to 7 days out, so the comparison that matters is within that horizon, where the gap also grows naturally with time to the close:

| 3–7 d, back overround | rows | exchange snapshots | gap std (pp) |
|---|---|---|---|
| up to 1% | 19,779 | 411 | 2.65 |
| 1–2% | 23,211 | 472 | 3.30 |
| 2–3% | 7,167 | 145 | 4.72 |
| 3–5% | 4,689 | 99 | 3.72 |
| 5–10% | 954 | 20 | 7.49 |
| over 10% | 1,212 | 30 | 15.95 |

The effect does not come from horizon: at the same horizon, books over 5% sit two to six times further from the close than books under 1%. The wide groups are small, 50 snapshots in all, about 1.8% of the exchange snapshots that rows use. Left in, they weaken the baseline exactly where the model can see `exchange_overround`, which is the effect ADR-0012's age cap prevents for stale prices.

## Decision

- An exchange snapshot whose back overround exceeds 5% (`odds_analytics.retail_ev.MAX_EXCHANGE_OVERROUND`) is skipped, as if it did not exist.
- A row then takes the latest narrow exchange price at or before t, subject to ADR-0012's 2-hour age cap, and is dropped when there is none.
- The close takes the latest narrow live Betfair price within 30 minutes of kick-off, else the FDUK close.

## Considered options

- **No cap.** It keeps 2% more rows, and the baseline's weighted MSE rises from 0.00339 to 0.00358.
- **Drop the row** whenever its latest exchange price is wide. On this data it gives the same 113,907 rows and baseline weighted MSE as the fallback, because no wide snapshot had a narrow one within the previous 2 hours. The comparison is weak: the script derives the dropped rows from the uncapped build, whose close is also uncapped, and the two policies would differ only where live Betfair snapshots are dense enough for a narrow price to precede a wide one within 2 hours. Falling back keeps rows whenever a narrow price within the age cap exists, and that price is as valid as any other price of that age (inferred).
- **A liquidity measure** such as back size or matched volume. It exists only in the live Betfair data, not for FDUK opening and closing prices. Overround applies to both.
- **The threshold.** Within 3 to 7 days the spread rises from 2.65 pp to 4.72 pp up to 3% overround, is 3.72 pp at 3–5%, and doubles above 5%. The 5% cut is a judgement on 50 wide snapshots; a 2% or 3% cut would also remove books whose spread is within the range of narrower ones.

## Consequences

- The development data has 338 wide snapshots, and the cap drops 2,166 of the 116,073 uncapped rows.
- The threshold is a constant. A different value means a new ADR, because it changes the rows that the keep bar is scored on.
