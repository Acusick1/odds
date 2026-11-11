---
type: reference
title: Odds API vendor facts
description: What The Odds API bills, when its historical snapshots fall, how it rate-limits, which books and fixtures its historical responses carry per sport, and how long a backfill call takes, each with its source.
sources:
  - odds_analytics.backfill_plan.UNITS_PER_REGION_MARKET
  - odds_analytics.backfill_plan.BOOKMAKERS_PER_REGION
  - odds_lambda.data_fetcher.RATE_LIMIT_WAIT
  - odds_analytics.backfill_executor.RATE_LIMIT_ATTEMPTS
status: stable
---

# Odds API vendor facts

Facts about The Odds API itself, as its documentation states them or as the acquisition key's calls measured them. "Measured 2026-11-11" marks results of paid historical calls and of a request ramp on the free `/sports` endpoint made that day; issue links mark earlier manifest probes. The client that calls the API is on [the Odds API client](odds-api-client.md), and the backfill that spends most of its units on [Odds API historical backfill](odds-api-backfill.md).

## Billing

- A historical odds call costs `10 x [markets specified] x [regions specified]` units ([historical odds costs](https://the-odds-api.com/liveapi/guides/v4/#usage-quota-costs-8), [historical odds data](https://the-odds-api.com/historical-odds-data/)); `odds_analytics.backfill_plan.UNITS_PER_REGION_MARKET` holds the figure the planner uses.
- "Every group of 10 bookmakers is the equivalent of 1 region", so 1 to 10 named books bill as one region and 11 to 20 as two, and when a request gives both `bookmakers` and `regions`, `bookmakers` takes priority ([get odds](https://the-odds-api.com/liveapi/guides/v4/#get-odds)); `odds_analytics.backfill_plan.BOOKMAKERS_PER_REGION` holds the group size.
- The cost is linear in markets, so markets requested together cost the same units as each requested in its own call: splitting markets costs calls, not units ([historical odds costs](https://the-odds-api.com/liveapi/guides/v4/#usage-quota-costs-8)).
- The cost formula counts only markets and regions, so filters such as `eventIds` and the commence-time bounds do not reduce it ([historical odds costs](https://the-odds-api.com/liveapi/guides/v4/#usage-quota-costs-8)).
- `/sports` does not count against the quota ([get sports](https://the-odds-api.com/liveapi/guides/v4/#get-sports)).
- "Responses with empty data do not count towards the usage quota" ([historical odds, more info](https://the-odds-api.com/liveapi/guides/v4/#more-info-3)). No historical call has measured it. The historical formula counts markets specified, where the event-odds endpoints count "unique markets returned" ([event odds costs](https://the-odds-api.com/liveapi/guides/v4/#usage-quota-costs-5)); whether a historical call is billed for a requested market it does not return is unverified.
- Whether a 429 response is billed is unverified; the backfill client retries it on the assumption that it is not (`odds_analytics.backfill_executor.RATE_LIMIT_ATTEMPTS`).
- Observed `x-requests-last` on historical calls (measured 2026-11-11):

| Request | Billed units |
| --- | --- |
| `h2h,totals` with 10 named bookmakers | 20 |
| `h2h` with 10 named bookmakers | 10 |
| `h2h` with `regions=uk` | 10 |

## Snapshots

- Historical snapshots are 10 minutes apart from 2020-06-06 and 5 minutes apart from September 2022 ([historical odds data](https://the-odds-api.com/historical-odds-data/)).
- A historical request returns "the closest snapshot equal to or earlier than the provided date" ([historical odds](https://the-odds-api.com/liveapi/guides/v4/#get-historical-odds)). The response carries the snapshot's own `timestamp`, and `previous_timestamp` and `next_timestamp` for its neighbours.
- 5-minute snapshots fall 37 to 42 s past each 5-minute mark, varying by date; 10-minute snapshots fall at 5, 15, 25 minutes past the hour and so on, with no seconds ([#729](https://github.com/Acusick1/odds/issues/729)).

## Rate limits

- The documentation states no numeric rate limit: requests are limited "to protect our systems from sudden bursts in traffic", a request over the limit gets HTTP 429, and the advice is to space requests "over several seconds" ([rate limiting](https://the-odds-api.com/liveapi/guides/v4/#rate-limiting-status-code-429)). Responses carry no rate-limit headers (measured 2026-11-11).
- A ramp on `/sports` (measured 2026-11-11) got 2 responses of 429 out of 60 requests at concurrency 20, a burst of about 40 requests per second, and none at a sustained rate of about 20 requests per second.
- The backfill client retries a 429 with jittered exponential backoff (`odds_lambda.data_fetcher.RATE_LIMIT_WAIT`, `odds_analytics.backfill_executor.RATE_LIMIT_ATTEMPTS`); live collection does not retry it ([retry and keys](odds-api-client.md#retry-and-keys)).

## Coverage

Measured on historical `h2h` and `totals` responses.

- **Exchange totals.** No probed sport or season carries `betfair_ex_uk` on `totals` ([#787](https://github.com/Acusick1/odds/issues/787)).
- **Football totals.** Historical football `totals` carry only Pinnacle and William Hill ([#787](https://github.com/Acusick1/odds/issues/787)).
- **Exchange start on EPL.** `betfair_ex_uk` on EPL `h2h` is absent on 2023-03-25 and present from 2023-04-01 ([#729](https://github.com/Acusick1/odds/issues/729)).
- **Fixture horizon.** How far ahead a snapshot lists fixtures ([#787](https://github.com/Acusick1/odds/issues/787)):

| Sport | Farthest fixture listed |
| --- | --- |
| MLB | 584 to 612 min |
| NBA | 604 to 2054 min |
| Championship | 125 to 172 h |
| La Liga | 127 to 346 h |

- **The `uk` region.** `regions=uk` on historical `h2h` returns these books (measured 2026-11-11). Pinnacle is not in the region.

| Sport and month | Books |
| --- | --- |
| EPL, 2025-09 | 19: `paddypower`, `smarkets`, `betway`, `sport888`, `williamhill`, `boylesports`, `betfair_ex_uk`, `leovegas`, `virginbet`, `unibet_uk`, `casumo`, `livescorebet`, `grosvenor`, `betvictor`, `coral`, `ladbrokes_uk`, `betfair_sb_uk`, `skybet`, `matchbook` |
| La Liga, 2024-09 | 20: the EPL list and `mrgreen` |
| MLB, 2024-07 | 17 |
| NBA, 2025-01 | 11 |

- **Betfred.** `betfred_uk` appears in no historical response, by name or in the `uk` region (measured 2026-11-11), so 10 named books including it return 8 or 9.
- How each key is stored is on [sport, market and bookmaker keys](sport-and-market-keys.md).

## Throughput

A backfill call measured 2026-11-11 took about 2.3 s: 0.5 to 0.8 s of API time, about 0.5 s of storage, and a fixed 1 s pause after the call. The executor pauses after a paid call only for `--pause` seconds ([odds backfill execute](odds-api-backfill.md#commands)), so a call without a pause takes about 1 to 1.3 s `(inferred)`.
