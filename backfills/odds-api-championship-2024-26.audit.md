# Audit: `odds-api-championship-2024-26.yaml`

2025-26 is sealed ([ADR-0021](../docs/adr/0021-sealed-holdout-and-one-shot-gate.md)): its section and this summary report counts, coverage and ages only, with no overround check, back-price check or other price-derived statistic.

Coverage of the EFL Championship Odds API historical run (`h2h` stored as `1x2`, `regions=uk`), read from the production database on 2026-11-12 after the run finished at 18:08Z. The tables below each season come from `uv run python scripts/audit_backfill.py --manifest backfills/odds-api-championship-2024-26.yaml` ([audit a finished manifest](../wiki/collection/howto-backfill-odds-api-history.md#audit-a-finished-manifest)). The near-kick-off Betfair Exchange figures are research inputs: whether they can stand in for the FDUK close is [#730](https://github.com/Acusick1/odds/issues/730)'s result, still pending.

## Summary

| | 2024-25 | 2025-26 |
|---|---|---|
| FINAL fixtures / with a stored snapshot | 552 / 552 | 552 / 552 |
| Unmatched listings | 6 (two swapped dates, four postponed fixtures) | 19 (eleven probe listings, eight postponed-fixture listings) |
| KO−2h band (10 min) served | 100.0% | 100.0% |
| KO−1d band (60 min) served | 100.0% (1 minute short) | 99.9% (450 minutes short) |
| KO−4d band (6 h) served | 98.2% | 99.7% |
| Exchange price within 5 min of KO | 521 / 552 | 504 / 552 |
| Exchange price within 10 min of KO | 552 / 552 | 540 / 552 |
| Provider gaps recorded | 1 | 2 |
| Off-phase snapshots | 1 | 3 |
| Planned calls / stored / covered by a gap / missing | 5,091 / 5,090 / 1 / 0 | 4,779 / 4,777 / 2 / 0 |
| Distinct stored snapshot times × 10 units | 50,910 | 47,790 |

## Findings

- **Billing.** `odds backfill execute` printed 9,888 paid calls at 10 units each, 98,880 units, against 9,870 planned. The 2025-04-15T18:55:38Z snapshot, a provider gap, was requested 21 times before gaps were recorded and skipped, so 20 calls (200 units) bought nothing new. Three off-phase snapshots each stopped the run until it was resumed. The two probe snapshots (2024-09-14 and 2025-09-13, `h2h,totals` with named books) are outside these counts.
- **Fixture horizon.** The band reaches 96 h before kick-off (KO−4d), and some rounds are first stored 71 to 102 h before it: 69 fixtures in 2024-25 and 43 in 2025-26, nearly all in midweek rounds (Tuesday to Friday) and the 2025-11-29, 2026-01-01 and 2026-04-03 rounds (fixture lists from a separate query). The API appears to list a round only once the previous one is played `(inferred)`. That leaves 1.8% (2024-25) and 0.2% (2025-26) of the KO−4d band's decisions without a snapshot, which no purchase can fill.
- **Final round.** Ipswich v Queens Park Rangers and Preston North End v Southampton (2026-05-02T11:30Z) have a stored snapshot at 2026-04-23T06:40:38Z and none until 2026-04-29T00:40:38Z. The response of 2026-04-24T00:40:38Z lists 12 games, neither of them, and no unmatched listing names them (separate query). They account for 1,580 of the 2025-26 KO−4d band's 1,582 unserved decisions classed other.
- **Postponed fixtures.** Eleven fixtures were listed at their original dates, in responses that attached to no event, and played later; `event_postponements` has no row for any of them (separate query).

  | Fixture | Listed at | Responses | Played | First stored before KO |
  |---|---|---|---|---|
  | Blackburn Rovers v Portsmouth | 2024-11-23T15:00Z | 127 | 2025-01-15T19:45Z | 262 h |
  | Portsmouth v Millwall | 2024-11-27T19:45Z | 137 | 2025-01-28T19:45Z | 71 h |
  | Cardiff City v Watford | 2024-12-07T15:00Z | 86 | 2025-01-14T19:45Z | 238 h |
  | Plymouth Argyle v Oxford United | 2024-12-07T15:00Z | 69 | 2025-01-14T19:45Z | 238 h |
  | Blackburn Rovers v Ipswich Town | 2025-09-20T14:00Z | 182 | 2025-12-02T19:45Z | 82 h |
  | Charlton Athletic v Portsmouth | 2025-12-06T12:30Z | 174 | 2026-02-17T19:45Z | 248 h |
  | Blackburn Rovers v Sheffield Wednesday | 2025-12-06T15:00Z | 183 | 2026-02-03T19:45Z | 126 h |
  | Sheffield United v Oxford United | 2026-01-04T12:00Z | 107 | 2026-02-03T19:45Z | 126 h |
  | Hull City v Watford | 2026-01-04T15:00Z | 120 | 2026-02-03T19:45Z | 126 h |
  | Portsmouth v Ipswich Town | 2026-01-04T15:00Z, 2026-02-03T19:45Z | 120, 127 | 2026-04-14T19:00Z | 126 h |
  | Ipswich Town v Hull City | 2026-02-14T15:00Z | 11 | 2026-03-03T19:45Z | 240 h |

- **Swapped dates.** Sheffield United v Swansea City (listed 2024-10-01T18:45Z in 2 responses) and Norwich City v Leeds United (2024-10-02T18:45Z in 1) were listed a day off, in responses from 2024-09-25 and 09-26 with no books; the stored kick-offs are 2024-10-02T18:45Z and 2024-10-01T18:45Z.
- **Probe listings.** The 11 unmatched listings of 2025-09-13T11:30Z to 2025-09-14T11:00Z are in 2 responses each, both repeat fetches of the 2025-09-13T09:55:39Z probe snapshot. Their kick-offs equal the stored ones, they carry the ids of stored events, and the first fetch stored a snapshot for each, so the repeats had nothing to attach `(inferred)`.
- **Provider gaps.** Three holes: 2025-04-15T18:55:38Z (returned 18:00:38Z), 2025-10-20T18:40:38Z (returned 16:31:34Z) and 2026-05-01T06:40:38Z (returned 06:05:37Z). The first left 12 fixtures 55 minutes without a 6-hour-fresh input. The second left 8 fixtures without a 60-minute-fresh input for about 57 minutes: 433 decision minutes are classed as gap, and 16 more (two per fixture) as other, because the next stored snapshot is stamped 19:42:57Z, 2 min 19 s after its planned 19:40:38Z. The third cost no decision.
- **Off-phase snapshots.** Four stored snapshots are off the grid's phase: 2025-02-03T15:56:33Z, 2025-10-03T05:51:42Z, 2025-10-20T16:31:34Z (the provider gap's return) and 2025-10-20T19:42:57Z. The calls for 2025-02-03T15:55:38Z, 2025-10-03T05:50:38Z and 2025-10-20T19:40:38Z each stopped the run until it was resumed. The first two leave one unserved KO−1d decision minute each, of Queens Park Rangers v Blackburn Rovers and Wrexham AFC v Birmingham City (separate query).
- **Books outside `UK_RETAIL_BOOKMAKERS`.** `casumo`, `grosvenor`, `leovegas`, `livescorebet`, `virginbet` (both seasons) and `mrgreen` (2024-25) are stored under their own keys and counted neither as UK retail nor as exchanges. They are UK-licensed sportsbooks `(inferred)`, so retail rules reading `UK_RETAIL_BOOKMAKERS` ignore up to six of the region's books.
- **Uneven retail books.** In 2024-25 `skybet` is on 91.8% of fixtures, `betfair_sportsbook` on 84.1% and `smarkets` on 71.7%; in 2025-26 `skybet` is in 55% and 53% of the snapshots of March and April 2026 and 80 to 99% of the other months' (monthly shares from a separate query).
- **Betfred.** `betfred` is stored on 50 of the 2025-26 fixtures, from 2026-04-13 to 2026-05-02 (dates from a separate query), and on none in 2024-25.
- **Pinnacle** appears only in the two probe snapshots, which named it; the `uk` region does not return it.
- **Exchange quotes without a draw.** When the exchange quotes only the two teams, the response's `h2h` keeps its key and is stored as an `h2h` snapshot on the soccer fixture: 5 snapshots on 5 fixtures in 2024-25 and 71 on 26 fixtures before kick-off in 2025-26. Readers of `1x2` do not see these partial exchange books. Listings with no books are stored as snapshots with no market (27 before kick-off in 2024-25, 31 in 2025-26).
- **Near kick-off exchange.** The latest exchange price is within 4.4 min of kick-off for 521 fixtures in 2024-25 and 504 in 2025-26, and 9.1 to 9.4 min old for the other 31 and 36, at the 19:00, 20:00 and 19:45 slots where the plan's closing calls are 10 minutes apart and none falls 4.4 min before kick-off. Twelve fixtures stored with a kick-off at :01 or :31 (2026-02-06 to 2026-02-09) have a price 10.4 min old, so they are outside the 10-minute share; the KO−2h band is still fully served. Every 2024-25 price found is complete and within the 5% overround limit. The run bought no snapshot nearer kick-off.

## 2024-25

### Fixtures, 2024-25

552 FINAL fixtures; 552 with a stored `1x2` snapshot at or before kick-off; 0 without.

Listings in stored responses that attached to no event: 6.

| Home | Away | Listed kick-off | Responses |
|---|---|---|---|
| Sheffield United | Swansea City | 2024-10-01T18:45:00Z | 2 |
| Norwich City | Leeds United | 2024-10-02T18:45:00Z | 1 |
| Blackburn Rovers | Portsmouth | 2024-11-23T15:00:00Z | 127 |
| Portsmouth | Millwall | 2024-11-27T19:45:00Z | 137 |
| Cardiff City | Watford | 2024-12-07T15:00:00Z | 86 |
| Plymouth Argyle | Oxford United | 2024-12-07T15:00:00Z | 69 |

Odds API historical snapshots of these fixtures outside `1x2`:

| Market | Stamped | Snapshots | Fixtures |
|---|---|---|---|
| h2h | before KO | 5 | 5 |
| totals | before KO | 11 | 11 |
| none | after KO | 6 | 6 |
| none | before KO | 27 | 24 |

### Bookmakers, 2024-25

Share of the 552 fixtures with the book in any snapshot at or before kick-off, and of the 76,863 such snapshots.

| Book | Fixtures | Snapshots | Role | Odds API keys aliased to it |
|---|---|---|---|---|
| `betfair_exchange` | 100.0% | 84.7% | exchange | `betfair`, `betfair_ex_eu`, `betfair_ex_uk` |
| `betvictor` | 100.0% | 89.8% | UK retail | none |
| `betway` | 100.0% | 92.6% | UK retail | none |
| `boylesports` | 100.0% | 82.2% | UK retail | none |
| `casumo` | 100.0% | 81.6% | neither | none |
| `coral` | 100.0% | 87.7% | UK retail | none |
| `grosvenor` | 100.0% | 81.6% | neither | none |
| `ladbrokes` | 100.0% | 87.7% | UK retail | `ladbrokes_uk` |
| `leovegas` | 100.0% | 81.7% | neither | none |
| `matchbook` | 100.0% | 61.3% | exchange | none |
| `paddypower` | 100.0% | 97.5% | UK retail | none |
| `unibet_uk` | 100.0% | 81.6% | UK retail | none |
| `williamhill` | 100.0% | 79.1% | UK retail | none |
| `skybet` | 91.8% | 81.1% | UK retail | none |
| `livescorebet` | 89.1% | 74.0% | neither | none |
| `virginbet` | 89.1% | 74.1% | neither | none |
| `betfair_sportsbook` | 84.1% | 43.4% | UK retail | `betfair_sb_uk` |
| `888sport` | 76.1% | 74.1% | UK retail | `sport888` |
| `smarkets` | 71.7% | 69.4% | exchange | none |
| `mrgreen` | 28.3% | 20.0% | neither | none |
| `pinnacle` | 2.0% | 0.0% | neither | none |

### Leads, 2024-25

Whole-minute decisions per fixture and band, served when a stored snapshot lies in `[t − max age, t]` at its actual time. Unserved decisions are split into those before the fixture's first stored snapshot (not yet listed), those a recorded gap's planned snapshot would have served, and the rest. The last column needs `betfair_exchange` and a UK retail book in the same snapshot.

| Band | Decisions | Served | Unserved: not listed | Unserved: gap | Unserved: other | Fixtures fully served | Served, exchange + retail |
|---|---|---|---|---|---|---|---|
| KO−1m..KO−120m, max age 10m | 66,240 | 100.0% | 0 | 0 | 0 | 552/552 | 100.0% |
| KO−121m..KO−1440m, max age 60m | 728,640 | 100.0% | 0 | 0 | 1 | 551/552 | 100.0% |
| KO−1441m..KO−5760m, max age 360m | 2,384,640 | 98.2% | 43,206 | 660 | 0 | 499/552 | 95.4% |

Fixtures first stored less than 102 h before kick-off: 69, first stored at leads (h: fixtures) 71: 1, 75: 7, 76: 5, 77: 12, 79: 3, 82: 10, 83: 2, 93: 1, 96: 2, 97: 8, 99: 4, 100: 5, 101: 7, 102: 2.

### Near kick-off `betfair_exchange`, 2024-25

Per fixture, the latest exchange snapshot at or before kick-off within 120 min; usable means a back price on every outcome with back overround at most 5%. Research inputs, not validated closing labels.

| Price | Fixtures with one | Within 5 min | Within 10 min | p50 age (min) | p75 age (min) | p90 age (min) | p99 age (min) |
|---|---|---|---|---|---|---|---|
| Any exchange quote | 552/552 | 94.4% | 100.0% | 4.4 | 4.4 | 4.4 | 9.4 |
| Usable | 552/552 | 94.4% | 100.0% | 4.4 | 4.4 | 4.4 | 9.4 |

Latest exchange snapshot missing an outcome's back price: 0 fixtures.

### Provider gaps and off-phase snapshots, 2024-25

Recorded gaps: 1.

| Planned | Returned | Next |
|---|---|---|
| 2025-04-15T18:55:38Z | 2025-04-15T18:00:38Z | 2025-04-15T22:25:39Z |

Stored snapshot times off the grid's phase (outside their nearest grid time's whole minute): 1 (2025-02-03T15:56:33Z).

### Calls, 2024-25

Planned 5,091; stored 5,090; covered by a recorded gap 1; missing 0. Distinct stored snapshot times, any market: 5,091, × 10 units = 50,910 units. `1x2` snapshot times matching no planned call (gap returns, probes): 1. Planned calls that serve only fixtures not yet stored at that time (not listed): 0.

## 2025-26

### Fixtures, 2025-26

552 FINAL fixtures; 552 with a stored `1x2` snapshot at or before kick-off; 0 without.

Listings in stored responses that attached to no event: 19.

| Home | Away | Listed kick-off | Responses |
|---|---|---|---|
| Charlton Athletic | Millwall | 2025-09-13T11:30:00Z | 2 |
| Oxford United | Leicester City | 2025-09-13T11:30:00Z | 2 |
| Preston North End | Middlesbrough | 2025-09-13T11:30:00Z | 2 |
| Coventry City | Norwich City | 2025-09-13T14:00:00Z | 2 |
| Sheffield Wednesday | Bristol City | 2025-09-13T14:00:00Z | 2 |
| Stoke City | Birmingham City | 2025-09-13T14:00:00Z | 2 |
| Swansea City | Hull City | 2025-09-13T14:00:00Z | 2 |
| Watford | Blackburn Rovers | 2025-09-13T14:00:00Z | 2 |
| West Bromwich Albion | Derby County | 2025-09-13T14:00:00Z | 2 |
| Wrexham AFC | Queens Park Rangers | 2025-09-13T14:00:00Z | 2 |
| Southampton | Portsmouth | 2025-09-14T11:00:00Z | 2 |
| Blackburn Rovers | Ipswich Town | 2025-09-20T14:00:00Z | 182 |
| Charlton Athletic | Portsmouth | 2025-12-06T12:30:00Z | 174 |
| Blackburn Rovers | Sheffield Wednesday | 2025-12-06T15:00:00Z | 183 |
| Sheffield United | Oxford United | 2026-01-04T12:00:00Z | 107 |
| Hull City | Watford | 2026-01-04T15:00:00Z | 120 |
| Portsmouth | Ipswich Town | 2026-01-04T15:00:00Z | 120 |
| Portsmouth | Ipswich Town | 2026-02-03T19:45:00Z | 127 |
| Ipswich Town | Hull City | 2026-02-14T15:00:00Z | 11 |

Odds API historical snapshots of these fixtures outside `1x2`:

| Market | Stamped | Snapshots | Fixtures |
|---|---|---|---|
| h2h | after KO | 14 | 14 |
| h2h | before KO | 71 | 26 |
| totals | before KO | 11 | 11 |
| none | after KO | 14 | 14 |
| none | before KO | 31 | 22 |

### Bookmakers, 2025-26

Share of the 552 fixtures with the book in any snapshot at or before kick-off, and of the 88,863 such snapshots.

| Book | Fixtures | Snapshots | Role | Odds API keys aliased to it |
|---|---|---|---|---|
| `betfair_exchange` | 100.0% | 77.7% | exchange | `betfair`, `betfair_ex_eu`, `betfair_ex_uk` |
| `betvictor` | 100.0% | 58.5% | UK retail | none |
| `betway` | 100.0% | 82.8% | UK retail | none |
| `boylesports` | 100.0% | 75.8% | UK retail | none |
| `casumo` | 100.0% | 94.9% | neither | none |
| `coral` | 100.0% | 75.1% | UK retail | none |
| `grosvenor` | 100.0% | 94.9% | neither | none |
| `ladbrokes` | 100.0% | 75.1% | UK retail | `ladbrokes_uk` |
| `leovegas` | 100.0% | 94.9% | neither | none |
| `matchbook` | 100.0% | 54.8% | exchange | none |
| `paddypower` | 100.0% | 81.4% | UK retail | none |
| `smarkets` | 100.0% | 87.1% | exchange | none |
| `williamhill` | 100.0% | 68.1% | UK retail | none |
| `skybet` | 99.8% | 76.2% | UK retail | none |
| `unibet_uk` | 99.8% | 80.4% | UK retail | none |
| `betfair_sportsbook` | 98.4% | 41.4% | UK retail | `betfair_sb_uk` |
| `livescorebet` | 95.7% | 92.0% | neither | none |
| `virginbet` | 95.7% | 92.0% | neither | none |
| `888sport` | 63.0% | 46.8% | UK retail | `sport888` |
| `betfred` | 9.1% | 5.3% | UK retail | `betfred_uk` |
| `pinnacle` | 2.0% | 0.0% | neither | none |

### Leads, 2025-26

Whole-minute decisions per fixture and band, served when a stored snapshot lies in `[t − max age, t]` at its actual time. Unserved decisions are split into those before the fixture's first stored snapshot (not yet listed), those a recorded gap's planned snapshot would have served, and the rest. The last column needs `betfair_exchange` and a UK retail book in the same snapshot.

| Band | Decisions | Served | Unserved: not listed | Unserved: gap | Unserved: other | Fixtures fully served | Served, exchange + retail |
|---|---|---|---|---|---|---|---|
| KO−1m..KO−120m, max age 10m | 66,240 | 100.0% | 0 | 0 | 0 | 552/552 | 100.0% |
| KO−121m..KO−1440m, max age 60m | 728,640 | 99.9% | 0 | 433 | 17 | 543/552 | 99.8% |
| KO−1441m..KO−5760m, max age 360m | 2,384,640 | 99.7% | 4,410 | 0 | 1,582 | 545/552 | 95.6% |

Fixtures first stored less than 102 h before kick-off: 43, first stored at leads (h: fixtures) 81: 3, 82: 2, 96: 7, 97: 1, 99: 13, 100: 10, 101: 7.

### Near kick-off `betfair_exchange`, 2025-26

Per fixture, the latest exchange snapshot at or before kick-off within 120 min, counted and aged only: this season is sealed ([ADR-0021](../docs/adr/0021-sealed-holdout-and-one-shot-gate.md)), so no price-derived statistic is reported. Research inputs, not validated closing labels.

| Price | Fixtures with one | Within 5 min | Within 10 min | p50 age (min) | p75 age (min) | p90 age (min) | p99 age (min) |
|---|---|---|---|---|---|---|---|
| Any exchange quote | 552/552 | 91.3% | 97.8% | 4.4 | 4.4 | 4.4 | 10.4 |

### Provider gaps and off-phase snapshots, 2025-26

Recorded gaps: 2.

| Planned | Returned | Next |
|---|---|---|
| 2025-10-20T18:40:38Z | 2025-10-20T16:31:34Z | 2025-10-20T19:32:57Z |
| 2026-05-01T06:40:38Z | 2026-05-01T06:05:37Z | 2026-05-01T07:20:38Z |

Stored snapshot times off the grid's phase (outside their nearest grid time's whole minute): 3 (2025-10-03T05:51:42Z, 2025-10-20T16:31:34Z, 2025-10-20T19:42:57Z).

### Calls, 2025-26

Planned 4,779; stored 4,777; covered by a recorded gap 2; missing 0. Distinct stored snapshot times, any market: 4,779, × 10 units = 47,790 units. `1x2` snapshot times matching no planned call (gap returns, probes): 2. Planned calls that serve only fixtures not yet stored at that time (not listed): 0.

