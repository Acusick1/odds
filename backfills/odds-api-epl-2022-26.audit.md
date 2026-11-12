# Audit: `odds-api-epl-2022-26.yaml`

Coverage of the EPL Odds API historical run (`h2h` stored as `1x2`, `regions=uk`), read from the production database on 2026-11-12 after the run finished at 21:07Z. The tables below each season come from `uv run python scripts/audit_backfill.py --manifest backfills/odds-api-epl-2022-26.yaml` ([audit a finished manifest](../wiki/collection/howto-backfill-odds-api-history.md#audit-a-finished-manifest)). The near-kick-off Betfair Exchange figures are research inputs: whether they can stand in for the FDUK close is [#730](https://github.com/Acusick1/odds/issues/730)'s result, still pending.

EPL 2022-23 and 2023-24 are sealed while `odds_analytics.holdout.EPL_2022_24_SEALED` is true ([ADR-0021](../docs/adr/0021-sealed-holdout-and-one-shot-gate.md)). Their sections and this summary report counts, coverage and ages only: no overround check, back-price check or other price-derived statistic. 2022-23 covers the fixtures from 2023-04-01.

## Summary

| | 2022-23 (from 2023-04-01) | 2023-24 | 2024-25 | 2025-26 |
|---|---|---|---|---|
| FINAL fixtures / with a stored snapshot | 110 / 110 | 380 / 380 | 380 / 380 | 380 / 380 |
| Unmatched listings | 1 (date a day off) | 13 (postponed fixtures, swapped dates) | 11 (provisional kick-offs, one postponed fixture) | 8 (provisional kick-offs) |
| KO−2h band (10 min) served | 100.0% | 100.0% (2 minutes short) | 100.0% | 99.7% (120 minutes short) |
| KO−1d band (60 min) served | 100.0% | 100.0% (15 minutes short) | 100.0% (206 minutes short) | 100.0% (180 minutes short) |
| KO−7d band (6 h) served | 99.5% | 99.9% | 99.9% | 99.3% |
| Exchange price within 5 min of KO | 104 / 110 | 358 / 380 | 365 / 380 | 363 / 380 |
| Exchange price within 10 min of KO | 110 / 110 | 380 / 380 | 380 / 380 | 379 / 380 |
| Provider gaps recorded | 0 | 1 | 1 | 3 |
| Off-phase snapshots | 1 | 1 | 1 | 1 |
| Planned calls / stored / covered by a gap / missing | 1,566 / 1,566 / 0 / 0 | 5,784 / 5,781 / 1 / 2 | 5,425 / 5,421 / 4 / 0 | 5,651 / 5,634 / 17 / 0 |
| Distinct stored snapshot times × 10 units | 15,660 | 57,840 | 54,220 | 56,370 |

## Findings

- **Billing.** The run's log has 17,933 paid calls at 10 units each, 179,330 units, for 17,871 distinct snapshot times. The 62 repeated requests, 620 units, bought nothing new: the 2024-02-25T19:25:38Z snapshot, a provider gap, was requested 21 times, and each of the two missing calls below 22 times, before gaps were recorded and skipped. The log skips the 537 earliest snapshot times (2023-03-25T11:25:38Z to 2023-04-21T16:55:39Z) as already stored, so their units are not in it: the next section's named-book calls. The probe snapshots (2023-04-01T09:55:38Z and 2025-09-13T09:55:38Z, `h2h,totals` with named books) are outside the manifest's call counts.
- **Named-book calls in 2022-23.** The 537 earliest snapshot times, 536 calls and the 2023-04-01T09:55:38Z probe, were stored while the run requested named books instead of `regions=uk`; they fall in the 54 fixtures from 2023-03-25 to 2023-04-21 and carry 9 books, from 2 to 9 per snapshot. Their coverage, counted on snapshots at or before kick-off:

  | Book | Fixtures (of 54) | Snapshots (of 11,526) |
  |---|---|---|
  | `paddypower` | 100.0% | 100.0% |
  | `pinnacle` | 100.0% | 97.9% |
  | `betfair_exchange` | 100.0% | 96.3% |
  | `888sport` | 100.0% | 94.4% |
  | `skybet` | 100.0% | 91.3% |
  | `williamhill` | 100.0% | 88.9% |
  | `coral` | 100.0% | 88.6% |
  | `ladbrokes` | 100.0% | 85.9% |
  | `betvictor` | 88.9% | 81.1% |

  The `uk` region's books are on the 74 fixtures with snapshots after the early period (18,647 snapshots, 1 to 18 books per snapshot; some of the 74 also have early snapshots), all but `betvictor` (90.5% of them) on 100.0% of those fixtures. That is why the 2022-23 book table shows `betway`, `boylesports`, `casumo`, `leovegas`, `livescorebet`, `matchbook`, `mrgreen`, `unibet_uk`, `virginbet` and `betfair_sportsbook` on 67.3% of fixtures (74 of 110), and `pinnacle` on 49.1% (54 of 110): the named-book calls requested it, and the `uk` region does not return it.
- **Grid step.** The manifest's one grid entry starts at 2022-09-01T00:00:38Z with a 5-minute step. Every audited snapshot time is from 2023-03-25, so the audit sees no 10-minute step and no `:x5:00` times, which the EPL data before 2022-09 has (the stored EPL history starts at 2020-09-19T09:55:00Z) and the audit's seasons exclude. The audit's phase check reads `Manifest.grid_step_at` and `Manifest.floor_to_grid`, so a manifest with a 10-minute entry from 2020-09-01T00:05:00Z followed by the 5-minute entry accepts `:x5:00` times before 2022-09-01 and a 5-minute `:x5:38` time after it (checked with that two-entry grid; the as-is manifest flags a `:x5:00` time).
- **Exchange key.** The legacy `betfair` key (and `betfair_ex_uk`, `betfair_ex_eu`) is stored as `betfair_exchange` ([#862](https://github.com/Acusick1/odds/issues/862)): the EPL Odds API historical snapshots of events before 2023-08 hold `betfair_exchange` and `betfair_sportsbook` among the `betfair*` keys and no other, and `betfair_exchange` is on 100.0% of the fixtures of all four seasons (and 100.0% of the 54 named-book fixtures). Its first stored price is on 2023-03-28, so the snapshots of 2023-03-25 to 2023-03-27 carry none; it is on 97.6%, 93.8%, 92.2% and 96.0% of the snapshots at or before kick-off, by season.
- **Fixture horizon.** Some rounds are first stored 114 to 173 h before kick-off, later than the KO−7d band's 168 h start: 24, 22, 22 and 38 fixtures by season (fixture lists from a separate query). The API appears to list a round only once the previous one is played `(inferred)`. That leaves 0.5%, 0.1%, 0.1% and 0.6% of the KO−7d band's decisions without a snapshot (`not listed`), which no purchase can fill.
- **Missing calls.** The calls for 2024-01-05T19:40:38Z and 2024-01-06T01:40:38Z serve only Burnley v Luton (2024-01-12T19:45Z), first stored 114 h before kick-off, after both calls; the executor's skip judged them not stored, so every resume requested them again (22 requests each).
- **Postponed fixtures.** Eight fixtures were listed at their original dates, in responses that attached to no event, and played later; `event_postponements` has no row for them (separate query).

  | Fixture | Listed at | Responses | Played | First stored before KO |
  |---|---|---|---|---|
  | Bournemouth v Luton | 2023-12-16T15:00Z, 15:00:24Z | 240, 11 | 2024-03-13T19:30Z | 388 h |
  | Manchester City v Brentford | 2023-12-23T15:00Z | 3 | 2024-02-20T19:30Z | 246 h |
  | Arsenal v Chelsea | 2024-03-16T12:30Z | 93 | 2024-04-23T19:00Z | 241 h |
  | Crystal Palace v Newcastle United | 2024-03-16T15:00Z | 92 | 2024-04-24T19:00Z | 265 h |
  | Manchester United v Sheffield United | 2024-03-16T15:00Z | 92 | 2024-04-24T19:00Z | 262 h |
  | Wolverhampton Wanderers v Bournemouth | 2024-03-16T15:00Z | 93 | 2024-04-24T18:45Z | 264 h |
  | Everton v Liverpool (2023-24) | 2024-03-17T14:00Z | 93 | 2024-04-24T19:00Z | 266 h |
  | Brighton and Hove Albion v Manchester City | 2024-03-17T16:30Z | 93 | 2024-04-25T19:00Z | 289 h |
  | Everton v Liverpool (2024-25) | 2024-12-07T12:30Z | 181 | 2025-02-12T19:30Z | 403 h |

- **Swapped and shifted dates.** Everton v Arsenal (listed 2023-09-16T16:30Z, 168 responses, played 2023-09-17T15:30Z) and Newcastle United v Brentford (listed 2023-09-17T15:30Z, 167, played 2023-09-16T16:30Z) swapped dates. Burnley v Luton was listed for 2024-01-15T19:45Z in 193 responses and played 2024-01-12T19:45Z, Tottenham Hotspur v Nottingham Forest for 2024-04-08T19:00Z (played 2024-04-07T17:00Z), Manchester City v Leeds United for 2023-05-07T13:00Z (played 2023-05-06T14:00Z).
- **Provisional kick-offs.** The other unmatched listings of 2024-25 and 2025-26 are listed 1 to 2 days from the stored kick-off, in 1 to 239 responses, mostly 1 day before it at 12:00 to 14:00Z: Manchester United v Leicester City, Leicester City v Crystal Palace, West Ham United v Tottenham Hotspur, Aston Villa v Tottenham Hotspur (listed 2 days after), Crystal Palace v Wolverhampton Wanderers, Manchester City v Bournemouth, Bournemouth v Newcastle United, Crystal Palace v Newcastle United, Nottingham Forest v Aston Villa, Aston Villa v Sunderland, Manchester City v Arsenal, Nottingham Forest v Burnley, Bournemouth v Crystal Palace and Aston Villa v Liverpool (listed 2 days after). Three games listed 2024-12-10T12:00Z in one response (Crystal Palace v Arsenal, Ipswich Town v Everton, Liverpool v West Ham United) and Nottingham Forest v Brentford (listed 2025-04-26T14:00Z in 176 responses, stored for 2025-05-01) are not matched to a played date more closely by the data. All 380 fixtures of each season have a stored snapshot.
- **Provider gaps.** Five holes: 2024-02-25T19:25:38Z (returned 19:10:38Z), 2025-04-15T18:25:38Z (returned 18:00:38Z), 2025-10-20T06:55:38Z (returned 06:45:38Z), 2025-10-20T16:55:38Z (returned 16:31:34Z) and 2026-05-01T06:55:38Z (returned 06:05:37Z). West Ham United v Brentford (2024-02-26) lost 15 minutes of 60-minute-fresh input, Newcastle United v Crystal Palace (2025-04-16) 205, Leeds United v Burnley (2026-05-01) 50, and 10 fixtures 144 minutes of 6-hour-fresh input after 2025-10-20. (Fixture lists from a separate query.)
- **West Ham United v Brentford, 2025-10-20.** The Monday 19:00 game fell inside the outage: no snapshot from 16:31:34Z to kick-off, so 119 decisions of the KO−2h band are unserved, 130 more of the KO−1d band lost to the 06:55:38Z hole, and the fixture has no exchange price within 120 min of kick-off: 379 of 380 in 2025-26.
- **Off-phase snapshots.** Three calls returned a snapshot stamped off the phase and stopped the run until it was resumed: 2023-05-12T00:55:38Z returned 00:57:50Z (leaving 2 decisions of the KO−7d band on each of 11 fixtures unserved), 2024-02-24T19:15:38Z returned 19:17:17Z (leaving a decision minute of Arsenal v Newcastle United unserved) and 2025-02-03T15:55:38Z returned 15:56:33Z. The 2025-10-20T16:31:34Z snapshot is the outage's return.
- **Books outside `UK_RETAIL_BOOKMAKERS`.** `casumo`, `grosvenor`, `leovegas`, `livescorebet`, `virginbet` and `mrgreen` are stored under their own keys and counted neither as UK retail nor as exchanges. They are UK-licensed sportsbooks `(inferred)`, so retail rules reading `UK_RETAIL_BOOKMAKERS` ignore up to six of the region's books.
- **Uneven books.** In 2023-24 `grosvenor` is on 78.9% of fixtures and `betfair_sportsbook` on 20.3%; in 2024-25 `888sport` is on 97.4% of fixtures and `smarkets` on 73.9%.
- **Betfred.** `betfred` is stored on 62 of the 2025-26 fixtures, from 2026-04-13 to 2026-05-24 (dates from a separate query), and on none in the earlier seasons.
- **Pinnacle** appears only in the named-book calls of 2022-23 and in the two probe snapshots; the `uk` region does not return it.
- **Exchange quotes without a draw.** When the exchange quotes only the two teams, the response's `h2h` keeps its key and is stored as an `h2h` snapshot on the soccer fixture: 29 snapshots on 29 fixtures before kick-off in 2023-24, 14 on 14 in 2024-25 and 55 on 40 in 2025-26. Readers of `1x2` do not see these partial exchange books. Listings with no books are stored as snapshots with no market (1, 16, 31 and 13 before kick-off, by season).
- **Near kick-off exchange.** The latest exchange price is 4.3 to 4.4 min before kick-off for 104, 358, 365 and 363 fixtures by season, and 9.3 to 9.4 min for the rest, where the plan's closing calls are 10 minutes apart and none falls 4.4 min before kick-off `(inferred)`. The 2024-25 and 2025-26 prices found are complete and within the 5% overround limit. The run bought no snapshot nearer kick-off.

## 2022-23

### Fixtures, 2022-23

110 FINAL fixtures; 110 with a stored `1x2` snapshot at or before kick-off; 0 without.

Listings in stored responses that attached to no event: 1.

| Home | Away | Listed kick-off | Responses |
|---|---|---|---|
| Manchester City | Leeds United | 2023-05-07T13:00:00Z | 18 |

Odds API historical snapshots of these fixtures outside `1x2`:

| Market | Stamped | Snapshots | Fixtures |
|---|---|---|---|
| none | before KO | 1 | 1 |

### Bookmakers, 2022-23

Share of the 110 fixtures with the book in any snapshot at or before kick-off, and of the 30,173 such snapshots.

| Book | Fixtures | Snapshots | Role | Odds API keys aliased to it |
|---|---|---|---|---|
| `888sport` | 100.0% | 91.0% | UK retail | `sport888` |
| `betfair_exchange` | 100.0% | 97.6% | exchange | `betfair`, `betfair_ex_eu`, `betfair_ex_uk` |
| `coral` | 100.0% | 88.4% | UK retail | none |
| `ladbrokes` | 100.0% | 87.3% | UK retail | `ladbrokes_uk` |
| `paddypower` | 100.0% | 99.4% | UK retail | none |
| `skybet` | 100.0% | 92.6% | UK retail | none |
| `williamhill` | 100.0% | 91.6% | UK retail | none |
| `betvictor` | 90.0% | 84.3% | UK retail | none |
| `betfair_sportsbook` | 67.3% | 39.7% | UK retail | `betfair_sb_uk` |
| `betway` | 67.3% | 58.9% | UK retail | none |
| `boylesports` | 67.3% | 30.1% | UK retail | none |
| `casumo` | 67.3% | 47.5% | neither | none |
| `leovegas` | 67.3% | 47.5% | neither | none |
| `livescorebet` | 67.3% | 61.0% | neither | none |
| `matchbook` | 67.3% | 34.1% | exchange | none |
| `mrgreen` | 67.3% | 47.5% | neither | none |
| `unibet_uk` | 67.3% | 47.5% | UK retail | none |
| `virginbet` | 67.3% | 61.0% | neither | none |
| `pinnacle` | 49.1% | 37.4% | neither | none |

### Leads, 2022-23

Whole-minute decisions per fixture and band, served when a stored snapshot lies in `[t − max age, t]` at its actual time. Unserved decisions are split into those before the fixture's first stored snapshot (not yet listed), those a recorded gap's planned snapshot would have served, and the rest. The last column needs `betfair_exchange` and a UK retail book in the same snapshot.

| Band | Decisions | Served | Unserved: not listed | Unserved: gap | Unserved: other | Fixtures fully served | Served, exchange + retail |
|---|---|---|---|---|---|---|---|
| KO−1m..KO−120m, max age 10m | 13,200 | 100.0% | 0 | 0 | 0 | 110/110 | 100.0% |
| KO−121m..KO−1440m, max age 60m | 145,200 | 100.0% | 0 | 0 | 0 | 110/110 | 100.0% |
| KO−1441m..KO−10080m, max age 360m | 950,400 | 99.5% | 4,820 | 0 | 22 | 89/110 | 95.0% |

Fixtures first stored less than 174 h before kick-off: 24, first stored at leads (h: fixtures) 148: 1, 150: 2, 165: 7, 168: 1, 171: 5, 173: 8.

### Near kick-off `betfair_exchange`, 2022-23

Per fixture, the latest exchange snapshot at or before kick-off within 120 min, counted and aged only: this season is sealed ([ADR-0021](../docs/adr/0021-sealed-holdout-and-one-shot-gate.md)), so no price-derived statistic is reported. Research inputs, not validated closing labels.

| Price | Fixtures with one | Within 5 min | Within 10 min | p50 age (min) | p75 age (min) | p90 age (min) | p99 age (min) |
|---|---|---|---|---|---|---|---|
| Any exchange quote | 110/110 | 94.5% | 100.0% | 4.3 | 4.3 | 4.4 | 9.3 |

### Provider gaps and off-phase snapshots, 2022-23

Recorded gaps: 0.

Stored snapshot times off the grid's phase (outside their nearest grid time's whole minute): 1 (2023-05-12T00:57:50Z).

### Calls, 2022-23

Planned 1,566; stored 1,566; covered by a recorded gap 0; missing 0. Distinct stored snapshot times, any market: 1,566, × 10 units = 15,660 units. `1x2` snapshot times matching no planned call (gap returns, probes): 0. Planned calls that serve only fixtures not yet stored at that time (not listed): 0.

## 2023-24

### Fixtures, 2023-24

380 FINAL fixtures; 380 with a stored `1x2` snapshot at or before kick-off; 0 without.

Listings in stored responses that attached to no event: 13.

| Home | Away | Listed kick-off | Responses |
|---|---|---|---|
| Everton | Arsenal | 2023-09-16T16:30:00Z | 168 |
| Newcastle United | Brentford | 2023-09-17T15:30:00Z | 167 |
| Bournemouth | Luton | 2023-12-16T15:00:00Z | 240 |
| Bournemouth | Luton | 2023-12-16T15:00:24Z | 11 |
| Manchester City | Brentford | 2023-12-23T15:00:00Z | 3 |
| Burnley | Luton | 2024-01-15T19:45:00Z | 193 |
| Arsenal | Chelsea | 2024-03-16T12:30:00Z | 93 |
| Crystal Palace | Newcastle United | 2024-03-16T15:00:00Z | 92 |
| Manchester United | Sheffield United | 2024-03-16T15:00:00Z | 92 |
| Wolverhampton Wanderers | Bournemouth | 2024-03-16T15:00:00Z | 93 |
| Everton | Liverpool | 2024-03-17T14:00:00Z | 93 |
| Brighton and Hove Albion | Manchester City | 2024-03-17T16:30:00Z | 93 |
| Tottenham Hotspur | Nottingham Forest | 2024-04-08T19:00:00Z | 19 |

Odds API historical snapshots of these fixtures outside `1x2`:

| Market | Stamped | Snapshots | Fixtures |
|---|---|---|---|
| h2h | after KO | 4 | 2 |
| h2h | before KO | 29 | 29 |
| none | after KO | 3 | 2 |
| none | before KO | 16 | 16 |

### Bookmakers, 2023-24

Share of the 380 fixtures with the book in any snapshot at or before kick-off, and of the 107,646 such snapshots.

| Book | Fixtures | Snapshots | Role | Odds API keys aliased to it |
|---|---|---|---|---|
| `888sport` | 100.0% | 99.3% | UK retail | `sport888` |
| `betfair_exchange` | 100.0% | 93.8% | exchange | `betfair`, `betfair_ex_eu`, `betfair_ex_uk` |
| `betvictor` | 100.0% | 88.4% | UK retail | none |
| `betway` | 100.0% | 93.7% | UK retail | none |
| `boylesports` | 100.0% | 68.1% | UK retail | none |
| `casumo` | 100.0% | 81.3% | neither | none |
| `coral` | 100.0% | 83.9% | UK retail | none |
| `ladbrokes` | 100.0% | 84.1% | UK retail | `ladbrokes_uk` |
| `livescorebet` | 100.0% | 87.9% | neither | none |
| `matchbook` | 100.0% | 51.9% | exchange | none |
| `mrgreen` | 100.0% | 81.3% | neither | none |
| `paddypower` | 100.0% | 99.2% | UK retail | none |
| `skybet` | 100.0% | 88.2% | UK retail | none |
| `unibet_uk` | 100.0% | 81.3% | UK retail | none |
| `virginbet` | 100.0% | 87.9% | neither | none |
| `williamhill` | 100.0% | 88.3% | UK retail | none |
| `leovegas` | 99.2% | 78.2% | neither | none |
| `grosvenor` | 78.9% | 64.1% | neither | none |
| `betfair_sportsbook` | 20.3% | 13.6% | UK retail | `betfair_sb_uk` |

### Leads, 2023-24

Whole-minute decisions per fixture and band, served when a stored snapshot lies in `[t − max age, t]` at its actual time. Unserved decisions are split into those before the fixture's first stored snapshot (not yet listed), those a recorded gap's planned snapshot would have served, and the rest. The last column needs `betfair_exchange` and a UK retail book in the same snapshot.

| Band | Decisions | Served | Unserved: not listed | Unserved: gap | Unserved: other | Fixtures fully served | Served, exchange + retail |
|---|---|---|---|---|---|---|---|
| KO−1m..KO−120m, max age 10m | 45,600 | 100.0% | 0 | 0 | 2 | 379/380 | 100.0% |
| KO−121m..KO−1440m, max age 60m | 501,600 | 100.0% | 0 | 15 | 0 | 379/380 | 100.0% |
| KO−1441m..KO−10080m, max age 360m | 3,283,200 | 99.9% | 4,224 | 0 | 0 | 371/380 | 99.8% |

Fixtures first stored less than 174 h before kick-off: 22, first stored at leads (h: fixtures) 114: 1, 164: 1, 165: 1, 166: 3, 167: 2, 168: 6, 169: 4, 171: 1, 172: 1, 173: 2.

### Near kick-off `betfair_exchange`, 2023-24

Per fixture, the latest exchange snapshot at or before kick-off within 120 min, counted and aged only: this season is sealed ([ADR-0021](../docs/adr/0021-sealed-holdout-and-one-shot-gate.md)), so no price-derived statistic is reported. Research inputs, not validated closing labels.

| Price | Fixtures with one | Within 5 min | Within 10 min | p50 age (min) | p75 age (min) | p90 age (min) | p99 age (min) |
|---|---|---|---|---|---|---|---|
| Any exchange quote | 380/380 | 94.2% | 100.0% | 4.3 | 4.4 | 4.4 | 9.4 |

### Provider gaps and off-phase snapshots, 2023-24

Recorded gaps: 1.

| Planned | Returned | Next |
|---|---|---|
| 2024-02-25T19:25:38Z | 2024-02-25T19:10:38Z | 2024-02-25T20:10:39Z |

Stored snapshot times off the grid's phase (outside their nearest grid time's whole minute): 1 (2024-02-24T19:17:17Z).

### Calls, 2023-24

Planned 5,784; stored 5,781; covered by a recorded gap 1; missing 2. Distinct stored snapshot times, any market: 5,784, × 10 units = 57,840 units. `1x2` snapshot times matching no planned call (gap returns, probes): 3. Planned calls that serve only fixtures not yet stored at that time (not listed): 2.

Missing calls: 2024-01-05T19:40:38Z, 2024-01-06T01:40:38Z

## 2024-25

### Fixtures, 2024-25

380 FINAL fixtures; 380 with a stored `1x2` snapshot at or before kick-off; 0 without.

Listings in stored responses that attached to no event: 11.

| Home | Away | Listed kick-off | Responses |
|---|---|---|---|
| Manchester United | Leicester City | 2024-11-09T15:00:00Z | 12 |
| Everton | Liverpool | 2024-12-07T12:30:00Z | 181 |
| Crystal Palace | Arsenal | 2024-12-10T12:00:00Z | 1 |
| Ipswich Town | Everton | 2024-12-10T12:00:00Z | 1 |
| Liverpool | West Ham United | 2024-12-10T12:00:00Z | 1 |
| Leicester City | Crystal Palace | 2025-01-14T19:45:00Z | 61 |
| Nottingham Forest | Brentford | 2025-04-26T14:00:00Z | 176 |
| West Ham United | Tottenham Hotspur | 2025-05-03T14:00:00Z | 203 |
| Aston Villa | Tottenham Hotspur | 2025-05-18T13:15:00Z | 34 |
| Crystal Palace | Wolverhampton Wanderers | 2025-05-18T14:00:00Z | 140 |
| Manchester City | Bournemouth | 2025-05-18T18:00:00Z | 23 |

Odds API historical snapshots of these fixtures outside `1x2`:

| Market | Stamped | Snapshots | Fixtures |
|---|---|---|---|
| h2h | before KO | 14 | 14 |
| none | after KO | 23 | 23 |
| none | before KO | 31 | 24 |

### Bookmakers, 2024-25

Share of the 380 fixtures with the book in any snapshot at or before kick-off, and of the 103,405 such snapshots.

| Book | Fixtures | Snapshots | Role | Odds API keys aliased to it |
|---|---|---|---|---|
| `betfair_exchange` | 100.0% | 92.2% | exchange | `betfair`, `betfair_ex_eu`, `betfair_ex_uk` |
| `betvictor` | 100.0% | 77.5% | UK retail | none |
| `betway` | 100.0% | 95.7% | UK retail | none |
| `boylesports` | 100.0% | 80.6% | UK retail | none |
| `casumo` | 100.0% | 86.7% | neither | none |
| `coral` | 100.0% | 82.0% | UK retail | none |
| `grosvenor` | 100.0% | 86.7% | neither | none |
| `ladbrokes` | 100.0% | 82.0% | UK retail | `ladbrokes_uk` |
| `leovegas` | 100.0% | 86.7% | neither | none |
| `matchbook` | 100.0% | 52.9% | exchange | none |
| `paddypower` | 100.0% | 94.8% | UK retail | none |
| `skybet` | 100.0% | 92.9% | UK retail | none |
| `unibet_uk` | 100.0% | 86.7% | UK retail | none |
| `williamhill` | 100.0% | 88.9% | UK retail | none |
| `betfair_sportsbook` | 99.5% | 36.0% | UK retail | `betfair_sb_uk` |
| `888sport` | 97.4% | 88.1% | UK retail | `sport888` |
| `livescorebet` | 89.5% | 75.2% | neither | none |
| `virginbet` | 89.5% | 75.3% | neither | none |
| `smarkets` | 73.9% | 72.4% | exchange | none |
| `mrgreen` | 28.9% | 19.3% | neither | none |

### Leads, 2024-25

Whole-minute decisions per fixture and band, served when a stored snapshot lies in `[t − max age, t]` at its actual time. Unserved decisions are split into those before the fixture's first stored snapshot (not yet listed), those a recorded gap's planned snapshot would have served, and the rest. The last column needs `betfair_exchange` and a UK retail book in the same snapshot.

| Band | Decisions | Served | Unserved: not listed | Unserved: gap | Unserved: other | Fixtures fully served | Served, exchange + retail |
|---|---|---|---|---|---|---|---|
| KO−1m..KO−120m, max age 10m | 45,600 | 100.0% | 0 | 0 | 0 | 380/380 | 100.0% |
| KO−121m..KO−1440m, max age 60m | 501,600 | 100.0% | 0 | 25 | 181 | 378/380 | 100.0% |
| KO−1441m..KO−10080m, max age 360m | 3,283,200 | 99.9% | 4,606 | 0 | 0 | 359/380 | 99.8% |

Fixtures first stored less than 174 h before kick-off: 22, first stored at leads (h: fixtures) 162: 1, 163: 3, 164: 10, 165: 4, 167: 2, 168: 2.

### Near kick-off `betfair_exchange`, 2024-25

Per fixture, the latest exchange snapshot at or before kick-off within 120 min; usable means a back price on every outcome with back overround at most 5%. Research inputs, not validated closing labels.

| Price | Fixtures with one | Within 5 min | Within 10 min | p50 age (min) | p75 age (min) | p90 age (min) | p99 age (min) |
|---|---|---|---|---|---|---|---|
| Any exchange quote | 380/380 | 96.1% | 100.0% | 4.4 | 4.4 | 4.4 | 9.4 |
| Usable | 380/380 | 96.1% | 100.0% | 4.4 | 4.4 | 4.4 | 9.4 |

Latest exchange snapshot missing an outcome's back price: 0 fixtures.

### Provider gaps and off-phase snapshots, 2024-25

Recorded gaps: 1.

| Planned | Returned | Next |
|---|---|---|
| 2025-04-15T18:25:38Z | 2025-04-15T18:00:38Z | 2025-04-15T22:25:39Z |

Stored snapshot times off the grid's phase (outside their nearest grid time's whole minute): 1 (2025-02-03T15:56:33Z).

### Calls, 2024-25

Planned 5,425; stored 5,421; covered by a recorded gap 4; missing 0. Distinct stored snapshot times, any market: 5,422, × 10 units = 54,220 units. `1x2` snapshot times matching no planned call (gap returns, probes): 1. Planned calls that serve only fixtures not yet stored at that time (not listed): 0.

## 2025-26

### Fixtures, 2025-26

380 FINAL fixtures; 380 with a stored `1x2` snapshot at or before kick-off; 0 without.

Listings in stored responses that attached to no event: 8.

| Home | Away | Listed kick-off | Responses |
|---|---|---|---|
| Bournemouth | Newcastle United | 2025-09-20T14:00:00Z | 57 |
| Crystal Palace | Newcastle United | 2026-04-11T14:00:00Z | 239 |
| Nottingham Forest | Aston Villa | 2026-04-11T14:00:00Z | 236 |
| Aston Villa | Sunderland | 2026-04-18T14:00:00Z | 59 |
| Manchester City | Arsenal | 2026-04-18T14:00:00Z | 59 |
| Nottingham Forest | Burnley | 2026-04-18T14:00:00Z | 59 |
| Bournemouth | Crystal Palace | 2026-05-02T14:00:00Z | 198 |
| Aston Villa | Liverpool | 2026-05-17T11:30:00Z | 57 |

Odds API historical snapshots of these fixtures outside `1x2`:

| Market | Stamped | Snapshots | Fixtures |
|---|---|---|---|
| h2h | after KO | 13 | 10 |
| h2h | before KO | 55 | 40 |
| totals | before KO | 20 | 20 |
| none | after KO | 22 | 22 |
| none | before KO | 13 | 13 |

### Bookmakers, 2025-26

Share of the 380 fixtures with the book in any snapshot at or before kick-off, and of the 107,390 such snapshots.

| Book | Fixtures | Snapshots | Role | Odds API keys aliased to it |
|---|---|---|---|---|
| `888sport` | 100.0% | 96.1% | UK retail | `sport888` |
| `betfair_exchange` | 100.0% | 96.0% | exchange | `betfair`, `betfair_ex_eu`, `betfair_ex_uk` |
| `betvictor` | 100.0% | 57.8% | UK retail | none |
| `betway` | 100.0% | 94.8% | UK retail | none |
| `boylesports` | 100.0% | 86.4% | UK retail | none |
| `casumo` | 100.0% | 85.4% | neither | none |
| `coral` | 100.0% | 53.6% | UK retail | none |
| `grosvenor` | 100.0% | 85.4% | neither | none |
| `ladbrokes` | 100.0% | 53.4% | UK retail | `ladbrokes_uk` |
| `leovegas` | 100.0% | 85.3% | neither | none |
| `paddypower` | 100.0% | 99.0% | UK retail | none |
| `skybet` | 100.0% | 85.9% | UK retail | none |
| `smarkets` | 100.0% | 97.1% | exchange | none |
| `unibet_uk` | 100.0% | 91.9% | UK retail | none |
| `williamhill` | 100.0% | 96.0% | UK retail | none |
| `betfair_sportsbook` | 99.7% | 38.0% | UK retail | `betfair_sb_uk` |
| `matchbook` | 99.5% | 37.5% | exchange | none |
| `livescorebet` | 94.7% | 80.8% | neither | none |
| `virginbet` | 94.7% | 80.8% | neither | none |
| `betfred` | 16.3% | 14.6% | UK retail | `betfred_uk` |
| `pinnacle` | 5.3% | 0.0% | neither | none |

### Leads, 2025-26

Whole-minute decisions per fixture and band, served when a stored snapshot lies in `[t − max age, t]` at its actual time. Unserved decisions are split into those before the fixture's first stored snapshot (not yet listed), those a recorded gap's planned snapshot would have served, and the rest. The last column needs `betfair_exchange` and a UK retail book in the same snapshot.

| Band | Decisions | Served | Unserved: not listed | Unserved: gap | Unserved: other | Fixtures fully served | Served, exchange + retail |
|---|---|---|---|---|---|---|---|
| KO−1m..KO−120m, max age 10m | 45,600 | 99.7% | 0 | 6 | 114 | 379/380 | 99.7% |
| KO−121m..KO−1440m, max age 60m | 501,600 | 100.0% | 0 | 60 | 120 | 378/380 | 99.8% |
| KO−1441m..KO−10080m, max age 360m | 3,283,200 | 99.3% | 20,826 | 240 | 1,200 | 339/380 | 99.2% |

Fixtures first stored less than 174 h before kick-off: 38, first stored at leads (h: fixtures) 126: 1, 130: 1, 147: 1, 148: 1, 152: 3, 153: 6, 154: 2, 156: 1, 163: 3, 164: 3, 165: 2, 166: 5, 167: 2, 168: 6, 170: 1.

### Near kick-off `betfair_exchange`, 2025-26

Per fixture, the latest exchange snapshot at or before kick-off within 120 min; usable means a back price on every outcome with back overround at most 5%. Research inputs, not validated closing labels.

| Price | Fixtures with one | Within 5 min | Within 10 min | p50 age (min) | p75 age (min) | p90 age (min) | p99 age (min) |
|---|---|---|---|---|---|---|---|
| Any exchange quote | 379/380 | 95.5% | 99.7% | 4.4 | 4.4 | 4.4 | 9.4 |
| Usable | 379/380 | 95.5% | 99.7% | 4.4 | 4.4 | 4.4 | 9.4 |

Latest exchange snapshot missing an outcome's back price: 0 fixtures.

### Provider gaps and off-phase snapshots, 2025-26

Recorded gaps: 3.

| Planned | Returned | Next |
|---|---|---|
| 2025-10-20T06:55:38Z | 2025-10-20T06:45:38Z | 2025-10-20T09:29:52Z |
| 2025-10-20T16:55:38Z | 2025-10-20T16:31:34Z | 2025-10-20T19:32:57Z |
| 2026-05-01T06:55:38Z | 2026-05-01T06:05:37Z | 2026-05-01T07:20:38Z |

Stored snapshot times off the grid's phase (outside their nearest grid time's whole minute): 1 (2025-10-20T16:31:34Z).

### Calls, 2025-26

Planned 5,651; stored 5,634; covered by a recorded gap 17; missing 0. Distinct stored snapshot times, any market: 5,637, × 10 units = 56,370 units. `1x2` snapshot times matching no planned call (gap returns, probes): 3. Planned calls that serve only fixtures not yet stored at that time (not listed): 0.

