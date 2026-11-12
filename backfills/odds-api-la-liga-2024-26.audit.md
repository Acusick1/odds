# Audit: `odds-api-la-liga-2024-26.yaml`

2025-26 is sealed ([ADR-0021](../docs/adr/0021-sealed-holdout-and-one-shot-gate.md)): its section and this summary report counts, coverage and ages only, with no overround check, back-price check or other price-derived statistic.

Coverage of the La Liga Odds API historical run (`h2h` stored as `1x2`, `regions=uk`), read from the production database on 2026-11-12 after the run finished at 19:02Z. The tables below each season come from `uv run python scripts/audit_backfill.py --manifest backfills/odds-api-la-liga-2024-26.yaml` ([audit a finished manifest](../wiki/collection/howto-backfill-odds-api-history.md#audit-a-finished-manifest)). The near-kick-off Betfair Exchange figures are research inputs: whether they can stand in for the FDUK close is [#730](https://github.com/Acusick1/odds/issues/730)'s result, still pending.

## Summary

| | 2024-25 | 2025-26 |
|---|---|---|
| FINAL fixtures / with a stored snapshot | 380 / 380 | 380 / 380 |
| Unmatched listings | 49 (postponed fixtures, provisional kick-offs, two Supercopa games) | 48 (postponed fixtures, provisional kick-offs, two Supercopa games) |
| KO−2h band (10 min) served | 100.0% (1 minute short) | 99.7% (121 minutes short) |
| KO−1d band (60 min) served | 100.0% (1 minute short) | 99.9% (306 minutes short) |
| KO−7d band (6 h) served | 98.3% | 97.9% |
| Exchange price within 5 min of KO | 377 / 380 | 379 / 380 |
| Exchange price within 10 min of KO | 380 / 380 | 379 / 380 |
| Provider gaps recorded | 2 | 16 |
| Off-phase snapshots | 2 | 2 |
| Planned calls / stored / covered by a gap / missing | 7,469 / 7,463 / 3 / 3 | 7,527 / 7,509 / 18 / 0 |
| Distinct stored snapshot times × 10 units | 74,680 | 75,130 |

## Findings

- **Billing.** `odds backfill execute` printed 15,071 paid calls at 10 units each, 150,710 units, against 14,996 planned. 92 of them, 920 units, bought nothing new: the 2025-04-15T18:55:38Z snapshot, a provider gap, was requested 21 times (20 calls, 200 units); each of the three missing calls below was requested 21 times (60 calls, 600 units); and 13 calls inside the 2025-10-20 outage, from 16:55:38Z to 18:55:38Z, each returned the 16:31:34Z snapshot, of which the first stored it and 12 (120 units) bought nothing new. The run stopped on five consecutive provider gaps twice before gaps were recorded and skipped. The two probe snapshots (2024-09-14 and 2025-09-13, `h2h,totals` with named books) are outside these counts.
- **Missing calls.** The calls for 2025-03-20T19:55:38Z, 2025-03-21T01:55:38Z and 2025-03-21T07:55:38Z each stored snapshots for 20 fixtures but none for Barcelona v CA Osasuna (played 2025-03-27T20:00Z), the fixture they serve, which the API did not list at its new date until 2025-03-21T13:55:38Z, 144 h before kick-off (listed instead at its original 2025-03-08 and 2025-03-09 dates). The executor's skip judged them not stored, so every resume requested them again.
- **Fixture horizon.** Some rounds are first stored 22 to 171 h before kick-off, later than the KO−7d band's 168 h start: 50 fixtures in 2024-25 and 27 in 2025-26 (fixture lists from a separate query), among them Sevilla v Girona (22 h, listed at its original 2026-02-07 date) and Valencia v Oviedo (24 h). The API appears to list a round only once the previous one is played `(inferred)`. That leaves 1.7% (2024-25) and 1.9% (2025-26) of the KO−7d band's decisions without a snapshot, which no purchase can fill.
- **Band under 98%.** The 2025-26 KO−7d band is 97.9% served: 62,293 decisions not yet listed, 1,440 behind provider gaps and 3,986 other. 3,985 of the others are Sevilla v Rayo Vallecano (2026-03-08T17:30Z), which has no snapshot between 2026-02-27T20:55:37Z and 2026-03-04T11:55:37Z; the API listed the game at 2026-03-07T20:00Z in 186 responses.
- **Alavés v Valencia, 2025-10-20.** The Monday 19:00 game fell inside the outage: no snapshot from 16:31:34Z to kick-off, so all 119 decisions of the KO−2h band after 16:31:34Z are unserved, it has no exchange price within 120 min of kick-off (latest 148.4 min before it), and the earlier 06:55:38Z hole cost 130 decisions of the KO−1d band. That is the 379 of 380 and 99.7% in the 2025-26 exchange rows.
- **Provider gaps.** 18 recorded holes: 2025-04-15T18:55:38Z (returned 18:00:38Z) and 2025-05-18T22:55:38Z (returned 19:20:37Z) in 2024-25; in 2025-26 2025-10-20T06:55:38Z (returned 06:45:38Z), the 13 calls of the 2025-10-20 outage from 16:55:38Z to 18:55:38Z (all returned 16:31:34Z, next stored 19:32:57Z), 2026-05-01T06:55:38Z (returned 06:05:37Z) and 2026-05-18T01:10:38Z (returned 2026-05-17T21:25:37Z). They cost decisions: 12 fixtures lost 55 minutes of 6-hour-fresh input after 2025-04-15, 10 fixtures lost 144 minutes after the outage, Girona v Mallorca lost 50 minutes of 60-minute-fresh input on 2026-05-01, and Alavés v Valencia lost the decisions above. (Fixture lists from a separate query.)
- **Off-phase snapshots.** The calls for 2024-08-15T18:25:38Z, 2025-02-03T15:55:38Z and 2026-01-17T15:00:38Z returned snapshots stamped 18:26:25Z, 15:56:33Z and 15:01:03Z, each of which stopped the run until it was resumed; the 2025-10-20T16:31:34Z snapshot is the outage's return. They leave one unserved decision minute each: Real Betis v Girona (KO−2h band, 2024-08-15), Girona v Las Palmas (KO−1d band, 2025-02-03) and, probably, Mallorca v Athletic Bilbao (KO−2h band, 2026-01-17) (separate query). The log has 28 `snapshot_key_conflict` warnings, each keeping the stored snapshot: 5 at 2025-12-13T15:00:41Z, 15 at 2026-01-17T14:00:53Z, 5 at 2026-01-17T14:30:48Z and 3 at 2026-01-17T18:15:50Z.
- **Postponed fixtures.** Eight fixtures were listed at their original dates, in responses that attached to no event, and played later; `event_postponements` has no row for them (separate query for the dates).

  | Fixture | Listed at | Responses | Played | First stored before KO |
  |---|---|---|---|---|
  | Real Betis v Getafe | 2024-08-28T00:00Z, T16:00Z, T19:00Z | 248, 215, 16 | 2024-09-18T17:00Z | 286 h |
  | Villarreal v Rayo Vallecano | 2024-11-02T17:30Z | 258 | 2024-12-18T20:30Z | 238 h |
  | Valencia v Real Madrid | 2024-11-02T20:00Z | 258 | 2025-01-03T20:00Z | 450 h |
  | Espanyol v Valencia | 2024-11-09T17:30Z | 257 | 2024-12-18T20:30Z | 238 h |
  | Barcelona v CA Osasuna | 2025-03-08T20:00Z, 2025-03-09T16:00Z | 283, 9 | 2025-03-27T20:00Z | 150 h |
  | Villarreal v Espanyol | 2025-03-03T20:00Z | 427 | 2025-04-27T14:15Z | 241 h |
  | Levante v Villarreal | 2025-12-14T17:30Z | 459 | 2026-02-18T19:00Z | 250 h |
  | Rayo Vallecano v Oviedo | 2026-02-07T13:00Z | 354 | 2026-03-04T18:00Z | 237 h |

- **Supercopa listings.** Athletic Bilbao v Real Madrid and Mallorca v Barcelona (2025-01-11T16:00Z, 6 responses), and Athletic Bilbao v Real Madrid and Barcelona v Atlético Madrid (2026-01-11T18:00Z, 224 responses) are pairings of clubs whose La Liga games between them were played on other dates, so they are not La Liga fixtures; they fall in the Supercopa de España weeks `(inferred)`.
- **Provisional kick-offs.** The other unmatched listings are fixtures listed 0.1 to 2 days from the stored kick-off, in 1 to 427 responses: mostly a Sunday slot (16:00 or 17:00 UTC) for games then played on the Friday, Saturday or Monday (2024-09-15, 2024-12-01, 2025-03-09, 2025-09-14, 2025-09-21 and 2026-01-25 rounds); a few at a 00:00Z placeholder (Getafe v Alavés, Espanyol v Mallorca, Espanyol v Sevilla, Valladolid v Villarreal, 2024-09-29 to 2024-10-27); and the games of a few days shifted by hours or by a day, among them Barcelona v Real Sociedad (listed 2025-03-01T17:30Z, played 2025-03-02T15:15Z) and Real Betis v Real Madrid (listed 2025-03-02T15:15Z, played 2025-03-01T17:30Z), which swapped dates. Leganés v Athletic Bilbao and Mallorca v Real Sociedad were listed for 2024-09-25T19:00Z in 48 responses and played on 2024-09-19 and 2024-09-17. All 380 fixtures of each season have a stored snapshot.
- **Books outside `UK_RETAIL_BOOKMAKERS`.** `casumo`, `grosvenor`, `leovegas`, `livescorebet`, `virginbet` (both seasons) and `mrgreen` (2024-25) are stored under their own keys and counted neither as UK retail nor as exchanges. They are UK-licensed sportsbooks `(inferred)`, so retail rules reading `UK_RETAIL_BOOKMAKERS` ignore up to six of the region's books.
- **Uneven retail books.** In 2024-25 `coral`, `ladbrokes`, `matchbook` and `skybet` are on 90.0% of fixtures; in 2025-26 `betvictor`, `coral` and `ladbrokes` are in 53 to 54% of snapshots, and `unibet_uk` is on 86.8% of fixtures.
- **Betfred.** `betfred` is stored on 71 of the 2025-26 fixtures, from 2026-04-13 to 2026-05-24 (dates from a separate query), and on none in 2024-25.
- **Pinnacle** appears only in the two probe snapshots, which named it; the `uk` region does not return it.
- **Exchange quotes without a draw.** When the exchange quotes only the two teams, the response's `h2h` keeps its key and is stored as an `h2h` snapshot on the soccer fixture: 15 snapshots on 14 fixtures in 2024-25 and 583 on 43 fixtures before kick-off in 2025-26. Readers of `1x2` do not see these partial exchange books. Listings with no books are stored as snapshots with no market (42 before kick-off in 2024-25, 41 in 2025-26).
- **Near kick-off exchange.** The latest exchange price is 4.4 min old for 377 fixtures in 2024-25 and 379 in 2025-26, all of the 2024-25 prices complete and within the 5% overround limit; it is 9.4 min old for Alavés v Real Betis, Espanyol v Rayo Vallecano and Getafe v Real Sociedad (17:15 kick-offs, where the plan's closing calls are 10 minutes apart) and 148.4 min old for Alavés v Valencia above. The run bought no snapshot nearer kick-off.

## 2024-25

### Fixtures, 2024-25

380 FINAL fixtures; 380 with a stored `1x2` snapshot at or before kick-off; 0 without.

Listings in stored responses that attached to no event: 49.

| Home | Away | Listed kick-off | Responses |
|---|---|---|---|
| Real Betis | Getafe | 2024-08-28T00:00:00Z | 248 |
| Real Betis | Getafe | 2024-08-28T16:00:00Z | 215 |
| Real Betis | Getafe | 2024-08-28T19:00:00Z | 16 |
| Atlético Madrid | Valencia | 2024-09-15T16:00:00Z | 41 |
| Celta Vigo | Valladolid | 2024-09-15T16:00:00Z | 41 |
| Mallorca | Villarreal | 2024-09-15T16:00:00Z | 41 |
| Rayo Vallecano | CA Osasuna | 2024-09-15T16:00:00Z | 41 |
| Real Betis | Leganés | 2024-09-15T16:00:00Z | 41 |
| Real Sociedad | Real Madrid | 2024-09-15T16:00:00Z | 41 |
| Sevilla | Getafe | 2024-09-15T16:00:00Z | 41 |
| Espanyol | Alavés | 2024-09-15T19:00:00Z | 41 |
| Leganés | Athletic Bilbao | 2024-09-25T19:00:00Z | 48 |
| Mallorca | Real Sociedad | 2024-09-25T19:00:00Z | 48 |
| Getafe | Alavés | 2024-09-29T00:00:00Z | 2 |
| Espanyol | Mallorca | 2024-10-06T00:00:00Z | 88 |
| Athletic Bilbao | Espanyol | 2024-10-20T14:15:00Z | 196 |
| Espanyol | Sevilla | 2024-10-27T00:00:00Z | 209 |
| Valladolid | Villarreal | 2024-10-27T00:00:00Z | 3 |
| Villarreal | Rayo Vallecano | 2024-11-02T17:30:00Z | 258 |
| Valencia | Real Madrid | 2024-11-02T20:00:00Z | 258 |
| Espanyol | Valencia | 2024-11-09T17:30:00Z | 257 |
| Getafe | Girona | 2024-11-09T20:00:00Z | 82 |
| Leganés | Sevilla | 2024-11-10T17:30:00Z | 81 |
| Alavés | Leganés | 2024-12-01T17:00:00Z | 23 |
| Barcelona | Las Palmas | 2024-12-01T17:00:00Z | 23 |
| Espanyol | Celta Vigo | 2024-12-01T17:00:00Z | 23 |
| Mallorca | Valencia | 2024-12-01T17:00:00Z | 23 |
| Real Sociedad | Real Betis | 2024-12-01T17:00:00Z | 23 |
| Valladolid | Atlético Madrid | 2024-12-01T17:00:00Z | 23 |
| Villarreal | Girona | 2024-12-01T17:00:00Z | 23 |
| Athletic Bilbao | Real Madrid | 2025-01-11T16:00:00Z | 6 |
| Mallorca | Barcelona | 2025-01-11T16:00:00Z | 6 |
| CA Osasuna | Rayo Vallecano | 2025-01-18T13:00:00Z | 96 |
| Celta Vigo | Athletic Bilbao | 2025-01-18T17:30:00Z | 81 |
| Real Betis | Alavés | 2025-01-19T13:00:00Z | 81 |
| Valencia | Real Sociedad | 2025-01-19T17:30:00Z | 132 |
| Girona | Sevilla | 2025-01-19T20:00:00Z | 96 |
| Barcelona | Real Sociedad | 2025-03-01T17:30:00Z | 228 |
| Real Betis | Real Madrid | 2025-03-02T15:15:00Z | 228 |
| Villarreal | Espanyol | 2025-03-03T20:00:00Z | 427 |
| Barcelona | CA Osasuna | 2025-03-08T20:00:00Z | 283 |
| Alavés | Villarreal | 2025-03-09T16:00:00Z | 9 |
| Barcelona | CA Osasuna | 2025-03-09T16:00:00Z | 9 |
| Celta Vigo | Leganés | 2025-03-09T16:00:00Z | 9 |
| Espanyol | Girona | 2025-03-09T16:00:00Z | 9 |
| Getafe | Atlético Madrid | 2025-03-09T16:00:00Z | 9 |
| Real Sociedad | Sevilla | 2025-03-09T16:00:00Z | 9 |
| Valencia | Valladolid | 2025-03-09T16:00:00Z | 9 |
| Athletic Bilbao | CA Osasuna | 2025-03-28T20:00:00Z | 200 |

Odds API historical snapshots of these fixtures outside `1x2`:

| Market | Stamped | Snapshots | Fixtures |
|---|---|---|---|
| h2h | before KO | 15 | 14 |
| totals | before KO | 12 | 12 |
| none | after KO | 15 | 15 |
| none | before KO | 42 | 36 |

### Bookmakers, 2024-25

Share of the 380 fixtures with the book in any snapshot at or before kick-off, and of the 124,870 such snapshots.

| Book | Fixtures | Snapshots | Role | Odds API keys aliased to it |
|---|---|---|---|---|
| `betfair_exchange` | 100.0% | 82.7% | exchange | `betfair`, `betfair_ex_eu`, `betfair_ex_uk` |
| `betfair_sportsbook` | 100.0% | 38.9% | UK retail | `betfair_sb_uk` |
| `betvictor` | 100.0% | 71.2% | UK retail | none |
| `betway` | 100.0% | 97.0% | UK retail | none |
| `boylesports` | 100.0% | 82.3% | UK retail | none |
| `casumo` | 100.0% | 88.7% | neither | none |
| `grosvenor` | 100.0% | 88.7% | neither | none |
| `leovegas` | 100.0% | 88.8% | neither | none |
| `paddypower` | 100.0% | 99.1% | UK retail | none |
| `unibet_uk` | 100.0% | 88.7% | UK retail | none |
| `williamhill` | 100.0% | 95.8% | UK retail | none |
| `coral` | 90.0% | 79.4% | UK retail | none |
| `ladbrokes` | 90.0% | 79.4% | UK retail | `ladbrokes_uk` |
| `matchbook` | 90.0% | 45.9% | exchange | none |
| `skybet` | 90.0% | 87.7% | UK retail | none |
| `livescorebet` | 86.8% | 76.0% | neither | none |
| `virginbet` | 86.8% | 75.9% | neither | none |
| `smarkets` | 68.9% | 68.7% | exchange | none |
| `888sport` | 46.3% | 44.2% | UK retail | `sport888` |
| `mrgreen` | 33.2% | 22.5% | neither | none |
| `pinnacle` | 3.2% | 0.0% | neither | none |

### Leads, 2024-25

Whole-minute decisions per fixture and band, served when a stored snapshot lies in `[t − max age, t]` at its actual time. Unserved decisions are split into those before the fixture's first stored snapshot (not yet listed), those a recorded gap's planned snapshot would have served, and the rest. The last column needs `betfair_exchange` and a UK retail book in the same snapshot.

| Band | Decisions | Served | Unserved: not listed | Unserved: gap | Unserved: other | Fixtures fully served | Served, exchange + retail |
|---|---|---|---|---|---|---|---|
| KO−1m..KO−120m, max age 10m | 45,600 | 100.0% | 0 | 0 | 1 | 379/380 | 100.0% |
| KO−121m..KO−1440m, max age 60m | 501,600 | 100.0% | 0 | 0 | 1 | 379/380 | 100.0% |
| KO−1441m..KO−10080m, max age 360m | 3,283,200 | 98.3% | 55,762 | 660 | 0 | 321/380 | 96.1% |

Fixtures first stored less than 174 h before kick-off: 50, first stored at leads (h: fixtures) 104: 1, 119: 1, 123: 1, 124: 1, 126: 2, 128: 3, 136: 1, 138: 1, 139: 1, 140: 1, 141: 3, 142: 1, 143: 2, 145: 2, 147: 1, 150: 1, 152: 1, 158: 1, 159: 1, 160: 11, 161: 1, 163: 3, 164: 1, 165: 3, 166: 1, 167: 1, 168: 2, 171: 1.

### Near kick-off `betfair_exchange`, 2024-25

Per fixture, the latest exchange snapshot at or before kick-off within 120 min; usable means a back price on every outcome with back overround at most 5%. Research inputs, not validated closing labels.

| Price | Fixtures with one | Within 5 min | Within 10 min | p50 age (min) | p75 age (min) | p90 age (min) | p99 age (min) |
|---|---|---|---|---|---|---|---|
| Any exchange quote | 380/380 | 99.2% | 100.0% | 4.4 | 4.4 | 4.4 | 4.4 |
| Usable | 380/380 | 99.2% | 100.0% | 4.4 | 4.4 | 4.4 | 4.4 |

Latest exchange snapshot missing an outcome's back price: 0 fixtures.

### Provider gaps and off-phase snapshots, 2024-25

Recorded gaps: 2.

| Planned | Returned | Next |
|---|---|---|
| 2025-04-15T18:55:38Z | 2025-04-15T18:00:38Z | 2025-04-15T22:25:39Z |
| 2025-05-18T22:55:38Z | 2025-05-18T19:20:37Z | 2025-05-19T06:20:37Z |

Stored snapshot times off the grid's phase (outside their nearest grid time's whole minute): 2 (2024-08-15T18:26:25Z, 2025-02-03T15:56:33Z).

### Calls, 2024-25

Planned 7,469; stored 7,463; covered by a recorded gap 3; missing 3. Distinct stored snapshot times, any market: 7,468, × 10 units = 74,680 units. `1x2` snapshot times matching no planned call (gap returns, probes): 5. Planned calls that serve only fixtures not yet stored at that time (not listed): 5.

Missing calls: 2025-03-20T19:55:38Z, 2025-03-21T01:55:38Z, 2025-03-21T07:55:38Z

## 2025-26

### Fixtures, 2025-26

380 FINAL fixtures; 380 with a stored `1x2` snapshot at or before kick-off; 0 without.

Listings in stored responses that attached to no event: 48.

| Home | Away | Listed kick-off | Responses |
|---|---|---|---|
| Alavés | Levante | 2025-08-16T15:00:00Z | 19 |
| Levante | Real Betis | 2025-09-13T14:15:00Z | 276 |
| Athletic Bilbao | Alavés | 2025-09-14T17:00:00Z | 276 |
| Atlético Madrid | Villarreal | 2025-09-14T17:00:00Z | 276 |
| Real Sociedad | Real Madrid | 2025-09-14T17:00:00Z | 276 |
| Alavés | Sevilla | 2025-09-21T17:00:00Z | 43 |
| Girona | Levante | 2025-09-21T17:00:00Z | 43 |
| Mallorca | Atlético Madrid | 2025-09-21T17:00:00Z | 43 |
| Rayo Vallecano | Celta Vigo | 2025-09-21T17:00:00Z | 43 |
| Real Betis | Real Sociedad | 2025-09-21T17:00:00Z | 43 |
| Real Madrid | Espanyol | 2025-09-21T17:00:00Z | 43 |
| Valencia | Athletic Bilbao | 2025-09-21T17:00:00Z | 43 |
| Villarreal | CA Osasuna | 2025-09-21T17:00:00Z | 43 |
| Alavés | Sevilla | 2025-09-21T19:00:00Z | 12 |
| Girona | Levante | 2025-09-21T19:00:00Z | 12 |
| Real Betis | Real Sociedad | 2025-09-21T19:00:00Z | 12 |
| Real Madrid | Espanyol | 2025-09-21T19:00:00Z | 12 |
| Villarreal | CA Osasuna | 2025-09-21T19:00:00Z | 12 |
| Celta Vigo | Real Betis | 2025-09-24T17:00:00Z | 46 |
| Valencia | Oviedo | 2025-09-29T19:00:00Z | 309 |
| Levante | Villarreal | 2025-12-14T17:30:00Z | 459 |
| Getafe | Real Sociedad | 2026-01-11T17:00:00Z | 39 |
| Athletic Bilbao | Real Madrid | 2026-01-11T18:00:00Z | 224 |
| Barcelona | Atlético Madrid | 2026-01-11T18:00:00Z | 224 |
| Alavés | Real Betis | 2026-01-25T17:00:00Z | 51 |
| Atlético Madrid | Mallorca | 2026-01-25T17:00:00Z | 50 |
| Girona | Getafe | 2026-01-25T17:00:00Z | 51 |
| Levante | Elche CF | 2026-01-25T17:00:00Z | 51 |
| Rayo Vallecano | CA Osasuna | 2026-01-25T17:00:00Z | 51 |
| Sevilla | Athletic Bilbao | 2026-01-25T17:00:00Z | 50 |
| Valencia | Espanyol | 2026-01-25T17:00:00Z | 40 |
| Villarreal | Real Madrid | 2026-01-25T17:00:00Z | 51 |
| Rayo Vallecano | Oviedo | 2026-02-07T13:00:00Z | 354 |
| Sevilla | Girona | 2026-02-07T17:30:00Z | 369 |
| Rayo Vallecano | Atlético Madrid | 2026-02-14T15:15:00Z | 93 |
| Getafe | Villarreal | 2026-02-15T15:15:00Z | 92 |
| Real Betis | Rayo Vallecano | 2026-02-22T17:00:00Z | 242 |
| CA Osasuna | Mallorca | 2026-03-06T20:00:00Z | 304 |
| Getafe | Real Betis | 2026-03-07T13:00:00Z | 309 |
| Celta Vigo | Real Madrid | 2026-03-07T20:00:00Z | 118 |
| Sevilla | Rayo Vallecano | 2026-03-07T20:00:00Z | 186 |
| Athletic Bilbao | Barcelona | 2026-03-08T15:15:00Z | 118 |
| Alavés | CA Osasuna | 2026-04-04T16:30:00Z | 119 |
| Real Betis | Espanyol | 2026-04-05T17:00:00Z | 7 |
| Real Betis | Espanyol | 2026-04-05T19:00:00Z | 83 |
| Barcelona | Celta Vigo | 2026-04-22T17:00:00Z | 11 |
| Real Betis | Real Madrid | 2026-04-26T17:00:00Z | 44 |
| Barcelona | Real Betis | 2026-05-17T17:00:00Z | 307 |

Odds API historical snapshots of these fixtures outside `1x2`:

| Market | Stamped | Snapshots | Fixtures |
|---|---|---|---|
| h2h | after KO | 9 | 8 |
| h2h | before KO | 583 | 43 |
| totals | before KO | 19 | 19 |
| none | after KO | 8 | 8 |
| none | before KO | 41 | 31 |

### Bookmakers, 2025-26

Share of the 380 fixtures with the book in any snapshot at or before kick-off, and of the 133,619 such snapshots.

| Book | Fixtures | Snapshots | Role | Odds API keys aliased to it |
|---|---|---|---|---|
| `betfair_exchange` | 100.0% | 79.3% | exchange | `betfair`, `betfair_ex_eu`, `betfair_ex_uk` |
| `betway` | 100.0% | 95.8% | UK retail | none |
| `boylesports` | 100.0% | 91.1% | UK retail | none |
| `casumo` | 100.0% | 86.9% | neither | none |
| `grosvenor` | 100.0% | 86.9% | neither | none |
| `leovegas` | 100.0% | 86.9% | neither | none |
| `paddypower` | 100.0% | 92.1% | UK retail | none |
| `smarkets` | 100.0% | 97.8% | exchange | none |
| `williamhill` | 100.0% | 92.0% | UK retail | none |
| `betvictor` | 99.7% | 53.8% | UK retail | none |
| `coral` | 99.7% | 52.7% | UK retail | none |
| `ladbrokes` | 99.7% | 52.7% | UK retail | `ladbrokes_uk` |
| `matchbook` | 99.7% | 49.4% | exchange | none |
| `skybet` | 99.5% | 85.2% | UK retail | none |
| `betfair_sportsbook` | 99.2% | 37.5% | UK retail | `betfair_sb_uk` |
| `livescorebet` | 94.7% | 80.9% | neither | none |
| `virginbet` | 94.7% | 81.5% | neither | none |
| `unibet_uk` | 86.8% | 78.7% | UK retail | none |
| `betfred` | 18.7% | 12.3% | UK retail | `betfred_uk` |
| `pinnacle` | 5.0% | 0.0% | neither | none |

### Leads, 2025-26

Whole-minute decisions per fixture and band, served when a stored snapshot lies in `[t − max age, t]` at its actual time. Unserved decisions are split into those before the fixture's first stored snapshot (not yet listed), those a recorded gap's planned snapshot would have served, and the rest. The last column needs `betfair_exchange` and a UK retail book in the same snapshot.

| Band | Decisions | Served | Unserved: not listed | Unserved: gap | Unserved: other | Fixtures fully served | Served, exchange + retail |
|---|---|---|---|---|---|---|---|
| KO−1m..KO−120m, max age 10m | 45,600 | 99.7% | 0 | 120 | 1 | 378/380 | 99.7% |
| KO−121m..KO−1440m, max age 60m | 501,600 | 99.9% | 116 | 60 | 130 | 377/380 | 99.9% |
| KO−1441m..KO−10080m, max age 360m | 3,283,200 | 97.9% | 62,293 | 1,440 | 3,986 | 345/380 | 96.1% |

Fixtures first stored less than 174 h before kick-off: 27, first stored at leads (h: fixtures) 22: 1, 24: 1, 73: 1, 81: 1, 103: 1, 114: 1, 132: 9, 141: 1, 143: 1, 145: 1, 147: 1, 156: 1, 160: 1, 164: 1, 166: 1, 167: 1, 168: 1, 171: 2.

### Near kick-off `betfair_exchange`, 2025-26

Per fixture, the latest exchange snapshot at or before kick-off within 120 min, counted and aged only: this season is sealed ([ADR-0021](../docs/adr/0021-sealed-holdout-and-one-shot-gate.md)), so no price-derived statistic is reported. Research inputs, not validated closing labels.

| Price | Fixtures with one | Within 5 min | Within 10 min | p50 age (min) | p75 age (min) | p90 age (min) | p99 age (min) |
|---|---|---|---|---|---|---|---|
| Any exchange quote | 379/380 | 99.7% | 99.7% | 4.4 | 4.4 | 4.4 | 4.4 |

### Provider gaps and off-phase snapshots, 2025-26

Recorded gaps: 16.

| Planned | Returned | Next |
|---|---|---|
| 2025-10-20T06:55:38Z | 2025-10-20T06:45:38Z | 2025-10-20T09:29:52Z |
| 2025-10-20T16:55:38Z | 2025-10-20T16:31:34Z | 2025-10-20T19:32:57Z |
| 2025-10-20T17:05:38Z | 2025-10-20T16:31:34Z | 2025-10-20T19:32:57Z |
| 2025-10-20T17:15:38Z | 2025-10-20T16:31:34Z | 2025-10-20T19:32:57Z |
| 2025-10-20T17:25:38Z | 2025-10-20T16:31:34Z | 2025-10-20T19:32:57Z |
| 2025-10-20T17:35:38Z | 2025-10-20T16:31:34Z | 2025-10-20T19:32:57Z |
| 2025-10-20T17:45:38Z | 2025-10-20T16:31:34Z | 2025-10-20T19:32:57Z |
| 2025-10-20T17:55:38Z | 2025-10-20T16:31:34Z | 2025-10-20T19:32:57Z |
| 2025-10-20T18:05:38Z | 2025-10-20T16:31:34Z | 2025-10-20T19:32:57Z |
| 2025-10-20T18:15:38Z | 2025-10-20T16:31:34Z | 2025-10-20T19:32:57Z |
| 2025-10-20T18:25:38Z | 2025-10-20T16:31:34Z | 2025-10-20T19:32:57Z |
| 2025-10-20T18:35:38Z | 2025-10-20T16:31:34Z | 2025-10-20T19:32:57Z |
| 2025-10-20T18:45:38Z | 2025-10-20T16:31:34Z | 2025-10-20T19:32:57Z |
| 2025-10-20T18:55:38Z | 2025-10-20T16:31:34Z | 2025-10-20T19:32:57Z |
| 2026-05-01T06:55:38Z | 2026-05-01T06:05:37Z | 2026-05-01T07:20:38Z |
| 2026-05-18T01:10:38Z | 2026-05-17T21:25:37Z | 2026-05-18T06:45:37Z |

Stored snapshot times off the grid's phase (outside their nearest grid time's whole minute): 2 (2025-10-20T16:31:34Z, 2026-01-17T15:01:03Z).

### Calls, 2025-26

Planned 7,527; stored 7,509; covered by a recorded gap 18; missing 0. Distinct stored snapshot times, any market: 7,513, × 10 units = 75,130 units. `1x2` snapshot times matching no planned call (gap returns, probes): 3. Planned calls that serve only fixtures not yet stored at that time (not listed): 1.

