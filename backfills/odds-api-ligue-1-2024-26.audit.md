# Audit: `odds-api-ligue-1-2024-26.yaml`

2025-26 is sealed ([ADR-0021](../docs/adr/0021-sealed-holdout-and-one-shot-gate.md)): its section and this summary report counts, coverage and ages only, with no overround check, back-price check or other price-derived statistic.

Coverage of the Ligue 1 Odds API historical run (`h2h` stored as `1x2`, `regions=uk`), read from the production database on 2026-11-12 after the run finished at 18:33Z. The tables below each season come from `uv run python scripts/audit_backfill.py --manifest backfills/odds-api-ligue-1-2024-26.yaml` ([audit a finished manifest](../wiki/collection/howto-backfill-odds-api-history.md#audit-a-finished-manifest)). The near-kick-off Betfair Exchange figures are research inputs: whether they can stand in for the FDUK close is [#730](https://github.com/Acusick1/odds/issues/730)'s result, still pending.

## Summary

| | 2024-25 | 2025-26 |
|---|---|---|
| FINAL fixtures / with a stored snapshot | 306 / 306 | 306 / 306 |
| Unmatched listings | 34 (provisional kick-offs, one postponed fixture) | 30 (provisional kick-offs, five postponed fixtures) |
| KO−2h band (10 min) served | 100.0% | 99.7% (121 minutes short) |
| KO−1d band (60 min) served | 100.0% (15 minutes short) | 100.0% (64 minutes short) |
| KO−7d band (6 h) served | 99.7% | 98.9% |
| Exchange price within 5 min of KO | 287 / 306 | 267 / 306 |
| Exchange price within 10 min of KO | 306 / 306 | 305 / 306 |
| Provider gaps recorded | 2 | 2 |
| Off-phase snapshots | 0 | 2 |
| Planned calls / stored / covered by a gap / missing | 5,410 / 5,408 / 2 / 0 | 5,275 / 5,273 / 2 / 0 |
| Distinct stored snapshot times × 10 units | 54,110 | 52,760 |

## Findings

- **Billing.** `odds backfill execute` printed 10,706 paid calls at 10 units each, 107,060 units, against 10,685 planned. The 2024-11-30T02:40:38Z snapshot, a provider gap, was requested 22 times before gaps were recorded and skipped, so 21 calls (210 units) bought nothing new. The two probe snapshots (2024-09-14 and 2025-09-13, `h2h,totals` with named books) are outside these counts.
- **Fixture horizon.** Some rounds are first stored 118 to 173 h before kick-off, later than the KO−7d band's 168 h start: 10 fixtures in 2024-25 and 29 in 2025-26 (fixture lists from a separate query), mostly the first three rounds of each season and the round after the November international break. The API appears to list a round only once the previous one is played `(inferred)`. That leaves 0.2% (2024-25) and 1.1% (2025-26) of the KO−7d band's decisions without a snapshot, which no purchase can fill.
- **Postponed fixtures.** Five fixtures were listed at their original dates, in responses that attached to no event, and played later; `event_postponements` has no row for any of them (separate query). Their replayed fixtures are first stored 31 to 388 h before kick-off.

  | Fixture | Listed at | Responses | Played | First stored before KO |
  |---|---|---|---|---|
  | Nantes v Paris Saint Germain | 2025-04-13T15:00Z | 161 | 2025-04-22T18:45Z | 225 h |
  | Marseille v Paris Saint Germain | 2025-09-21T15:00Z, 2025-09-21T18:45Z, 2025-09-22T18:00Z | 41, 302, 9 | 2025-09-22T21:00Z | 31 h |
  | Paris Saint Germain v Nantes | 2026-03-15T16:00Z | 78 | 2026-04-22T17:00Z | 388 h |
  | RC Lens v Paris Saint Germain | 2026-04-11T15:00Z, 2026-05-12T17:00Z | 180, 2 | 2026-05-13T19:00Z | 222 h |
  | Brest v Strasbourg | 2026-04-12T15:15Z | 231 | 2026-05-13T17:00Z | 94 h |

- **Marseille v Paris Saint Germain.** The last stored snapshot is 244 min before its 21:00 kick-off (16:55:39Z on 2025-09-22). The 9 responses from 17:55Z to 20:05Z listed the game at 18:00, three hours before the stored kick-off, under the stored event's own Odds API id, and stored nothing for it; the storer appears to drop a game whose listed kick-off differs from the event's `(inferred)`. This leaves the 2025-26 KO−2h band's 121 unserved decisions (120 of them this fixture), 64 of the KO−1d band's, and no exchange price within 120 min of kick-off for this fixture: 305 of 306.
- **Provisional kick-offs.** The other 56 unmatched listings are 54 fixtures listed in 1 to 162 responses at a kick-off 0.1 to 2 days from the stored one. Most sit in the Sunday 15:00 or 16:00 UTC slot of their round (2024-08-18, 08-25, 09-15, 2025-01-12, 02-02, 02-09, 02-16, 02-23, 03-09, 09-14, 09-21, 2026-02-15 and 02-22) and were played on the Friday, Saturday or Sunday evening. A few moved by hours: Le Havre v Nantes (listed 2025-11-08T20:05Z, played 18:00Z), Rennes v Brest (2025-12-14T15:45Z, played 2025-12-13T16:00Z), Nantes v Marseille (2026-05-01T18:45Z, played 2026-05-02T13:00Z), Lyon v Angers (listed 2025-09-21T13:00Z and T15:00Z, played 2025-09-19T18:45Z) and Nantes v Brest (listed 2025-02-08T18:00Z and 2025-02-09T16:00Z, played 2025-02-07T18:00Z). All 306 fixtures of each season have a stored snapshot.
- **Provider gaps.** Four holes: 2024-11-30T02:40:38Z (returned 02:35:38Z), 2025-04-15T18:40:38Z (returned 18:00:38Z), 2025-10-20T18:40:38Z (returned 16:31:34Z) and 2026-05-01T06:40:38Z (returned 06:05:37Z). Each left the fixtures it served without a fresh snapshot until the next stored one: 3 fixtures lost 5 minutes of 60-minute-fresh input, and 10, 9 and 9 fixtures lost 40, 129 and 35 minutes of 6-hour-fresh input (fixture lists from a separate query).
- **Off-phase snapshots.** The call for 2025-10-20T18:40:38Z returned the 16:31:34Z snapshot, a provider gap. The call for 2026-04-03T18:00:38Z returned a snapshot stamped 18:01:01Z, which stopped the run until it was resumed and left one unserved KO−2h decision minute, of Paris Saint Germain v Toulouse (separate query), without a 10-minute-fresh input.
- **Key conflicts.** The log has 11 `snapshot_key_conflict` warnings, each keeping the stored snapshot: 2 at 2024-09-13T17:30:39Z, 1 at 2026-01-17T17:35:42Z and 8 at the off-phase 2026-04-03T18:01:01Z.
- **Books outside `UK_RETAIL_BOOKMAKERS`.** `casumo`, `grosvenor`, `leovegas`, `livescorebet`, `virginbet` (both seasons) and `mrgreen` (2024-25) are stored under their own keys and counted neither as UK retail nor as exchanges. They are UK-licensed sportsbooks `(inferred)`, so retail rules reading `UK_RETAIL_BOOKMAKERS` ignore up to six of the region's books.
- **Uneven retail books.** `betvictor` is in 23% and 5% of the snapshots of August and September 2024 (86.6% of 2024-25 fixtures) and in 37% and 24% of those of December 2025 and January 2026. `coral` is in 85 to 99% of each 2024-25 month's snapshots and in 44 to 65% of each month's from September 2025; `ladbrokes` has the same overall share (monthly shares from a separate query).
- **Betfred.** `betfred` is stored on 48 of the 2025-26 fixtures, from 2026-04-13 to 2026-05-17 (dates from a separate query), and on none in 2024-25.
- **Pinnacle** appears only in the two probe snapshots, which named it; the `uk` region does not return it.
- **Exchange quotes without a draw.** When the exchange quotes only the two teams, the response's `h2h` keeps its key and is stored as an `h2h` snapshot on the soccer fixture: 2 snapshots on 2 fixtures in 2024-25 and 929 on 51 fixtures before kick-off in 2025-26. Readers of `1x2` do not see these partial exchange books. Listings with no books are stored as snapshots with no market (8 before kick-off in 2024-25, 25 in 2025-26).
- **Near kick-off exchange.** The latest exchange price is 4.4 min before kick-off for 287 fixtures in 2024-25 and 267 in 2025-26, and 9.4 min for the other 19 and 38, all at the 19:05 and 20:05 kick-off slots, where the plan's closing calls are 10 minutes apart and none falls 4.4 min before kick-off. The 39th 2025-26 miss is Marseille v Paris Saint Germain above. Every 2024-25 price found is complete and within the 5% overround limit. The run bought no snapshot nearer kick-off.

## 2024-25

### Fixtures, 2024-25

306 FINAL fixtures; 306 with a stored `1x2` snapshot at or before kick-off; 0 without.

Listings in stored responses that attached to no event: 34.

| Home | Away | Listed kick-off | Responses |
|---|---|---|---|
| Le Havre | Paris Saint Germain | 2024-08-18T15:00:00Z | 5 |
| Lille | Angers | 2024-08-25T15:00:00Z | 1 |
| Lyon | AS Monaco | 2024-08-25T15:00:00Z | 55 |
| Saint Etienne | Le Havre | 2024-08-25T15:00:00Z | 42 |
| Auxerre | AS Monaco | 2024-09-15T15:00:00Z | 149 |
| Marseille | Nice | 2024-09-15T15:00:00Z | 149 |
| Paris Saint Germain | Brest | 2024-09-15T15:00:00Z | 149 |
| RC Lens | Lyon | 2024-09-15T15:00:00Z | 149 |
| Saint Etienne | Lille | 2024-09-15T15:00:00Z | 149 |
| Auxerre | Lille | 2025-01-12T16:00:00Z | 66 |
| Brest | Lyon | 2025-01-12T16:00:00Z | 66 |
| Nantes | AS Monaco | 2025-01-12T16:00:00Z | 46 |
| Paris Saint Germain | Saint Etienne | 2025-01-12T16:00:00Z | 43 |
| Rennes | Marseille | 2025-01-12T16:00:00Z | 66 |
| Stade de Reims | Nice | 2025-01-12T16:00:00Z | 66 |
| AS Monaco | Auxerre | 2025-02-02T16:00:00Z | 11 |
| Lille | Saint Etienne | 2025-02-02T16:00:00Z | 11 |
| Montpellier | RC Lens | 2025-02-02T16:00:00Z | 11 |
| Nantes | Brest | 2025-02-08T18:00:00Z | 58 |
| Lille | Le Havre | 2025-02-09T16:00:00Z | 162 |
| Nantes | Brest | 2025-02-09T16:00:00Z | 162 |
| AS Monaco | Nantes | 2025-02-16T16:00:00Z | 2 |
| Brest | Auxerre | 2025-02-16T16:00:00Z | 2 |
| Marseille | Saint Etienne | 2025-02-16T16:00:00Z | 2 |
| Rennes | Lille | 2025-02-16T16:00:00Z | 2 |
| Toulouse | Paris Saint Germain | 2025-02-16T16:00:00Z | 2 |
| Auxerre | Marseille | 2025-02-23T16:00:00Z | 1 |
| Lille | AS Monaco | 2025-02-23T16:00:00Z | 1 |
| Lyon | Paris Saint Germain | 2025-02-23T16:00:00Z | 1 |
| Rennes | Stade de Reims | 2025-02-23T16:00:00Z | 1 |
| Saint Etienne | Angers | 2025-02-23T16:00:00Z | 1 |
| Stade de Reims | Auxerre | 2025-03-08T16:00:00Z | 4 |
| Nice | Lyon | 2025-03-09T16:00:00Z | 2 |
| Nantes | Paris Saint Germain | 2025-04-13T15:00:00Z | 161 |

Odds API historical snapshots of these fixtures outside `1x2`:

| Market | Stamped | Snapshots | Fixtures |
|---|---|---|---|
| h2h | before KO | 2 | 2 |
| totals | before KO | 17 | 17 |
| none | after KO | 9 | 9 |
| none | before KO | 8 | 8 |

### Bookmakers, 2024-25

Share of the 306 fixtures with the book in any snapshot at or before kick-off, and of the 79,302 such snapshots.

| Book | Fixtures | Snapshots | Role | Odds API keys aliased to it |
|---|---|---|---|---|
| `betfair_exchange` | 100.0% | 70.2% | exchange | `betfair`, `betfair_ex_eu`, `betfair_ex_uk` |
| `betfair_sportsbook` | 100.0% | 45.4% | UK retail | `betfair_sb_uk` |
| `betway` | 100.0% | 98.5% | UK retail | none |
| `boylesports` | 100.0% | 84.3% | UK retail | none |
| `casumo` | 100.0% | 92.5% | neither | none |
| `coral` | 100.0% | 95.0% | UK retail | none |
| `grosvenor` | 100.0% | 92.5% | neither | none |
| `ladbrokes` | 100.0% | 95.0% | UK retail | `ladbrokes_uk` |
| `leovegas` | 100.0% | 92.5% | neither | none |
| `matchbook` | 100.0% | 45.3% | exchange | none |
| `paddypower` | 100.0% | 98.8% | UK retail | none |
| `skybet` | 100.0% | 92.9% | UK retail | none |
| `unibet_uk` | 100.0% | 92.4% | UK retail | none |
| `williamhill` | 100.0% | 94.5% | UK retail | none |
| `betvictor` | 86.6% | 70.6% | UK retail | none |
| `livescorebet` | 85.3% | 80.6% | neither | none |
| `virginbet` | 85.3% | 80.6% | neither | none |
| `smarkets` | 70.6% | 70.3% | exchange | none |
| `888sport` | 61.1% | 56.1% | UK retail | `sport888` |
| `mrgreen` | 32.4% | 21.3% | neither | none |
| `pinnacle` | 5.6% | 0.0% | neither | none |

### Leads, 2024-25

Whole-minute decisions per fixture and band, served when a stored snapshot lies in `[t − max age, t]` at its actual time. Unserved decisions are split into those before the fixture's first stored snapshot (not yet listed), those a recorded gap's planned snapshot would have served, and the rest. The last column needs `betfair_exchange` and a UK retail book in the same snapshot.

| Band | Decisions | Served | Unserved: not listed | Unserved: gap | Unserved: other | Fixtures fully served | Served, exchange + retail |
|---|---|---|---|---|---|---|---|
| KO−1m..KO−120m, max age 10m | 36,720 | 100.0% | 0 | 0 | 0 | 306/306 | 99.9% |
| KO−121m..KO−1440m, max age 60m | 403,920 | 100.0% | 0 | 15 | 0 | 303/306 | 100.0% |
| KO−1441m..KO−10080m, max age 360m | 2,643,840 | 99.7% | 6,606 | 400 | 0 | 290/306 | 92.0% |

Fixtures first stored less than 174 h before kick-off: 10, first stored at leads (h: fixtures) 118: 1, 142: 1, 146: 1, 161: 1, 165: 2, 168: 2, 169: 2.

### Near kick-off `betfair_exchange`, 2024-25

Per fixture, the latest exchange snapshot at or before kick-off within 120 min; usable means a back price on every outcome with back overround at most 5%. Research inputs, not validated closing labels.

| Price | Fixtures with one | Within 5 min | Within 10 min | p50 age (min) | p75 age (min) | p90 age (min) | p99 age (min) |
|---|---|---|---|---|---|---|---|
| Any exchange quote | 306/306 | 93.8% | 100.0% | 4.4 | 4.4 | 4.4 | 9.4 |
| Usable | 306/306 | 93.8% | 100.0% | 4.4 | 4.4 | 4.4 | 9.4 |

Latest exchange snapshot missing an outcome's back price: 0 fixtures.

### Provider gaps and off-phase snapshots, 2024-25

Recorded gaps: 2.

| Planned | Returned | Next |
|---|---|---|
| 2024-11-30T02:40:38Z | 2024-11-30T02:35:38Z | 2024-11-30T02:50:38Z |
| 2025-04-15T18:40:38Z | 2025-04-15T18:00:38Z | 2025-04-15T22:25:39Z |

Stored snapshot times off the grid's phase (outside their nearest grid time's whole minute): 0.

### Calls, 2024-25

Planned 5,410; stored 5,408; covered by a recorded gap 2; missing 0. Distinct stored snapshot times, any market: 5,411, × 10 units = 54,110 units. `1x2` snapshot times matching no planned call (gap returns, probes): 3. Planned calls that serve only fixtures not yet stored at that time (not listed): 0.

## 2025-26

### Fixtures, 2025-26

306 FINAL fixtures; 306 with a stored `1x2` snapshot at or before kick-off; 0 without.

Listings in stored responses that attached to no event: 30.

| Home | Away | Listed kick-off | Responses |
|---|---|---|---|
| Auxerre | AS Monaco | 2025-09-14T15:00:00Z | 41 |
| Marseille | Lorient | 2025-09-14T15:00:00Z | 41 |
| Nice | Nantes | 2025-09-14T15:00:00Z | 41 |
| Rennes | Lyon | 2025-09-14T15:00:00Z | 41 |
| Lyon | Angers | 2025-09-21T13:00:00Z | 11 |
| Brest | Nice | 2025-09-21T15:00:00Z | 41 |
| Lyon | Angers | 2025-09-21T15:00:00Z | 41 |
| Marseille | Paris Saint Germain | 2025-09-21T15:00:00Z | 41 |
| Nantes | Rennes | 2025-09-21T15:00:00Z | 41 |
| RC Lens | Lille | 2025-09-21T15:00:00Z | 41 |
| Marseille | Paris Saint Germain | 2025-09-21T18:45:00Z | 302 |
| Marseille | Paris Saint Germain | 2025-09-22T18:00:00Z | 9 |
| Le Havre | Nantes | 2025-11-08T20:05:00Z | 72 |
| Rennes | Brest | 2025-12-14T15:45:00Z | 2 |
| AS Monaco | Nantes | 2026-02-15T16:00:00Z | 140 |
| Lille | Brest | 2026-02-15T16:00:00Z | 144 |
| Lyon | Nice | 2026-02-15T16:00:00Z | 144 |
| Marseille | Strasbourg | 2026-02-15T16:00:00Z | 145 |
| Paris FC | RC Lens | 2026-02-15T16:00:00Z | 144 |
| Rennes | Paris Saint Germain | 2026-02-15T16:00:00Z | 140 |
| Brest | Marseille | 2026-02-22T16:00:00Z | 2 |
| Paris Saint Germain | Metz | 2026-02-22T16:00:00Z | 2 |
| RC Lens | AS Monaco | 2026-02-22T16:00:00Z | 2 |
| Strasbourg | Lyon | 2026-02-22T16:00:00Z | 2 |
| Toulouse | Paris FC | 2026-02-22T16:00:00Z | 2 |
| Paris Saint Germain | Nantes | 2026-03-15T16:00:00Z | 78 |
| RC Lens | Paris Saint Germain | 2026-04-11T15:00:00Z | 180 |
| Brest | Strasbourg | 2026-04-12T15:15:00Z | 231 |
| Nantes | Marseille | 2026-05-01T18:45:00Z | 7 |
| RC Lens | Paris Saint Germain | 2026-05-12T17:00:00Z | 2 |

Odds API historical snapshots of these fixtures outside `1x2`:

| Market | Stamped | Snapshots | Fixtures |
|---|---|---|---|
| h2h | after KO | 4 | 4 |
| h2h | before KO | 929 | 51 |
| totals | before KO | 16 | 16 |
| none | after KO | 12 | 12 |
| none | before KO | 25 | 23 |

### Bookmakers, 2025-26

Share of the 306 fixtures with the book in any snapshot at or before kick-off, and of the 80,502 such snapshots.

| Book | Fixtures | Snapshots | Role | Odds API keys aliased to it |
|---|---|---|---|---|
| `betfair_exchange` | 100.0% | 88.4% | exchange | `betfair`, `betfair_ex_eu`, `betfair_ex_uk` |
| `betfair_sportsbook` | 100.0% | 45.7% | UK retail | `betfair_sb_uk` |
| `betvictor` | 100.0% | 67.8% | UK retail | none |
| `betway` | 100.0% | 95.2% | UK retail | none |
| `boylesports` | 100.0% | 92.5% | UK retail | none |
| `casumo` | 100.0% | 90.8% | neither | none |
| `coral` | 100.0% | 51.6% | UK retail | none |
| `grosvenor` | 100.0% | 90.8% | neither | none |
| `ladbrokes` | 100.0% | 51.6% | UK retail | `ladbrokes_uk` |
| `leovegas` | 100.0% | 90.8% | neither | none |
| `matchbook` | 100.0% | 48.8% | exchange | none |
| `paddypower` | 100.0% | 89.1% | UK retail | none |
| `smarkets` | 100.0% | 97.0% | exchange | none |
| `unibet_uk` | 100.0% | 86.0% | UK retail | none |
| `williamhill` | 100.0% | 94.2% | UK retail | none |
| `skybet` | 99.7% | 82.1% | UK retail | none |
| `livescorebet` | 94.1% | 87.5% | neither | none |
| `virginbet` | 94.1% | 87.8% | neither | none |
| `888sport` | 61.4% | 61.8% | UK retail | `sport888` |
| `betfred` | 15.7% | 12.3% | UK retail | `betfred_uk` |
| `pinnacle` | 5.2% | 0.0% | neither | none |

### Leads, 2025-26

Whole-minute decisions per fixture and band, served when a stored snapshot lies in `[t − max age, t]` at its actual time. Unserved decisions are split into those before the fixture's first stored snapshot (not yet listed), those a recorded gap's planned snapshot would have served, and the rest. The last column needs `betfair_exchange` and a UK retail book in the same snapshot.

| Band | Decisions | Served | Unserved: not listed | Unserved: gap | Unserved: other | Fixtures fully served | Served, exchange + retail |
|---|---|---|---|---|---|---|---|
| KO−1m..KO−120m, max age 10m | 36,720 | 99.7% | 0 | 0 | 121 | 304/306 | 99.7% |
| KO−121m..KO−1440m, max age 60m | 403,920 | 100.0% | 0 | 0 | 64 | 305/306 | 99.9% |
| KO−1441m..KO−10080m, max age 360m | 2,643,840 | 98.9% | 28,029 | 1,476 | 0 | 274/306 | 97.2% |

Fixtures first stored less than 174 h before kick-off: 29, first stored at leads (h: fixtures) 31: 1, 94: 1, 127: 1, 129: 1, 147: 2, 148: 1, 149: 2, 150: 1, 151: 2, 153: 1, 159: 1, 168: 3, 169: 1, 170: 1, 171: 7, 173: 3.

### Near kick-off `betfair_exchange`, 2025-26

Per fixture, the latest exchange snapshot at or before kick-off within 120 min, counted and aged only: this season is sealed ([ADR-0021](../docs/adr/0021-sealed-holdout-and-one-shot-gate.md)), so no price-derived statistic is reported. Research inputs, not validated closing labels.

| Price | Fixtures with one | Within 5 min | Within 10 min | p50 age (min) | p75 age (min) | p90 age (min) | p99 age (min) |
|---|---|---|---|---|---|---|---|
| Any exchange quote | 305/306 | 87.3% | 99.7% | 4.4 | 4.4 | 9.3 | 9.4 |

### Provider gaps and off-phase snapshots, 2025-26

Recorded gaps: 2.

| Planned | Returned | Next |
|---|---|---|
| 2025-10-20T18:40:38Z | 2025-10-20T16:31:34Z | 2025-10-20T19:32:57Z |
| 2026-05-01T06:40:38Z | 2026-05-01T06:05:37Z | 2026-05-01T07:20:38Z |

Stored snapshot times off the grid's phase (outside their nearest grid time's whole minute): 2 (2025-10-20T16:31:34Z, 2026-04-03T18:01:01Z).

### Calls, 2025-26

Planned 5,275; stored 5,273; covered by a recorded gap 2; missing 0. Distinct stored snapshot times, any market: 5,276, × 10 units = 52,760 units. `1x2` snapshot times matching no planned call (gap returns, probes): 3. Planned calls that serve only fixtures not yet stored at that time (not listed): 0.

