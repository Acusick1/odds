# Audit: `odds-api-serie-a-2024-26.yaml`

Coverage of the Serie A Odds API historical run (`h2h` stored as `1x2`, `regions=uk`), read from the production database on 2026-11-12 after the run finished at 20:59Z. The tables below each season come from `uv run python scripts/audit_backfill.py --manifest backfills/odds-api-serie-a-2024-26.yaml` ([audit a finished manifest](../wiki/collection/howto-backfill-odds-api-history.md#audit-a-finished-manifest)). The near-kick-off Betfair Exchange figures are research inputs: whether they can stand in for the FDUK close is [#730](https://github.com/Acusick1/odds/issues/730)'s result, still pending.

Serie A 2025-26 is sealed ([ADR-0021](../docs/adr/0021-sealed-holdout-and-one-shot-gate.md)): its section and this summary report counts, coverage and ages only, with no overround check, back-price check or other price-derived statistic.

## Summary

| | 2024-25 | 2025-26 |
|---|---|---|
| FINAL fixtures / with a stored snapshot | 380 / 379 | 380 / 380 |
| Unmatched listings | 105 (provisional kick-offs, ten postponed fixtures) | 61 (provisional kick-offs, five postponed fixtures) |
| KO−2h band (10 min) served | 99.7% (120 minutes short) | 99.7% (121 minutes short) |
| KO−1d band (60 min) served | 99.7% (1,451 minutes short) | 100.0% (150 minutes short) |
| KO−7d band (6 h) served | 95.8% | 98.2% |
| Exchange price within 5 min of KO | 379 / 380 | 379 / 380 |
| Exchange price within 10 min of KO | 379 / 380 | 379 / 380 |
| Provider gaps recorded | 3 | 8 |
| Off-phase snapshots | 0 | 2 |
| Planned calls / stored / covered by a gap / missing | 7,090 / 7,084 / 4 / 2 | 7,143 / 7,125 / 18 / 0 |
| Distinct stored snapshot times × 10 units | 70,870 | 71,290 |

## Findings

- **Billing.** `odds backfill execute` printed 14,241 paid calls at 10 units each, 142,410 units, against 14,233 planned, for 14,220 distinct snapshot times. 25 calls, 250 units, bought nothing new: the 2024-11-30T02:40:38Z snapshot, a provider gap, was requested 22 times (21 calls), and the 2025-10-20 outage's 5 calls, 16:40:38Z to 17:20:38Z, each returned the 16:31:34Z snapshot, of which the first stored it and 4 bought nothing new. The run stopped on 5 consecutive provider gaps once. The two probe snapshots are outside these counts.
- **Fiorentina v Inter Milan.** The one FINAL fixture without a stored snapshot at or before kick-off (2025-02-06T17:00Z as stored): its only stored snapshot is stamped 19:40:38Z, 2 h 40 min after the stored kick-off, and the one response that listed it at the replayed date lists 19:47Z. The match was listed at its original 2024-12-01T17:00Z date in 371 and 3 responses (17:01Z). That fixture is the missing 380th in the 2024-25 exchange row, and the unserved 120 KO−2h decisions.
- **Band under 98%.** The 2024-25 KO−7d band is 95.8% served: 133,154 decisions not yet listed (4.1%), 240 behind provider gaps and 3,566 other, 3,565 of them Inter Milan v AS Roma (2025-04-27T13:00Z), which has no snapshot from 2025-04-19T09:40:39Z to 2025-04-23T00:25:38Z; the listings at 2025-04-26T16:00Z (139 responses) and 2025-04-27T16:00Z (54) attached to no event. The 2025-26 band is 98.2%: 58,536 not listed, 490 behind gaps and 800 other (the outage below).
- **Fixture horizon.** Serie A rounds are first stored later than the other leagues': 57 fixtures in 2024-25 and 47 in 2025-26 first stored less than 174 h before kick-off (fixture lists from a separate query), 10 of the 104 less than 100 h and 61 at 140 h or more; the round of 2025-05-18 (nine fixtures) is first stored 120 h before. The API appears to list a round only once the previous one is played `(inferred)`.
- **Missing calls.** The calls for 2025-05-13T00:40:38Z and 2025-05-13T06:40:38Z precede the first stored snapshot of the fixtures of 2025-05-17 and 2025-05-18, first stored 102 h and 120 h before kick-off; the executor's skip judged them not stored.
- **Postponed fixtures.** Fifteen fixtures were listed at their original dates, in responses that attached to no event, and played later; `event_postponements` has no row for them (separate query for the played dates).

  | Fixture | Listed at | Responses | Played | First stored before KO |
  |---|---|---|---|---|
  | Atalanta v Como | 2024-09-22T16:00Z, 2024-09-23T18:45Z, 19:45Z | 21, 373, 1 | 2024-09-24T18:45Z | 22 h |
  | Bologna v AC Milan | 2024-10-26T16:00Z, 2025-02-26T17:30Z | 236, 138 | 2025-02-27T19:45Z | 192 h |
  | Fiorentina v Inter Milan | 2024-12-01T17:00Z, 17:01Z | 371, 3 | 2025-02-06 (stored 17:00Z) | none before kick-off |
  | Atalanta v Juventus | 2025-01-05T17:00Z | 193 | 2025-01-14T19:45Z | 221 h |
  | Como v AC Milan | 2025-01-05T17:00Z | 193 | 2025-01-14T17:30Z | 219 h |
  | Inter Milan v Bologna | 2025-01-05T17:00Z | 193 | 2025-01-15T19:45Z | 245 h |
  | Torino v Udinese | 2025-04-21T10:30Z | 388 | 2025-04-23T16:30Z | 46 h |
  | Cagliari v Fiorentina | 2025-04-21T13:00Z | 387 | 2025-04-23T16:30Z | 46 h |
  | Genoa v Lazio | 2025-04-21T16:00Z | 387 | 2025-04-23T16:30Z | 46 h |
  | Parma v Juventus | 2025-04-21T18:45Z | 387 | 2025-04-23T16:30Z | 46 h |
  | Como v AC Milan (2025-26) | 2025-12-21T14:00Z | 34 | 2026-01-15T19:45Z | 150 h |
  | Hellas Verona v Bologna | 2025-12-21T14:00Z | 34 | 2026-01-15T17:30Z | 148 h |
  | Inter Milan v Lecce | 2025-12-21T14:00Z | 34 | 2026-01-14T19:45Z | 126 h |
  | Napoli v Parma | 2025-12-21T14:00Z | 34 | 2026-01-14T17:30Z | 124 h |
  | AC Milan v Como (2025-26) | 2026-02-08T14:00Z | 223 | 2026-02-18T19:45Z | 417 h |

- **Provisional kick-offs.** The other unmatched listings are 1 to 2 days from the stored kick-off, in 1 to 387 responses, nearly all at a Sunday 13:00 to 17:00 UTC slot for games played on the Friday, Saturday, Sunday or Monday (rounds of 2024-09-15, 2025-02-09, 2025-03-02, 2025-04-27, 2025-05-11, 2025-09-21, 2026-03-08, 2026-04-12 and 2026-05-17 among them); the nine games listed for 2026-05-24T13:00Z in 1 response and 2026-06-04 listings of Udinese v Como, Juventus v Genoa and Napoli v AC Milan (1 response) name dates after the last stored kick-offs; Inter Milan v AS Roma was listed for 2026-05-04T18:45Z in 200 responses though its stored fixture was played 2026-04-05T18:45Z, which the data does not explain. All 380 fixtures of 2025-26 and 379 of 2024-25 have a stored snapshot.
- **Provider gaps.** Eleven recorded holes: 2024-11-30T02:40:38Z (returned 02:35:38Z), 2025-04-15T18:40:38Z (returned 18:00:38Z), 2025-05-19T00:40:38Z (returned 2025-05-18T21:00:37Z), 2025-10-20T07:40:38Z (returned 06:45:38Z), five consecutive calls of the 2025-10-20 outage from 16:40:38Z to 17:20:38Z (returned 16:31:34Z), 2026-05-01T06:40:38Z (returned 06:05:37Z) and 2026-05-18T00:40:38Z (returned 2026-05-17T20:55:36Z). They cost decisions: 3 fixtures lost 5 minutes of 60-minute-fresh input after 2024-11-30 and 6 fixtures 40 minutes of 6-hour-fresh input after 2025-04-15; after 2025-10-20 10 fixtures lost 129 minutes of 6-hour-fresh input (49 classed as gap and 80 as other: the recorded gaps end at 17:20:38Z and the next stored snapshot is 19:32:57Z) and Pisa v Lecce 35 minutes of 60-minute-fresh input on 2026-05-01. (Fixture lists from a separate query.)
- **Cremonese v Udinese, 2025-10-20.** The Monday 18:45 game fell inside the outage: no snapshot from 16:31:34Z to kick-off, so 119 decisions of the KO−2h band are unserved, 115 of the KO−1d band lost to the 07:40:38Z hole, and the fixture has no exchange price within 120 min of kick-off (the latest is 133.4 min old): 379 of 380 in the 2025-26 exchange row.
- **Off-phase snapshots.** The 2025-10-20T16:31:34Z snapshot is the outage's return, and 2026-01-17T19:31:03Z is stamped off the grid's phase. The log has 13 `snapshot_key_conflict` warnings, each keeping the stored snapshot: 1 at 2026-01-03T18:00:49Z, 11 at 2026-01-17T19:31:03Z and 1 at 2026-04-18T17:00:50Z.
- **Books outside `UK_RETAIL_BOOKMAKERS`.** `casumo`, `grosvenor`, `leovegas`, `livescorebet`, `virginbet` (both seasons) and `mrgreen` (2024-25) are stored under their own keys and counted neither as UK retail nor as exchanges. They are UK-licensed sportsbooks `(inferred)`, so retail rules reading `UK_RETAIL_BOOKMAKERS` ignore up to six of the region's books.
- **Uneven books.** In 2024-25 `coral` and `ladbrokes` are on 94.7% of fixtures and `smarkets` on 71.1%.
- **Betfred.** `betfred` is stored on 61 of the 2025-26 fixtures, from 2026-04-13 to 2026-05-24 (dates from a separate query), and on none in 2024-25.
- **Pinnacle** appears only in the two probe snapshots, which named it; the `uk` region does not return it.
- **Exchange quotes without a draw.** When the exchange quotes only the two teams, the response's `h2h` keeps its key and is stored as an `h2h` snapshot on the soccer fixture: 17 snapshots on 17 fixtures in 2024-25 and 343 on 23 fixtures before kick-off in 2025-26. Readers of `1x2` do not see these partial exchange books. Listings with no books are stored as snapshots with no market (29 before kick-off in 2024-25, 12 in 2025-26).
- **Near kick-off exchange.** The latest exchange price is 4.4 min before kick-off for 379 fixtures in each season, apart from Fiorentina v Inter Milan (no snapshot at or before kick-off) and Cremonese v Udinese above. The 2024-25 prices found are complete and within the 5% overround limit. The run bought no snapshot nearer kick-off.

## 2024-25

### Fixtures, 2024-25

380 FINAL fixtures; 379 with a stored `1x2` snapshot at or before kick-off; 1 without.

| Fixture | Kick-off |
|---|---|
| Fiorentina v Inter Milan | 2025-02-06T17:00:00Z |

Listings in stored responses that attached to no event: 105.

| Home | Away | Listed kick-off | Responses |
|---|---|---|---|
| AC Milan | Venezia | 2024-09-15T13:00:00Z | 19 |
| Cagliari | Napoli | 2024-09-15T13:00:00Z | 19 |
| Como | Bologna | 2024-09-15T13:00:00Z | 19 |
| Empoli | Juventus | 2024-09-15T13:00:00Z | 19 |
| Genoa | AS Roma | 2024-09-15T13:00:00Z | 18 |
| Lazio | Hellas Verona | 2024-09-15T13:00:00Z | 18 |
| Monza | Inter Milan | 2024-09-15T13:00:00Z | 18 |
| Parma | Udinese | 2024-09-15T13:00:00Z | 18 |
| Atalanta BC | Como | 2024-09-22T16:00:00Z | 21 |
| Juventus | Napoli | 2024-09-22T16:00:00Z | 3 |
| Atalanta BC | Como | 2024-09-23T18:45:00Z | 373 |
| Atalanta BC | Como | 2024-09-23T19:45:00Z | 1 |
| Bologna | AC Milan | 2024-10-26T16:00:00Z | 236 |
| Fiorentina | Inter Milan | 2024-12-01T17:00:00Z | 371 |
| Fiorentina | Inter Milan | 2024-12-01T17:01:00Z | 3 |
| Atalanta BC | Juventus | 2025-01-05T17:00:00Z | 193 |
| Como | AC Milan | 2025-01-05T17:00:00Z | 193 |
| Inter Milan | Bologna | 2025-01-05T17:00:00Z | 193 |
| Fiorentina | Inter Milan | 2025-02-06T19:47:00Z | 1 |
| Como | Juventus | 2025-02-09T13:00:00Z | 60 |
| Empoli | AC Milan | 2025-02-09T13:00:00Z | 60 |
| Hellas Verona | Atalanta BC | 2025-02-09T13:00:00Z | 60 |
| Inter Milan | Fiorentina | 2025-02-09T13:00:00Z | 60 |
| Torino | Genoa | 2025-02-09T13:00:00Z | 60 |
| Como | Juventus | 2025-02-09T14:00:00Z | 130 |
| Empoli | AC Milan | 2025-02-09T14:00:00Z | 130 |
| Hellas Verona | Atalanta BC | 2025-02-09T14:00:00Z | 130 |
| Inter Milan | Fiorentina | 2025-02-09T14:00:00Z | 193 |
| Lecce | Bologna | 2025-02-09T14:00:00Z | 122 |
| Napoli | Udinese | 2025-02-09T14:00:00Z | 122 |
| Torino | Genoa | 2025-02-09T14:00:00Z | 130 |
| Venezia | AS Roma | 2025-02-09T14:00:00Z | 122 |
| Bologna | AC Milan | 2025-02-26T17:30:00Z | 138 |
| AC Milan | Lazio | 2025-03-02T14:00:00Z | 55 |
| AS Roma | Como | 2025-03-02T14:00:00Z | 55 |
| Atalanta BC | Venezia | 2025-03-02T14:00:00Z | 193 |
| Fiorentina | Lecce | 2025-03-02T14:00:00Z | 170 |
| Juventus | Hellas Verona | 2025-03-02T14:00:00Z | 228 |
| Monza | Torino | 2025-03-02T14:00:00Z | 55 |
| Napoli | Inter Milan | 2025-03-02T14:00:00Z | 174 |
| Udinese | Parma | 2025-03-02T14:00:00Z | 193 |
| AC Milan | Lazio | 2025-03-02T17:00:00Z | 37 |
| Atalanta BC | Venezia | 2025-03-02T17:00:00Z | 37 |
| Bologna | Cagliari | 2025-03-02T17:00:00Z | 2 |
| Fiorentina | Lecce | 2025-03-02T17:00:00Z | 37 |
| Genoa | Empoli | 2025-03-02T17:00:00Z | 37 |
| Juventus | Hellas Verona | 2025-03-02T17:00:00Z | 2 |
| Monza | Torino | 2025-03-02T17:00:00Z | 37 |
| Napoli | Inter Milan | 2025-03-02T17:00:00Z | 37 |
| Udinese | Parma | 2025-03-02T17:00:00Z | 37 |
| Hellas Verona | Bologna | 2025-03-09T00:00:00Z | 5 |
| Torino | Udinese | 2025-04-21T10:30:00Z | 388 |
| Cagliari | Fiorentina | 2025-04-21T13:00:00Z | 387 |
| Genoa | Lazio | 2025-04-21T16:00:00Z | 387 |
| Parma | Juventus | 2025-04-21T18:45:00Z | 387 |
| Atalanta BC | Lecce | 2025-04-25T18:45:00Z | 161 |
| Como | Genoa | 2025-04-26T13:00:00Z | 139 |
| Inter Milan | AS Roma | 2025-04-26T16:00:00Z | 139 |
| Lazio | Parma | 2025-04-26T18:45:00Z | 139 |
| Atalanta BC | Lecce | 2025-04-27T13:00:00Z | 67 |
| Como | Genoa | 2025-04-27T13:00:00Z | 14 |
| Hellas Verona | Cagliari | 2025-04-27T13:00:00Z | 14 |
| Juventus | Monza | 2025-04-27T13:00:00Z | 3 |
| Lazio | Parma | 2025-04-27T13:00:00Z | 14 |
| Napoli | Torino | 2025-04-27T13:00:00Z | 3 |
| Udinese | Bologna | 2025-04-27T13:00:00Z | 14 |
| Venezia | AC Milan | 2025-04-27T13:00:00Z | 3 |
| Atalanta BC | Lecce | 2025-04-27T16:00:00Z | 54 |
| Como | Genoa | 2025-04-27T16:00:00Z | 54 |
| Fiorentina | Empoli | 2025-04-27T16:00:00Z | 54 |
| Hellas Verona | Cagliari | 2025-04-27T16:00:00Z | 54 |
| Inter Milan | AS Roma | 2025-04-27T16:00:00Z | 54 |
| Lazio | Parma | 2025-04-27T16:00:00Z | 54 |
| Napoli | Torino | 2025-04-27T16:00:00Z | 54 |
| Udinese | Bologna | 2025-04-27T16:00:00Z | 54 |
| Venezia | AC Milan | 2025-04-27T16:00:00Z | 54 |
| AC Milan | Bologna | 2025-05-11T16:00:00Z | 47 |
| Atalanta BC | AS Roma | 2025-05-11T16:00:00Z | 219 |
| Como | Cagliari | 2025-05-11T16:00:00Z | 47 |
| Empoli | Parma | 2025-05-11T16:00:00Z | 47 |
| Hellas Verona | Lecce | 2025-05-11T16:00:00Z | 47 |
| Lazio | Juventus | 2025-05-11T16:00:00Z | 47 |
| Napoli | Genoa | 2025-05-11T16:00:00Z | 302 |
| Udinese | Monza | 2025-05-11T16:00:00Z | 47 |
| Venezia | Fiorentina | 2025-05-11T16:00:00Z | 219 |
| AS Roma | AC Milan | 2025-05-18T13:00:00Z | 244 |
| Cagliari | Venezia | 2025-05-18T13:00:00Z | 244 |
| Fiorentina | Bologna | 2025-05-18T13:00:00Z | 244 |
| Genoa | Atalanta BC | 2025-05-18T13:00:00Z | 247 |
| Hellas Verona | Como | 2025-05-18T13:00:00Z | 244 |
| Inter Milan | Lazio | 2025-05-18T13:00:00Z | 244 |
| Juventus | Udinese | 2025-05-18T13:00:00Z | 244 |
| Lecce | Torino | 2025-05-18T13:00:00Z | 244 |
| Monza | Empoli | 2025-05-18T13:00:00Z | 244 |
| Parma | Napoli | 2025-05-18T13:00:00Z | 244 |
| AC Milan | Monza | 2025-05-25T13:00:00Z | 2 |
| Atalanta BC | Parma | 2025-05-25T13:00:00Z | 1 |
| Bologna | Genoa | 2025-05-25T13:00:00Z | 2 |
| Como | Inter Milan | 2025-05-25T13:00:00Z | 2 |
| Empoli | Hellas Verona | 2025-05-25T13:00:00Z | 2 |
| Lazio | Lecce | 2025-05-25T13:00:00Z | 1 |
| Napoli | Cagliari | 2025-05-25T13:00:00Z | 2 |
| Torino | AS Roma | 2025-05-25T13:00:00Z | 2 |
| Udinese | Fiorentina | 2025-05-25T13:00:00Z | 2 |
| Venezia | Juventus | 2025-05-25T13:00:00Z | 1 |

Odds API historical snapshots of these fixtures outside `1x2`:

| Market | Stamped | Snapshots | Fixtures |
|---|---|---|---|
| h2h | before KO | 17 | 17 |
| totals | before KO | 19 | 19 |
| none | after KO | 27 | 27 |
| none | before KO | 29 | 29 |

### Bookmakers, 2024-25

Share of the 380 fixtures with the book in any snapshot at or before kick-off, and of the 117,441 such snapshots.

| Book | Fixtures | Snapshots | Role | Odds API keys aliased to it |
|---|---|---|---|---|
| `888sport` | 99.7% | 89.3% | UK retail | `sport888` |
| `betfair_exchange` | 99.7% | 78.3% | exchange | `betfair`, `betfair_ex_eu`, `betfair_ex_uk` |
| `betfair_sportsbook` | 99.7% | 40.7% | UK retail | `betfair_sb_uk` |
| `betvictor` | 99.7% | 73.7% | UK retail | none |
| `betway` | 99.7% | 97.5% | UK retail | none |
| `boylesports` | 99.7% | 82.0% | UK retail | none |
| `casumo` | 99.7% | 86.6% | neither | none |
| `grosvenor` | 99.7% | 86.7% | neither | none |
| `leovegas` | 99.7% | 86.6% | neither | none |
| `matchbook` | 99.7% | 48.2% | exchange | none |
| `paddypower` | 99.7% | 99.9% | UK retail | none |
| `skybet` | 99.7% | 92.1% | UK retail | none |
| `unibet_uk` | 99.7% | 86.6% | UK retail | none |
| `williamhill` | 99.7% | 94.8% | UK retail | none |
| `coral` | 94.7% | 87.1% | UK retail | none |
| `ladbrokes` | 94.7% | 87.1% | UK retail | `ladbrokes_uk` |
| `livescorebet` | 89.2% | 78.2% | neither | none |
| `virginbet` | 89.2% | 78.2% | neither | none |
| `smarkets` | 71.1% | 71.7% | exchange | none |
| `mrgreen` | 28.7% | 18.2% | neither | none |
| `pinnacle` | 4.7% | 0.0% | neither | none |

### Leads, 2024-25

Whole-minute decisions per fixture and band, served when a stored snapshot lies in `[t − max age, t]` at its actual time. Unserved decisions are split into those before the fixture's first stored snapshot (not yet listed), those a recorded gap's planned snapshot would have served, and the rest. The last column needs `betfair_exchange` and a UK retail book in the same snapshot.

| Band | Decisions | Served | Unserved: not listed | Unserved: gap | Unserved: other | Fixtures fully served | Served, exchange + retail |
|---|---|---|---|---|---|---|---|
| KO−1m..KO−120m, max age 10m | 45,600 | 99.7% | 120 | 0 | 0 | 379/380 | 99.7% |
| KO−121m..KO−1440m, max age 60m | 501,600 | 99.7% | 1,436 | 15 | 0 | 375/380 | 99.7% |
| KO−1441m..KO−10080m, max age 360m | 3,283,200 | 95.8% | 133,154 | 240 | 3,566 | 327/380 | 93.4% |

Fixtures first stored less than 174 h before kick-off: 57, first stored at leads (h: fixtures) -3: 1, 22: 1, 33: 1, 46: 4, 74: 1, 96: 2, 102: 1, 106: 1, 108: 1, 120: 9, 123: 1, 126: 1, 132: 1, 138: 4, 143: 1, 144: 3, 146: 1, 149: 1, 150: 1, 153: 1, 156: 1, 162: 2, 164: 1, 165: 1, 166: 3, 167: 1, 168: 3, 170: 3, 171: 1, 172: 3, 174: 1.

### Near kick-off `betfair_exchange`, 2024-25

Per fixture, the latest exchange snapshot at or before kick-off within 120 min; usable means a back price on every outcome with back overround at most 5%. Research inputs, not validated closing labels.

| Price | Fixtures with one | Within 5 min | Within 10 min | p50 age (min) | p75 age (min) | p90 age (min) | p99 age (min) |
|---|---|---|---|---|---|---|---|
| Any exchange quote | 379/380 | 99.7% | 99.7% | 4.4 | 4.4 | 4.4 | 4.4 |
| Usable | 379/380 | 99.7% | 99.7% | 4.4 | 4.4 | 4.4 | 4.4 |

Latest exchange snapshot missing an outcome's back price: 0 fixtures.

### Provider gaps and off-phase snapshots, 2024-25

Recorded gaps: 3.

| Planned | Returned | Next |
|---|---|---|
| 2024-11-30T02:40:38Z | 2024-11-30T02:35:38Z | 2024-11-30T02:50:38Z |
| 2025-04-15T18:40:38Z | 2025-04-15T18:00:38Z | 2025-04-15T22:25:39Z |
| 2025-05-19T00:40:38Z | 2025-05-18T21:00:37Z | 2025-05-19T07:10:37Z |

Stored snapshot times off the grid's phase (outside their nearest grid time's whole minute): 0.

### Calls, 2024-25

Planned 7,090; stored 7,084; covered by a recorded gap 4; missing 2. Distinct stored snapshot times, any market: 7,087, × 10 units = 70,870 units. `1x2` snapshot times matching no planned call (gap returns, probes): 3. Planned calls that serve only fixtures not yet stored at that time (not listed): 4.

Missing calls: 2025-05-13T00:40:38Z, 2025-05-13T06:40:38Z

## 2025-26

### Fixtures, 2025-26

380 FINAL fixtures; 380 with a stored `1x2` snapshot at or before kick-off; 0 without.

Listings in stored responses that attached to no event: 61.

| Home | Away | Listed kick-off | Responses |
|---|---|---|---|
| Bologna | Genoa | 2025-09-21T13:00:00Z | 8 |
| Hellas Verona | Juventus | 2025-09-21T13:00:00Z | 8 |
| Lecce | Cagliari | 2025-09-21T13:00:00Z | 8 |
| Napoli | Pisa | 2025-09-21T13:00:00Z | 8 |
| Udinese | AC Milan | 2025-09-21T13:00:00Z | 8 |
| Atalanta BC | AC Milan | 2025-10-29T13:00:00Z | 6 |
| Como | AC Milan | 2025-12-21T14:00:00Z | 34 |
| Hellas Verona | Bologna | 2025-12-21T14:00:00Z | 34 |
| Inter Milan | Lecce | 2025-12-21T14:00:00Z | 34 |
| Napoli | Parma | 2025-12-21T14:00:00Z | 34 |
| Bologna | AC Milan | 2026-02-01T14:00:00Z | 14 |
| Udinese | AS Roma | 2026-02-01T14:00:00Z | 2 |
| AC Milan | Como | 2026-02-08T14:00:00Z | 223 |
| AC Milan | Inter Milan | 2026-03-08T14:00:00Z | 123 |
| Atalanta BC | Udinese | 2026-03-08T14:00:00Z | 126 |
| Cagliari | Como | 2026-03-08T14:00:00Z | 122 |
| Genoa | AS Roma | 2026-03-08T14:00:00Z | 123 |
| Juventus | Pisa | 2026-03-08T14:00:00Z | 123 |
| Lazio | Sassuolo | 2026-03-08T14:00:00Z | 309 |
| Lecce | Cremonese | 2026-03-08T14:00:00Z | 123 |
| Napoli | Torino | 2026-03-08T14:00:00Z | 135 |
| Hellas Verona | Genoa | 2026-03-15T14:00:00Z | 35 |
| Hellas Verona | Fiorentina | 2026-04-04T13:00:00Z | 3 |
| Lazio | Parma | 2026-04-04T13:00:00Z | 3 |
| Lecce | Atalanta BC | 2026-04-04T13:00:00Z | 200 |
| Udinese | Como | 2026-04-06T13:00:00Z | 204 |
| AC Milan | Udinese | 2026-04-12T13:00:00Z | 13 |
| AS Roma | Pisa | 2026-04-12T13:00:00Z | 16 |
| Atalanta BC | Juventus | 2026-04-12T13:00:00Z | 13 |
| Bologna | Lecce | 2026-04-12T13:00:00Z | 13 |
| Cagliari | Cremonese | 2026-04-12T13:00:00Z | 13 |
| Como | Inter Milan | 2026-04-12T13:00:00Z | 17 |
| Fiorentina | Lazio | 2026-04-12T13:00:00Z | 13 |
| Genoa | Sassuolo | 2026-04-12T13:00:00Z | 16 |
| Torino | Hellas Verona | 2026-04-12T13:00:00Z | 13 |
| Bologna | Cagliari | 2026-05-03T13:00:00Z | 9 |
| Inter Milan | Parma | 2026-05-03T13:00:00Z | 9 |
| Juventus | Hellas Verona | 2026-05-03T13:00:00Z | 9 |
| Cremonese | Lazio | 2026-05-04T13:30:00Z | 9 |
| Inter Milan | AS Roma | 2026-05-04T18:45:00Z | 200 |
| AS Roma | Lazio | 2026-05-17T13:00:00Z | 244 |
| Atalanta BC | Bologna | 2026-05-17T13:00:00Z | 244 |
| Cagliari | Torino | 2026-05-17T13:00:00Z | 244 |
| Como | Parma | 2026-05-17T13:00:00Z | 244 |
| Genoa | AC Milan | 2026-05-17T13:00:00Z | 244 |
| Juventus | Fiorentina | 2026-05-17T13:00:00Z | 244 |
| Pisa | Napoli | 2026-05-17T13:00:00Z | 244 |
| Sassuolo | Lecce | 2026-05-17T13:00:00Z | 244 |
| Udinese | Cremonese | 2026-05-17T13:00:00Z | 244 |
| AC Milan | Cagliari | 2026-05-24T13:00:00Z | 1 |
| Bologna | Inter Milan | 2026-05-24T13:00:00Z | 1 |
| Cremonese | Como | 2026-05-24T13:00:00Z | 1 |
| Fiorentina | Atalanta BC | 2026-05-24T13:00:00Z | 1 |
| Hellas Verona | AS Roma | 2026-05-24T13:00:00Z | 1 |
| Lazio | Pisa | 2026-05-24T13:00:00Z | 1 |
| Lecce | Genoa | 2026-05-24T13:00:00Z | 1 |
| Napoli | Udinese | 2026-05-24T13:00:00Z | 1 |
| Torino | Juventus | 2026-05-24T13:00:00Z | 1 |
| Udinese | Como | 2026-06-04T13:00:00Z | 1 |
| Juventus | Genoa | 2026-06-04T16:00:00Z | 1 |
| Napoli | AC Milan | 2026-06-04T18:45:00Z | 1 |

Odds API historical snapshots of these fixtures outside `1x2`:

| Market | Stamped | Snapshots | Fixtures |
|---|---|---|---|
| h2h | after KO | 20 | 15 |
| h2h | before KO | 343 | 23 |
| totals | before KO | 20 | 20 |
| none | after KO | 35 | 34 |
| none | before KO | 12 | 12 |

### Bookmakers, 2025-26

Share of the 380 fixtures with the book in any snapshot at or before kick-off, and of the 125,405 such snapshots.

| Book | Fixtures | Snapshots | Role | Odds API keys aliased to it |
|---|---|---|---|---|
| `888sport` | 100.0% | 93.8% | UK retail | `sport888` |
| `betfair_exchange` | 100.0% | 81.0% | exchange | `betfair`, `betfair_ex_eu`, `betfair_ex_uk` |
| `betfair_sportsbook` | 100.0% | 39.5% | UK retail | `betfair_sb_uk` |
| `betvictor` | 100.0% | 54.7% | UK retail | none |
| `betway` | 100.0% | 96.4% | UK retail | none |
| `boylesports` | 100.0% | 90.5% | UK retail | none |
| `casumo` | 100.0% | 90.7% | neither | none |
| `coral` | 100.0% | 55.8% | UK retail | none |
| `grosvenor` | 100.0% | 90.7% | neither | none |
| `ladbrokes` | 100.0% | 55.7% | UK retail | `ladbrokes_uk` |
| `leovegas` | 100.0% | 90.6% | neither | none |
| `matchbook` | 100.0% | 52.0% | exchange | none |
| `paddypower` | 100.0% | 92.5% | UK retail | none |
| `skybet` | 100.0% | 84.8% | UK retail | none |
| `smarkets` | 100.0% | 97.7% | exchange | none |
| `unibet_uk` | 100.0% | 93.6% | UK retail | none |
| `williamhill` | 100.0% | 93.9% | UK retail | none |
| `livescorebet` | 97.4% | 88.6% | neither | none |
| `virginbet` | 97.4% | 88.7% | neither | none |
| `betfred` | 16.1% | 11.7% | UK retail | `betfred_uk` |
| `pinnacle` | 5.3% | 0.0% | neither | none |

### Leads, 2025-26

Whole-minute decisions per fixture and band, served when a stored snapshot lies in `[t − max age, t]` at its actual time. Unserved decisions are split into those before the fixture's first stored snapshot (not yet listed), those a recorded gap's planned snapshot would have served, and the rest. The last column needs `betfair_exchange` and a UK retail book in the same snapshot.

| Band | Decisions | Served | Unserved: not listed | Unserved: gap | Unserved: other | Fixtures fully served | Served, exchange + retail |
|---|---|---|---|---|---|---|---|
| KO−1m..KO−120m, max age 10m | 45,600 | 99.7% | 0 | 46 | 75 | 378/380 | 99.7% |
| KO−121m..KO−1440m, max age 60m | 501,600 | 100.0% | 0 | 90 | 60 | 378/380 | 99.9% |
| KO−1441m..KO−10080m, max age 360m | 3,283,200 | 98.2% | 58,536 | 490 | 800 | 334/380 | 97.3% |

Fixtures first stored less than 174 h before kick-off: 47, first stored at leads (h: fixtures) 102: 1, 117: 1, 120: 1, 124: 1, 126: 1, 129: 5, 135: 1, 138: 3, 140: 2, 141: 1, 143: 1, 144: 6, 146: 2, 148: 1, 149: 1, 150: 2, 152: 1, 156: 1, 162: 1, 164: 1, 167: 1, 168: 3, 170: 6, 171: 1, 173: 2.

### Near kick-off `betfair_exchange`, 2025-26

Per fixture, the latest exchange snapshot at or before kick-off within 120 min, counted and aged only: this season is sealed ([ADR-0021](../docs/adr/0021-sealed-holdout-and-one-shot-gate.md)), so no price-derived statistic is reported. Research inputs, not validated closing labels.

| Price | Fixtures with one | Within 5 min | Within 10 min | p50 age (min) | p75 age (min) | p90 age (min) | p99 age (min) |
|---|---|---|---|---|---|---|---|
| Any exchange quote | 379/380 | 99.7% | 99.7% | 4.4 | 4.4 | 4.4 | 4.4 |

### Provider gaps and off-phase snapshots, 2025-26

Recorded gaps: 8.

| Planned | Returned | Next |
|---|---|---|
| 2025-10-20T07:40:38Z | 2025-10-20T06:45:38Z | 2025-10-20T09:29:52Z |
| 2025-10-20T16:40:38Z | 2025-10-20T16:31:34Z | 2025-10-20T19:32:57Z |
| 2025-10-20T16:50:38Z | 2025-10-20T16:31:34Z | 2025-10-20T19:32:57Z |
| 2025-10-20T17:00:38Z | 2025-10-20T16:31:34Z | 2025-10-20T19:32:57Z |
| 2025-10-20T17:10:38Z | 2025-10-20T16:31:34Z | 2025-10-20T19:32:57Z |
| 2025-10-20T17:20:38Z | 2025-10-20T16:31:34Z | 2025-10-20T19:32:57Z |
| 2026-05-01T06:40:38Z | 2026-05-01T06:05:37Z | 2026-05-01T07:20:38Z |
| 2026-05-18T00:40:38Z | 2026-05-17T20:55:36Z | 2026-05-18T07:05:37Z |

Stored snapshot times off the grid's phase (outside their nearest grid time's whole minute): 2 (2025-10-20T16:31:34Z, 2026-01-17T19:31:03Z).

### Calls, 2025-26

Planned 7,143; stored 7,125; covered by a recorded gap 18; missing 0. Distinct stored snapshot times, any market: 7,129, × 10 units = 71,290 units. `1x2` snapshot times matching no planned call (gap returns, probes): 4. Planned calls that serve only fixtures not yet stored at that time (not listed): 2.

