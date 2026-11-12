# Audit: `odds-api-bundesliga-2024-26.yaml`

2025-26 is sealed ([ADR-0021](../docs/adr/0021-sealed-holdout-and-one-shot-gate.md)): its section and this summary report counts, coverage and ages only, with no overround check, back-price check or other price-derived statistic.

Coverage of the Bundesliga Odds API historical run (`h2h` stored as `1x2`, `regions=uk`), read from the production database on 2026-11-12 after the run finished at 16:55Z. The tables below each season come from `uv run python scripts/audit_backfill.py --manifest backfills/odds-api-bundesliga-2024-26.yaml` ([audit a finished manifest](../wiki/collection/howto-backfill-odds-api-history.md#audit-a-finished-manifest)). The near-kick-off Betfair Exchange figures are research inputs: whether they can stand in for the FDUK close is [#730](https://github.com/Acusick1/odds/issues/730)'s result, still pending.

## Summary

| | 2024-25 | 2025-26 |
|---|---|---|
| FINAL fixtures / with a stored snapshot | 306 / 306 | 306 / 306 |
| Unmatched listings | 0 | 3 (postponed fixtures at their original dates) |
| KO−2h band (10 min) served | 100.0% | 100.0% |
| KO−1d band (60 min) served | 100.0% | 100.0% (1 minute short) |
| KO−7d band (6 h) served | 98.6% | 99.3% |
| Exchange price within 5 min of KO | 306 / 306 | 306 / 306 |
| Provider gaps recorded | 2 | 1 |
| Off-phase snapshots | 0 | 1 |
| Planned calls / stored / covered by a gap / missing | 4,642 / 4,639 / 3 / 0 | 4,752 / 4,751 / 1 / 0 |
| Distinct stored snapshot times × 10 units | 46,420 | 47,530 |

## Findings

- **Billing.** `odds backfill execute` printed 9,414 paid calls at 10 units each, 94,140 units, against 9,394 planned. 22 of them requested the 2025-04-15T21:25:38Z snapshot, a provider gap, before gaps were recorded and skipped, so 21 calls (210 units) bought nothing new. The two probe snapshots (2024-09-14 and 2025-09-13, `h2h,totals` with named books) are outside these counts.
- **Fixture horizon.** Some rounds are first stored 122 to 173 h before kick-off, 16 fixtures a season later than the KO−7d band's 168 h start: the first round after the winter break and the final round (fixture lists from a separate query). The API appears to list a round only once the previous one is played `(inferred)`. That leaves 1.3% (2024-25) and 0.6% (2025-26) of the KO−7d band's decisions without a snapshot, which no purchase can fill. Two 2024-25 planned calls served only fixtures not yet listed.
- **Provider gaps.** Two of the three holes cost decisions, all in the KO−7d band: after the 2025-04-15 and 2025-10-20 holes the next planned call was 6 h after the missing one, so 9 fixtures each lost 205 and 160 minutes of 6-hour-fresh input. The 2025-05-11 hole fell before its fixtures were listed.
- **Off-phase snapshot.** The call for 2026-05-08T04:25:38Z returned a snapshot stamped 04:26:09Z, which stopped the run until it was resumed. It leaves the one unserved KO−1d decision minute, of Borussia Dortmund v Eintracht Frankfurt (separate query), without a 60-minute-fresh input.
- **Postponed fixtures.** St. Pauli v Leipzig and Werder Bremen v Hoffenheim (listed 2026-01-10) and Hamburger SV v Leverkusen (listed 2026-01-13) were listed at their original dates in 204 to 232 responses and attached to no event; their replayed fixtures (2026-01-27 and 2026-03-04) are first stored more than 200 h ahead, and no postponement is recorded in `event_postponements` (separate queries).
- **Books outside `UK_RETAIL_BOOKMAKERS`.** `casumo`, `grosvenor`, `leovegas`, `livescorebet`, `virginbet` (both seasons, on 75 to 94% of snapshots) and `mrgreen` (2024-25) are stored under their own keys and counted neither as UK retail nor as exchanges. They are UK-licensed sportsbooks `(inferred)`, so retail rules reading `UK_RETAIL_BOOKMAKERS` ignore up to six of the region's books.
- **Betfred.** `betfred` is stored on 45 of the 2025-26 fixtures, from 2026-04-13 to the season's end (dates from a separate query), and on none in 2024-25.
- **Pinnacle** appears only in the two probe snapshots, which named it; the `uk` region does not return it.
- **Exchange quotes without a draw.** When the exchange quotes only the two teams, the response's `h2h` keeps its key and is stored as an `h2h` snapshot on the soccer fixture: 219 snapshots on 23 fixtures before kick-off in 2025-26. Readers of `1x2` do not see these partial exchange books. Listings with no books are stored as snapshots with no market (47 in 2024-25, 34 in 2025-26).
- **Near kick-off exchange.** The last planned call before each kick-off is the grid snapshot 4 min 22 s before it, so every fixture's latest exchange price is 4.4 min old, all of the 2024-25 prices complete and within the 5% overround limit. The run bought no snapshot nearer kick-off.

## 2024-25

### Fixtures, 2024-25

306 FINAL fixtures; 306 with a stored `1x2` snapshot at or before kick-off; 0 without.

Listings in stored responses that attached to no event: 0.

Odds API historical snapshots of these fixtures outside `1x2`:

| Market | Stamped | Snapshots | Fixtures |
|---|---|---|---|
| totals | before KO | 17 | 17 |
| none | after KO | 27 | 26 |
| none | before KO | 20 | 20 |

### Bookmakers, 2024-25

Share of the 306 fixtures with the book in any snapshot at or before kick-off, and of the 69,700 such snapshots.

| Book | Fixtures | Snapshots | Role | Odds API keys aliased to it |
|---|---|---|---|---|
| `betfair_exchange` | 100.0% | 82.6% | exchange | `betfair`, `betfair_ex_eu`, `betfair_ex_uk` |
| `betfair_sportsbook` | 100.0% | 47.8% | UK retail | `betfair_sb_uk` |
| `betvictor` | 100.0% | 85.0% | UK retail | none |
| `betway` | 100.0% | 98.4% | UK retail | none |
| `boylesports` | 100.0% | 77.4% | UK retail | none |
| `casumo` | 100.0% | 84.9% | neither | none |
| `coral` | 100.0% | 91.1% | UK retail | none |
| `grosvenor` | 100.0% | 85.0% | neither | none |
| `ladbrokes` | 100.0% | 91.1% | UK retail | `ladbrokes_uk` |
| `leovegas` | 100.0% | 84.9% | neither | none |
| `matchbook` | 100.0% | 43.3% | exchange | none |
| `paddypower` | 100.0% | 99.8% | UK retail | none |
| `skybet` | 100.0% | 96.4% | UK retail | none |
| `unibet_uk` | 100.0% | 84.9% | UK retail | none |
| `williamhill` | 100.0% | 95.0% | UK retail | none |
| `livescorebet` | 88.6% | 75.2% | neither | none |
| `virginbet` | 88.6% | 75.2% | neither | none |
| `smarkets` | 73.5% | 71.0% | exchange | none |
| `888sport` | 61.8% | 57.7% | UK retail | `sport888` |
| `mrgreen` | 29.4% | 17.7% | neither | none |
| `pinnacle` | 5.6% | 0.0% | neither | none |

### Leads, 2024-25

Whole-minute decisions per fixture and band, served when a stored snapshot lies in `[t − max age, t]` at its actual time. Unserved decisions are split into those before the fixture's first stored snapshot (not yet listed), those a recorded gap's planned snapshot would have served, and the rest. The last column needs `betfair_exchange` and a UK retail book in the same snapshot.

| Band | Decisions | Served | Unserved: not listed | Unserved: gap | Unserved: other | Fixtures fully served | Served, exchange + retail |
|---|---|---|---|---|---|---|---|
| KO−1m..KO−120m, max age 10m | 36,720 | 100.0% | 0 | 0 | 0 | 306/306 | 99.9% |
| KO−121m..KO−1440m, max age 60m | 403,920 | 100.0% | 0 | 0 | 0 | 306/306 | 100.0% |
| KO−1441m..KO−10080m, max age 360m | 2,643,840 | 98.6% | 34,556 | 1,845 | 0 | 281/306 | 96.7% |

Fixtures first stored less than 174 h before kick-off: 19, first stored at leads (h: fixtures) 122: 9, 128: 1, 147: 5, 150: 1, 168: 1, 171: 1, 173: 1.

### Near kick-off `betfair_exchange`, 2024-25

Per fixture, the latest exchange snapshot at or before kick-off within 120 min; usable means a back price on every outcome with back overround at most 5%. Research inputs, not validated closing labels.

| Price | Fixtures with one | Within 5 min | Within 10 min | p50 age (min) | p75 age (min) | p90 age (min) | p99 age (min) |
|---|---|---|---|---|---|---|---|
| Any exchange quote | 306/306 | 100.0% | 100.0% | 4.4 | 4.4 | 4.4 | 4.4 |
| Usable | 306/306 | 100.0% | 100.0% | 4.4 | 4.4 | 4.4 | 4.4 |

Latest exchange snapshot missing an outcome's back price: 0 fixtures.

### Provider gaps and off-phase snapshots, 2024-25

Recorded gaps: 2.

| Planned | Returned | Next |
|---|---|---|
| 2025-04-15T21:25:38Z | 2025-04-15T18:00:38Z | 2025-04-15T22:25:39Z |
| 2025-05-11T23:25:38Z | 2025-05-11T19:40:37Z | 2025-05-12T06:15:38Z |

Stored snapshot times off the grid's phase (outside their nearest grid time's whole minute): 0.

### Calls, 2024-25

Planned 4,642; stored 4,639; covered by a recorded gap 3; missing 0. Distinct stored snapshot times, any market: 4,642, × 10 units = 46,420 units. `1x2` snapshot times matching no planned call (gap returns, probes): 2. Planned calls that serve only fixtures not yet stored at that time (not listed): 2.

## 2025-26

### Fixtures, 2025-26

306 FINAL fixtures; 306 with a stored `1x2` snapshot at or before kick-off; 0 without.

Listings in stored responses that attached to no event: 3.

| Home | Away | Listed kick-off | Responses |
|---|---|---|---|
| FC St. Pauli | RB Leipzig | 2026-01-10T14:30:00Z | 232 |
| Werder Bremen | TSG Hoffenheim | 2026-01-10T14:30:00Z | 231 |
| Hamburger SV | Bayer Leverkusen | 2026-01-13T19:30:00Z | 204 |

Odds API historical snapshots of these fixtures outside `1x2`:

| Market | Stamped | Snapshots | Fixtures |
|---|---|---|---|
| h2h | after KO | 8 | 7 |
| h2h | before KO | 219 | 23 |
| totals | before KO | 17 | 17 |
| none | after KO | 26 | 26 |
| none | before KO | 8 | 8 |

### Bookmakers, 2025-26

Share of the 306 fixtures with the book in any snapshot at or before kick-off, and of the 72,488 such snapshots.

| Book | Fixtures | Snapshots | Role | Odds API keys aliased to it |
|---|---|---|---|---|
| `betfair_exchange` | 100.0% | 93.6% | exchange | `betfair`, `betfair_ex_eu`, `betfair_ex_uk` |
| `betfair_sportsbook` | 100.0% | 47.0% | UK retail | `betfair_sb_uk` |
| `betvictor` | 100.0% | 68.8% | UK retail | none |
| `betway` | 100.0% | 96.9% | UK retail | none |
| `boylesports` | 100.0% | 93.6% | UK retail | none |
| `casumo` | 100.0% | 93.8% | neither | none |
| `coral` | 100.0% | 92.7% | UK retail | none |
| `grosvenor` | 100.0% | 93.8% | neither | none |
| `ladbrokes` | 100.0% | 93.2% | UK retail | `ladbrokes_uk` |
| `leovegas` | 100.0% | 93.8% | neither | none |
| `matchbook` | 100.0% | 48.5% | exchange | none |
| `paddypower` | 100.0% | 88.9% | UK retail | none |
| `skybet` | 100.0% | 80.9% | UK retail | none |
| `smarkets` | 100.0% | 97.8% | exchange | none |
| `unibet_uk` | 100.0% | 92.0% | UK retail | none |
| `williamhill` | 100.0% | 97.2% | UK retail | none |
| `livescorebet` | 97.1% | 90.9% | neither | none |
| `virginbet` | 97.1% | 90.9% | neither | none |
| `888sport` | 67.3% | 63.9% | UK retail | `sport888` |
| `betfred` | 14.7% | 10.3% | UK retail | `betfred_uk` |
| `pinnacle` | 5.6% | 0.0% | neither | none |

### Leads, 2025-26

Whole-minute decisions per fixture and band, served when a stored snapshot lies in `[t − max age, t]` at its actual time. Unserved decisions are split into those before the fixture's first stored snapshot (not yet listed), those a recorded gap's planned snapshot would have served, and the rest. The last column needs `betfair_exchange` and a UK retail book in the same snapshot.

| Band | Decisions | Served | Unserved: not listed | Unserved: gap | Unserved: other | Fixtures fully served | Served, exchange + retail |
|---|---|---|---|---|---|---|---|
| KO−1m..KO−120m, max age 10m | 36,720 | 100.0% | 0 | 0 | 0 | 306/306 | 100.0% |
| KO−121m..KO−1440m, max age 60m | 403,920 | 100.0% | 0 | 0 | 1 | 305/306 | 99.9% |
| KO−1441m..KO−10080m, max age 360m | 2,643,840 | 99.3% | 16,476 | 1,440 | 0 | 281/306 | 99.2% |

Fixtures first stored less than 174 h before kick-off: 19, first stored at leads (h: fixtures) 126: 1, 145: 5, 148: 1, 151: 1, 158: 8, 168: 1, 169: 1, 171: 1.

### Near kick-off `betfair_exchange`, 2025-26

Per fixture, the latest exchange snapshot at or before kick-off within 120 min, counted and aged only: this season is sealed ([ADR-0021](../docs/adr/0021-sealed-holdout-and-one-shot-gate.md)), so no price-derived statistic is reported. Research inputs, not validated closing labels.

| Price | Fixtures with one | Within 5 min | Within 10 min | p50 age (min) | p75 age (min) | p90 age (min) | p99 age (min) |
|---|---|---|---|---|---|---|---|
| Any exchange quote | 306/306 | 100.0% | 100.0% | 4.4 | 4.4 | 4.4 | 4.4 |

### Provider gaps and off-phase snapshots, 2025-26

Recorded gaps: 1.

| Planned | Returned | Next |
|---|---|---|
| 2025-10-20T09:25:38Z | 2025-10-20T06:45:38Z | 2025-10-20T09:29:52Z |

Stored snapshot times off the grid's phase (outside their nearest grid time's whole minute): 1 (2026-05-08T04:26:09Z).

### Calls, 2025-26

Planned 4,752; stored 4,751; covered by a recorded gap 1; missing 0. Distinct stored snapshot times, any market: 4,753, × 10 units = 47,530 units. `1x2` snapshot times matching no planned call (gap returns, probes): 2. Planned calls that serve only fixtures not yet stored at that time (not listed): 0.

