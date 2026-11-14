# Audit: `odds-api-nba-2024-26.yaml`

Coverage of the NBA Odds API historical run (`h2h`, `regions=uk`, seasons 2024-25 and 2025-26), read from the production database on 2026-11-14 after the run finished at 00:41Z (2024-25 from 2026-11-13T20:06Z to 21:56Z, 2025-26 to 2026-11-14T00:41Z; both exited 0). The tables below each season come from `uv run python scripts/audit_backfill.py --manifest backfills/odds-api-nba-2024-26.yaml` ([audit a finished manifest](../wiki/collection/howto-backfill-odds-api-history.md#audit-a-finished-manifest)); the figures in the Findings come from the same read-only audit and the run's log (`logs/backfill/nba-2024-26.log`, outside git). No NBA window is in `odds_analytics.holdout`, so no season is sealed and every section reports prices. The near-tip-off Betfair Exchange figures are research inputs; whether they can stand in for a scoring close is [#730](https://github.com/Acusick1/odds/issues/730)'s result.

## Summary

| | 2024-25 | 2025-26 |
|---|---|---|
| FINAL fixtures / with a stored snapshot | 1,321 / 1,321 | 1,322 / 1,322 |
| Unmatched listings | 10 | 7 |
| KO−2h band (10 min) served | 99.9% | 100.0% |
| KO−8h band (60 min) served | 99.9% | 100.0% |
| Exchange price within 5 min of tip-off | 99.5% | 100.0% |
| Exchange price within 10 min of tip-off | 100.0% | 100.0% |
| Fixtures with no exchange price near tip-off | 0 | 0 |
| Provider gaps recorded | 3 | 0 |
| Off-phase snapshots | 1 | 0 |
| Planned calls / stored / covered by a gap / missing | 7,928 / 7,916 / 11 / 1 | 8,189 / 8,189 / 0 / 0 |
| Calls paid in the log / units | 7,919 / 79,190 | 8,189 / 81,890 |
| Distinct stored snapshot times × 10 units | 79,180 | 81,900 |

## Findings

- **No fixture lacks a snapshot, and the bands are served.** All 1,321 and 1,322 FINAL fixtures have a stored `h2h` snapshot at or before tip-off. The KO−2h band serves 99.9% (2024-25) and 100.0% (2025-26) of decisions, the KO−8h band 99.9% and 100.0%; 5 and 2 fixtures of 2024-25 are not fully served in the two bands, 0 and 1 of 2025-26.
- **Fixture horizon.** 41 (2024-25) and 62 (2025-26) fixtures are first stored less than 9 h before tip-off: at 8 h, 29 and 40; at 9 h, 12 and 21; 1 of 2025-26 at 6 h. The API lists only about the next 10 h of games ([#787](https://github.com/Acusick1/odds/issues/787)). In 2025-26 this leaves 146 KO−8h decisions with no snapshot that no purchase can fill; 2024-25's KO−8h band has none not listed.
- **Betfair Exchange is on every fixture and sits 4.4 min before tip-off.** `betfair_exchange` is on 100.0% of fixtures in both seasons, but on 69.5% (2024-25) and 77.4% (2025-26) of the snapshots stored at or before tip-off, so it is listed only part of the way in (the lead at which it first appears is not tabulated). The latest exchange snapshot within 120 min of tip-off is 4.4 min old at the 50th to 99th percentile in both seasons: 99.5% of 2024-25 fixtures have one within 5 min and all have one within 10 min; 100.0% of 2025-26 fixtures within 5 min. Every such price is usable (a back price on every outcome, overround at most 5%) and none misses an outcome. The 5-minute share is the run's 10-minute cadence in the last two hours, as in [MLB](odds-api-mlb-2024-26.audit.md) `(inferred)`.
- **Provider gaps.** 2024-25 records 3: 2024-11-30T02:45:38Z (returned 02:35:38Z, next 02:50:38Z), 2025-01-19T01:35:38Z (returned 01:25:38Z, next 01:40:37Z) and 2025-04-15T18:25:38Z (returned 18:00:38Z, next 22:25:39Z). The third is a 4 h 25 min hole in the provider's history: the run skipped its 8 following planned calls (19:25:38Z to 22:15:38Z) as covered by the gap record. The gap column of the band tables counts 40 KO−2h and 50 KO−8h decisions; the remaining 76 and 304 unserved decisions of 2024-25 have no other recorded cause and most are plausibly the 2025-04-15 hole, which the gap column counts only within a band's max age `(inferred)`. 2025-26 has no gap.
- **Missing call.** One planned call stored nothing: 2024-11-30T02:35:38Z. The log shows it paid (2 fixtures, 10 units), and the 02:45:38Z call returned that same snapshot as a provider gap, so the time is the gap's returned snapshot (the audit counts 2 `h2h` snapshot times matching no planned call in 2024-25, and 1 in 2025-26, the probe at 2025-12-13T15:55:38Z `(inferred)`).
- **Calls against the plan.** The plan gave 16,117 calls and 161,170 units (7,928 and 8,189 calls; 79,280 and 81,890 units). The log has 16,108 `fetched` lines and 161,080 units (79,190 and 81,890): 9 planned calls were skipped, 8 covered by the 2025-04-15 gap record and 1 the probe 2025-01-18T15:55:38Z already stored, which is 90 units under the plan. Distinct stored snapshot times × 10 units are the same 161,080 (79,180 and 81,900).
- **Quota drew down 1,130 units more than the run billed.** The log's first fetch leaves 3,899,030, so the key held 3,899,040 at the start, and the last leaves 3,736,830: 162,210 units, against 161,080 billed. The extra 1,130 units are 91 calls on 2024-10-23 to 2024-10-25 (25, 35 and 31 calls) whose quota fell by 20 to 50 units while each `fetched` line says 10; no other call differs, and the 2024-25 summary's 3,818,720 carries into 2025-26's first call. Another consumer of the same key during those minutes is the likely cause `(inferred)`; the log does not name it.
- **Books outside `UK_RETAIL_BOOKMAKERS`.** `livescorebet` and `virginbet` (66.4% of 2024-25 fixtures, 100.0% of 2025-26), `leovegas` (66.2%, 99.8%), `casumo` and `grosvenor` (43.8% of 2024-25 only) are stored under their own keys and are counted neither as UK retail nor as exchanges. `smarkets` (91.3%, 99.7%) and `matchbook` (94.9%, 94.7%) are exchanges. `pinnacle` is on 0.5% and 0.2% of fixtures and 0.0% of snapshots, so only the probe snapshots carry it `(inferred)`. `skybet` appears only in 2025-26 (84.9% of fixtures) and `betvictor` is thin in 2024-25 (1.5%). `betway`, `paddypower`, `williamhill`, `coral` and `ladbrokes` are on 99.9 to 100.0% of fixtures in both seasons.
- **Unmatched listings.** 10 (2024-25) and 7 (2025-26) listings in stored responses attached to no event, the same as the summaries' dropped games (10 and 7); the largest are New York Knicks v Detroit Pistons at 2025-04-19T17:10Z (32 responses) and Oklahoma City Thunder v New York Knicks at 2026-06-04T00:40Z (57). The listed kick-offs differ from the stored events' (rescheduled start times) `(inferred)`.
- **A draw outcome from `boylesports`.** 45 `unmatched_outcome` warnings, all in 2024-25, say `boylesports` returned an outcome named `Draw` on an NBA `h2h` market: 2 for Chicago Bulls v Orlando Magic (event 4987), 8 for Portland Trail Blazers v Dallas Mavericks, 34 for Phoenix Suns v Denver Nuggets and 1 for Los Angeles Lakers v Atlanta Hawks. The matcher logged each and stored no price for the outcome.
- **Snapshot key conflicts.** The 2026-04-10T22:15:38Z call returned 24 fixtures where its neighbours returned 15, and 9 of its snapshots (`snapshot_key_conflict`, time 22:15:40Z) differed from a snapshot already stored at that key; the stored one was kept in each case. The audit counts no off-phase snapshot in 2025-26.
- **Off-phase snapshot.** 2025-02-03T15:56:33Z, returned by the 15:55:38Z call (2024-25).
- **Rate limit.** One HTTP 429 (`api_rate_limited`, attempt 1, 2026-11-14T00:00:08Z) was retried and the run carried on; the log has no stop, no `SNAPSHOT-MISMATCH` and no resume, and one `discord_alert_sent` at 2026-11-13T21:56:15Z, 3 s before 2024-25 exited.
- **Non-`h2h` snapshots.** `totals` is stored in 6 (2024-25) and 10 (2025-26) snapshots before tip-off, though the manifest requests only `h2h`. Snapshots of these fixtures with no market stored: before tip-off 5,792 (570 fixtures) and 3,033 (60), after 103 and 138.

## 2024-25

### Fixtures, 2024-25

1321 FINAL fixtures; 1321 with a stored `h2h` snapshot at or before kick-off; 0 without.

Listings in stored responses that attached to no event: 10.

| Home | Away | Listed kick-off | Responses |
|---|---|---|---|
| New York Knicks | Atlanta Hawks | 2024-12-10T22:00:00Z | 1 |
| Brooklyn Nets | Oklahoma City Thunder | 2025-02-26T00:30:00Z | 2 |
| New York Knicks | Detroit Pistons | 2025-04-19T17:10:00Z | 32 |
| Indiana Pacers | Milwaukee Bucks | 2025-04-19T23:10:00Z | 6 |
| Miami Heat | Cleveland Cavaliers | 2025-04-29T02:00:00Z | 9 |
| Milwaukee Bucks | Indiana Pacers | 2025-05-03T02:00:00Z | 2 |
| Cleveland Cavaliers | Indiana Pacers | 2025-05-03T23:00:00Z | 6 |
| Denver Nuggets | Los Angeles Clippers | 2025-05-04T01:35:00Z | 16 |
| Cleveland Cavaliers | Indiana Pacers | 2025-05-04T17:00:00Z | 32 |
| Cleveland Cavaliers | Indiana Pacers | 2025-05-05T23:00:00Z | 6 |

Odds API historical snapshots of these fixtures outside `h2h`:

| Market | Stamped | Snapshots | Fixtures |
|---|---|---|---|
| totals | before KO | 6 | 6 |
| none | after KO | 103 | 89 |
| none | before KO | 5792 | 570 |

### Bookmakers, 2024-25

Share of the 1321 fixtures with the book in any snapshot at or before kick-off, and of the 87,233 such snapshots.

| Book | Fixtures | Snapshots | Role | Odds API keys aliased to it |
|---|---|---|---|---|
| `betfair_exchange` | 100.0% | 69.5% | exchange | `betfair`, `betfair_ex_eu`, `betfair_ex_uk` |
| `betway` | 100.0% | 74.1% | UK retail | none |
| `paddypower` | 100.0% | 77.6% | UK retail | none |
| `williamhill` | 100.0% | 77.0% | UK retail | none |
| `coral` | 99.9% | 63.3% | UK retail | none |
| `ladbrokes` | 99.9% | 63.2% | UK retail | `ladbrokes_uk` |
| `888sport` | 94.9% | 78.1% | UK retail | `sport888` |
| `matchbook` | 94.9% | 48.6% | exchange | none |
| `smarkets` | 91.3% | 67.6% | exchange | none |
| `boylesports` | 80.0% | 39.2% | UK retail | none |
| `betfair_sportsbook` | 74.2% | 34.9% | UK retail | `betfair_sb_uk` |
| `livescorebet` | 66.4% | 45.4% | neither | none |
| `virginbet` | 66.4% | 45.4% | neither | none |
| `unibet_uk` | 66.2% | 45.4% | UK retail | none |
| `leovegas` | 66.2% | 45.3% | neither | none |
| `casumo` | 43.8% | 30.8% | neither | none |
| `grosvenor` | 43.8% | 30.8% | neither | none |
| `betvictor` | 1.5% | 0.7% | UK retail | none |
| `pinnacle` | 0.5% | 0.0% | neither | none |

### Leads, 2024-25

Whole-minute decisions per fixture and band, served when a stored snapshot lies in `[t − max age, t]` at its actual time. Unserved decisions are split into those before the fixture's first stored snapshot (not yet listed), those a recorded gap's planned snapshot would have served, and the rest. The last column needs `betfair_exchange` and a UK retail book in the same snapshot.

| Band | Decisions | Served | Unserved: not listed | Unserved: gap | Unserved: other | Fixtures fully served | Served, exchange + retail |
|---|---|---|---|---|---|---|---|
| KO−1m..KO−120m, max age 10m | 158,520 | 99.9% | 0 | 40 | 76 | 1316/1321 | 99.9% |
| KO−121m..KO−480m, max age 60m | 475,560 | 99.9% | 0 | 50 | 304 | 1319/1321 | 99.9% |

Fixtures first stored less than 9 h before kick-off: 41, first stored at leads (h: fixtures) 8: 29, 9: 12.

### Near kick-off `betfair_exchange`, 2024-25

Per fixture, the latest exchange snapshot at or before kick-off within 120 min; usable means a back price on every outcome with back overround at most 5%. Research inputs, not validated closing labels.

| Price | Fixtures with one | Within 5 min | Within 10 min | p50 age (min) | p75 age (min) | p90 age (min) | p99 age (min) |
|---|---|---|---|---|---|---|---|
| Any exchange quote | 1321/1321 | 99.5% | 100.0% | 4.4 | 4.4 | 4.4 | 4.4 |
| Usable | 1321/1321 | 99.5% | 100.0% | 4.4 | 4.4 | 4.4 | 4.4 |

Latest exchange snapshot missing an outcome's back price: 0 fixtures.

### Provider gaps and off-phase snapshots, 2024-25

Recorded gaps: 3.

| Planned | Returned | Next |
|---|---|---|
| 2024-11-30T02:45:38Z | 2024-11-30T02:35:38Z | 2024-11-30T02:50:38Z |
| 2025-01-19T01:35:38Z | 2025-01-19T01:25:38Z | 2025-01-19T01:40:37Z |
| 2025-04-15T18:25:38Z | 2025-04-15T18:00:38Z | 2025-04-15T22:25:39Z |

Stored snapshot times off the grid's phase (outside their nearest grid time's whole minute): 1 (2025-02-03T15:56:33Z).

### Calls, 2024-25

Planned 7,928; stored 7,916; covered by a recorded gap 11; missing 1. Distinct stored snapshot times, any market: 7,918, × 10 units = 79,180 units. `h2h` snapshot times matching no planned call (gap returns, probes): 2. Planned calls that serve only fixtures not yet stored at that time (not listed): 0.

Missing calls: 2024-11-30T02:35:38Z

## 2025-26

### Fixtures, 2025-26

1322 FINAL fixtures; 1322 with a stored `h2h` snapshot at or before kick-off; 0 without.

Listings in stored responses that attached to no event: 7.

| Home | Away | Listed kick-off | Responses |
|---|---|---|---|
| Charlotte Hornets | San Antonio Spurs | 2026-01-31T20:10:00Z | 3 |
| San Antonio Spurs | Orlando Magic | 2026-02-01T21:10:00Z | 22 |
| Charlotte Hornets | New Orleans Pelicans | 2026-02-03T00:10:00Z | 26 |
| Memphis Grizzlies | Minnesota Timberwolves | 2026-02-03T02:40:00Z | 28 |
| Boston Celtics | Minnesota Timberwolves | 2026-04-01T00:13:00Z | 2 |
| Charlotte Hornets | Miami Heat | 2026-04-15T23:40:00Z | 2 |
| Oklahoma City Thunder | New York Knicks | 2026-06-04T00:40:00Z | 57 |

Odds API historical snapshots of these fixtures outside `h2h`:

| Market | Stamped | Snapshots | Fixtures |
|---|---|---|---|
| totals | before KO | 10 | 10 |
| none | after KO | 138 | 105 |
| none | before KO | 3033 | 60 |

### Bookmakers, 2025-26

Share of the 1322 fixtures with the book in any snapshot at or before kick-off, and of the 72,273 such snapshots.

| Book | Fixtures | Snapshots | Role | Odds API keys aliased to it |
|---|---|---|---|---|
| `betfair_exchange` | 100.0% | 77.4% | exchange | `betfair`, `betfair_ex_eu`, `betfair_ex_uk` |
| `betway` | 100.0% | 64.1% | UK retail | none |
| `boylesports` | 100.0% | 82.3% | UK retail | none |
| `coral` | 100.0% | 73.2% | UK retail | none |
| `ladbrokes` | 100.0% | 73.2% | UK retail | `ladbrokes_uk` |
| `livescorebet` | 100.0% | 81.0% | neither | none |
| `virginbet` | 100.0% | 81.0% | neither | none |
| `williamhill` | 100.0% | 86.8% | UK retail | none |
| `betvictor` | 99.9% | 55.2% | UK retail | none |
| `paddypower` | 99.9% | 90.4% | UK retail | none |
| `leovegas` | 99.8% | 80.6% | neither | none |
| `smarkets` | 99.7% | 79.9% | exchange | none |
| `unibet_uk` | 98.9% | 92.7% | UK retail | none |
| `betfair_sportsbook` | 96.4% | 67.5% | UK retail | `betfair_sb_uk` |
| `matchbook` | 94.7% | 65.7% | exchange | none |
| `skybet` | 84.9% | 76.7% | UK retail | none |
| `888sport` | 77.2% | 71.0% | UK retail | `sport888` |
| `pinnacle` | 0.2% | 0.0% | neither | none |

### Leads, 2025-26

Whole-minute decisions per fixture and band, served when a stored snapshot lies in `[t − max age, t]` at its actual time. Unserved decisions are split into those before the fixture's first stored snapshot (not yet listed), those a recorded gap's planned snapshot would have served, and the rest. The last column needs `betfair_exchange` and a UK retail book in the same snapshot.

| Band | Decisions | Served | Unserved: not listed | Unserved: gap | Unserved: other | Fixtures fully served | Served, exchange + retail |
|---|---|---|---|---|---|---|---|
| KO−1m..KO−120m, max age 10m | 158,640 | 100.0% | 0 | 0 | 0 | 1322/1322 | 100.0% |
| KO−121m..KO−480m, max age 60m | 475,920 | 100.0% | 146 | 0 | 0 | 1321/1322 | 99.9% |

Fixtures first stored less than 9 h before kick-off: 62, first stored at leads (h: fixtures) 6: 1, 8: 40, 9: 21.

### Near kick-off `betfair_exchange`, 2025-26

Per fixture, the latest exchange snapshot at or before kick-off within 120 min; usable means a back price on every outcome with back overround at most 5%. Research inputs, not validated closing labels.

| Price | Fixtures with one | Within 5 min | Within 10 min | p50 age (min) | p75 age (min) | p90 age (min) | p99 age (min) |
|---|---|---|---|---|---|---|---|
| Any exchange quote | 1322/1322 | 100.0% | 100.0% | 4.4 | 4.4 | 4.4 | 4.4 |
| Usable | 1322/1322 | 100.0% | 100.0% | 4.4 | 4.4 | 4.4 | 4.4 |

Latest exchange snapshot missing an outcome's back price: 0 fixtures.

### Provider gaps and off-phase snapshots, 2025-26

Recorded gaps: 0.

Stored snapshot times off the grid's phase (outside their nearest grid time's whole minute): 0.

### Calls, 2025-26

Planned 8,189; stored 8,189; covered by a recorded gap 0; missing 0. Distinct stored snapshot times, any market: 8,190, × 10 units = 81,900 units. `h2h` snapshot times matching no planned call (gap returns, probes): 1. Planned calls that serve only fixtures not yet stored at that time (not listed): 0.

