# Audit: `odds-api-mlb-2024-26.yaml`

Coverage of the MLB Odds API historical run (`h2h`, `regions=uk`, calendar-year windows 2024, 2025 and 2026), read from the production database on 2026-11-13 after the run finished at 12:58Z. The tables below each season come from `uv run python scripts/audit_backfill.py --manifest backfills/odds-api-mlb-2024-26.yaml` ([audit a finished manifest](../wiki/collection/howto-backfill-odds-api-history.md#audit-a-finished-manifest)); the figures in the Findings come from further read-only queries and the run's log (`logs/backfill/mlb-2024-26.log`). No MLB window is in `odds_analytics.holdout` (its windows are the five leagues' 2025-26), so no season is sealed and every section reports prices. The near-kick-off Betfair Exchange figures are research inputs: MLB has no FDUK close, so a scoring close would depend on [#730](https://github.com/Acusick1/odds/issues/730)'s NBA sample ([#787](https://github.com/Acusick1/odds/issues/787)).

## Summary

| | 2024 | 2025 | 2026 |
|---|---|---|---|
| FINAL fixtures / with a stored snapshot | 2,472 / 2,472 | 2,477 / 2,477 | 2,470 / 2,469 (the All-Star Game) |
| Doubleheader games / with a stored snapshot | 58 / 58 | 56 / 56 | 50 / 50 |
| Unmatched listings | 135 | 139 | 101 |
| KO−2h band (10 min) served | 99.8% | 99.6% | 99.9% |
| KO−8h band (60 min) served | 99.7% | 99.6% | 99.9% |
| Exchange price within 5 min of first pitch | 1,373 / 2,472 | 1,327 / 2,477 | 1,399 / 2,470 |
| Exchange price within 10 min of first pitch | 2,454 / 2,472 | 2,463 / 2,477 | 2,456 / 2,470 |
| Fixtures with no exchange price at any lead | 1 | 6 | 5 |
| Provider gaps recorded | 2 | 2 | 0 |
| Off-phase snapshots | 1 | 3 | 3 |
| Planned calls / stored / covered by a gap / missing | 10,768 / 10,735 / 3 / 30 | 10,734 / 10,717 / 16 / 1 | 10,594 / 10,572 / 0 / 22 |
| Distinct stored snapshot times × 10 units | 107,670 | 107,200 | 105,750 |

## Findings

- **Exchange coverage does not thin from 2025-04-26.** From 2025-04-26 the exchange is on 2,091 of 2,097 games with a stored price, and on 98.2 to 99.6% of the snapshots stored 0 to 9 h before first pitch, the same as before it (2024: 99.3 to 99.9%; 2025 before the 26th: 97.8 to 100.0%; 2026: 99.4 to 99.7%). The exchange first appears a median 11 to 13 h before first pitch (10th to 90th percentile 8 to 23 h) for games with an exchange price, so only 11 to 17% of snapshots stored 9 h or more ahead carry it. The one change near that date is 2025-04-26 itself: no snapshot from 09:05Z to 17:15Z carries the exchange (20 snapshots, 16 to 17 listed games each), and the exchange is present from the next snapshot after 17:15Z. This audit reads only the Odds API's historical `uk` region; it does not test the exchange's own listing.

  | Share of stored snapshots with the exchange, by hours before first pitch | 0-1 | 1-2 | 2-3 | 3-4 | 4-5 | 5-6 | 6-7 | 7-8 | 8-9 | 9+ |
  |---|---|---|---|---|---|---|---|---|---|---|
  | 2024 | 99.9 | 99.9 | 99.9 | 99.9 | 99.8 | 99.7 | 99.6 | 99.4 | 99.3 | 17.0 |
  | 2025 to 2025-04-25 | 100.0 | 100.0 | 100.0 | 100.0 | 99.8 | 98.9 | 98.7 | 98.3 | 97.8 | 11.1 |
  | 2025 from 2025-04-26 | 99.6 | 99.6 | 99.5 | 99.4 | 99.2 | 99.3 | 98.9 | 98.8 | 98.2 | 12.3 |
  | 2026 | 99.7 | 99.6 | 99.7 | 99.6 | 99.7 | 99.5 | 99.4 | 99.5 | 99.5 | 12.1 |

- **The 5-minute share is the purchase, not the exchange.** The run bought a 10-minute cadence in the last two hours (the band's max age), so a fixture's latest snapshot is 4.4 min old only when a call happens to fall there, and 9.4 min old otherwise: 4,099 of 7,419 fixtures (55%) have an exchange price within 5 min and 4,112 have a snapshot of any kind within 5 min, so the exchange is missing from only 13 of those snapshots. The share is the same for first pitches on the 10-minute and the 5-minute mark (56% and 53%). Within 10 min the share is 99.3 to 99.4% in all three seasons. Putting an exchange price at most 4.4 min before first pitch on the 3,307 fixtures with no snapshot that near would cost one call at the 5-minute grid time before first pitch, deduplicated across games and net of stored times: 718 (2024), 777 (2025) and 736 (2026) calls, 22,310 units `(inferred)` from the stored data, assuming the exchange is in those snapshots as it is in 99.7% of near-kick-off ones.
- **Doubleheaders.** There are 82 doubleheaders (29, 28 and 25), 164 games (58, 56 and 50), and all 164 have an event with stored snapshots: none is unmatched. The listing is the problem: when the API lists a pair as one row (`1 rows for 2 games`, twice `3 rows for 2 games`) the matcher logs `mlb_game_unresolved` with reason `ambiguous_doubleheader` and attaches nothing. The run's log has 1,058 such warnings over at least 39 series-dates (19 in 2024, 10 in 2025, 10 in 2026), plus 248 `no_game` and 116 `start_too_far`. Games on doubleheaders are 2.2% of fixtures but 16 of the 20 fixtures not fully served in 2024's KO−2h band, whose coverage is 91.4% on doubleheaders against 99.99% elsewhere (2025: 97.6% against 99.7%; 2026: 97.6% against 99.96%). 33 planned calls stored nothing because every fixture they serve was a doubleheader game with an unresolved listing, and the 2024-09-30 Mets–Braves doubleheader accounts for 24 (09:05Z to 20:45Z); its two games (17:10Z and 20:50Z) hold 60 and 73 snapshots from other calls, and 150 KO−2h decisions are unserved.
- **Missing calls.** 53 planned calls stored nothing: 30 in 2024 (2024-04-04T08:05, T09:05 and T10:05Z, Mets–Tigers; 2024-07-28T01:55, T02:05 and T02:15Z, Giants–Rockies; 24 on 2024-09-30, Mets–Braves), 1 in 2025 (2025-05-14T08:00:38Z, Orioles–Twins) and 22 in 2026 (2026-04-26T20:50 and T21:00Z, Mets–Rockies; 2026-08-17T09:35:38Z, whose cause is not logged as ambiguous; and 19 on 2026-07-14 for the All-Star break, below). 33 are doubleheader listings; the 2026-08-17 call and the break's 19 are not. The audit prints only the first 20 missing calls of a season.
- **All-Star Game.** The 2026-07-15T00:00Z National League v American League event has no stored snapshot. The 19 calls from 2026-07-14T15:55:38Z to 23:55:38Z returned no games, stored nothing and bought nothing: no listed game, so no price. Treat the event as outside MLB fixtures `(inferred)`; its listed kick-off is a placeholder.
- **Billing and units that bought nothing new.** The run's log has 32,187 paid requests at 10 units each, 321,870 units, against 32,096 planned calls and 32,062 distinct stored snapshot times (320,620 units). 164 requests, 1,640 units, left no snapshot at their planned time: the 53 missing calls were requested 143 times (the three 2024-04-04 calls 25 times each, because each rerun of the 2024 season requeued a call that had stored nothing; 90 repeats on missing calls in all) and the 2024-07-21T00:25:38Z call, a provider gap, 22 times (21 repeats, 210 units, across 20 resumes after `SNAPSHOT-MISMATCH` stops before gaps were recorded and skipped). 111 requests in total were repeats of a time already requested (109 in 2024, 2 in 2026). The two probe snapshots (2024-07-20T15:55:38Z and 2026-07-18T15:55:38Z) are outside these counts.
- **Calls with no exchange.** 44 stored snapshot times carry no exchange price, each shared by at least five fixtures: 10 in 2024 (2024-05-22, 2024-06-09, 2024-07-04, 2024-09-15), 29 in 2025 (20 on 2025-04-26, 4 on 2025-04-06, 4 on 2025-04-27, 1 on 2025-05-17) and 5 in 2026 (2026-08-06, 2026-08-22). Apart from 2025-04-26, each is 5 to 8 h before the nearest first pitch, except 2024-09-15T14:55Z, 1 h 14 min before. A run of such calls stopped the 2025 run at the default limit (`10 consecutive sparse responses, the last lacks betfair_exchange on h2h`); the run then used `--max-consecutive-gaps 100`. 12 fixtures have no exchange price at any lead (1, 6 and 5).
- **Provider gaps.** 2025-04-15T18:35:38Z returned the 18:00:38Z snapshot and the next call stored was 22:25:39Z, so no snapshot lies between 18:00:38Z and 22:25:39Z and 12 fixtures that evening lose 920 KO−2h and 1,540 KO−8h decisions. The 2025-10-20 gap (17:05:38Z planned, next 19:32:57Z, 16 planned calls covered by gaps in all 2025) and the 2024 gaps (2024-07-21T00:25:38Z, 2024-08-03T16:20:38Z) cost far less; 2026 has none. The band tables' gap column counts only decisions within a band's max age after a planned time, so most of these decisions appear under "other".
- **Fixture horizon.** 88 (2024), 66 (2025) and 58 (2026) fixtures are first stored less than 9 h before first pitch, 59, 42 and 41 of them at 8 h. The API lists only about the next 10 h of games ([#787](https://github.com/Acusick1/odds/issues/787)), which leaves 1,518, 1,121 and 376 KO−8h decisions without a snapshot that no purchase can fill.
- **Unmatched listings.** 135, 139 and 101 listings in stored responses attached to no event, (responses per listing in the tables below): 43, 25 and 28 on dates with a doubleheader, 84, 67 and 52 on dates with no event for the pairing (postponed games listed at their original dates), and 8, 47 and 21 on dates with one game (rescheduled start times). The classification is by Eastern date and pairing `(inferred)`.
- **Books outside `UK_RETAIL_BOOKMAKERS`.** `casumo`, `grosvenor`, `leovegas`, `livescorebet`, `virginbet` (all seasons), `mrgreen` (2024) and `betano_uk` (42% of 2026 fixtures) are stored under their own keys and counted neither as UK retail nor as exchanges. `smarkets` and `matchbook` are exchanges; `smarkets` is on 35.9% of 2025 fixtures and 100% of 2026's.
- **Betfred.** `betfred` is on 18.5% of 2026 fixtures and none in 2024 or 2025. **Pinnacle** appears only in the two probe snapshots, which named it.
- **Off-phase snapshots.** 2024-08-15T18:26:25Z; 2025-07-03T00:56:31Z, 2025-10-20T16:31:34Z (the gap's returned snapshot) and 2025-10-20T20:06:49Z; 2026-04-03T15:01:03Z, T18:01:01Z and T19:01:01Z. The 2026-04-03 times leave 7 KO−2h and 10 KO−8h fixtures one decision minute short each.

## 2024

### Fixtures, 2024

2472 FINAL fixtures; 2472 with a stored `h2h` snapshot at or before kick-off; 0 without.

Listings in stored responses that attached to no event: 135.

| Home | Away | Listed kick-off | Responses |
|---|---|---|---|
| New York Mets | Detroit Tigers | 2024-04-02T23:10:00Z | 81 |
| New York Mets | Detroit Tigers | 2024-04-03T00:45:00Z | 3 |
| New York Mets | Detroit Tigers | 2024-04-03T01:00:00Z | 2 |
| New York Mets | Detroit Tigers | 2024-04-03T01:15:00Z | 2 |
| Philadelphia Phillies | Cincinnati Reds | 2024-04-03T17:05:00Z | 37 |
| Chicago White Sox | Atlanta Braves | 2024-04-03T18:10:00Z | 17 |
| New York Mets | Detroit Tigers | 2024-04-03T23:10:00Z | 46 |
| New York Mets | Detroit Tigers | 2024-04-03T23:30:00Z | 6 |
| New York Mets | Detroit Tigers | 2024-04-03T23:45:00Z | 1 |
| New York Mets | Detroit Tigers | 2024-04-04T00:00:00Z | 3 |
| New York Mets | Detroit Tigers | 2024-04-04T16:10:00Z | 13 |
| New York Mets | Detroit Tigers | 2024-04-04T20:10:00Z | 2 |
| New York Mets | Detroit Tigers | 2024-04-04T20:20:00Z | 11 |
| Minnesota Twins | Cleveland Guardians | 2024-04-07T18:10:00Z | 36 |
| Atlanta Braves | New York Mets | 2024-04-10T23:20:00Z | 41 |
| Cincinnati Reds | Milwaukee Brewers | 2024-04-11T17:10:00Z | 36 |
| Detroit Tigers | Minnesota Twins | 2024-04-11T17:10:00Z | 26 |
| Cleveland Guardians | New York Yankees | 2024-04-12T23:10:00Z | 20 |
| Detroit Tigers | Minnesota Twins | 2024-04-13T16:45:00Z | 20 |
| Detroit Tigers | Minnesota Twins | 2024-04-13T17:10:00Z | 15 |
| Detroit Tigers | Minnesota Twins | 2024-04-13T21:00:00Z | 1 |
| Detroit Tigers | Minnesota Twins | 2024-04-13T21:05:33Z | 14 |
| Detroit Tigers | Minnesota Twins | 2024-04-13T21:15:00Z | 1 |
| Detroit Tigers | Minnesota Twins | 2024-04-13T21:55:00Z | 1 |
| Detroit Tigers | Minnesota Twins | 2024-04-13T22:55:00Z | 1 |
| Chicago White Sox | Kansas City Royals | 2024-04-16T23:40:00Z | 39 |
| Chicago White Sox | Kansas City Royals | 2024-04-17T18:10:00Z | 17 |
| Chicago White Sox | Kansas City Royals | 2024-04-17T21:10:00Z | 13 |
| Chicago Cubs | Miami Marlins | 2024-04-18T23:40:00Z | 25 |
| Colorado Rockies | Seattle Mariners | 2024-04-20T00:40:00Z | 43 |
| Detroit Tigers | St. Louis Cardinals | 2024-04-29T22:40:00Z | 27 |
| Detroit Tigers | St. Louis Cardinals | 2024-04-30T22:40:00Z | 20 |
| St. Louis Cardinals | New York Mets | 2024-05-08T08:15:00Z | 2 |
| St. Louis Cardinals | New York Mets | 2024-05-08T17:15:00Z | 41 |
| St. Louis Cardinals | New York Mets | 2024-05-08T17:16:00Z | 7 |
| St. Louis Cardinals | New York Mets | 2024-05-08T17:56:00Z | 12 |
| St. Louis Cardinals | New York Mets | 2024-05-08T18:16:00Z | 5 |
| St. Louis Cardinals | New York Mets | 2024-05-08T19:16:00Z | 7 |
| St. Louis Cardinals | New York Mets | 2024-05-08T20:16:00Z | 2 |
| Oakland Athletics | Texas Rangers | 2024-05-08T22:37:00Z | 4 |
| Oakland Athletics | Texas Rangers | 2024-05-08T22:38:00Z | 11 |
| Chicago White Sox | Washington Nationals | 2024-05-13T23:40:00Z | 42 |
| Baltimore Orioles | Toronto Blue Jays | 2024-05-14T22:35:00Z | 73 |
| Chicago White Sox | Washington Nationals | 2024-05-15T00:04:21Z | 8 |
| Chicago White Sox | Washington Nationals | 2024-05-15T00:05:00Z | 4 |
| Atlanta Braves | San Diego Padres | 2024-05-18T23:15:00Z | 92 |
| Atlanta Braves | San Diego Padres | 2024-05-18T23:45:00Z | 3 |
| Atlanta Braves | San Diego Padres | 2024-05-19T00:30:00Z | 6 |
| Atlanta Braves | San Diego Padres | 2024-05-19T00:50:00Z | 1 |
| Atlanta Braves | San Diego Padres | 2024-05-19T01:00:00Z | 1 |
| St. Louis Cardinals | Chicago Cubs | 2024-05-25T00:15:00Z | 53 |
| St. Louis Cardinals | Chicago Cubs | 2024-05-25T00:45:00Z | 4 |
| St. Louis Cardinals | Chicago Cubs | 2024-05-25T00:55:00Z | 5 |
| St. Louis Cardinals | Chicago Cubs | 2024-05-25T01:20:00Z | 1 |
| St. Louis Cardinals | Chicago Cubs | 2024-05-25T01:35:00Z | 1 |
| St. Louis Cardinals | Chicago Cubs | 2024-05-25T01:50:00Z | 1 |
| New York Mets | Los Angeles Dodgers | 2024-05-27T20:10:00Z | 46 |
| Detroit Tigers | Pittsburgh Pirates | 2024-05-28T22:40:00Z | 68 |
| New York Mets | Los Angeles Dodgers | 2024-05-28T23:40:00Z | 3 |
| New York Mets | Los Angeles Dodgers | 2024-05-28T23:41:42Z | 12 |
| New York Mets | Los Angeles Dodgers | 2024-05-28T23:43:00Z | 1 |
| Detroit Tigers | Pittsburgh Pirates | 2024-05-29T17:10:00Z | 20 |
| Detroit Tigers | Pittsburgh Pirates | 2024-05-29T19:40:00Z | 19 |
| Detroit Tigers | Pittsburgh Pirates | 2024-05-29T19:45:00Z | 1 |
| Cleveland Guardians | Kansas City Royals | 2024-06-05T22:40:00Z | 38 |
| St. Louis Cardinals | Colorado Rockies | 2024-06-08T00:18:00Z | 7 |
| Minnesota Twins | Oakland Athletics | 2024-06-15T18:10:00Z | 54 |
| St. Louis Cardinals | Atlanta Braves | 2024-06-25T23:45:00Z | 58 |
| St. Louis Cardinals | Atlanta Braves | 2024-06-26T00:15:00Z | 6 |
| St. Louis Cardinals | Atlanta Braves | 2024-06-26T00:45:00Z | 2 |
| St. Louis Cardinals | Atlanta Braves | 2024-06-26T01:15:00Z | 3 |
| St. Louis Cardinals | Atlanta Braves | 2024-06-26T02:00:00Z | 2 |
| Baltimore Orioles | Texas Rangers | 2024-06-30T17:36:00Z | 2 |
| St. Louis Cardinals | Kansas City Royals | 2024-07-09T23:46:00Z | 47 |
| Chicago White Sox | Minnesota Twins | 2024-07-10T00:11:00Z | 62 |
| Chicago White Sox | Minnesota Twins | 2024-07-10T18:11:00Z | 14 |
| Chicago White Sox | Minnesota Twins | 2024-07-10T21:31:00Z | 21 |
| American League | National League | 2024-07-17T00:11:00Z | 182 |
| Atlanta Braves | St. Louis Cardinals | 2024-07-19T23:20:00Z | 37 |
| Atlanta Braves | St. Louis Cardinals | 2024-07-19T23:59:00Z | 3 |
| Atlanta Braves | St. Louis Cardinals | 2024-07-20T00:15:00Z | 3 |
| Atlanta Braves | St. Louis Cardinals | 2024-07-20T00:25:00Z | 1 |
| Atlanta Braves | St. Louis Cardinals | 2024-07-20T00:45:00Z | 1 |
| Atlanta Braves | St. Louis Cardinals | 2024-07-20T01:00:00Z | 1 |
| Atlanta Braves | St. Louis Cardinals | 2024-07-20T01:30:00Z | 4 |
| Atlanta Braves | St. Louis Cardinals | 2024-07-20T02:00:00Z | 3 |
| Atlanta Braves | Cincinnati Reds | 2024-07-23T23:21:00Z | 56 |
| Atlanta Braves | Cincinnati Reds | 2024-07-23T23:51:00Z | 7 |
| Atlanta Braves | Cincinnati Reds | 2024-07-24T00:21:00Z | 1 |
| Atlanta Braves | Cincinnati Reds | 2024-07-24T00:51:00Z | 5 |
| Atlanta Braves | Cincinnati Reds | 2024-07-24T22:06:00Z | 45 |
| San Francisco Giants | Colorado Rockies | 2024-07-27T23:05:00Z | 1 |
| San Francisco Giants | Colorado Rockies | 2024-07-28T02:06:00Z | 1 |
| San Francisco Giants | Colorado Rockies | 2024-07-28T02:16:00Z | 2 |
| Baltimore Orioles | Toronto Blue Jays | 2024-07-29T22:51:00Z | 4 |
| Baltimore Orioles | Toronto Blue Jays | 2024-07-29T23:10:00Z | 1 |
| Baltimore Orioles | Toronto Blue Jays | 2024-07-29T23:40:00Z | 3 |
| Baltimore Orioles | Toronto Blue Jays | 2024-07-30T00:16:00Z | 2 |
| Baltimore Orioles | Toronto Blue Jays | 2024-07-30T00:31:00Z | 2 |
| Baltimore Orioles | Toronto Blue Jays | 2024-07-30T00:33:00Z | 7 |
| Baltimore Orioles | Toronto Blue Jays | 2024-07-30T00:37:00Z | 2 |
| Chicago Cubs | St. Louis Cardinals | 2024-08-04T18:20:00Z | 11 |
| Cleveland Guardians | Arizona Diamondbacks | 2024-08-06T22:40:00Z | 2 |
| Cleveland Guardians | Arizona Diamondbacks | 2024-08-06T22:41:00Z | 54 |
| New York Yankees | Los Angeles Angels | 2024-08-06T23:06:00Z | 50 |
| Cleveland Guardians | Arizona Diamondbacks | 2024-08-06T23:26:00Z | 1 |
| Cleveland Guardians | Arizona Diamondbacks | 2024-08-07T00:01:00Z | 11 |
| Cleveland Guardians | Arizona Diamondbacks | 2024-08-07T00:21:00Z | 2 |
| Cleveland Guardians | Arizona Diamondbacks | 2024-08-07T00:41:00Z | 2 |
| Cleveland Guardians | Arizona Diamondbacks | 2024-08-07T17:10:00Z | 9 |
| Cleveland Guardians | Arizona Diamondbacks | 2024-08-07T17:11:00Z | 21 |
| New York Yankees | Los Angeles Angels | 2024-08-07T20:05:00Z | 12 |
| Cleveland Guardians | Arizona Diamondbacks | 2024-08-07T20:30:00Z | 19 |
| New York Yankees | Los Angeles Angels | 2024-08-07T23:36:00Z | 18 |
| New York Yankees | Texas Rangers | 2024-08-09T23:06:00Z | 46 |
| New York Yankees | Texas Rangers | 2024-08-10T20:41:00Z | 1 |
| New York Yankees | Texas Rangers | 2024-08-10T20:46:00Z | 19 |
| New York Mets | Baltimore Orioles | 2024-08-19T00:20:00Z | 2 |
| Boston Red Sox | Toronto Blue Jays | 2024-08-26T18:05:00Z | 3 |
| Cleveland Guardians | Kansas City Royals | 2024-08-26T20:10:00Z | 1 |
| Pittsburgh Pirates | Washington Nationals | 2024-09-06T22:41:00Z | 72 |
| Houston Astros | Arizona Diamondbacks | 2024-09-08T18:10:00Z | 37 |
| Boston Red Sox | Minnesota Twins | 2024-09-21T20:11:00Z | 75 |
| Boston Red Sox | Minnesota Twins | 2024-09-21T21:11:00Z | 5 |
| Boston Red Sox | Minnesota Twins | 2024-09-22T17:36:00Z | 20 |
| Detroit Tigers | Tampa Bay Rays | 2024-09-24T22:41:00Z | 34 |
| Atlanta Braves | New York Mets | 2024-09-25T23:21:00Z | 64 |
| Atlanta Braves | New York Mets | 2024-09-26T23:20:00Z | 10 |
| Cleveland Guardians | Houston Astros | 2024-09-29T19:11:00Z | 16 |
| Cleveland Guardians | Houston Astros | 2024-09-29T19:41:00Z | 4 |
| Atlanta Braves | New York Mets | 2024-09-30T17:11:00Z | 21 |
| Atlanta Braves | New York Mets | 2024-09-30T21:10:00Z | 3 |
| New York Yankees | Kansas City Royals | 2024-10-04T23:39:00Z | 1 |
| Cleveland Guardians | Detroit Tigers | 2024-10-12T20:38:00Z | 3 |
| Cleveland Guardians | Detroit Tigers | 2024-10-13T00:09:00Z | 2 |

Odds API historical snapshots of these fixtures outside `h2h`:

| Market | Stamped | Snapshots | Fixtures |
|---|---|---|---|
| totals | before KO | 16 | 16 |
| none | after KO | 147 | 94 |
| none | before KO | 1029 | 355 |

### Bookmakers, 2024

Share of the 2472 fixtures with the book in any snapshot at or before kick-off, and of the 164,883 such snapshots.

| Book | Fixtures | Snapshots | Role | Odds API keys aliased to it |
|---|---|---|---|---|
| `casumo` | 100.0% | 61.8% | neither | none |
| `grosvenor` | 100.0% | 61.7% | neither | none |
| `matchbook` | 100.0% | 65.3% | exchange | none |
| `mrgreen` | 100.0% | 61.8% | neither | none |
| `paddypower` | 100.0% | 99.3% | UK retail | none |
| `unibet_uk` | 100.0% | 61.8% | UK retail | none |
| `virginbet` | 100.0% | 79.0% | neither | none |
| `betfair_exchange` | 100.0% | 57.8% | exchange | `betfair`, `betfair_ex_eu`, `betfair_ex_uk` |
| `888sport` | 99.9% | 87.3% | UK retail | `sport888` |
| `williamhill` | 99.9% | 73.8% | UK retail | none |
| `coral` | 99.8% | 70.0% | UK retail | none |
| `ladbrokes` | 99.8% | 70.0% | UK retail | `ladbrokes_uk` |
| `boylesports` | 99.7% | 51.0% | UK retail | none |
| `livescorebet` | 98.8% | 73.5% | neither | none |
| `leovegas` | 95.3% | 58.5% | neither | none |
| `betway` | 93.1% | 59.5% | UK retail | none |
| `betfair_sportsbook` | 62.6% | 49.2% | UK retail | `betfair_sb_uk` |
| `betvictor` | 3.9% | 3.3% | UK retail | none |
| `pinnacle` | 0.6% | 0.0% | neither | none |

### Leads, 2024

Whole-minute decisions per fixture and band, served when a stored snapshot lies in `[t − max age, t]` at its actual time. Unserved decisions are split into those before the fixture's first stored snapshot (not yet listed), those a recorded gap's planned snapshot would have served, and the rest. The last column needs `betfair_exchange` and a UK retail book in the same snapshot.

| Band | Decisions | Served | Unserved: not listed | Unserved: gap | Unserved: other | Fixtures fully served | Served, exchange + retail |
|---|---|---|---|---|---|---|---|
| KO−1m..KO−120m, max age 10m | 296,640 | 99.8% | 0 | 20 | 609 | 2452/2472 | 99.7% |
| KO−121m..KO−480m, max age 60m | 889,920 | 99.7% | 1,518 | 0 | 1,248 | 2452/2472 | 99.2% |

Fixtures first stored less than 9 h before kick-off: 88, first stored at leads (h: fixtures) 4: 1, 5: 3, 6: 3, 7: 7, 8: 59, 9: 15.

### Near kick-off `betfair_exchange`, 2024

Per fixture, the latest exchange snapshot at or before kick-off within 120 min; usable means a back price on every outcome with back overround at most 5%. Research inputs, not validated closing labels.

| Price | Fixtures with one | Within 5 min | Within 10 min | p50 age (min) | p75 age (min) | p90 age (min) | p99 age (min) |
|---|---|---|---|---|---|---|---|
| Any exchange quote | 2468/2472 | 55.5% | 99.3% | 4.4 | 9.4 | 9.4 | 9.4 |
| Usable | 2468/2472 | 55.5% | 99.3% | 4.4 | 9.4 | 9.4 | 9.4 |

Latest exchange snapshot missing an outcome's back price: 0 fixtures.

### Provider gaps and off-phase snapshots, 2024

Recorded gaps: 2.

| Planned | Returned | Next |
|---|---|---|
| 2024-07-21T00:25:38Z | 2024-07-21T00:15:37Z | 2024-07-21T00:40:37Z |
| 2024-08-03T16:20:38Z | 2024-08-03T16:15:38Z | 2024-08-03T16:25:38Z |

Stored snapshot times off the grid's phase (outside their nearest grid time's whole minute): 1 (2024-08-15T18:26:25Z).

### Calls, 2024

Planned 10,768; stored 10,735; covered by a recorded gap 3; missing 30. Distinct stored snapshot times, any market: 10,767, × 10 units = 107,670 units. `h2h` snapshot times matching no planned call (gap returns, probes): 32. Planned calls that serve only fixtures not yet stored at that time (not listed): 0.

Missing calls: 2024-04-04T08:05:38Z, 2024-04-04T09:05:38Z, 2024-04-04T10:05:38Z, 2024-07-28T01:55:38Z, 2024-07-28T02:05:38Z, 2024-07-28T02:15:38Z, 2024-09-30T09:05:38Z, 2024-09-30T10:05:38Z, 2024-09-30T11:05:38Z, 2024-09-30T12:05:38Z, 2024-09-30T13:05:38Z, 2024-09-30T14:05:38Z, 2024-09-30T15:05:38Z, 2024-09-30T15:15:38Z, 2024-09-30T15:25:38Z, 2024-09-30T15:35:38Z, 2024-09-30T15:45:38Z, 2024-09-30T15:55:38Z, 2024-09-30T16:05:38Z, 2024-09-30T16:15:38Z

## 2025

### Fixtures, 2025

2477 FINAL fixtures; 2477 with a stored `h2h` snapshot at or before kick-off; 0 without.

Listings in stored responses that attached to no event: 139.

| Home | Away | Listed kick-off | Responses |
|---|---|---|---|
| Boston Red Sox | St. Louis Cardinals | 2025-04-05T20:10:00Z | 72 |
| Boston Red Sox | St. Louis Cardinals | 2025-04-05T20:11:00Z | 10 |
| Boston Red Sox | St. Louis Cardinals | 2025-04-05T20:45:00Z | 2 |
| Boston Red Sox | St. Louis Cardinals | 2025-04-05T21:15:00Z | 2 |
| Boston Red Sox | St. Louis Cardinals | 2025-04-05T21:45:00Z | 3 |
| Boston Red Sox | St. Louis Cardinals | 2025-04-05T22:15:00Z | 3 |
| Boston Red Sox | St. Louis Cardinals | 2025-04-05T22:45:00Z | 2 |
| Atlanta Braves | Miami Marlins | 2025-04-06T17:35:00Z | 44 |
| Atlanta Braves | Miami Marlins | 2025-04-06T17:36:00Z | 18 |
| Detroit Tigers | New York Yankees | 2025-04-07T22:40:00Z | 15 |
| New York Mets | Miami Marlins | 2025-04-08T23:10:00Z | 8 |
| New York Mets | Miami Marlins | 2025-04-08T23:11:00Z | 34 |
| Baltimore Orioles | Toronto Blue Jays | 2025-04-11T23:05:00Z | 7 |
| Baltimore Orioles | Toronto Blue Jays | 2025-04-11T23:06:00Z | 33 |
| New York Yankees | San Francisco Giants | 2025-04-12T19:05:00Z | 63 |
| New York Yankees | San Francisco Giants | 2025-04-12T19:06:00Z | 2 |
| New York Yankees | San Francisco Giants | 2025-04-12T19:09:00Z | 2 |
| Colorado Rockies | Washington Nationals | 2025-04-19T00:40:00Z | 17 |
| Colorado Rockies | Washington Nationals | 2025-04-19T00:41:00Z | 29 |
| Chicago Cubs | Arizona Diamondbacks | 2025-04-19T15:21:00Z | 8 |
| Kansas City Royals | Colorado Rockies | 2025-04-23T23:40:00Z | 62 |
| Kansas City Royals | Colorado Rockies | 2025-04-23T23:41:00Z | 19 |
| Kansas City Royals | Colorado Rockies | 2025-04-24T00:30:00Z | 5 |
| Kansas City Royals | Colorado Rockies | 2025-04-24T01:30:00Z | 6 |
| Kansas City Royals | Colorado Rockies | 2025-04-24T02:00:00Z | 5 |
| Kansas City Royals | Colorado Rockies | 2025-04-24T18:10:00Z | 25 |
| Kansas City Royals | Colorado Rockies | 2025-04-24T18:11:00Z | 11 |
| Kansas City Royals | Colorado Rockies | 2025-04-24T21:26:21Z | 12 |
| Kansas City Royals | Colorado Rockies | 2025-04-24T21:30:00Z | 5 |
| Detroit Tigers | Baltimore Orioles | 2025-04-25T22:40:00Z | 61 |
| Cleveland Guardians | Boston Red Sox | 2025-04-25T23:10:00Z | 55 |
| Cleveland Guardians | Boston Red Sox | 2025-04-25T23:11:00Z | 13 |
| New York Yankees | Toronto Blue Jays | 2025-04-26T17:05:00Z | 12 |
| New York Yankees | Toronto Blue Jays | 2025-04-26T17:06:00Z | 32 |
| New York Yankees | Toronto Blue Jays | 2025-04-27T17:36:00Z | 3 |
| New York Yankees | Toronto Blue Jays | 2025-04-27T21:25:00Z | 2 |
| New York Yankees | Toronto Blue Jays | 2025-04-27T21:26:13Z | 1 |
| New York Yankees | Toronto Blue Jays | 2025-04-27T22:35:00Z | 2 |
| Cincinnati Reds | St. Louis Cardinals | 2025-04-29T22:40:00Z | 12 |
| Cincinnati Reds | St. Louis Cardinals | 2025-04-29T22:41:00Z | 56 |
| St. Louis Cardinals | New York Mets | 2025-05-03T18:15:00Z | 50 |
| St. Louis Cardinals | New York Mets | 2025-05-03T18:16:00Z | 11 |
| New York Yankees | Tampa Bay Rays | 2025-05-04T15:06:00Z | 14 |
| Philadelphia Phillies | Arizona Diamondbacks | 2025-05-04T15:06:00Z | 17 |
| Washington Nationals | Cleveland Guardians | 2025-05-05T22:45:00Z | 43 |
| Washington Nationals | Cleveland Guardians | 2025-05-05T23:30:00Z | 7 |
| Washington Nationals | Cleveland Guardians | 2025-05-06T00:00:00Z | 2 |
| Washington Nationals | Cleveland Guardians | 2025-05-06T22:45:00Z | 13 |
| Washington Nationals | Cleveland Guardians | 2025-05-06T23:10:00Z | 4 |
| Washington Nationals | Cleveland Guardians | 2025-05-06T23:10:09Z | 14 |
| Colorado Rockies | Detroit Tigers | 2025-05-07T00:40:00Z | 51 |
| Colorado Rockies | Detroit Tigers | 2025-05-07T00:41:00Z | 4 |
| Colorado Rockies | Detroit Tigers | 2025-05-08T19:11:00Z | 44 |
| Colorado Rockies | Detroit Tigers | 2025-05-08T22:10:53Z | 14 |
| Colorado Rockies | Detroit Tigers | 2025-05-08T22:40:00Z | 3 |
| Baltimore Orioles | Minnesota Twins | 2025-05-13T22:35:00Z | 43 |
| Philadelphia Phillies | St. Louis Cardinals | 2025-05-13T22:45:00Z | 43 |
| Baltimore Orioles | Minnesota Twins | 2025-05-14T16:06:00Z | 25 |
| Baltimore Orioles | Minnesota Twins | 2025-05-14T19:45:00Z | 6 |
| Baltimore Orioles | Minnesota Twins | 2025-05-14T19:46:00Z | 13 |
| Baltimore Orioles | Minnesota Twins | 2025-05-14T22:36:00Z | 3 |
| Minnesota Twins | Cleveland Guardians | 2025-05-20T22:11:00Z | 29 |
| Minnesota Twins | Cleveland Guardians | 2025-05-20T23:41:00Z | 42 |
| Minnesota Twins | Cleveland Guardians | 2025-05-21T17:10:00Z | 40 |
| Minnesota Twins | Cleveland Guardians | 2025-05-21T17:11:00Z | 2 |
| Washington Nationals | Atlanta Braves | 2025-05-21T22:46:00Z | 78 |
| Minnesota Twins | Cleveland Guardians | 2025-05-21T23:10:00Z | 5 |
| Boston Red Sox | Baltimore Orioles | 2025-05-22T22:45:00Z | 31 |
| Boston Red Sox | Baltimore Orioles | 2025-05-22T22:46:00Z | 51 |
| Boston Red Sox | Baltimore Orioles | 2025-05-23T23:11:00Z | 43 |
| Boston Red Sox | Baltimore Orioles | 2025-05-24T20:11:00Z | 18 |
| Philadelphia Phillies | Atlanta Braves | 2025-05-28T22:46:00Z | 67 |
| New York Mets | Chicago White Sox | 2025-05-28T23:10:00Z | 6 |
| New York Mets | Chicago White Sox | 2025-05-28T23:11:00Z | 10 |
| Baltimore Orioles | Chicago White Sox | 2025-05-30T23:05:00Z | 1 |
| Baltimore Orioles | Chicago White Sox | 2025-05-30T23:06:00Z | 39 |
| St. Louis Cardinals | Detroit Tigers | 2025-05-31T23:45:00Z | 1 |
| St. Louis Cardinals | Kansas City Royals | 2025-06-04T23:46:00Z | 73 |
| Cincinnati Reds | Arizona Diamondbacks | 2025-06-06T23:11:00Z | 51 |
| Cincinnati Reds | Arizona Diamondbacks | 2025-06-06T23:11:11Z | 5 |
| Cincinnati Reds | Arizona Diamondbacks | 2025-06-06T23:12:00Z | 12 |
| Baltimore Orioles | Los Angeles Angels | 2025-06-13T00:15:00Z | 2 |
| Los Angeles Dodgers | San Francisco Giants | 2025-06-15T20:10:00Z | 40 |
| Los Angeles Dodgers | San Francisco Giants | 2025-06-15T20:11:00Z | 7 |
| Detroit Tigers | Pittsburgh Pirates | 2025-06-18T22:41:00Z | 61 |
| Chicago White Sox | St. Louis Cardinals | 2025-06-18T23:41:00Z | 68 |
| Chicago Cubs | Milwaukee Brewers | 2025-06-19T00:06:00Z | 68 |
| Chicago White Sox | St. Louis Cardinals | 2025-06-19T18:10:00Z | 9 |
| Chicago White Sox | St. Louis Cardinals | 2025-06-19T18:11:00Z | 28 |
| Chicago White Sox | St. Louis Cardinals | 2025-06-19T21:41:00Z | 1 |
| Chicago White Sox | St. Louis Cardinals | 2025-06-19T21:46:00Z | 1 |
| Chicago White Sox | St. Louis Cardinals | 2025-06-19T21:51:00Z | 17 |
| Philadelphia Phillies | San Diego Padres | 2025-07-01T22:30:00Z | 52 |
| Washington Nationals | Detroit Tigers | 2025-07-01T22:45:00Z | 52 |
| Boston Red Sox | Cincinnati Reds | 2025-07-01T23:10:00Z | 69 |
| New York Mets | Milwaukee Brewers | 2025-07-01T23:10:00Z | 56 |
| Boston Red Sox | Cincinnati Reds | 2025-07-01T23:11:00Z | 16 |
| Boston Red Sox | Cincinnati Reds | 2025-07-02T18:30:00Z | 21 |
| Boston Red Sox | Cincinnati Reds | 2025-07-02T18:45:00Z | 1 |
| New York Mets | New York Yankees | 2025-07-05T17:11:00Z | 14 |
| New York Mets | New York Yankees | 2025-07-06T13:41:00Z | 16 |
| Detroit Tigers | Tampa Bay Rays | 2025-07-09T17:10:00Z | 6 |
| Detroit Tigers | Tampa Bay Rays | 2025-07-09T17:11:00Z | 36 |
| Baltimore Orioles | New York Mets | 2025-07-09T23:06:00Z | 75 |
| Chicago White Sox | Cleveland Guardians | 2025-07-10T23:41:00Z | 99 |
| Chicago White Sox | Cleveland Guardians | 2025-07-11T00:00:00Z | 2 |
| Chicago White Sox | Cleveland Guardians | 2025-07-11T00:20:00Z | 2 |
| Chicago White Sox | Cleveland Guardians | 2025-07-11T01:00:00Z | 4 |
| Chicago White Sox | Cleveland Guardians | 2025-07-11T01:40:00Z | 2 |
| Chicago White Sox | Cleveland Guardians | 2025-07-11T01:45:00Z | 1 |
| Chicago White Sox | Cleveland Guardians | 2025-07-12T00:30:00Z | 3 |
| Chicago White Sox | Cleveland Guardians | 2025-07-12T00:34:00Z | 9 |
| National League | American League | 2025-07-16T00:00:00Z | 25 |
| Kansas City Royals | Cleveland Guardians | 2025-07-26T00:10:00Z | 2 |
| Kansas City Royals | Cleveland Guardians | 2025-07-26T00:11:00Z | 73 |
| Chicago White Sox | Philadelphia Phillies | 2025-07-30T08:10:00Z | 2 |
| Cincinnati Reds | Atlanta Braves | 2025-08-02T23:16:00Z | 99 |
| Cincinnati Reds | Atlanta Braves | 2025-08-03T00:40:00Z | 3 |
| Cincinnati Reds | Atlanta Braves | 2025-08-03T01:00:00Z | 4 |
| Cincinnati Reds | Atlanta Braves | 2025-08-03T01:40:00Z | 6 |
| Cincinnati Reds | Atlanta Braves | 2025-08-03T17:25:00Z | 14 |
| Chicago Cubs | Milwaukee Brewers | 2025-08-19T00:06:00Z | 41 |
| San Francisco Giants | Chicago White Sox | 2025-08-27T01:45:00Z | 1 |
| Boston Red Sox | New York Yankees | 2025-09-13T16:11:00Z | 19 |
| New York Mets | Texas Rangers | 2025-09-13T17:11:00Z | 17 |
| Washington Nationals | Atlanta Braves | 2025-09-16T23:15:00Z | 5 |
| Washington Nationals | Atlanta Braves | 2025-09-16T23:45:00Z | 4 |
| Washington Nationals | Atlanta Braves | 2025-09-17T00:15:00Z | 2 |
| Washington Nationals | Atlanta Braves | 2025-09-17T00:45:00Z | 13 |
| Houston Astros | Seattle Mariners | 2025-09-21T18:11:00Z | 46 |
| Chicago Cubs | San Diego Padres | 2025-10-03T02:01:00Z | 1 |
| Cleveland Guardians | Detroit Tigers | 2025-10-03T02:01:00Z | 5 |
| New York Yankees | Boston Red Sox | 2025-10-03T02:01:00Z | 1 |
| Philadelphia Phillies | Los Angeles Dodgers | 2025-10-04T18:30:00Z | 44 |
| Seattle Mariners | Detroit Tigers | 2025-10-04T18:30:00Z | 13 |
| Milwaukee Brewers | Chicago Cubs | 2025-10-06T18:30:00Z | 2 |
| New York Yankees | Toronto Blue Jays | 2025-10-07T18:30:00Z | 3 |
| Detroit Tigers | Seattle Mariners | 2025-10-10T19:09:00Z | 2 |
| Seattle Mariners | Detroit Tigers | 2025-10-10T20:40:00Z | 19 |

Odds API historical snapshots of these fixtures outside `h2h`:

| Market | Stamped | Snapshots | Fixtures |
|---|---|---|---|
| none | after KO | 164 | 116 |
| none | before KO | 676 | 333 |

### Bookmakers, 2025

Share of the 2477 fixtures with the book in any snapshot at or before kick-off, and of the 169,024 such snapshots.

| Book | Fixtures | Snapshots | Role | Odds API keys aliased to it |
|---|---|---|---|---|
| `888sport` | 100.0% | 99.1% | UK retail | `sport888` |
| `matchbook` | 100.0% | 60.3% | exchange | none |
| `paddypower` | 100.0% | 99.3% | UK retail | none |
| `coral` | 100.0% | 70.2% | UK retail | none |
| `ladbrokes` | 100.0% | 70.2% | UK retail | `ladbrokes_uk` |
| `betfair_exchange` | 99.8% | 53.7% | exchange | `betfair`, `betfair_ex_eu`, `betfair_ex_uk` |
| `williamhill` | 99.6% | 78.2% | UK retail | none |
| `boylesports` | 99.4% | 46.5% | UK retail | none |
| `livescorebet` | 99.4% | 58.6% | neither | none |
| `virginbet` | 99.4% | 58.6% | neither | none |
| `grosvenor` | 99.4% | 58.6% | neither | none |
| `betway` | 99.3% | 75.2% | UK retail | none |
| `casumo` | 98.4% | 57.8% | neither | none |
| `leovegas` | 98.4% | 57.8% | neither | none |
| `unibet_uk` | 56.5% | 34.0% | UK retail | none |
| `smarkets` | 35.9% | 31.8% | exchange | none |
| `betvictor` | 15.6% | 10.8% | UK retail | none |

### Leads, 2025

Whole-minute decisions per fixture and band, served when a stored snapshot lies in `[t − max age, t]` at its actual time. Unserved decisions are split into those before the fixture's first stored snapshot (not yet listed), those a recorded gap's planned snapshot would have served, and the rest. The last column needs `betfair_exchange` and a UK retail book in the same snapshot.

| Band | Decisions | Served | Unserved: not listed | Unserved: gap | Unserved: other | Fixtures fully served | Served, exchange + retail |
|---|---|---|---|---|---|---|---|
| KO−1m..KO−120m, max age 10m | 297,240 | 99.6% | 0 | 0 | 1,106 | 2454/2477 | 99.3% |
| KO−121m..KO−480m, max age 60m | 891,720 | 99.6% | 1,121 | 559 | 1,901 | 2447/2477 | 98.2% |

Fixtures first stored less than 9 h before kick-off: 66, first stored at leads (h: fixtures) 5: 2, 6: 4, 7: 5, 8: 42, 9: 13.

### Near kick-off `betfair_exchange`, 2025

Per fixture, the latest exchange snapshot at or before kick-off within 120 min; usable means a back price on every outcome with back overround at most 5%. Research inputs, not validated closing labels.

| Price | Fixtures with one | Within 5 min | Within 10 min | p50 age (min) | p75 age (min) | p90 age (min) | p99 age (min) |
|---|---|---|---|---|---|---|---|
| Any exchange quote | 2470/2477 | 53.6% | 99.4% | 4.4 | 9.4 | 9.4 | 9.4 |
| Usable | 2470/2477 | 53.5% | 99.4% | 4.4 | 9.4 | 9.4 | 9.4 |

Latest exchange snapshot missing an outcome's back price: 0 fixtures.

### Provider gaps and off-phase snapshots, 2025

Recorded gaps: 2.

| Planned | Returned | Next |
|---|---|---|
| 2025-04-15T18:35:38Z | 2025-04-15T18:00:38Z | 2025-04-15T22:25:39Z |
| 2025-10-20T17:05:38Z | 2025-10-20T16:31:34Z | 2025-10-20T19:32:57Z |

Stored snapshot times off the grid's phase (outside their nearest grid time's whole minute): 3 (2025-07-03T00:56:31Z, 2025-10-20T16:31:34Z, 2025-10-20T20:06:49Z).

### Calls, 2025

Planned 10,734; stored 10,717; covered by a recorded gap 16; missing 1. Distinct stored snapshot times, any market: 10,720, × 10 units = 107,200 units. `h2h` snapshot times matching no planned call (gap returns, probes): 3. Planned calls that serve only fixtures not yet stored at that time (not listed): 1.

Missing calls: 2025-05-14T08:00:38Z

## 2026

### Fixtures, 2026

2470 FINAL fixtures; 2469 with a stored `h2h` snapshot at or before kick-off; 1 without.

| Fixture | Kick-off |
|---|---|
| National League v American League | 2026-07-15T00:00:00Z |

Listings in stored responses that attached to no event: 101.

| Home | Away | Listed kick-off | Responses |
|---|---|---|---|
| Chicago White Sox | Toronto Blue Jays | 2026-04-02T20:10:00Z | 3 |
| Kansas City Royals | Milwaukee Brewers | 2026-04-03T23:46:00Z | 69 |
| Kansas City Royals | Milwaukee Brewers | 2026-04-04T20:11:00Z | 23 |
| Cleveland Guardians | Chicago Cubs | 2026-04-04T23:15:00Z | 74 |
| Cleveland Guardians | Chicago Cubs | 2026-04-05T17:11:00Z | 9 |
| Cleveland Guardians | Chicago Cubs | 2026-04-05T17:40:00Z | 8 |
| Cleveland Guardians | Chicago Cubs | 2026-04-05T17:41:00Z | 17 |
| Cleveland Guardians | Chicago Cubs | 2026-04-05T20:41:00Z | 12 |
| Cleveland Guardians | Chicago Cubs | 2026-04-05T20:42:49Z | 13 |
| Cleveland Guardians | Chicago Cubs | 2026-04-05T23:10:00Z | 9 |
| Cleveland Guardians | Kansas City Royals | 2026-04-07T22:10:00Z | 13 |
| New York Mets | Arizona Diamondbacks | 2026-04-07T23:11:00Z | 13 |
| Chicago White Sox | Baltimore Orioles | 2026-04-07T23:41:00Z | 16 |
| Boston Red Sox | Detroit Tigers | 2026-04-19T17:36:00Z | 43 |
| Los Angeles Angels | Toronto Blue Jays | 2026-04-22T07:00:00Z | 2 |
| New York Mets | Colorado Rockies | 2026-04-25T20:11:00Z | 40 |
| New York Mets | Colorado Rockies | 2026-04-26T17:41:00Z | 12 |
| New York Mets | Colorado Rockies | 2026-04-26T20:55:00Z | 1 |
| New York Mets | Colorado Rockies | 2026-04-26T20:58:00Z | 1 |
| Kansas City Royals | Los Angeles Angels | 2026-04-26T23:21:00Z | 48 |
| Philadelphia Phillies | San Francisco Giants | 2026-04-29T22:11:00Z | 16 |
| Baltimore Orioles | Houston Astros | 2026-04-29T22:36:00Z | 68 |
| Philadelphia Phillies | San Francisco Giants | 2026-04-29T22:41:00Z | 69 |
| Baltimore Orioles | Houston Astros | 2026-04-30T16:36:00Z | 8 |
| Philadelphia Phillies | San Francisco Giants | 2026-04-30T17:06:00Z | 24 |
| Baltimore Orioles | Houston Astros | 2026-04-30T19:46:00Z | 2 |
| Baltimore Orioles | Houston Astros | 2026-04-30T19:56:00Z | 2 |
| Baltimore Orioles | Houston Astros | 2026-04-30T19:58:00Z | 18 |
| St. Louis Cardinals | Milwaukee Brewers | 2026-05-05T23:46:00Z | 51 |
| Colorado Rockies | New York Mets | 2026-05-06T00:41:00Z | 39 |
| Boston Red Sox | Tampa Bay Rays | 2026-05-09T20:11:00Z | 53 |
| Boston Red Sox | Tampa Bay Rays | 2026-05-09T22:01:00Z | 1 |
| Cincinnati Reds | St. Louis Cardinals | 2026-05-22T22:41:00Z | 60 |
| New York Yankees | Tampa Bay Rays | 2026-05-23T17:36:00Z | 29 |
| Baltimore Orioles | Detroit Tigers | 2026-05-23T20:06:00Z | 56 |
| Cincinnati Reds | St. Louis Cardinals | 2026-05-24T17:41:00Z | 66 |
| Cincinnati Reds | St. Louis Cardinals | 2026-05-24T18:11:00Z | 4 |
| Cincinnati Reds | St. Louis Cardinals | 2026-05-24T18:41:00Z | 3 |
| New York Yankees | Boston Red Sox | 2026-06-06T23:36:00Z | 82 |
| New York Yankees | Boston Red Sox | 2026-06-07T00:06:00Z | 9 |
| Chicago White Sox | Atlanta Braves | 2026-06-11T23:41:00Z | 60 |
| Cleveland Guardians | Detroit Tigers | 2026-06-14T17:41:00Z | 65 |
| Atlanta Braves | San Francisco Giants | 2026-06-16T23:16:00Z | 65 |
| Atlanta Braves | San Francisco Giants | 2026-06-16T23:16:45Z | 11 |
| Atlanta Braves | San Francisco Giants | 2026-06-17T18:05:00Z | 32 |
| Atlanta Braves | San Francisco Giants | 2026-06-18T23:16:00Z | 70 |
| Chicago Cubs | Toronto Blue Jays | 2026-06-21T18:21:00Z | 52 |
| New York Mets | Chicago Cubs | 2026-06-22T23:11:00Z | 47 |
| St. Louis Cardinals | Arizona Diamondbacks | 2026-06-25T23:46:00Z | 92 |
| Baltimore Orioles | Chicago Cubs | 2026-07-08T22:36:00Z | 85 |
| Baltimore Orioles | Chicago Cubs | 2026-07-08T22:36:51Z | 3 |
| Baltimore Orioles | Chicago Cubs | 2026-07-09T22:36:00Z | 42 |
| Pittsburgh Pirates | Milwaukee Brewers | 2026-07-10T22:41:00Z | 45 |
| Pittsburgh Pirates | Milwaukee Brewers | 2026-07-10T22:50:00Z | 5 |
| Pittsburgh Pirates | Milwaukee Brewers | 2026-07-10T23:20:00Z | 3 |
| Pittsburgh Pirates | Milwaukee Brewers | 2026-07-10T23:35:00Z | 1 |
| Pittsburgh Pirates | Milwaukee Brewers | 2026-07-10T23:50:00Z | 2 |
| Pittsburgh Pirates | Milwaukee Brewers | 2026-07-11T00:20:00Z | 2 |
| Pittsburgh Pirates | Milwaukee Brewers | 2026-07-11T00:45:00Z | 2 |
| Pittsburgh Pirates | Milwaukee Brewers | 2026-07-11T04:50:00Z | 4 |
| Pittsburgh Pirates | Milwaukee Brewers | 2026-07-11T23:00:00Z | 3 |
| Cleveland Guardians | Pittsburgh Pirates | 2026-07-17T23:11:00Z | 41 |
| Cleveland Guardians | Pittsburgh Pirates | 2026-07-18T20:11:00Z | 3 |
| Cleveland Guardians | Pittsburgh Pirates | 2026-07-18T23:40:00Z | 3 |
| Cleveland Guardians | Pittsburgh Pirates | 2026-07-18T23:50:00Z | 5 |
| Cleveland Guardians | Pittsburgh Pirates | 2026-07-19T00:05:00Z | 1 |
| New York Yankees | Los Angeles Dodgers | 2026-07-19T00:09:00Z | 72 |
| Cleveland Guardians | Pittsburgh Pirates | 2026-07-19T00:25:00Z | 1 |
| Cleveland Guardians | Pittsburgh Pirates | 2026-07-19T00:25:17Z | 11 |
| Cleveland Guardians | Pittsburgh Pirates | 2026-07-19T00:30:00Z | 1 |
| New York Yankees | Pittsburgh Pirates | 2026-07-21T23:06:00Z | 40 |
| Boston Red Sox | Baltimore Orioles | 2026-07-21T23:11:00Z | 14 |
| Cincinnati Reds | Cleveland Guardians | 2026-07-27T23:10:00Z | 57 |
| Cincinnati Reds | Cleveland Guardians | 2026-07-27T23:40:00Z | 4 |
| Cincinnati Reds | Cleveland Guardians | 2026-07-27T23:45:00Z | 2 |
| Cincinnati Reds | Cleveland Guardians | 2026-07-28T00:00:00Z | 3 |
| Cincinnati Reds | Cleveland Guardians | 2026-07-28T00:15:00Z | 2 |
| New York Mets | Atlanta Braves | 2026-07-28T23:11:00Z | 57 |
| Cleveland Guardians | Detroit Tigers | 2026-09-04T23:46:00Z | 22 |
| Chicago White Sox | Detroit Tigers | 2026-09-19T23:10:00Z | 2 |
| Chicago White Sox | Detroit Tigers | 2026-09-19T23:11:00Z | 29 |
| Baltimore Orioles | Toronto Blue Jays | 2026-09-22T22:36:00Z | 50 |
| New York Yankees | Baltimore Orioles | 2026-09-25T20:05:00Z | 1 |
| New York Yankees | Baltimore Orioles | 2026-09-25T23:30:00Z | 3 |
| New York Yankees | Baltimore Orioles | 2026-09-25T23:31:12Z | 16 |
| Philadelphia Phillies | Tampa Bay Rays | 2026-09-26T19:50:11Z | 8 |
| Washington Nationals | New York Mets | 2026-09-26T20:05:00Z | 19 |
| Washington Nationals | New York Mets | 2026-09-26T20:06:00Z | 19 |
| Philadelphia Phillies | Tampa Bay Rays | 2026-09-27T02:50:11Z | 2 |
| New York Yankees | Baltimore Orioles | 2026-09-27T17:05:00Z | 15 |
| New York Yankees | Baltimore Orioles | 2026-09-27T17:10:00Z | 3 |
| New York Yankees | Baltimore Orioles | 2026-09-27T17:40:00Z | 3 |
| New York Yankees | Baltimore Orioles | 2026-09-27T18:00:00Z | 2 |
| New York Yankees | Baltimore Orioles | 2026-09-27T18:30:00Z | 5 |
| New York Yankees | Baltimore Orioles | 2026-09-27T18:55:00Z | 1 |
| New York Yankees | Baltimore Orioles | 2026-09-27T19:20:00Z | 28 |
| New York Yankees | Baltimore Orioles | 2026-09-27T19:55:00Z | 2 |
| Atlanta Braves | Philadelphia Phillies | 2026-10-01T18:00:00Z | 25 |
| Chicago White Sox | Cleveland Guardians | 2026-10-08T21:00:00Z | 15 |
| Cleveland Guardians | Tampa Bay Rays | 2026-10-15T22:00:00Z | 1 |
| Cleveland Guardians | Tampa Bay Rays | 2026-10-16T22:00:00Z | 15 |

Odds API historical snapshots of these fixtures outside `h2h`:

| Market | Stamped | Snapshots | Fixtures |
|---|---|---|---|
| totals | before KO | 16 | 16 |
| none | after KO | 102 | 75 |
| none | before KO | 235 | 150 |

### Bookmakers, 2026

Share of the 2470 fixtures with the book in any snapshot at or before kick-off, and of the 164,483 such snapshots.

| Book | Fixtures | Snapshots | Role | Odds API keys aliased to it |
|---|---|---|---|---|
| `betway` | 100.0% | 78.3% | UK retail | none |
| `casumo` | 100.0% | 64.9% | neither | none |
| `grosvenor` | 100.0% | 64.9% | neither | none |
| `paddypower` | 100.0% | 99.2% | UK retail | none |
| `smarkets` | 100.0% | 98.4% | exchange | none |
| `coral` | 99.9% | 77.7% | UK retail | none |
| `ladbrokes` | 99.9% | 77.7% | UK retail | `ladbrokes_uk` |
| `betvictor` | 99.8% | 87.5% | UK retail | none |
| `livescorebet` | 99.8% | 64.6% | neither | none |
| `virginbet` | 99.8% | 64.6% | neither | none |
| `betfair_exchange` | 99.8% | 54.7% | exchange | `betfair`, `betfair_ex_eu`, `betfair_ex_uk` |
| `boylesports` | 99.5% | 52.9% | UK retail | none |
| `888sport` | 99.2% | 98.1% | UK retail | `sport888` |
| `unibet_uk` | 98.7% | 95.4% | UK retail | none |
| `matchbook` | 97.5% | 63.8% | exchange | none |
| `williamhill` | 94.9% | 74.8% | UK retail | none |
| `leovegas` | 89.2% | 49.2% | neither | none |
| `betfair_sportsbook` | 45.1% | 16.7% | UK retail | `betfair_sb_uk` |
| `betano_uk` | 42.1% | 33.4% | neither | none |
| `betfred` | 18.5% | 16.7% | UK retail | `betfred_uk` |
| `pinnacle` | 0.6% | 0.0% | neither | none |

### Leads, 2026

Whole-minute decisions per fixture and band, served when a stored snapshot lies in `[t − max age, t]` at its actual time. Unserved decisions are split into those before the fixture's first stored snapshot (not yet listed), those a recorded gap's planned snapshot would have served, and the rest. The last column needs `betfair_exchange` and a UK retail book in the same snapshot.

| Band | Decisions | Served | Unserved: not listed | Unserved: gap | Unserved: other | Fixtures fully served | Served, exchange + retail |
|---|---|---|---|---|---|---|---|
| KO−1m..KO−120m, max age 10m | 296,400 | 99.9% | 120 | 0 | 149 | 2457/2470 | 99.6% |
| KO−121m..KO−480m, max age 60m | 889,200 | 99.9% | 376 | 0 | 569 | 2454/2470 | 99.5% |

Fixtures first stored less than 9 h before kick-off: 58, first stored at leads (h: fixtures) 8: 41, 9: 17.

### Near kick-off `betfair_exchange`, 2026

Per fixture, the latest exchange snapshot at or before kick-off within 120 min; usable means a back price on every outcome with back overround at most 5%. Research inputs, not validated closing labels.

| Price | Fixtures with one | Within 5 min | Within 10 min | p50 age (min) | p75 age (min) | p90 age (min) | p99 age (min) |
|---|---|---|---|---|---|---|---|
| Any exchange quote | 2461/2470 | 56.6% | 99.4% | 4.4 | 9.4 | 9.4 | 9.4 |
| Usable | 2461/2470 | 56.6% | 99.4% | 4.4 | 9.4 | 9.4 | 9.4 |

Latest exchange snapshot missing an outcome's back price: 0 fixtures.

### Provider gaps and off-phase snapshots, 2026

Recorded gaps: 0.

Stored snapshot times off the grid's phase (outside their nearest grid time's whole minute): 3 (2026-04-03T15:01:03Z, 2026-04-03T18:01:01Z, 2026-04-03T19:01:01Z).

### Calls, 2026

Planned 10,594; stored 10,572; covered by a recorded gap 0; missing 22. Distinct stored snapshot times, any market: 10,575, × 10 units = 105,750 units. `h2h` snapshot times matching no planned call (gap returns, probes): 3. Planned calls that serve only fixtures not yet stored at that time (not listed): 19.

Missing calls: 2026-04-26T20:50:38Z, 2026-04-26T21:00:38Z, 2026-07-14T15:55:38Z, 2026-07-14T16:55:38Z, 2026-07-14T17:55:38Z, 2026-07-14T18:55:38Z, 2026-07-14T19:55:38Z, 2026-07-14T20:55:38Z, 2026-07-14T21:55:38Z, 2026-07-14T22:05:38Z, 2026-07-14T22:15:38Z, 2026-07-14T22:25:38Z, 2026-07-14T22:35:38Z, 2026-07-14T22:45:38Z, 2026-07-14T22:55:38Z, 2026-07-14T23:05:38Z, 2026-07-14T23:15:38Z, 2026-07-14T23:25:38Z, 2026-07-14T23:35:38Z, 2026-07-14T23:45:38Z

