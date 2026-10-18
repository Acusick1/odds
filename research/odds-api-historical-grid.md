# Odds API historical backfill: snapshot grid and cost (ticket #506)

Verdict: with 10 named bookmakers a call costs 10 units, and a shared snapshot grid covers all six EPL seasons at 10 horizons in 78,220 units. That fits one month of the $59 100K plan. The one unproven assumption is that a historical odds snapshot returns at least a week of upcoming fixtures.

## 1. `bookmakers` billing

Source: https://the-odds-api.com/liveapi/guides/v4/ (live `/odds` and historical odds sections).
- Live odds: "Every group of 10 bookmakers is the equivalent of 1 region." and "Specifying between 11 and 20 bookmakers counts as 2 regions."
- Historical odds: its parameters "are the same as for the /odds endpoint", including "If both `bookmakers` and `regions` are both specified, `bookmakers` takes priority." Cost: "The usage quota cost for historical odds is 10 per region per market."; cost = 10 x [markets] x [regions].
- So 10 named bookmakers across regions (Pinnacle is `eu`, the rest `uk`) is 1 region-equivalent: 10 units per call for h2h, against 20 for `regions=uk,eu`. The historical section does not restate the 10-bookmaker rule; it is inherited from `/odds`. `research/odds-api-historical-cost.md` ("bookmakers are a filter and do not change cost") misses this rule. Verify with `x-requests-last` on the first paid call.
- The docs do not state how many days of fixtures a snapshot returns. #510 saw ~2 weeks on the (free) historical events endpoint; the odds endpoint is unproven. The 7d horizon needs at least 7 days.

## 2. Snapshot grid

Snapshot interval: 10 minutes from 2020-06-06, 5 minutes from September 2022 ("From September 2022, historical odds snapshots are available at 5 minute intervals"; the exact day is not given, the script uses 2022-09-01). `date` returns the closest snapshot at or before it.

Method (`research/odds-api-historical-grid/grid.py`): each (match, horizon) is a time interval `[K-h-tol, K-h+tol]` with tol = 15 min for h <= 3h and 1h beyond. One call at T serves every interval containing T, whatever the match or horizon, so the minimum call count is the minimum point set stabbing all intervals (greedy by right endpoint, optimal). T is floored to the snapshot grid, so 2020-21 and 2021-22 (10-min snapshots) lose at most 10 minutes of precision, inside the tolerance.

Local DB (dev, localhost:5433): every season 2020-21 to 2025-26 is present with 380 matches after dropping 10 duplicate fixtures in 2025-26 (same pairing, two event ids; the later kick-off is kept). The 2019-20 restart (to 2020-07-26) and 2026-27 are excluded.

Units = calls x 10 (h2h, 10 bookmakers: pinnacle, betfair_ex_uk, williamhill, paddypower, skybet, ladbrokes_uk, coral, betfred_uk, betvictor, betway). "Per-match" is one call per match per horizon, the no-sharing bound.

Horizons: 10 = 7d, 3d, 2d, 24h, 12h, 6h, 3h, 1h, 30m, 10m; 4 = 24h, 6h, 1h, 10m.

| season | matches | distinct kick-offs | 10-horizon calls | units | 4-horizon calls | units | per-match 10h | per-match 4h |
|---|---|---|---|---|---|---|---|---|
| 2020-21 | 380 | 339 | 1568 | 15,680 | 1006 | 10,060 | 38,000 | 15,200 |
| 2021-22 | 380 | 230 | 1260 | 12,600 | 736 | 7,360 | 38,000 | 15,200 |
| 2022-23 | 380 | 229 | 1225 | 12,250 | 722 | 7,220 | 38,000 | 15,200 |
| 2023-24 | 380 | 235 | 1287 | 12,870 | 751 | 7,510 | 38,000 | 15,200 |
| 2024-25 | 380 | 216 | 1202 | 12,020 | 703 | 7,030 | 38,000 | 15,200 |
| 2025-26 | 380 | 228 | 1280 | 12,800 | 750 | 7,500 | 38,000 | 15,200 |
| total | 2280 | | 7822 | 78,220 | 4668 | 46,680 | 228,000 | 91,200 |

Cumulative newest-first (10-horizon / 4-horizon units): back to 2025-26 12,800 / 7,500; 2024-25 24,820 / 14,530; 2023-24 37,690 / 22,040; 2022-23 49,940 / 29,260; 2021-22 62,540 / 36,620; 2020-21 78,220 / 46,680.

Sharing saves 66% at 10 horizons and 49% at 4 against per-match. 2020-21 is the costliest season (339 distinct kick-off slots under COVID scheduling).

## 3. Plans

Plans (from #510, https://the-odds-api.com/): 20K $30, 100K $59, 5M $119.

| Option | Units | Plan | Months |
|---|---|---|---|
| 6 seasons, 10 horizons, shared grid | 78,220 | 100K ($59) | 1 (21,780 spare) |
| 6 seasons, 4 horizons, shared grid | 46,680 | 100K ($59) | 1 |
| 5 seasons from 2021-22, 10 horizons | 62,540 | 100K | 1 |
| 3 newest seasons, 10 horizons | 37,690 | 100K | 1 |
| 6 seasons, 4 horizons, per-match | 91,200 | 100K | 1 (8,800 spare) |
| 6 seasons, 10 horizons, per-match | 228,000 | 100K x3 ($177) or 5M ($119) | 3 or 1 |
| 2 newest seasons, 4 horizons, shared | 14,530 | 20K ($30) | 1 |

- The 5M plan is not needed for the shared grid.
- Adding `totals` doubles cost (156,440 for six seasons, about two months on 100K).
- Bookmaker coverage in the historical data (Pinnacle, `betfair_ex_uk` and each UK retailer back to 2020) is unproven; bookmakers appear only from when they were added to the live API. Spend the first paid call on a 2020-09 date to check.

## Quota

Units spent by this research: 0 (docs reads and the local DB only).
