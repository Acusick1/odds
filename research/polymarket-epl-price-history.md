# Polymarket `/prices-history` for finished EPL matches (ticket #518)

Measured 2026-10-18 against the public CLOB and Gamma APIs. No authentication, no spend, 0 Odds API units. The local dev DB (localhost:5433) was read only.

## Verdict

**PROVEN: usable as a free historical near-sharp reference for finished EPL matches at 7 d, 3 d, 1 d, 6 h, 1 h and kick-off, with one required trick (`startTs`/`endTs` windows, not `interval=max`) and one caveat (7 d prices are unreliable in about 7% of matches).** All 449 closed 1X2 matches from 2025-08-15 to 2026-10-18 returned a series for all three markets. A 449-match horizon table and the comparison output are in `polymarket-epl-price-history/`.

## How to call it

Docs: [`GET /prices-history`](https://docs.polymarket.com/api-reference/markets/get-prices-history). Parameters are `market` (the Yes token id), `startTs`, `endTs`, `interval` (`max|all|1m|1w|1d|6h|1h`) and `fidelity` (minutes). Response is `{"history":[{"t":unix_s,"p":float}]}`. The docs do not say whether `p` is a midpoint or a last trade, give no limits, and say nothing about resolved markets. Everything below is observed behaviour, not documented.

- Token ids come from Gamma `GET /events?series_id=10188&closed=true` (`markets[i].clobTokenIds[0]` is the Yes token). Market order is always home, draw, away (449 of 449 events).
- Gamma and CLOB returned HTTP 403 to Python's default `urllib` user agent; `curl`'s user agent worked.
- **`interval=max` returns an empty `history` for old markets.** It returned data only for the most recent matches (about 2 weeks; 10 Oct 2026 had 321 points, 2025-08 to 2026-09 samples had 0). `interval=all` was also empty.
- **`startTs`/`endTs` works for every finished match tested**, back to the first match, 2025-08-15.
- Maximum window: 14 days accepted, 20 days rejected (`'startTs' and 'endTs' interval is too long`). I did not find the exact limit between 15 and 19 days.
- `interval=1w` with a small `fidelity` is rejected (`minimum 'fidelity' for '1w' range is 5`); explicit timestamps avoid this.
- Fidelity 1 over 7 days returned 10,260 points per market (about the 10.5K row range of the trades cap, but no error). Fidelity 10 returns about 1,020.

## Resolution, coverage and gaps (449 matches x 3 markets, 14 d window, fidelity 10)

- **Coverage: 449 of 449 matches have all three series, no HTTP errors, no empty series.** Series start at least 64 h before kick-off for every match; 446 start before T-7 d and 447 before T-3 d (the window was capped at 14 d, so true start is earlier).
- **The series is a resampled grid, not trade ticks.** Spacing is 600 s (598-602) at fidelity 10 and about 60 s at fidelity 1. The price changes between consecutive points in only about 5.5% of points (median), so values are carried forward between changes.
- **Gaps exist in the grid.** Longest gap per market: median 1,207 s, 95th percentile 6,629 s, maximum 12,601 s (3.5 h).
- As-of staleness (horizon minus the last grid point at or before it) is a median of 590 s at the coarse horizons and 52 s at 1 h and kick-off (fidelity 1). Maximum 2,348 s at 1 d.
- The last 3 h before kick-off at fidelity 1 is what gives usable 1 h and kick-off values.

## Do the three series sum to about 1?

Yes from 3 d out. Sum of the home, draw and away Yes prices at each horizon (as-of price, 449 matches; `horizon_prices.csv`):

| Horizon | Matches | Mean sum | Min | Max | Matches with abs(sum-1) > 0.03 |
|---|---|---|---|---|---|
| 7 d | 446 | 1.016 | 0.885 | 1.525 | 33 |
| 3 d | 447 | 1.003 | 0.985 | 1.025 | 0 |
| 1 d | 449 | 1.002 | 0.985 | 1.015 | 0 |
| 6 h | 449 | 1.002 | 0.985 | 1.020 | 0 |
| 1 h | 449 | 1.001 | 0.985 | 1.015 | 0 |
| kick-off | 449 | 1.001 | 0.975 | 1.020 | 0 |

The 33 bad 7 d cases cluster in 2025-08 (7 of 27), 2025-12 (6 of 56), 2026-01 (13 of 49) and 2026-05 (5 of 41). Examples: `epl-wes-che-2025-08-22` 0.495/0.485/0.500 (sum 1.48), `epl-bou-wol-2025-08-23` 0.435/0.395/0.435. These look like untraded or placeholder quotes on an empty book **(inferred; I did not inspect the books)**. Normalising is not enough there. Use 7 d only with the sum filter, or treat 3 d as the earliest reliable horizon.

## Comparison with Pinnacle and Betfair Exchange (local dev DB)

Method: for every Pinnacle or Betfair Exchange h2h/1x2 snapshot taken 0-200 h before kick-off, take the Polymarket as-of price at that snapshot's time (at most 30 min stale; fidelity 1 inside 3 h, fidelity 10 outside), normalise the three Yes prices to sum to 1, devig the reference proportionally, and compare per outcome. All 449 matches were matched to DB events by team name and kick-off time. Odds API snapshots (`snapshot_time` is the fetch time) and OddsPortal snapshots (`op_*` ids) are reported separately; I did not separate bookmaker `last_update` from fetch time, so reference staleness is not removed. Full table: `pm_vs_refs.txt`.

Mean absolute difference per outcome, in percentage points (mean of the per-snapshot max in brackets):

| Hours to kick-off | Pinnacle, Odds API | Betfair Exchange, Odds API | Betfair Exchange, OddsPortal |
|---|---|---|---|
| under 0.5 | 0.48 (0.71), 12 events | 0.39 (0.58), 45 events | 0.40 (0.60), 154 events |
| 0.5-2 | 0.48 (0.73), 18 events | 0.38 (0.58), 47 events | 0.49 (0.74), 279 events |
| 2-8 | 0.46 (0.69), 15 events | 0.36 (0.54), 50 events | 0.37 (0.56), 77 events |
| 8-30 | 0.59 (0.88), 18 events | 0.39 (0.59), 53 events | 0.35 (0.52), 381 events |
| 30-80 | 0.54 (0.81), 13 events | 0.35 (0.52), 57 events | 0.35 (0.52), 98 events |
| 80-200 | 0.69 (1.04), 20 events | 0.57 (0.85), 57 events | 0.83 (1.25), 101 events |

- Polymarket tracks Betfair Exchange to about 0.35-0.5 pp per outcome and Pinnacle to about 0.5-0.7 pp from 80 h out to kick-off, a bit tighter than the ticket #509 snapshot estimate (about 1 pp).
- OddsPortal Pinnacle rows are noisier (0.6-0.95 pp) and are less trustworthy as a reference, since they are scraped values.
- Pinnacle Odds API coverage is thin: h2h only from 2026-04-18 and absent after Pinnacle's Jan 2026 shutdown for much of the season (see ADR/FDUK ticket #511); Betfair Exchange covers every match.
- Part of the residual is Polymarket's 0.5-1 c tick, so sub-0.5 pp differences are not meaningful.
- This supports "a free, multi-horizon price close to the sharp reference". It does not show Polymarket adds independent information, and it is not a CLV test.

## Cost and limits

- Odds API units: 0. Money: $0. Roughly 2,700 CLOB requests (449 x 3 markets x 2 windows) took 51 s with 8 threads and hit no rate limits (no 429s).
- Window cap of 14-19 days per call, so horizons earlier than that need chunked calls.
- Fidelity 1 is available for old markets but is a resampled grid; there is no tick-level history and no order-book history.
- Prices are Yes-token prices only (midpoint-vs-trade semantics undocumented). No volume or spread per point.
- Sample size for Pinnacle comparisons is small (12-20 events per bucket) because of the DB's Pinnacle coverage.

## What the map can build on

1. A Polymarket EPL 1X2 backfill is cheap and complete: one loop of about 1,350 calls, about a minute, 449 matches back to 2025-08-15, giving the 3 d, 1 d, 6 h, 1 h and kick-off horizons with sum within 0.03 of 1. 7 d needs the sum filter.
2. Polymarket can act as a second sharp reference at those horizons (about 0.4 pp from Betfair Exchange, which covers every match), and as the only free multi-horizon reference for the period after Pinnacle's shutdown.
3. Coverage is one season plus (2025-08 onwards), so it cannot extend the training range before that.
4. Not tested: lead-lag against Pinnacle or Betfair Exchange, whether `p` is a midpoint or last trade, and series for markets older than 2025-08.
