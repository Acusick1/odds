# Odds API historical backfill: cost for EPL at several horizons (ticket #510)

Verdict: usable, but only on a paid plan, and `bet365` is not carried. Historical odds are refused on the free key (HTTP 401, 0 units), so bookmaker coverage in the historical odds could not be probed; it comes from the vendor docs.

## Proven with real calls (free key, 2026-10-18)

| Call | Result | Units |
|---|---|---|
| `GET /v4/historical/sports/soccer_epl/events?date=2025-03-01T12:00:00Z` | 200, EPL fixtures for the coming ~2 weeks | 1 |
| same, `date=2020-09-01T12:00:00Z` | 200, 18 events (first on 2020-09-12) | 1 |
| same, `date=2022-08-01T12:00:00Z` | 200, 21 events | 1 |
| `/historical/sports/soccer_epl/odds?regions=uk&markets=h2h` | 401 `Historical odds are only available on paid usage plans.` | 0 |
| same with `regions=eu` | 401, same message | 0 |

- Quota spent: **3 units** (`x-requests-remaining` went 204 to 201 on `ODDS_API_KEY`; `x-requests-last: 1` per events call). `odds status quota` printed nothing (see `wiki/collection/odds-api-quota.md#where-the-views-disagree`).
- The pricing page shows "Historical Odds" on the free Starter plan, but the API refuses it. The API and the docs ("only available on paid usage plans") are the authority.
- `/historical/events` works on the free tier and costs 1 unit, so event discovery and kickoff times for a backfill plan are cheap.
- A snapshot returns the fixtures for the next ~2 weeks, so one call serves many matches at different horizons. The events response proves this; the odds response is presumed identical (unproven).

## Facts from the docs

Source: https://the-odds-api.com/liveapi/guides/v4/#get-historical-odds
- Data from 2020-06-06. Snapshots every 10 minutes, every 5 minutes from September 2022. `date` returns the closest snapshot at or before it; the response carries `previous_timestamp` and `next_timestamp`.
- Cost per call: `10 x markets x regions`. Empty responses are free. Bookmakers are a filter and do not change cost.
- Bookmakers, sports and markets appear in the historical data only from when they were added to the live API. Early snapshots can contain data errors.
- Markets `h2h`, `spreads`, `totals` are valid on the featured endpoint.

Source: https://the-odds-api.com/sports-odds-data/bookmaker-apis.html
- **bet365: not available** in uk or eu (only `bet365_au`, paid, AFL/NRL). The current EPL target bookmaker cannot be backfilled from this source.
- **Pinnacle**: `pinnacle`, region `eu` only ("odds are from public website which may incur a delay").
- **Betfair Exchange**: `betfair_ex_uk`, `betfair_ex_eu`. Matchbook (uk, eu) and Smarkets (uk) also listed.
- **UK retail in `uk`**: `williamhill`, `paddypower`, `skybet`, `ladbrokes_uk`, `coral`, `betfred_uk`, `betvictor`, `betway`, `sport888`, `unibet_uk`, `boylesports`, `virginbet`, `livescorebet`, `betano_uk`, `leovegas`, `grosvenor`, `casumo`.
- Per-bookmaker start dates for EPL are not published. Not proven; the docs only say the list is bounded by when each was added.

## Cost model

Cost per call is `10 x markets x regions`: h2h, one region = 10; h2h+totals, one region = 20; h2h+totals, uk+eu = 40.

Calls needed, two bounds:
- **Per-match**: one call per match per horizon, `380 x S x H`.
- **Shared**: one call per distinct (kickoff - horizon) timestamp, so matches in the same kickoff slot share. Distinct kickoff slots in the local DB (`events`, `soccer_epl`, newest seasons first): 2024-25 = 216, 2023-24 = 235, 2022-23 = 229, 2021-22 = 230, 2020-21 = 339. Cumulative S=1: 216, S=3: 680, S=5: 1,249.
- A fixed snapshot grid (for example every 6 h) could go lower still, since each call returns ~2 weeks of fixtures; this is not costed here.

Units (`pm` = per-match, `sh` = shared slots):

| markets, regions | units/call | S | matches | H=3 pm | H=3 sh | H=6 pm | H=6 sh | H=10 pm | H=10 sh |
|---|---|---|---|---|---|---|---|---|---|
| h2h, uk (or eu) | 10 | 1 | 380 | 11,400 | 6,480 | 22,800 | 12,960 | 38,000 | 21,600 |
| h2h, uk (or eu) | 10 | 3 | 1140 | 34,200 | 20,400 | 68,400 | 40,800 | 114,000 | 68,000 |
| h2h, uk (or eu) | 10 | 5 | 1900 | 57,000 | 37,470 | 114,000 | 74,940 | 190,000 | 124,900 |
| h2h, uk+eu | 20 | 1 | 380 | 22,800 | 12,960 | 45,600 | 25,920 | 76,000 | 43,200 |
| h2h, uk+eu | 20 | 3 | 1140 | 68,400 | 40,800 | 136,800 | 81,600 | 228,000 | 136,000 |
| h2h, uk+eu | 20 | 5 | 1900 | 114,000 | 74,940 | 228,000 | 149,880 | 380,000 | 249,800 |
| h2h+totals, uk (or eu) | 20 | 1 | 380 | 22,800 | 12,960 | 45,600 | 25,920 | 76,000 | 43,200 |
| h2h+totals, uk (or eu) | 20 | 3 | 1140 | 68,400 | 40,800 | 136,800 | 81,600 | 228,000 | 136,000 |
| h2h+totals, uk (or eu) | 20 | 5 | 1900 | 114,000 | 74,940 | 228,000 | 149,880 | 380,000 | 249,800 |
| h2h+totals, uk+eu | 40 | 1 | 380 | 45,600 | 25,920 | 91,200 | 51,840 | 152,000 | 86,400 |
| h2h+totals, uk+eu | 40 | 3 | 1140 | 136,800 | 81,600 | 273,600 | 163,200 | 456,000 | 272,000 |
| h2h+totals, uk+eu | 40 | 5 | 1900 | 228,000 | 149,880 | 456,000 | 299,760 | 760,000 | 499,600 |

## Plans

Source: https://the-odds-api.com/ (pricing section), prices in USD per month, billed immediately, auto-renewing, cancellable any time:

| Plan | Price | Credits/month |
|---|---|---|
| Starter | free | 500 (no historical odds in practice) |
| 20K | $30 | 20,000 |
| 100K | $59 | 100,000 |
| 5M | $119 | 5,000,000 |
| 15M | $249 | 15,000,000 |

- One month of **20K ($30)** covers: h2h one region up to 3 seasons x 3 horizons shared (20,400 is just over; 2 seasons fits), or 1 season at 6 horizons.
- **100K ($59)** covers every h2h-only cell except 5 seasons x 10 horizons per-match (190,000), and every h2h+totals cell up to 3 seasons x 6 horizons shared (81,600) or 5 seasons x 3 horizons shared (74,940).
- **5M ($119)** covers everything above, including the largest cell (760,000).
- Whether credits roll over or unused credits are refunded is not stated in what was read. Not proven.

## Gaps

- Historical odds could not be called (paid only), so per-bookmaker presence and start dates for EPL are unproven; first thing to do in a paid month is one 10-unit probe per region at a known date.
- The in-repo planner estimates 30 units per snapshot (a literal), which is wrong for any region/market choice here (`wiki/collection/odds-api-backfill.md#limits`).
