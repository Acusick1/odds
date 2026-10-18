# Betfair free (Basic) historical data for EPL (ticket #507)

Verdict: PROVEN for coverage, format and cost; NOT YET PROVEN for bulk download. Basic covers EPL match-odds from 2015-05 at about 1-minute last-traded-price (LTP) only: no best back/lay, no volume. The bulk download needs a free "purchase" of the data on the Betfair account, which has not been made.

## Evidence

All service facts come from historicdata.betfair.com itself (its served JS bundle, translation file and public sample files) and from authenticated calls to its API with the repo's existing Betfair cert login. Nothing was purchased or downloaded beyond the public sample.

### Cost

- `GET /api/GetStandardRateCard` returns Soccer `basicCost: 0`, `advancedCost: 69`, `proCost: 230` (per sport per month, account currency).
- `GET /api/GetPricePlan` lists month-specific discounts: all plans free for 2015-05 to 2016-04; Advanced free for 2020-01 to 2020-05; Advanced 49 in the 2017-09 to 2018-07 rows. Basic is 0 in every row.
- Portal text: "All data from May 2015 - April 2016 is free to purchase". Rate card: Basic frequency "1 minute", volume "N/A"; Advanced "1 second", volume yes; Pro "Tick Data (50ms)" (`translations/en_GB`, keys `STANDARD_RATE_CARD`, `PLANS_DESCRIPTION`).

### Coverage

- Data starts May 2015 (portal `INTRO` and `DATA_PROVIDED` text). `GetAdvBasketDataSize` returns 0 files for Soccer MATCH_ODDS in 2005 to 2014 and non-zero from 2015.
- Soccer MATCH_ODDS Basic files per calendar year, from `GetCollectionOptions`: 2015 61,712; 2016 91,595; 2017 98,259; 2018 66,028; 2020 53,101; 2022 80,908; 2024 74,774; 2025 67,113; 2026 (to October) 51,392. Counts cover all countries and leagues.
- Country `GB`, Aug to Dec 2024: 4,245 files, 13 MB (Basic) vs 460 MB (Advanced). Basic is about 3 KB per market.
- No competition filter. Filters are sport, dates, eventId, eventName, marketType, countryCode, fileType (`DATA_DELIVERY`). `GB` includes lower English leagues and Scotland, and `marketDefinition` has no competition field (see sample), so EPL must be identified by team names in `eventName` (e.g. "Middlesbrough v Man City"). `eventName="English Premier League"` returned nothing.
- Availability lag: 5 days after event settlement, and only once every market in the event has settled (`UPDATE_FREQUENCY`).

### Fields (real sample files: Middlesbrough v Man City, 2017-04-30, a Premier League match)

Public free samples: `https://historicdata.betfair.com/files/football-{basic,adv,pro}-sample.bz2`. The Basic sample is saved here (57 KB).

| plan | per-runner keys in `rc` | MATCH_ODDS timestamps in sample | median gap |
|---|---|---|---|
| Basic | `ltp` | 24 (changes only) | 6 min (min 1 min) |
| Advanced | `ltp`, `tv`, `trd`, `batb`, `batl` (top 3 levels) | 6,326 | 1.9 s |
| Pro | `atb`, `atl` (full depth), `trd`, `ltp`, `tv`, `spn`, `spf` | 24,665 | 0.2 s |

- Files are Stream API recordings: newline-delimited JSON (`op: mcm`, `pt` epoch ms, `mc[].rc[]`). An event file interleaves all markets of the event; a market file holds one (`FILES_FORMAT`, `TIPS_DOWNLOAD`). The Basic sample is an event file with 77 markets, 1.47 MB uncompressed.
- Basic is delta-encoded: a runner's `ltp` appears only when it changed, so the price at time T is the last update at or before T (forward-fill) and can be stale.
- Market definition gives `marketTime` (scheduled KO), `status`, `inPlay`, runner names and final `WINNER`/`LOSER`, so results come free.
- LTP is a last trade, not a quoted back or lay price. No spread or overround can be derived from Basic.

### Prices at horizons before kick-off (Basic sample, MATCH_ODDS, KO 2017-04-30 13:05Z)

Reproduce with `python3 -I read_sample.py football-basic-sample.bz2`. Age is minutes since that runner's last update.

| horizon | Middlesbrough | Draw | Man City |
|---|---|---|---|
| T-24h, T-12h, T-6h | none yet | 6.4 (stale) | none yet |
| T-2h | 14.0 (age 10) | 6.4 | 1.31 (age 13) |
| T-1h | 14.5 (age 54) | 6.4 | 1.31 (age 73) |
| T-30m | 12.5 (age 22) | 6.8 (age 14) | 1.31 (age 103) |
| T-10m | 12.5 (age 42) | 6.8 (age 34) | 1.31 (age 123) |
| T-5m | 12.0 (age 0) | 6.8 (age 39) | 1.31 (age 128) |
| T-1m | 14.0 (age 1) | 6.8 (age 43) | 1.29 (age 2) |

Reading: a price exists at most horizons only after forward-fill, and a thinly traded runner's last trade can be hours old. One match is not a distribution; staleness over a season is still to measure.

### Download mechanics

- Auth: cert login as the repo does, then header `ssoid: <session token>` (portal `DownloadFile` example). `GetMyData` returned 200 `[]` (nothing purchased).
- Endpoints (portal `API_DOCS_PAGE`): `GetMyData`, `GetCollectionOptions`, `GetAdvBasketDataSize`, `DownloadListOfFiles`, `DownloadFile?filePath=/data/xds/historic/BASIC/<eventId>/<marketId>.bz2`. `GetMyData` and `GetStandardRateCard` are GET; the rest of the filter calls are POST with JSON `{sport, plan, fromDay/Month/Year, toDay/Month/Year, marketTypesCollection, countriesCollection, fileTypeCollection}`, `plan` = `"Basic Plan"`. `fromYear`, `toYear` and a valid date range are mandatory.
- Measured: `GetCollectionOptions` and `GetAdvBasketDataSize` work without a purchase (GB MATCH_ODDS Aug 2023: 847 files, 3 MB). `DownloadListOfFiles` returns 400 `UserDoesNotHaveDataAccessException` ("has not purchased all items for filter") for this account, so listing and downloading need the purchase first.
- The purchase is made on the portal (`purchaseBasket`; per sport, plan and month range; Basic costs 0). I did not make it because it accepts the licence terms below on the user's behalf.
- Bulk: `DownloadListOfFiles` for the range, then `DownloadFile` per path (the portal recommends this over the My Data tar build). Use file type `E` or `M`, not both, since both doubles the size. Rate limits are not documented; none were hit.

### Licence

Portal terms (`PERSONAL_USE`): data is "AS IS"; "personal, internal use only"; commercial use needs a licence; "resale or distribution to ... another organization or individual is strictly prohibited"; Betfair may suspend the service at any time; downloading or using the data accepts the terms. Consequence: keep Basic data out of the repo beyond tiny samples and never publish it. The committed 57 KB file is Betfair's own public sample.

## What the map can build on

- Free from 2015-05 to now: EPL match odds as about 1-minute LTP, with results. GB Basic is about 13 MB per five months, so a full backfill is small.
- Fits a Betfair-exchange traded-price series across horizons. Does not fit quoted back/lay or volume: that needs Advanced (Soccer 69 per month at the current rate card), which is a costed-burst question, not a free one.
- Open steps: make the free Basic purchase, download one season, match EPL events by team name to the `events` table, and measure LTP staleness per horizon across many matches.

## Spend

No money. Odds API quota: 0 units. Betfair account: read-only API calls only.
