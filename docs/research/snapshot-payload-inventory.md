# Snapshot payload inventory

Asset for [Inventory every source's snapshot payload and the fields readers use](https://github.com/Acusick1/odds/issues/659), part of [Redesign the odds data model before MLB and NBA build on it](https://github.com/Acusick1/odds/issues/658). Code read at `b4762da6`; data measured on the local copy of production (latest snapshot 2026-10-24, schema before #646).

## Observed in the stored data

| source | snapshots | JSON size | outcome entries (back-price rows) | Betfair runners (lay rows) | avg books/snapshot |
|---|---|---|---|---|---|
| polymarket | 139,291 | 70 MB | 417,873 | 0 | 1.0 |
| oddsportal_live | 36,577 | 22 MB | 1,252,579 | 0 | 14.5 |
| betfair_api | 33,060 | 28 MB | 88,530 | 88,530 | 1.0 |
| Odds API (no tag) | 25,498 | 18 MB | 721,640 | 0 | 5.2 |
| oddsportal | 19,211 | 21 MB | 365,691 | 0 | 8.7 |
| football_data_uk | 8,455 | 5.5 MB | 178,818 | 0 | 7.0 |
| **total** | **262,092** | **~165 MB** | **~3.02M** | **88,530** | |

- Expanding every stored snapshot gives about 3.1M price rows (back plus Betfair lay), against today's `odds` table of 2.23M rows using 964 MB. The legacy `polymarket_price_snapshots` add 196,409 two-outcome rows (about 390k price rows).
- The stored data differs from current writer code in three places:
  - **491 Odds API `h2h_lay` markets** (Betfair via the Odds API, older data). No current code requests or reads them.
  - **29 oddsportal_live outcomes carry American `price`** instead of `price_decimal` (legacy rows).
  - **`betfair_matched` appears on exactly the 22,859 oddsportal_live `betfair_exchange` entries** and on no other book. It holds per-outcome matched volume, e.g. `{"home": 5117, "draw": 16, "away": 5117}`.
- `polymarket_meta.best_bid`/`best_ask` are on 3,079 of 139,291 Polymarket snapshots (live only).
- Bookmaker keys per source: the Odds API keys are mostly US books (NBA), and FDUK includes the `market_max`/`market_avg` pseudo-books. OddsPortal live shows near-duplicate keys (`7bet`/`7betuk`, `betmgm`/`betmgmuk`, `unibet`/`unibet_uk`, `1xbetie`/`1xbetir`) that only read-time aliasing partly merges.

Paths are relative to `/home/andrew/code/odds`. `core` = `packages/odds-core/odds_core`, `lambda` = `packages/odds-lambda/odds_lambda`, `analytics` = `packages/odds-analytics/odds_analytics`, `mcp` = `packages/odds-mcp/odds_mcp`.

Common envelope (all sources): `raw_data = {"bookmakers": [ {key, title, last_update, markets:[{key, outcomes:[{name, <price>, point?}]}], <extra per-book keys>} ], <top-level tags>}`. Price field is `price_decimal` (`PRICE_DECIMAL_KEY`, core/odds_math.py:14) for every source except the Odds API, which stores integer American `price` (`PRICE_AMERICAN_KEY`, core/odds_math.py:15). `outcome_price_decimal()` (core/odds_math.py:57) accepts either.

## Sources

### Odds API (no `source` tag)
- **Writer**: `lambda/ingestion.py:120-147` (live) and `analytics/backfill_executor.py:251-281` (historical). Both pass the API event dict unchanged as `raw_data` to `OddsWriter.store_odds_snapshot`. Fetch is in `lambda/data_fetcher.py:174-234` (live) and `:313-356` (historical).
- **Shape** (the API's event dict, stored whole): top level `id, sport_key, sport_title, commence_time, home_team, away_team, bookmakers[]` (event fields are duplicated from the `events` table). Per bookmaker `key, title, last_update, markets[]`. Per market `key` (and, per the API, a market-level `last_update`; not code-verified), `outcomes[]`. Per outcome `name`, `price` (integer American, because `odds_format="american"` is the default at data_fetcher.py:179/313 and ingestion.py passes no override), optional `point` (spreads/totals).
- **Markets requested**: `h2h, spreads, totals` (core/config.py:83-85). Bookmaker list per sport from `core/config.py:75-82`. No current code requests `h2h_lay` or reads lay data; 491 `h2h_lay` markets exist in older stored data.
- **Post-processing before storage**: `relabel_three_way_markets` (core/utils.py:27-46) turns an `h2h` market that has a "Draw" outcome into `1x2` for soccer and normalises team names with `normalize_team_name`. Two-way sports are untouched. `canonicalize_bookmaker_keys` (data_fetcher.py:40-43) rewrites `betfair_ex_uk`/`betfair_ex_eu` to `betfair_exchange`, `ladbrokes_uk` to `ladbrokes`, `betfred_uk` to `betfred`, `sport888` to `888sport` in the stored dict (`ODDS_API_BOOKMAKER_KEY_MAP`, data_fetcher.py:31-37).
- **Other columns**: `snapshot_time = response.timestamp` (client-side `datetime.now(UTC)`, data_fetcher.py:236; for historical, the requested time `parse_api_datetime(snapshot_time)`, backfill_executor.py:268). `bookmaker_count = len(raw_data["bookmakers"])` (writers.py:330-340). `fetch_tier` and `hours_until_commence` computed from the Event's `commence_time` via `calculate_tier_from_timestamps` / `calculate_hours_until_commence` (writers.py:307-314, 336-343). `api_request_id` is not set (None).
- **Odds rows**: yes, via `store_odds_snapshot` (see "Odds rows" below).

### oddsportal (historical backfill, tag `oddsportal`)
- **Writer**: `scripts/ingest_oddsportal.py:392-430` builds the `OddsSnapshot` directly (not through `OddsWriter`), using `build_raw_data` in `lambda/oddsportal_common.py:124-221`. Two snapshots per event/market (opening, then closing).
- **Shape**: `{"bookmakers":[{key, title, last_update, markets:[{key: db_market, outcomes:[...]}]}], "source":"oddsportal"}`. `_snapshot_time` is a temporary top-level key that the script pops before storing (ingest_oddsportal.py:410, oddsportal_common.py:219).
  - `key = normalize_bookmaker_key(name)` (oddsportal_common.py:64-66; `BOOKMAKER_KEY_MAP` at :23-56, fallback is a lowercased alphanumeric slug). `title` is the OddsPortal display name. `last_update` is the fixed-year odds timestamp (`fix_odds_timestamp`, :95-121).
  - One market per bookmaker entry. `db_market` is `h2h` (team names), `1x2` (home team, `"Draw"`, away team) or `totals` (`"Over"`/`"Under"` with `point = line`) (ingest_oddsportal.py:86-112).
  - Outcome: `{"name", "price_decimal", "point"?}`. Entries with any price <= 1.0 or any missing outcome are dropped (oddsportal_common.py:23-29).
  - No `timing` or other tag. Opening vs closing is not recorded in `raw_data`; it exists only in `snapshot_time` and `fetch_tier`.
  - The market is encoded only in `api_request_id`, not in `raw_data["source"]` (see below).
- **Other columns**: `snapshot_time` = the latest fixed odds timestamp among the first outcomes (oddsportal_common.py:49-53). `fetch_tier = hours_to_tier(hours_before)` (oddsportal_common.py:77-87; the thresholds are <3, <12, <24, <72, else opening). `hours_until_commence = max(0, hours_before)`. `bookmaker_count = len(bookmakers)`. `api_request_id = "oddsportal"` for h2h, else `f"oddsportal_{db_market}"` (ingest_oddsportal.py:380-381, 421-426); used only as a dedupe key (:385-392).
- **Odds rows**: none are written.

### oddsportal_live (tag `oddsportal_live`)
- **Writer**: `lambda/oddsportal_adapter.py:64-121` (`convert_market`) builds `MatchOdds.raw_data`. `lambda/jobs/fetch_oddsportal.py:271-277` stores it through `OddsWriter.store_odds_snapshot`. One snapshot per event per market per scrape.
- **Shape**:
  ```
  {"bookmakers":[{"key","title","last_update","markets":[{"key","outcomes":[...]}], "betfair_matched"?: {<outcome label>: volume}}],
   "source":"oddsportal_live", "oddsportal_event_id": <id>}
  ```
  - `markets[].key` is `spec.pipeline_key` from `FEED_MARKETS` (lambda/oddsportal_client.py:158-181): `1x2` for `1x2`, `totals` for `over_under_2_5` (point 2.5), `h2h` for `home_away` (MLB).
  - Outcome: `{"name", "price_decimal"}` plus `"point"` when `spec.point` is set (adapter.py:41-57). Names are the team names from the listing, `"Draw"`, `"Over"`, `"Under"`.
  - `last_update = min(bk.last_change, scraped_at)` (adapter.py:89), ISO with `Z`.
  - `betfair_matched` (adapter.py:101-102) is written only when the feed carries `volumes`. It is `dict(zip(spec.outcomes, volumes))`, keyed `home|draw|away|over|under`. It is likely present only for the Betfair Exchange entry (not verified). It is a bookmaker-level key, not an outcome key.
  - `oddsportal_event_id` is the top-level OddsPortal event id.
- **Other columns**: `snapshot_time = match.scraped_date`. `fetch_tier`, `hours_until_commence` and `bookmaker_count` come from the writer. `api_request_id = f"{LIVE_REQUEST_ID_PREFIX}{YYYY-MM-DDTHH:MM}"` is set after the store (fetch_oddsportal.py:226-228, 277).
- **Odds rows**: yes. Because every outcome row is written, the OP `betfair_exchange` entry also produces Odds rows (same key as the real Betfair API).

### betfair_api (tag `betfair_api`)
- **Writer**: `lambda/betfair/adapter.py:187-234` builds the bookmaker entry and `:232-234` the envelope. `lambda/jobs/fetch_betfair_exchange.py:233-246` stores `{"bookmakers":[entry], "source":"betfair_api", "betfair_event_id": book.betfair_event_id}` through `store_odds_snapshot`.
- **Shape of the single bookmaker entry**:
  - `key:"betfair_exchange"`, `title:"Betfair Exchange"`, `last_update = snapshot_time.isoformat()`.
  - `markets:[{"key": sport_cfg.market_key, "outcomes":[{"name","price_decimal": runner.best_back}]}]`. `market_key` is `1x2` for EPL and `h2h` for MLB (betfair/constants.py:49-60). Names are the canonical home/away team or `"Draw"`. Runners without a best back price are dropped (adapter.py:160-161). A book with fewer than 2 or 3 priced outcomes is dropped (adapter.py:205-215).
  - `betfair_meta` (adapter.py:222-228): `market_id, market_status, inplay, total_matched, runners:[asdict(BetfairRunner)]`. `BetfairRunner` (betfair/client.py:42-51) has fields `selection_id, runner_name, best_back, best_lay, back_size, lay_size, last_price_traded`. The module docstring at adapter.py:20-25 says `name` where the code actually writes `runner_name`.
  - Only the BACK side is a top-level outcome price. LAY price and sizes exist only inside `betfair_meta.runners`. There is no `h2h_lay` market.
- **Other columns**: `snapshot_time` is the job's snapshot time. Tier and hours come from the writer. `api_request_id = f"{LIVE_REQUEST_ID_PREFIX}{snapshot_time.isoformat()}"` (fetch_betfair_exchange.py:244). `bookmaker_count = 1`.
- **Odds rows**: yes, back price only. One Odds row per outcome with `bookmaker_key="betfair_exchange"`. `betfair_meta` is not carried into Odds.

### football_data_uk (tag `football_data_uk`; opening vs closing via `timing.basis`)
- **Writer**: `scripts/ingest_football_data_uk.py:318-363` (`_build_snapshot_raw_data`) and `:500-570` (snapshot construction). `OddsSnapshot` is built directly (not through `OddsWriter`).
- **Shape**:
  ```
  {"bookmakers":[{"key","title"(=key),"last_update","markets":[{"key":"1x2","outcomes":[home, Draw, away]}]}],
   "source":"football_data_uk",
   "timing":{"basis":"fduk_collection_deadline"|"kickoff","kickoff_time_assumed":bool},
   "match_stats"?:{home_shots, away_shots, ..._ht_goals, referee}}
  ```
  - Bookmaker entries from CSV column prefixes (`BOOKMAKER_PREFIXES`, :72-89): PS=pinnacle, B365=bet365, BW=bwin, BFE=betfair_exchange, BF=betfair_sportsbook, IW=interwetten, WH=williamhill, VC and BV=betvictor, BFD=betfred, PP=paddypower, SKB=skybet, BMGM=betmgm, CL=coral, LB=ladbrokes, 1XB=onexbet. Aggregate pseudo-books `market_max` and `market_avg` come from `Max/Avg/BbMx/BbAv` (:92-97). The first key wins (`seen`, :342-348).
  - Closing uses column prefix + `C` (:340-342). `use_closing=False` is the opening.
  - Outcome: `{"name", "price_decimal"}`, names are home team, `"Draw"`, away team (:271-275). No `point`.
  - `match_stats` is post-match data stored in the pre-match snapshot (:303-315, 359-361).
  - `last_update` is filled to the snapshot time with `Z` (:541-542). `title` equals `key`.
- **Other columns**:
  - Closing: `snapshot_time = commence_time`, tier `hours_to_tier(hours_before)` (the closing tier).
  - Opening: `snapshot_time = opening_snapshot_time(...)`, the latest documented FDUK collection deadline (Friday 17:00 or Tuesday 13:00 UK). It is withheld (no snapshot) when the deadline is at or after kickoff (:207-218). `fetch_tier = opening_fetch_tier(...)` never returns the closing tier (:221-224).
  - `hours_until_commence = hours_before`. `bookmaker_count = len(bookmakers)`. `api_request_id = "football_data_uk"` (:548).
  - Existing snapshots and their Odds rows are deleted and replaced if content differs (:563-583).
- **Odds rows**: none are written by this script (its delete of Odds only clears old rows linked to the replaced snapshots).

### polymarket (tag `polymarket`; Gamma/CLOB, EPL 1x2 and MLB h2h)
- **Writer**: `lambda/polymarket_odds.py:299-344` (`build_raw_data`). Live: `lambda/jobs/fetch_polymarket_odds.py:157-165` via `store_odds_snapshot`. Backfill: `lambda/polymarket_odds_backfill.py:70-82`, building `OddsSnapshot` directly.
- **Shape**:
  ```
  {"bookmakers":[{"key":"polymarket","title":"Polymarket","last_update": snapshot_time.isoformat(),
     "markets":[{"key": sport.market_key, "outcomes":[{"name": home|"Draw"|away, "price_decimal": 1/p}]}],
     "polymarket_meta":{"pm_event_id","ticker","yes_probabilities":{home,draw,away},"probability_sum",
                        "best_bid"?:{home,draw,away},"best_ask"?:{home,draw,away}}}],
   "source":"polymarket"}
  ```
  - `price_decimal = 1 / midpoint`. The Yes probabilities are stored as both a price and a probability. There is no sizing, volume or depth.
  - `best_bid` / `best_ask` are written only for live snapshots (`include_quotes=True`, fetch_polymarket_odds.py:158-160). Backfill snapshots lack them.
  - Market key `1x2` (EPL) or `h2h` (MLB) from `lambda/polymarket_sports.py:36,45`. Live snapshots are accepted only when each side's `best_bid`/`best_ask` yields a midpoint in (0,1), the max spread is <= 0.03 and the Yes sum is within 0.03 of 1 (`MAX_QUOTE_SPREAD`, `SUM_TOLERANCE`, polymarket_odds.py:47-51). The backfill resamples a grid (:242-296).
- **Other columns**:
  - Live: `snapshot_time` = the job time; `fetch_tier` and `hours_until_commence` come from the writer; `api_request_id = f"polymarket_live_{snapshot_time.isoformat()}"` (:164).
  - Backfill: `snapshot_time = grid point`, `api_request_id = "polymarket_history"`, `fetch_tier = calculate_tier(hours).value`, `hours_until_commence`, `bookmaker_count = 1` (backfill.py:73-82).
- **Odds rows**: live yes (writer); backfill none.

### Odds rows (`OddsWriter._parse_and_store_odds`, lambda/storage/writers.py:367-427)
- Called from `store_odds_snapshot` at writers.py:349-354. Applies to Odds API, oddsportal_live, betfair_api and polymarket live. It does not apply to oddsportal historical, FDUK or polymarket backfill.
- Fields per (bookmaker, market, outcome):
  - `event_id`, `snapshot_id`.
  - `bookmaker_key = str(key).strip()`. It is NOT canonicalised with `canonical_bookmaker_key`; entries with an empty key are skipped.
  - `bookmaker_title = title or key`.
  - `market_key`, `outcome_name`.
  - `price_decimal = outcome_price_decimal(outcome)` (writers.py:~417).
  - `point = outcome.get("point")`.
  - `odds_timestamp = snapshot_time`.
  - `last_update = parse_api_datetime(bookmaker.last_update)`, falling back to `snapshot_time`.
  - `is_valid = True`.
- `Odds` has no source or kind column. Source is only recoverable via `snapshot_id` to `raw_data["source"]`. A `CheckConstraint(price_decimal > 1.0)` exists (models.py:~227).
- `extract_odds_from_snapshot` (core/snapshot_utils.py:16-61) re-derives equivalent un-persisted `Odds` objects from any snapshot, including historical OP and FDUK ones that have no stored Odds rows.

## Readers

Reads of `raw_data`:

| reader | fields | purpose | file:line |
|---|---|---|---|
| `snapshot_kind` | `source`, `timing.basis` | classify snapshot into `SnapshotKind` (raises on unknown tag) | core/snapshot_sources.py:49-63 |
| `book_prices` | `bookmakers[].key`, `markets[].key`, `outcomes[].name`, price (`price_decimal` or `price`) | per-book decimal price per home/draw/away; keeps prices > 1.0; first entry wins per canonical key; only markets `1x2` and `h2h` are defined | core/market_prices.py:39-62 |
| `is_reference_only_snapshot` | `bookmakers[].key` | detect polymarket-only snapshots | core/bookmakers.py:68-71 |
| `is_in_play`, `complete_exchange_prices`, `exchange_price`, `event_close` | `bookmakers[].key`, `bookmakers[].betfair_meta.inplay`, prices via `book_prices` | the Betfair close (last non-in-play live Betfair snapshot within 30 min of kickoff, else FDUK closing), devig and overround | core/betfair_close.py:32-117 |
| `raw_data_has_market` | `bookmakers[].markets[].key` | filter snapshots by market key | core/utils.py:9-20 (used in storage/readers.py:347,707,809; analytics/feature_groups.py:148) |
| `relabel_three_way_markets` | `markets[].key`, `outcomes[].name` | write-time relabel h2h to 1x2 and normalise team names | core/utils.py:27-46 |
| `extract_odds_from_snapshot` | `bookmakers[].key/title/last_update`, `markets[].key`, `outcomes[].name/price/price_decimal/point` | rebuild `Odds` objects from JSON (no alias canonicalisation) | core/snapshot_utils.py:16-61 |
| `build_market_state` / `load_market_state` | via `snapshot_kind`, `book_prices`, `is_in_play`, `exchange_price`; kind sets from `as_of.py` | live and backtest market state at decision time; retail roster filtered to `UK_RETAIL_BOOKMAKERS` | lambda/candidate_rules/market_state.py:66-171 (raw_data uses at :98,103,107,120) |
| `build_event_rows` (retail EV) | same as above plus `polymarket_meta.best_bid/best_ask` | decision rows: retail quotes, exchange price at t, Polymarket price and spread, dynamics, close | analytics/retail_ev.py:301-347 (polymarket at :218-232) |
| retail EV evaluation | `snapshot_kind(raw_data)` | per-event flags: has FDUK closing / OP live / live Betfair near kickoff | analytics/retail_ev_evaluation.py:256 |
| candidate report | SQL JSON path `raw_data->source`, `raw_data->timing->basis` | select the snapshots `event_close` can pick | analytics/candidate_report.py:229-235 |
| `snapshot_has_bookmaker` | `bookmakers[].key`, `markets[].key` | check a bookmaker quotes a market (raw key, no aliasing) | analytics/feature_groups.py:108-118 |
| `collect_event_data` | `markets[].key`, `bookmakers[].key` (via `is_reference_only_snapshot`), `fetch_tier` column | filter snapshots to the market and drop polymarket-only; closing snapshot = last snapshot whose `fetch_tier` equals the closing tier | analytics/feature_groups.py:148-155 |
| feature groups (tabular, target) | via `extract_odds_from_snapshot` (all of bookmaker key/title, market, outcome name, price, point) | tabular features and CLV target | analytics/feature_groups.py:525, 580, 685-689 |
| `get_sharp_prices` | via `raw_data_has_market` + `extract_odds_from_snapshot` | first snapshot with a sharp book, in priority order, per outcome | lambda/storage/readers.py:658-750 (raw at :707, :~716) |
| `get_latest_book_prices` | same | per `(bookmaker_key, outcome_name, point)` newest price within lookback | lambda/storage/readers.py:752-838 (raw at :809, key tuple :826) |
| `get_latest_snapshot` | `markets[].key` | newest snapshot containing a market | lambda/storage/readers.py:321-349 (:347) |
| MCP `get_current_odds` | `raw_data` blob returned wholesale if `include_raw_data`; per-book prices via reader | surface to the agent | mcp/server.py:143, 274 |
| MCP `get_odds_history` | via `extract_odds_from_snapshot` | per-snapshot structured odds | mcp/server.py:352 |
| MCP `find_retail_edges` | via reader entries; `Odds.bookmaker_key/outcome_name/point/price_decimal/american_price/implied_probability`; `NON_BETTABLE_BOOKMAKERS` | retail vs sharp edge ranking, per-book holds | mcp/server.py:975-1060 |
| `snapshot_prices` and pairing | via `extract_odds_from_snapshot`: (bookmaker_key stripped, market_key, outcome_name, point) to price_decimal; `raw_data["source"]` | find identical or conflicting same-time snapshot pairs and delete same-time duplicates (MLB oddsportal_live conflicts) | lambda/storage/same_time_pairs.py:91-110, 99-103 |
| orphan Odds SQL | JSON path `bookmakers[].key`, `markets[].key`, `outcomes[].name`, `point` | find Odds rows no snapshot quotes | lambda/storage/same_time_pairs.py:336-352 |
| health monitor | SQL `raw_data->source` (as_string), `betfair_meta.inplay`, prices | last snapshot per source (staleness), missing Betfair close alert | lambda/health_monitor.py:269, 329, 353-395 |
| collection calendar | SQL `raw_data->source` | latest snapshot time per source | lambda/collection_calendar.py:238-253 (:241) |
| event name repair | `source = betfair_api`, `bookmakers[].betfair_meta.runners[].runner_name` | canonical team names from the latest Betfair snapshot | lambda/storage/event_name_repair.py:46-63 |
| copy from production | whole snapshot via `model_dump` | clone snapshots and Odds rows to local DB | lambda/copy_from_production.py:144-196 |
| writer validation | `bookmakers[].key/last_update`, `markets[].key`, `outcomes[].name`, price | quality log warnings (future timestamps, bad prices, vig on 2-outcome h2h/spreads/totals) | lambda/storage/validators.py:~55-110 |

Reads of `Odds` objects and snapshot columns (no raw_data):

| reader | fields | purpose | file:line |
|---|---|---|---|
| `OddsReader.get_odds_at_time`, `get_line_movement`, `get_best_odds` | `Odds` table: `event_id`, `odds_timestamp`, `bookmaker_key`, `market_key`, `outcome_name`, `price_decimal` | direct SQL on the Odds table (closest time, line history, best price). Callers not traced. | lambda/storage/readers.py:168-319 |
| `get_database_stats` | `count(Odds.id)`, `count(OddsSnapshot.id)` | status counts | lambda/storage/readers.py:450-470; odds-cli `commands/status.py:66` |
| tier readers | `OddsSnapshot.fetch_tier` | tier coverage, first and last snapshot in tier | lambda/storage/readers.py:490-615 |
| quality metrics | `OddsSnapshot.fetch_tier`, snapshot counts, `Odds.bookmaker_key/title/event_id/id` | tier coverage and bookmaker coverage | analytics/quality_metrics.py:397-454, 529-575 |
| tabular feature extraction | `Odds.market_key`, `outcome_name`, `bookmaker_key`, `implied_probability`, `american_price` | average price and probability, sharp vs retail diffs | analytics/feature_extraction.py:84-150, 364-436 |
| CLV targets | `Odds.implied_probability`, `Odds.point` | opening vs closing probability delta, line movement | analytics/targets.py:22-63 |
| MCP serialization | `Odds.bookmaker_key/title`, `market_key`, `outcome_name`, `american_price`, `price_decimal`, `point`, `last_update`; snapshot `bookmaker_count`, `fetch_tier`, `hours_until_commence`, `created_at` | tool output | mcp/server.py:101-141, 272-315 |
| `fetch_tier`, `hours_until_commence` recalculation | snapshot columns | recompute when an event's commence time changes | lambda/storage/event_merge.py:143-154 |
| scrape CLI | `bookmaker_count` (of the `MatchOdds`, not the column) | display | odds-cli `commands/scrape.py:81` |
| scripts | `fetch_tier`, `hours_until_commence` | tier distribution and coverage checks | scripts/check_tier_distribution.py:31-47; scripts/check_game_tier_coverage.py:34-90 |
| backfill dedupe | `api_request_id` | skip already-ingested events | scripts/ingest_oddsportal.py:385-392; scripts/ingest_football_data_uk.py:563; lambda/polymarket_odds_backfill.py:130 |

Not found reading raw_data: `analytics/targets.py`, `analytics/utils.py`, `analytics/backfill_executor.py` (writes only), `analytics/training/data_preparation.py`. `snapshot_utils.extract_odds_from_snapshot` is the main bridge from `raw_data` to `Odds`-shaped objects for analytics and MCP.

## Written but unread

No non-test reader was found for these (grep for the key names outside tests):
- FDUK `match_stats` (all 14 stat keys and `referee`) (written ingest_football_data_uk.py:359-361; zero grep hits for `match_stats` in packages/). Probably left over from the removed `match_stats` feature group.
- FDUK `timing.kickoff_time_assumed`. Only `timing.basis` is read.
- FDUK pseudo-books `market_max` / `market_avg`. They are never in `UK_RETAIL_BOOKMAKERS`. Whether they leak into Odds-based feature code (`extract_odds_from_snapshot`) is unverified.
- Betfair `betfair_meta.market_id`, `market_status`, `total_matched`, and all runner fields except `runner_name` (`selection_id`, `best_back`, `best_lay`, `back_size`, `lay_size`, `last_price_traded`). Only `betfair_meta.inplay` (betfair_close.py:34) and `runners[].runner_name` (event_name_repair.py:60-61) are read. No lay price or size is read by any code.
- Betfair top-level `betfair_event_id`; OddsPortal live `oddsportal_event_id`; OP live `betfair_matched` volumes.
- Polymarket `polymarket_meta.pm_event_id`, `ticker`, `yes_probabilities`, `probability_sum`. Only `best_bid`/`best_ask` are read (retail_ev.py:224-232).
- Bookmaker `title` is read only when writing `Odds.bookmaker_title` and by the MCP/quality metrics. `last_update` reaches only `Odds.last_update` and the validator.
- Odds API: event-level duplicates (`id, sport_key, sport_title, commence_time, home_team, away_team`), market-level `last_update` if present, and any `spreads`/`totals` content in the typed read path. `market_outcomes` (core/market_prices.py:24-29) raises for markets other than `1x2`/`h2h`, so `totals`/`spreads`/`point` are only reachable through `extract_odds_from_snapshot` and `Odds` rows (feature code, MCP).
- `Odds.is_valid`, `Odds.validation_notes`, `Odds.created_at`: no non-test reader seen.
- `OddsSnapshot.api_request_id` is never used to interpret prices, only as an ingest dedupe key, and for OP historical it holds the market (`oddsportal_totals`) while `raw_data["source"]` does not.

## Legacy Polymarket tables (core/polymarket_models.py)

These are separate from the `polymarket`-tagged OddsSnapshots above. They were an NBA-oriented pipeline (docstrings mention NBA tickers). Consumers: `lambda/jobs/backfill_polymarket.py`, `lambda/storage/polymarket_writer.py`, `lambda/storage/polymarket_reader.py`, `analytics/polymarket_features.py`, `analytics/feature_groups.py` (PM feature group). Gated by `settings.polymarket.enabled` (health_monitor.py:70-71).

**PolymarketEvent** (`polymarket_events`):
- Maps to a price row (event / time context): `event_id` (FK to `events.id`, nullable, matched asynchronously), `start_date`.
- Does not map: `pm_event_id`, `ticker`, `slug`, `title`, `end_date`, `active`, `closed`, `volume`, `liquidity`, `markets_count`, `created_at`, `updated_at`.

**PolymarketMarket** (`polymarket_markets`):
- Maps to market / outcome / line: `market_type` (moneyline, spread, total, player_prop, other), `point` (line), `outcomes` (JSON list of labels, index-aligned with `clob_token_ids`), `clob_token_ids` (JSON list).
- Does not map: `pm_market_id`, `condition_id`, `question`, `group_item_title`, `active`, `closed`, `accepting_orders`, `created_at`.

**PolymarketPriceSnapshot** (`polymarket_price_snapshots`):
- Maps to time and price: `snapshot_time`; `outcome_0_price` and `outcome_1_price` (implied probabilities 0-1, i.e. a price per outcome for exactly two outcomes, wide not long); `polymarket_market_id` (to market).
- Does not map directly: `volume` and `liquidity` (market-level, not per-outcome or per-side size); `fetch_tier`, `hours_until_commence`. Two-outcome only, so a 3-way (draw) market cannot be represented; the new snapshot path instead uses one market per outcome.

**PolymarketOrderBookSnapshot** (`polymarket_orderbook_snapshots`):
- Maps to price row with side and size: `snapshot_time`, `token_id` (outcome identity), `best_bid` and `best_ask` (back/lay prices; bid and ask sides), `midpoint`; `raw_book` JSON `{bids:[...], asks:[...]}` holds per-level price and size, i.e. the only source of per-side size; `bid_depth_total` and `ask_depth_total` are aggregated size.
- Does not map: `spread`, `bid_levels`, `ask_levels`, `imbalance`, `weighted_mid` (derived), `fetch_tier`, `hours_until_commence`.

Also `PolymarketFetchLog` (not a price table).

## Bookmaker and outcome identity

**Bookmaker keys**
- `canonical_bookmaker_key(key, kind)` (core/bookmakers.py:62-65): for `SnapshotKind.ODDS_API` only, `betmgm` becomes `betmgm_us` (`ODDS_API_KEY_ALIASES`, :59). For all kinds `BOOKMAKER_ALIASES` (:19-28) is applied: `betfair_ex_uk` to `betfair_exchange`, `betfair` and `betfair_sb_uk` to `betfair_sportsbook`, `betfred_uk` to `betfred`, `betmgmuk` to `betmgm`, `ladbrokes_uk` to `ladbrokes`, `sport888` to `888sport`, `7betuk` to `7bet`.
- It is used only in `book_prices` (market_prices.py:45) and `is_in_play` (betfair_close.py:36). Every other reader (`extract_odds_from_snapshot`, `Odds.bookmaker_key`, `snapshot_has_bookmaker`, `get_sharp_prices`, `is_reference_only_snapshot`, MCP, same_time_pairs) uses raw stored keys.
- Per source, at write time:
  - Odds API: `canonicalize_bookmaker_keys` rewrites a subset (`betfair_ex_uk/eu`, `ladbrokes_uk`, `betfred_uk`, `sport888`) in the stored JSON (data_fetcher.py:31-43). `betmgm` (US) is stored as `betmgm`, the same key OP and FDUK use for BetMGM UK, which is why the read-time aliasing exists.
  - OP (both): `normalize_bookmaker_key` (`BOOKMAKER_KEY_MAP`, oddsportal_common.py:23-66, slug fallback).
  - Betfair API: fixed `betfair_exchange`.
  - FDUK: fixed prefix map (ingest_football_data_uk.py:72-89), plus `market_max` and `market_avg`.
  - Polymarket: `polymarket`.
- Roles are frozensets (`EXCHANGE_BOOKMAKERS`, `PREDICTION_MARKET_BOOKMAKERS`, `ZERO_MARGIN_BOOKMAKERS`, `NON_BETTABLE_BOOKMAKERS`, `UK_RETAIL_BOOKMAKERS`) in core/bookmakers.py:9-57. `UK_RETAIL_BOOKMAKERS` is defined by canonical key.

**Outcome identity**
- `outcome_names(event, market)` (core/market_prices.py:32-36) maps `{event.home_team: HOME, event.away_team: AWAY}` and, for `1x2` only, `{"Draw": DRAW}` (`DRAW_OUTCOME_NAME`, :13). It is a name match against the Event's team strings (no ids). Outcomes whose name is not in this dict are silently skipped (`book_prices`, :55-57).
- `Outcome` enum is only `HOME, DRAW, AWAY` (market_prices.py:17-21). `market_outcomes` supports only `1x2` and `h2h` (:24-29). There is no `over` / `under` mapping in the typed path.
- Over/Under exist only as literal `"Over"` / `"Under"` outcome names with `point`, written by OP historical and live (`totals`) and the Odds API. The Odds API `totals` and `spreads` also use `name` = team or Over/Under plus `point`.
- Draw is always the literal `"Draw"` across Odds API, OP, Betfair (runner "The Draw" mapped at adapter.py:47,164-166), FDUK and Polymarket (`DRAW_OUTCOME`, polymarket_odds.py:34).
- Team names are canonicalised: Odds API via `relabel_three_way_markets` only for three-way (soccer) sports (two-way sport names are stored unchanged); Betfair via `normalize_team_name` on runner names; OP via `normalize_team_name` in the Event and listing; Polymarket via `canonical_team` (polymarket_odds.py:138-139) then replaced by `event.home_team`/`away_team` strings in `build_raw_data` (:307-311).
- Side semantics are source-specific: OP, Odds API, FDUK and Polymarket store a single price per outcome with no side; Betfair stores the BACK price as the outcome price and the LAY and sizes only in `betfair_meta.runners`; Polymarket stores the midpoint as a decimal price (1/p) and, for live snapshots only, bid and ask per side in `polymarket_meta`.
- `book_prices` first-entry-wins semantics (:42-46) means a book duplicated under an alias is silently dropped after the first. For `h2h`, `book_prices` takes the first market with the matching key and ignores `point`.

## Uncertainties
- Whether the Odds API response includes market-level `last_update` is from API knowledge, not from this repo's code.
- `betfair_matched` appears only on the Betfair Exchange entry (verified in the stored data above).
- Line numbers labelled `~` are approximate; others were read directly. Callers of `OddsReader.get_odds_at_time/get_line_movement/get_best_odds` were not traced.
- Existing production data shape may differ from current writer code (older rows, and rows from before the `source` tag and `price_decimal` changes, e.g. Odds API rows hold `price`).
