---
type: reference
title: OddsPortal scraper
description: The OddsPortal HTTP feed client, the adapter that converts its prices, the shared oddsportal_common helpers, the results page archive reader, and odds scrape upcoming.
sources:
  - odds_lambda.oddsportal_client
  - odds_lambda.oddsportal_client.OddsPortalClient
  - odds_lambda.oddsportal_client.OddsPortalClient.fetch_listing
  - odds_lambda.oddsportal_client.OddsPortalClient.fetch_bookmaker_context
  - odds_lambda.oddsportal_client.OddsPortalClient.fetch_market
  - odds_lambda.oddsportal_client.parse_listing
  - odds_lambda.oddsportal_client.extract_market
  - odds_lambda.oddsportal_client.select_main_line_window
  - odds_lambda.oddsportal_client.decrypt_payload
  - odds_lambda.oddsportal_client.FEED_MARKETS
  - odds_lambda.oddsportal_client.OddsPortalError
  - odds_lambda.oddsportal_adapter.convert_market
  - odds_lambda.oddsportal_client.OddsPortalClient.fetch_results
  - odds_lambda.oddsportal_common.build_envelope
  - odds_lambda.oddsportal_common.event_id_from_match_link
  - odds_lambda.oddsportal_common.BOOKMAKER_KEY_MAP
  - odds_lambda.oddsportal_common.normalize_bookmaker_key
  - odds_core.odds_math.PRICE_DECIMAL_KEY
  - odds_cli.commands.scrape.scrape_upcoming
  - pyproject.toml
status: stable
---

# OddsPortal scraper

Live pre-match odds from OddsPortal (UK bookmakers [the Odds API](odds-api-client.md) lacks) and finished-match results are read over plain HTTP, with no browser and no OddsHarvester dependency (none of the `pyproject.toml` files declares it). The HTTP client serves [the fetch-oddsportal job](fetch-oddsportal-job.md) and `odds scrape upcoming`. Historical bulk loads use scripts instead ([historical-ingest-scripts](historical-ingest-scripts.md)). The reasons for reading the feed are in [ADR 0009](../../docs/adr/0009-oddsportal-feed-client.md).

## Modules

- `odds_lambda.jobs.fetch_oddsportal`: `odds_lambda.jobs.fetch_oddsportal.LEAGUE_SPECS`, `odds_lambda.jobs.fetch_oddsportal.scrape_league`, `odds_lambda.jobs.fetch_oddsportal.ingest_league`. `odds_lambda.jobs.fetch_oddsportal.LEAGUE_SPEC_BY_NAME`, keyed by each spec's `league` (`england-premier-league`, `mlb`, `nba`), serves the CLI and MCP scrape tools. `odds_lambda.jobs.fetch_oddsportal.main` scrapes only the league of its required `ctx.sport`. A queued run calls `odds_lambda.jobs.fetch_oddsportal.ingest_league` directly, so it skips the job's dormancy gate, scheduling (including its [runs before candidate decisions](fetch-oddsportal-job.md#runs-before-candidate-decisions)), alert context, heartbeat and its empty-collection check, which live in `odds_lambda.jobs.fetch_oddsportal.main` ([empty collection](../operations/alerting-and-heartbeats.md#empty-collection)); its snapshots are stored as source `oddsportal`, time basis `observed`, like the job's.
- `odds_lambda.oddsportal_client`: `odds_lambda.oddsportal_client.OddsPortalClient` (async context manager owning one `aiohttp` session), the pure functions `odds_lambda.oddsportal_client.parse_listing`, `odds_lambda.oddsportal_client.extract_market`, `odds_lambda.oddsportal_client.decrypt_payload`, `odds_lambda.oddsportal_client.parse_archive_path` and `odds_lambda.oddsportal_client.parse_results_page`, and the market table `odds_lambda.oddsportal_client.FEED_MARKETS`. Every failure it cannot recover from is an `odds_lambda.oddsportal_client.OddsPortalError`.
- `odds_lambda.oddsportal_adapter`: pure conversion of feed prices to the envelope shape (`odds_lambda.oddsportal_adapter.convert_market`).
- `odds_lambda.oddsportal_common`: `odds_lambda.oddsportal_common.BOOKMAKER_KEY_MAP`/`odds_lambda.oddsportal_common.normalize_bookmaker_key`, `odds_lambda.oddsportal_common.build_envelope` (historical records), `odds_lambda.oddsportal_common.event_id_from_match_link` (the event id a match link ends with, or the fragment of a head-to-head link, else none).

## Feed client

One scrape of a league makes these requests, each with a desktop Chrome `User-Agent`:

1. `odds_lambda.oddsportal_client.OddsPortalClient.fetch_listing` GETs the league's listing page (`odds_lambda.jobs.fetch_oddsportal.LeagueSpec.listing_path`). `odds_lambda.oddsportal_client.parse_listing` joins the strings of every `self.__next_f.push([1, ...])` script, reads each `<id>:<json>` line that parses as JSON, and collects every object carrying `encodeEventId`, `home-name`, `away-name`, `date-start-timestamp`, `url` and `sport-id`, deduplicated by event id. Kickoff is the UTC `date-start-timestamp`. When no event is found and the payload carries a `productErrorType`, it logs `oddsportal_listing_empty` and returns an empty list. With no event and no `productErrorType` the page format has drifted, and it raises `odds_lambda.oddsportal_client.OddsPortalError`.
2. `odds_lambda.oddsportal_client.OddsPortalClient.fetch_bookmaker_context` GETs `/proxy/api/home-data` for `hashedBookmakers` and `geoIPcode`, and `/proxy/ajax-providers-bonus-data/0/` for the bookmaker id to display-name map. The bookmaker set is whatever OddsPortal serves to the caller's region.
3. `odds_lambda.oddsportal_client.OddsPortalClient.fetch_market` GETs `/proxy/match-event/1-{sportId}-{eventId}-{bettingType}-{scope}-{hashedBookmakers}.dat?geo={geo}&lang=en` with the event page URL, minus its fragment, as `Referer`, once per scope of the market. `odds_lambda.oddsportal_client.FEED_MARKETS` is the single market table, keyed by LeagueSpec market key. Each `odds_lambda.oddsportal_client.FeedMarketSpec` holds the feed `betting_type`, `scopes` and `handicap` (as it appears in the feed key), the pipeline market key (`pipeline_key`), the outcome labels in feed order (`outcomes`), the totals line (`point`, `None` otherwise) and, for a ladder, `ladder`, `mirror_line` and `line_window`: `1x2` (type 1, scope 2, `home`/`draw`/`away`), `over_under_2_5` (type 2, scope 2, handicap 2.5, `over`/`under`), `home_away` (type 3, scope 1, `home`/`away`), `totals_ladder` (type 2, scopes 1 and 2, `over`/`under`) and `spreads_ladder` (type 5, scopes 1 and 2, `home`/`away`, `mirror_line`). It sleeps `request_delay` (1 s by default) before each feed request.

### Ladder markets

Basketball totals and handicaps have no fixed line: one request returns every line the books quote (keys `E-{type}-{scope}-0-{line}-0`, 40 to 85 lines per game, a handicap line signed as the home side's), so a `ladder` spec carries the line as data. The feed key's line is lossy, because the site strips trailing zeros (`E-5-1-0-1-0` is handicap 10, `E-2-1-0-23-0` is a total of 230), so `odds_lambda.oddsportal_client.extract_market` uses the key only to filter betting type and scope and reads each line from the market's `handicapValue`; a ladder market without one is skipped with a warning. Bookmakers are split between scopes 1 and 2 for these markets (scope 1 holds bet365, 10bet and bwin, scope 2 Sky Bet, William Hill, Paddy Power and the Betfair Exchange handicap) with no overlap seen. The site's event metadata labels scope 1 "FT including OT" and scope 2 "Full Time", but in three live games scope 2's over/under and handicap books priced the same main lines and the same de-vigged P(over) within about 1 percentage point of scope 1's, where a regulation-only market would sit 2 to 3 points lower, so both scopes are overtime-inclusive and are read as one market (the regulation-only markets, such as NBA three-way, are only in scope 2), so `odds_lambda.oddsportal_client.OddsPortalClient.fetch_market` requests both and keeps a bookmaker's first scope on a repeat. `odds_lambda.oddsportal_client.select_main_line_window` then picks the main line, the one the most bookmakers quote (ties to the line whose two prices are closest in implied probability, median across bookmakers), and keeps every line within `line_window` points (2.0) of it, so a bad main-line pick still has its neighbours and the stored ladder is about 8 lines, not 40 to 85. Whole-number lines (2 or 3 bookmakers) and the Exchange's one-book handicap ladder fall inside or outside the window like any other. Each bookmaker's rows carry their `line` and their own change time and volumes; fixed markets carry the spec's `point`.

The three `/proxy/` endpoints also send the site's XHR headers (`X-Requested-With`, `x-client-platform: next-op`, `x-language`).

Behaviour:

- Responses from the `/proxy/` endpoints are base64 of `ciphertext:ivHex`, AES-256-CBC with a PBKDF2-SHA256 key, optionally gzipped; `odds_lambda.oddsportal_client.decrypt_payload` reverses that and also accepts plain JSON. The password and salt are constants in the site's JavaScript. A body that fails to decrypt raises `odds_lambda.oddsportal_client.OddsPortalError`, so a changed key fails loudly.
- HTTP 429, 5xx, connection errors and timeouts are retried three times with exponential backoff; any other non-200 status raises immediately. A `/proxy/` endpoint can also answer HTTP 200 with a plain-text body `URL:<path> Status: <code>` reporting an upstream error: a 429 or 5xx code there is retried like the HTTP status, and any other code raises immediately, naming the code and URL. Requests time out after 30 s.
- A feed whose `d` object holds an `E` key is an error envelope. The client waits the envelope's `refresh` seconds (default 20, capped at 30) and retries twice before raising.
- `odds_lambda.oddsportal_client.extract_market` reads `oddsdata.back["E-{type}-{scope}-0-{line}-0"]`. A missing market key, or an `oddsdata` or `back` that is absent or an empty list (no odds posted yet), yields no bookmakers. A `d` that is not an object, or an `oddsdata` or `back` of any other type, raises `odds_lambda.oddsportal_client.OddsPortalError`. It skips a bookmaker whose `act` flag is false (a suspended price), whose id is not in the name map, or whose prices are missing, at or below 1.0 or the wrong length. Prices and volumes arrive either as lists or as dicts keyed by outcome index, and both are accepted. `volume` (Betfair Exchange matched amounts) and the latest `changeTime` are carried through when present.

## Results archive

`odds_lambda.oddsportal_client.OddsPortalClient.fetch_results` reads a league's results page the same way as a listing: the page's RSC payload carries the URL of the paginated archive feed, which `odds_lambda.oddsportal_client.parse_archive_path` returns as a `/proxy/` path. Each `<path>/page/N/` response is encrypted like the odds feed and holds up to `onePage` rows (50) plus `total`, in descending kick-off order. A row has team names, a kick-off timestamp, a status id (3 is finished, 5 cancelled) and `homeResult` / `awayResult`. `odds_lambda.oddsportal_client.parse_results_page` turns finished rows into `odds_lambda.oddsportal_client.FinishedMatch` values, cancelled rows into `odds_lambda.oddsportal_client.CancelledMatch` values, and counts the rows it saw. `odds_lambda.oddsportal_client.OddsPortalClient.fetch_results` returns them as `odds_lambda.oddsportal_client.FetchedResults`; a page that fails after the first leaves `complete` false and keeps the earlier pages, and a failing first page raises. A real archive page is recorded in `tests/fixtures/oddsportal/mlb_results_feed.json`. The results page lists the current season only ([results-jobs](results-jobs.md#fetch-oddsportal-results)).

## Adapter

- `odds_lambda.oddsportal_adapter.convert_market` emits an envelope in The Odds API shape with `source: oddsportal_live` and the feed's encoded event id as `oddsportal_event_id`, so a stored snapshot can be traced to its OddsPortal game, stored, one snapshot per market with its price rows by `odds_lambda.storage.writers.OddsWriter.store_envelope`, and a snapshot already stored under its key is skipped ([odds-writer](../storage/odds-writer.md), [Writing price rows](../storage/price-tables.md#writing-price-rows)). It looks the market up in `odds_lambda.oddsportal_client.FEED_MARKETS`: the pipeline key is `1x2`, `totals`, `spreads` or `h2h`, outcome names come from the spec labels (`home` and `away` are the event teams), every outcome carries its row's `point` (a ladder bookmaker's entry holds one market element per line, each with its own `last_update` and `betfair_matched`; a `mirror_line` spec gives the away outcome the negative of the home line, so a spread is stored home -5.5, away +5.5, [price-tables](../storage/price-tables.md#spreads-line-sign)), and an unsupported market raises `ValueError("Unsupported market")` ([sport-and-market-keys](sport-and-market-keys.md#market-key-dialects)).
- Feed decimal prices are stored unconverted under `price_decimal`, with no `price` key (`odds_core.odds_math.PRICE_DECIMAL_KEY`); `odds_lambda.oddsportal_common.build_envelope` writes the same encoding for historical records, skipping a bookmaker with any price at or below 1.0 ([odds-math](../foundations/odds-math.md)). Each bookmaker's `last_update` is its latest feed change time, capped at the scrape time, or the scrape time when the feed gives none. Bookmakers with volumes get a `betfair_matched` entry keyed by the spec outcome labels, and in a ladder market each line's market element carries its own. Each `odds_lambda.oddsportal_adapter.MatchOdds` carries the market key it was converted for and its `envelope`. A market with no bookmakers converts to nothing.
- Events resolve through the [event resolver](../storage/event-resolver.md), which records the `oddsportal_event_id` as the OddsPortal external id (nothing merges): for MLB the event holding the game's MLB `gamePk`, else the event holding the OddsPortal id, else the nearest event within ±2 h ([event-matching](../storage/event-matching.md#live-mlb-events)). The historical ingest resolves through the same resolver with a ±24 h window and records the id its match link ends with (`odds_lambda.oddsportal_common.event_id_from_match_link`) ([historical-ingest-scripts](historical-ingest-scripts.md#matching-ids-and-idempotency)).

## odds scrape upcoming

`odds scrape upcoming --league --market --dry-run` (`odds_cli.commands.scrape.scrape_upcoming`) replaces the spec's markets with `--market` values, which must be keys of `odds_lambda.oddsportal_client.FEED_MARKETS`; `--dry-run` scrapes and prints the converted matches without storing them. `odds_lambda.jobs.fetch_oddsportal.scrape_league` raises `odds_lambda.oddsportal_client.OddsPortalError` when there were upcoming events and every market fetch failed, so a changed key or a blocked feed fails the league instead of returning zero matches; individual failures are otherwise recorded in the scrape errors. The scrape also counts the events the listing contained and how many of them had not started (`odds_lambda.jobs.fetch_oddsportal.LeagueScrape.upcoming_listed`) and the latest start time listed (`odds_lambda.jobs.fetch_oddsportal.LeagueScrape.latest_listed_start`); only upcoming events are fetched. The MCP `odds_mcp.server.refresh_scrape` tool queues `odds_lambda.jobs.fetch_oddsportal.ingest_league` in the local scheduler process and needs it running ([mcp-scheduling-tools](../agent/mcp-server.md#scrape-and-scheduling-tools)).

## Related

- [fetch-oddsportal job](fetch-oddsportal-job.md)
- [Sport, market and bookmaker keys](sport-and-market-keys.md)
- [Betfair Exchange client and job](betfair-exchange.md)
- [ADR 0009: read OddsPortal live odds from its JSON feed](../../docs/adr/0009-oddsportal-feed-client.md)
