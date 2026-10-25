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
  - odds_lambda.oddsportal_client.decrypt_payload
  - odds_lambda.oddsportal_client.FEED_MARKETS
  - odds_lambda.oddsportal_client.OddsPortalError
  - odds_lambda.oddsportal_adapter.convert_market
  - odds_lambda.oddsportal_client.OddsPortalClient.fetch_results
  - odds_lambda.oddsportal_common.build_raw_data
  - odds_lambda.oddsportal_common.find_existing_event
  - odds_lambda.oddsportal_common.BOOKMAKER_KEY_MAP
  - odds_lambda.oddsportal_common.normalize_bookmaker_key
  - odds_core.odds_math.PRICE_DECIMAL_KEY
  - odds_lambda.oddsportal_common.hours_to_tier
  - odds_cli.commands.scrape.scrape_upcoming
  - pyproject.toml
status: stable
---

# OddsPortal scraper

Live pre-match odds from OddsPortal (UK bookmakers [the Odds API](odds-api-client.md) lacks) and finished-match results are read over plain HTTP, with no browser and no OddsHarvester dependency (none of the `pyproject.toml` files declares it). The HTTP client serves [the fetch-oddsportal job](fetch-oddsportal-job.md) and `odds scrape upcoming`. Historical bulk loads use scripts instead ([historical-ingest-scripts](historical-ingest-scripts.md)). The reasons for reading the feed are in [ADR 0009](../../docs/adr/0009-oddsportal-feed-client.md).

## Modules

- `odds_lambda.jobs.fetch_oddsportal`: `odds_lambda.jobs.fetch_oddsportal.LEAGUE_SPECS`, `odds_lambda.jobs.fetch_oddsportal.scrape_league`, `odds_lambda.jobs.fetch_oddsportal.ingest_league`. `odds_lambda.jobs.fetch_oddsportal.LEAGUE_SPEC_BY_NAME`, keyed by each spec's `league`, serves the CLI and MCP scrape tools. `odds_lambda.jobs.fetch_oddsportal.main` scrapes only the league of its required `ctx.sport`. A queued run calls `odds_lambda.jobs.fetch_oddsportal.ingest_league` directly, so it skips the job's dormancy gate, alert context, heartbeat and its empty-collection check, which live in `odds_lambda.jobs.fetch_oddsportal.main` ([empty collection](../operations/alerting-and-heartbeats.md#empty-collection)).
- `odds_lambda.oddsportal_client`: `odds_lambda.oddsportal_client.OddsPortalClient` (async context manager owning one `aiohttp` session), the pure functions `odds_lambda.oddsportal_client.parse_listing`, `odds_lambda.oddsportal_client.extract_market`, `odds_lambda.oddsportal_client.decrypt_payload`, `odds_lambda.oddsportal_client.parse_archive_path` and `odds_lambda.oddsportal_client.parse_results_page`, and the market table `odds_lambda.oddsportal_client.FEED_MARKETS`. Every failure it cannot recover from is an `odds_lambda.oddsportal_client.OddsPortalError`.
- `odds_lambda.oddsportal_adapter`: pure conversion of feed prices to the `raw_data` shape (`odds_lambda.oddsportal_adapter.convert_market`).
- `odds_lambda.oddsportal_common`: `odds_lambda.oddsportal_common.BOOKMAKER_KEY_MAP`/`odds_lambda.oddsportal_common.normalize_bookmaker_key`, `odds_lambda.oddsportal_common.hours_to_tier`, `odds_lambda.oddsportal_common.build_raw_data` (historical records), `odds_lambda.oddsportal_common.find_existing_event`. Besides these helpers it holds a DB query. `odds_lambda.betfair.adapter.build_raw_data` is an unrelated function of the same name.

## Feed client

One scrape of a league makes these requests, each with a desktop Chrome `User-Agent`:

1. `odds_lambda.oddsportal_client.OddsPortalClient.fetch_listing` GETs the league's listing page (`odds_lambda.jobs.fetch_oddsportal.LeagueSpec.listing_path`). `odds_lambda.oddsportal_client.parse_listing` joins the strings of every `self.__next_f.push([1, ...])` script, reads each `<id>:<json>` line that parses as JSON, and collects every object carrying `encodeEventId`, `home-name`, `away-name`, `date-start-timestamp`, `url` and `sport-id`, deduplicated by event id. Kickoff is the UTC `date-start-timestamp`. When no event is found and the payload carries a `productErrorType`, it logs `oddsportal_listing_empty` and returns an empty list. With no event and no `productErrorType` the page format has drifted, and it raises `odds_lambda.oddsportal_client.OddsPortalError`.
2. `odds_lambda.oddsportal_client.OddsPortalClient.fetch_bookmaker_context` GETs `/proxy/api/home-data` for `hashedBookmakers` and `geoIPcode`, and `/proxy/ajax-providers-bonus-data/0/` for the bookmaker id to display-name map. The bookmaker set is whatever OddsPortal serves to the caller's region.
3. `odds_lambda.oddsportal_client.OddsPortalClient.fetch_market` GETs `/proxy/match-event/1-{sportId}-{eventId}-{bettingType}-{scope}-{hashedBookmakers}.dat?geo={geo}&lang=en` with the event page URL, minus its fragment, as `Referer`. `odds_lambda.oddsportal_client.FEED_MARKETS` is the single market table, keyed by LeagueSpec market key. Each `odds_lambda.oddsportal_client.FeedMarketSpec` holds the feed `betting_type`, `scope` and `handicap` (as it appears in the feed key), the pipeline market key (`pipeline_key`), the outcome labels in feed order (`outcomes`) and the totals line (`point`, `None` otherwise): `1x2` (type 1, scope 2, `home`/`draw`/`away`), `over_under_2_5` (type 2, scope 2, handicap 2.5, `over`/`under`) and `home_away` (type 3, scope 1, `home`/`away`). It sleeps `request_delay` (1 s by default) before each feed request.

The three `/proxy/` endpoints also send the site's XHR headers (`X-Requested-With`, `x-client-platform: next-op`, `x-language`).

Behaviour:

- Responses from the `/proxy/` endpoints are base64 of `ciphertext:ivHex`, AES-256-CBC with a PBKDF2-SHA256 key, optionally gzipped; `odds_lambda.oddsportal_client.decrypt_payload` reverses that and also accepts plain JSON. The password and salt are constants in the site's JavaScript. A body that fails to decrypt raises `odds_lambda.oddsportal_client.OddsPortalError`, so a changed key fails loudly.
- HTTP 429, 5xx, connection errors and timeouts are retried three times with exponential backoff; any other non-200 status raises immediately. A `/proxy/` endpoint can also answer HTTP 200 with a plain-text body `URL:<path> Status: <code>` reporting an upstream error: a 429 or 5xx code there is retried like the HTTP status, and any other code raises immediately, naming the code and URL. Requests time out after 30 s.
- A feed whose `d` object holds an `E` key is an error envelope. The client waits the envelope's `refresh` seconds (default 20, capped at 30) and retries twice before raising.
- `odds_lambda.oddsportal_client.extract_market` reads `oddsdata.back["E-{type}-{scope}-0-{line}-0"]`. A missing market key, or an `oddsdata` or `back` that is absent or an empty list (no odds posted yet), yields no bookmakers. A `d` that is not an object, or an `oddsdata` or `back` of any other type, raises `odds_lambda.oddsportal_client.OddsPortalError`. It skips a bookmaker whose `act` flag is false (a suspended price), whose id is not in the name map, or whose prices are missing, at or below 1.0 or the wrong length. Prices and volumes arrive either as lists or as dicts keyed by outcome index, and both are accepted. `volume` (Betfair Exchange matched amounts) and the latest `changeTime` are carried through when present.

## Results archive

`odds_lambda.oddsportal_client.OddsPortalClient.fetch_results` reads a league's results page the same way as a listing: the page's RSC payload carries the URL of the paginated archive feed, which `odds_lambda.oddsportal_client.parse_archive_path` returns as a `/proxy/` path. Each `<path>/page/N/` response is encrypted like the odds feed and holds up to `onePage` rows (50) plus `total`, in descending kick-off order. A row has team names, a kick-off timestamp, a status id (3 is finished, 5 cancelled) and `homeResult` / `awayResult`. `odds_lambda.oddsportal_client.parse_results_page` turns finished rows into `odds_lambda.oddsportal_client.FinishedMatch` values, cancelled rows into `odds_lambda.oddsportal_client.CancelledMatch` values, and counts the rows it saw. `odds_lambda.oddsportal_client.OddsPortalClient.fetch_results` returns them as `odds_lambda.oddsportal_client.FetchedResults`; a page that fails after the first leaves `complete` false and keeps the earlier pages, and a failing first page raises. A real archive page is recorded in `tests/fixtures/oddsportal/mlb_results_feed.json`. The results page lists the current season only ([results-jobs](results-jobs.md#fetch-oddsportal-results)).

## Adapter

- `odds_lambda.oddsportal_adapter.convert_market` emits The Odds API `raw_data` shape with `source: oddsportal_live` and the feed's encoded event id as `oddsportal_event_id`, so a stored snapshot can be traced to its OddsPortal game, stored by `odds_lambda.storage.writers.OddsWriter.store_odds_snapshot` ([odds-writer](../storage/odds-writer.md)). It looks the market up in `odds_lambda.oddsportal_client.FEED_MARKETS`: the pipeline key is `1x2`, `totals` (with point 2.5) or `h2h`, outcome names come from the spec labels (`home` and `away` are the event teams), and an unsupported market raises `ValueError("Unsupported market")` ([sport-and-market-keys](sport-and-market-keys.md#market-key-dialects)).
- Feed decimal prices are stored unconverted under `price_decimal`, with no `price` key (`odds_core.odds_math.PRICE_DECIMAL_KEY`); `odds_lambda.oddsportal_common.build_raw_data` writes the same encoding for historical records, skipping a bookmaker with any price at or below 1.0 ([odds-math](../foundations/odds-math.md)). Each bookmaker's `last_update` is its latest feed change time, capped at the scrape time, or the scrape time when the feed gives none. Bookmakers with volumes get a `betfair_matched` entry keyed by the spec outcome labels. Each `odds_lambda.oddsportal_adapter.MatchOdds` carries the market key it was converted for. A market with no bookmakers converts to nothing.
- Events resolve by `odds_lambda.storage.writers.OddsWriter.find_or_create_event` (`op_live_` ids from `odds_core.event_ids.LIVE_EVENT_ID_PREFIX`, never merging): for MLB the event holding the game's MLB `gamePk`, else the ±2 h canonical event, the Odds API one when it exists ([event-matching](../storage/event-matching.md#live-mlb-events)). `odds_lambda.oddsportal_common.find_existing_event` (±24 h, same sport, read-only) is a different matcher from the writer's.

## Shared helpers

- `odds_lambda.oddsportal_common.hours_to_tier` has different edge behaviour from the copy used by live writes ([fetch-tier](../storage/fetch-tier.md#boundary-encodings)).

## odds scrape upcoming

`odds scrape upcoming --league --market --dry-run` (`odds_cli.commands.scrape.scrape_upcoming`) replaces the spec's markets with `--market` values, which must be keys of `odds_lambda.oddsportal_client.FEED_MARKETS`; `--dry-run` scrapes and prints the converted matches without storing them. `odds_lambda.jobs.fetch_oddsportal.scrape_league` raises `odds_lambda.oddsportal_client.OddsPortalError` when there were upcoming events and every market fetch failed, so a changed key or a blocked feed fails the league instead of returning zero matches; individual failures are otherwise recorded in the scrape errors. The scrape also counts the events the listing contained and how many of them had not started (`odds_lambda.jobs.fetch_oddsportal.LeagueScrape.upcoming_listed`) and the latest start time listed (`odds_lambda.jobs.fetch_oddsportal.LeagueScrape.latest_listed_start`); only upcoming events are fetched. The MCP `odds_mcp.server.refresh_scrape` tool queues `odds_lambda.jobs.fetch_oddsportal.ingest_league` in the local scheduler process and needs it running ([mcp-scheduling-tools](../agent/mcp-server.md#scrape-and-scheduling-tools)).

## Related

- [fetch-oddsportal job](fetch-oddsportal-job.md)
- [Sport, market and bookmaker keys](sport-and-market-keys.md)
- [Betfair Exchange client and job](betfair-exchange.md)
- [ADR 0009: read OddsPortal live odds from its JSON feed](../../docs/adr/0009-oddsportal-feed-client.md)
