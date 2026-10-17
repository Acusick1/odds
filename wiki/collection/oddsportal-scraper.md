---
type: reference
title: OddsPortal scraper
description: The OddsPortal HTTP feed client, the adapter that converts its prices, the shared oddsportal_common helpers, the results job's OddsHarvester retries, and odds scrape upcoming.
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
  - odds_lambda.oddsportal_common.run_scraper_with_retry
  - odds_lambda.oddsportal_common.build_raw_data
  - odds_lambda.oddsportal_common.find_existing_event
  - odds_lambda.oddsportal_common.BOOKMAKER_KEY_MAP
  - odds_lambda.oddsportal_common.normalize_bookmaker_key
  - odds_lambda.oddsportal_common.decimal_to_american
  - odds_lambda.oddsportal_common.hours_to_tier
  - odds_lambda.oddsportal_common.MAX_SCRAPER_RETRIES
  - odds_lambda.oddsportal_common.MAX_FAILED_URL_RETRY_PASSES
  - odds_cli.commands.scrape.scrape_upcoming
  - pyproject.toml
status: stable
---

# OddsPortal scraper

Live pre-match odds from OddsPortal (UK bookmakers [the Odds API](odds-api-client.md) lacks) are read over plain HTTP, with no browser. The results job ([results-jobs](results-jobs.md)) still drives headless Chromium through the OddsHarvester fork, pinned in `pyproject.toml` (git tag `odds-fork-2026.05.29`). The HTTP client serves [the fetch-oddsportal job](fetch-oddsportal-job.md) and `odds scrape upcoming`. Historical bulk loads use scripts instead ([historical-ingest-scripts](historical-ingest-scripts.md)). The reasons for reading the feed are in [ADR 0009](../../docs/adr/0009-oddsportal-feed-client.md).

## Modules

- `odds_lambda.jobs.fetch_oddsportal`: `odds_lambda.jobs.fetch_oddsportal.LEAGUE_SPECS`, `odds_lambda.jobs.fetch_oddsportal.scrape_league`, `odds_lambda.jobs.fetch_oddsportal.ingest_league`. `odds_lambda.jobs.fetch_oddsportal.LEAGUE_SPEC_BY_NAME` serves the CLI and MCP scrape tools.
- `odds_lambda.oddsportal_client`: `odds_lambda.oddsportal_client.OddsPortalClient` (async context manager owning one `aiohttp` session), the pure functions `odds_lambda.oddsportal_client.parse_listing`, `odds_lambda.oddsportal_client.extract_market` and `odds_lambda.oddsportal_client.decrypt_payload`, and the market table `odds_lambda.oddsportal_client.FEED_MARKETS`. Every failure it cannot recover from is an `odds_lambda.oddsportal_client.OddsPortalError`.
- `odds_lambda.oddsportal_adapter`: pure conversion of feed prices to the `raw_data` shape (`odds_lambda.oddsportal_adapter.convert_market`).
- `odds_lambda.oddsportal_common`: `odds_lambda.oddsportal_common.BOOKMAKER_KEY_MAP`/`odds_lambda.oddsportal_common.normalize_bookmaker_key`, `odds_lambda.oddsportal_common.decimal_to_american`, `odds_lambda.oddsportal_common.hours_to_tier`, `odds_lambda.oddsportal_common.build_raw_data` (historical records), `odds_lambda.oddsportal_common.run_scraper_with_retry` (results job), `odds_lambda.oddsportal_common.find_existing_event`. Besides conversion helpers it holds the retry state machine and a DB query. `odds_lambda.betfair.adapter.build_raw_data` is an unrelated function of the same name.

## Feed client

One scrape of a league makes these requests, each with a desktop Chrome `User-Agent`:

1. `odds_lambda.oddsportal_client.OddsPortalClient.fetch_listing` GETs the league's listing page (`odds_lambda.jobs.fetch_oddsportal.LeagueSpec.listing_path`). `odds_lambda.oddsportal_client.parse_listing` joins the strings of every `self.__next_f.push([1, ...])` script, reads each `<id>:<json>` line that parses as JSON, and collects every object carrying `encodeEventId`, `home-name`, `away-name`, `date-start-timestamp`, `url` and `sport-id`, deduplicated by event id. Kickoff is the UTC `date-start-timestamp`. When no event is found and the payload carries a `productErrorType`, it logs `oddsportal_listing_empty` and returns an empty list.
2. `odds_lambda.oddsportal_client.OddsPortalClient.fetch_bookmaker_context` GETs `/proxy/api/home-data` for `hashedBookmakers` and `geoIPcode`, and `/proxy/ajax-providers-bonus-data/0/` for the bookmaker id to display-name map. The bookmaker set is whatever OddsPortal serves to the caller's region.
3. `odds_lambda.oddsportal_client.OddsPortalClient.fetch_market` GETs `/proxy/match-event/1-{sportId}-{eventId}-{bettingType}-{scope}-{hashedBookmakers}.dat?geo={geo}&lang=en` with the event page URL, minus its fragment, as `Referer`. The betting type and scope come from `odds_lambda.oddsportal_client.FEED_MARKETS`: `1x2` (type 1, scope 2, 3 outcomes), `over_under_2_5` (type 2, scope 2, line 2.5) and `home_away` (type 3, scope 1). It sleeps `request_delay` (1 s by default) before each feed request.

The three `/proxy/` endpoints also send the site's XHR headers (`X-Requested-With`, `x-client-platform: next-op`, `x-language`).

Behaviour:

- Responses from the `/proxy/` endpoints are base64 of `ciphertext:ivHex`, AES-256-CBC with a PBKDF2-SHA256 key, optionally gzipped; `odds_lambda.oddsportal_client.decrypt_payload` reverses that and also accepts plain JSON. The password and salt are constants in the site's JavaScript. A body that fails to decrypt raises `odds_lambda.oddsportal_client.OddsPortalError`, so a changed key fails loudly.
- HTTP 429, 5xx, connection errors and timeouts are retried three times with exponential backoff; any other non-200 status raises immediately. Requests time out after 30 s.
- A feed whose `d` object holds an `E` key is an error envelope. The client waits the envelope's `refresh` seconds (default 20, capped at 30) and retries twice before raising.
- `odds_lambda.oddsportal_client.extract_market` reads `oddsdata.back["E-{type}-{scope}-0-{line}-0"]`. A missing key means the market is not offered and yields no bookmakers. It skips a bookmaker whose `act` flag is false (a suspended price), whose id is not in the name map, or whose prices are missing, non-positive or the wrong length. Prices and volumes arrive either as lists or as dicts keyed by outcome index, and both are accepted. `volume` (Betfair Exchange matched amounts) and the latest `changeTime` are carried through when present.

## Adapter

- `odds_lambda.oddsportal_adapter.convert_market` emits The Odds API `raw_data` shape with `source: oddsportal_live`, stored by `odds_lambda.storage.writers.OddsWriter.store_odds_snapshot` ([odds-writer](../storage/odds-writer.md)). Market keys are `1x2`, `totals` (with point 2.5) and `h2h`; unsupported markets raise `ValueError("Unsupported market")` ([sport-and-market-keys](sport-and-market-keys.md#market-key-dialects)).
- Decimal prices become American through `odds_lambda.oddsportal_common.decimal_to_american`. Each bookmaker's `last_update` is its latest feed change time, capped at the scrape time, or the scrape time when the feed gives none. Bookmakers with volumes get a `betfair_matched` entry (`home`/`draw`/`away`, `over`/`under` or `home`/`away`). A market with no bookmakers converts to nothing.
- Events resolve by `odds_lambda.storage.writers.OddsWriter.find_or_create_event` (±2 h, `op_live_` ids, preferring an Odds API event when both exist) ([event-matching](../storage/event-matching.md)). `odds_lambda.oddsportal_common.find_existing_event` (±24 h, returns `None` on ambiguity) is a different matcher from the writer's.

## Results job and shared helpers

- `odds_lambda.oddsportal_common.run_scraper_with_retry` retries an all-empty OddsHarvester scrape and then re-scrapes failed URLs (except 404s) in fresh browser sessions, with a fresh `fake_useragent` Chrome UA each time. It raises `RuntimeError` if OddsHarvester returns `None`. `odds_lambda.oddsportal_common.MAX_SCRAPER_RETRIES` and `odds_lambda.oddsportal_common.MAX_FAILED_URL_RETRY_PASSES` are 1, so an empty scrape is not retried and failed URLs get one pass; the docstring's "up to 3 passes" does not match (#447).
- `odds_lambda.oddsportal_common.decimal_to_american` and `odds_lambda.oddsportal_common.hours_to_tier` have different edge behaviour from the copies used by Betfair and live writes ([odds-math](../foundations/odds-math.md#conversion-copies), [fetch-tier](../storage/fetch-tier.md#boundary-encodings)).

## odds scrape upcoming

`odds scrape upcoming --league --market --dry-run` (`odds_cli.commands.scrape.scrape_upcoming`) replaces the spec's markets with `--market` values, which must be keys of `odds_lambda.oddsportal_client.FEED_MARKETS`; `--dry-run` scrapes and prints the converted matches without storing them. The MCP `odds_mcp.server.refresh_scrape` tool queues `odds_lambda.jobs.fetch_oddsportal.ingest_league` in the local scheduler process and needs it running ([mcp-scheduling-tools](../agent/mcp-server.md#scrape-and-scheduling-tools)).

## Related

- [fetch-oddsportal job](fetch-oddsportal-job.md)
- [Sport, market and bookmaker keys](sport-and-market-keys.md)
- [Betfair Exchange client and job](betfair-exchange.md)
- [ADR 0009: read OddsPortal live odds from its JSON feed](../../docs/adr/0009-oddsportal-feed-client.md)
