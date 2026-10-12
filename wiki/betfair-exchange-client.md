---
type: component
title: Betfair Exchange client
summary: BetfairExchangeClient, certificate loading and the adapter that turns best back Match Odds into betfair_exchange snapshots.
sources:
  - packages/odds-lambda/odds_lambda/betfair/client.py::BetfairExchangeClient
  - packages/odds-lambda/odds_lambda/betfair/adapter.py::betfair_book_to_bookmaker_entry
  - packages/odds-lambda/odds_lambda/betfair/cert_loader.py::resolve_cert_paths
  - packages/odds-lambda/odds_lambda/betfair/constants.py::SPORT_CONFIG
  - packages/odds-lambda/odds_lambda/jobs/fetch_betfair_exchange.py::main
  - packages/odds-core/odds_core/config.py::BetfairConfig
  - docs/ARCHITECTURE.md#betfair-exchange
  - docs/DATA_MODELS.md#sportsbook-odds
  - docs/DEVELOPMENT.md#other-settings
verified_at: 7887039
---

# Betfair Exchange client

Package `packages/odds-lambda/odds_lambda/betfair/`, driven by the [fetch-betfair-exchange job](job-fetch-betfair-exchange.md). It writes snapshots under bookmaker key `betfair_exchange`, so existing readers and the [sharp benchmark](sharp-benchmark.md) fallbacks consume them unchanged.

## Pieces

- `client.py::BetfairExchangeClient` wraps `betfairlightweight` (read-only, delayed app key). `login()` uses cert login when both cert paths are set, else `login_interactive()` (dev only; prompts for 2FA). `list_events(sport_cfg, lookahead_hours)` filters by competition; `list_match_odds_books` fetches the catalogue then `EX_BEST_OFFERS` books in batches of 25, returning `BetfairBook` / `BetfairRunner` dataclasses. `betfair_session` is the login/logout context manager.
- `constants.py::SPORT_CONFIG` holds per-sport config: EPL (`MATCH_ODDS`, market key `1x2`, draw runner) and MLB (market key `h2h`, `"Away @ Home"` event names). `competition_id` is pinned; name hints are a fallback.
- `adapter.py`: `resolve_teams` parses the event name; `betfair_book_to_bookmaker_entry` emits one bookmaker entry; `build_raw_data` wraps it with `source: betfair_api`. Runner names pass through `odds_core.team.normalize_team_name` after stripping MLB pitcher annotations such as "(TBD)" ([Event matching](event-matching.md)).
- `cert_loader.py::resolve_cert_paths`: SSM SecureString params (`cert_pem_ssm_param` + `key_pem_ssm_param`) are written to `/tmp/betfair.crt` and `/tmp/betfair.key` (mode 600, cached across warm starts); otherwise `cert_file` / `cert_key` pass through.
- `BetfairConfig` (`BETFAIR_` prefix; see [Configuration](configuration.md)): `enabled` defaults to false; enabling without username, password and app key fails validation. `lookahead_hours` defaults to 168; `sports` defaults to every key in `SPORT_CONFIG`.

The job runs the synchronous client in `asyncio.to_thread`, stores via `OddsWriter.find_or_create_event` and `store_odds_snapshot` ([Odds writer](odds-writer.md)), tags `api_request_id` `betfair_api_<time>`, and writes a `FetchLog` row without quota ([Odds API quota](odds-api-quota.md)).

## Hazards

- **Only best back is the price.** Outcome `price` is `best_back` converted to American (`odds_core.odds_math.decimal_to_american`, which truncates); best lay, sizes and last traded price live only in `betfair_meta.runners`. The docs' "best back/lay" refers to that metadata. Runners without a back price are dropped, and a book with fewer outcomes than expected returns `None` (nothing stored).
- **Delayed key**: prices lag 1-180 s (`BetfairConfig` docstring); in-play and suspended books are still fetched (`betfair_meta.inplay`, `market_status`).
- **Unknown runner names fall back to the normalized name** with a warning, so an alias gap yields an outcome that matches neither team.
- **Market key differs by sport** (`1x2` vs `h2h`) and from the Odds API's football `h2h` ([Data sources](data-sources.md)).
- `betfair_exchange` is also written by FDUK and OddsPortal ingestion; see [Historical odds ingestion](historical-odds-ingestion.md).
- Whether the job and the SSM cert parameters exist in AWS cannot be checked from the repo: `odds scheduler list-jobs --backend aws`.

## Related

- [Job deployment matrix](job-deployment-matrix.md)
- [Self-scheduling](self-scheduling.md)
