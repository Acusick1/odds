---
type: reference
title: The Odds API client and per-sport keys
description: TheOddsAPIClient methods, retry, the key each sport spends, and unit cost per endpoint.
sources:
  - odds_lambda.data_fetcher.TheOddsAPIClient
  - odds_core.config.APIConfig
  - odds_core.config.APIConfig.key_for
  - deployment/aws/terraform/lambda.tf
status: stable
---

# The Odds API client and per-sport keys

`odds_lambda.data_fetcher.TheOddsAPIClient` is an async aiohttp client used as `async with TheOddsAPIClient() as client`; it rewrites response bookmaker keys through `odds_core.bookmakers.ODDS_API_BOOKMAKER_KEY_MAP` ([sport and market keys](sport-and-market-keys.md)). Quota is spent only by this client; only the `odds_lambda.data_fetcher.TheOddsAPIClient.get_events` and `odds_lambda.data_fetcher.TheOddsAPIClient.get_historical_events` costs below are stated in method docstrings in `odds_lambda.data_fetcher`; the rest are the vendor's published pricing `(inferred)`, and actual billing is not verifiable from code. Settings are in `odds_core.config.APIConfig` (`ODDS_API_` prefix; [configuration](../foundations/configuration.md)). How usage is recorded and alerted is on [odds-api-quota](odds-api-quota.md).

## Methods

`odds_lambda.data_fetcher.TheOddsAPIClient.get_odds`, `odds_lambda.data_fetcher.TheOddsAPIClient.get_events` (free), `odds_lambda.data_fetcher.TheOddsAPIClient.get_scores` (always sends `daysFrom`, default 1), `odds_lambda.data_fetcher.TheOddsAPIClient.get_historical_odds`, `odds_lambda.data_fetcher.TheOddsAPIClient.get_historical_events`. The current-odds methods default regions, markets and the sport's bookmakers from `odds_core.config.DataCollectionConfig` (`fetch-odds` calls them for EPL and NBA) ([sport-and-market-keys](sport-and-market-keys.md#odds-api-scope)). `odds_lambda.data_fetcher.TheOddsAPIClient.get_historical_odds` has no defaults: the caller passes the markets and exactly one of regions or bookmakers (a `ValueError` otherwise), and only that one is sent. Responses carry `quota_remaining` and `quota_used` from the `x-requests-remaining` and `x-requests-used` headers; their sum is the key's plan size. A historical response also carries `units_used` from the `x-requests-last` header (the call's cost, `None` when absent) and `snapshot_time`, the response's own `timestamp`: the latest snapshot at or before the requested date (`odds_core.api_models.HistoricalOddsResponse`). `odds_lambda.data_fetcher.TheOddsAPIClient.get_odds`, `odds_lambda.data_fetcher.TheOddsAPIClient.get_events` and `odds_lambda.data_fetcher.TheOddsAPIClient.get_historical_odds` return the listed games as `odds_core.api_models.ApiEvent` (built by `odds_core.api_models.parse_api_event`, team names normalized, the API's id in `odds_api_id`), not as events; the API's id becomes an `odds_api` external id when the game is resolved to an event ([odds-ingestion-service](odds-ingestion-service.md)). `odds_lambda.data_fetcher.TheOddsAPIClient.get_odds` and `odds_lambda.data_fetcher.TheOddsAPIClient.get_historical_odds` relabel soccer three-way `h2h` markets to `1x2`, with canonical team names, in `raw_events_data`, then rename bookmaker keys to pipeline keys with `odds_lambda.data_fetcher.canonicalize_bookmaker_keys` ([market-key dialects](sport-and-market-keys.md#market-key-dialects)).

## Retry and keys

- `odds_lambda.data_fetcher.TheOddsAPIClient._send` makes one request and is the only retried layer: tenacity, 3 attempts, exponential wait 2-10 s, `reraise=True`, only on connection errors, `TimeoutError` and 5xx (`odds_lambda.data_fetcher._is_transient_error`); 30 s request timeout. Other HTTP errors, including 401 and 429, are not retried because they cannot succeed immediately, and each retry could spend quota `(inferred)`; the vendor's billing rules for failed calls are not checked here. Exhausted retries re-raise the original `aiohttp.ClientError`.
- `odds_lambda.data_fetcher.TheOddsAPIClient._make_request` takes the key of the sport it is called for from `odds_core.config.APIConfig.key_for` and calls `odds_lambda.data_fetcher.TheOddsAPIClient._send` once. A 401 or 429 logs `api_request_failed` with the status and raises the `aiohttp.ClientResponseError`; nothing retries it on another key.
- `odds_core.config.APIConfig.keys` (`ODDS_API_KEYS`) lists the keys in index order, falling back to the one `odds_core.config.APIConfig.key` (`ODDS_API_KEY`); `odds_lambda.data_fetcher.TheOddsAPIClient` raises `ValueError` at construction when neither is set, and `odds_core.config.APIConfig.key_for` raises one when a sport's index is beyond the list. `odds_core.config.APIConfig.key_index_by_sport` (`ODDS_API_KEY_INDEX_BY_SPORT`, JSON) gives each sport its index, `basketball_nba` 1 and every other sport (and a call with no sport) index 0. An explicit `api_key` argument overrides both.
- A sport therefore spends one key every time, and a key's quota is one sport's quota: EPL spends key 0 and NBA key 1 whatever the order of runs, and a 429 on one key cannot spend the other's units. Keys that rotated on exhaustion would have to be shared between sports, so a run's key would depend on process memory (the production host has no SSM parameter to persist it) and EPL's last units could go to NBA. The balance a sport's runs record is its key's, which the quota alert and `odds status quota` read per key ([odds-api-quota](odds-api-quota.md#where-quota-is-recorded-alerted-and-checked)).
- Terraform passes `ODDS_API_KEY` and `ODDS_API_KEYS` to both Lambdas as environment variables (`deployment/aws/terraform/lambda.tf`), so the same index rule applies there ([aws-infrastructure](../operations/aws-infrastructure.md)).

## Cost per call

| Endpoint | Client method | Units |
|---|---|---|
| `/odds` | `odds_lambda.data_fetcher.TheOddsAPIClient.get_odds` | 1 per region per market, regardless of game count (default 1 region x 3 markets = 3; every 10 bookmakers count as one region) ([vendor](https://the-odds-api.com/liveapi/guides/v4/#usage-quota-costs)) |
| `/events` | `odds_lambda.data_fetcher.TheOddsAPIClient.get_events` | free (`odds_lambda.data_fetcher.TheOddsAPIClient.get_events` docstring) |
| `/scores` | `odds_lambda.data_fetcher.TheOddsAPIClient.get_scores` | 1 without `daysFrom`, 2 with ([vendor](https://the-odds-api.com/liveapi/guides/v4/#usage-quota-costs)) |
| `/historical/odds` | `odds_lambda.data_fetcher.TheOddsAPIClient.get_historical_odds` | 10 per region per market (`odds_lambda.data_fetcher.TheOddsAPIClient.get_historical_events` docstring) |
| `/historical/events` | `odds_lambda.data_fetcher.TheOddsAPIClient.get_historical_events` | 1 (`odds_lambda.data_fetcher.TheOddsAPIClient.get_historical_events` docstring) |

- `odds_lambda.data_fetcher.TheOddsAPIClient.get_scores` always sends `daysFrom`, so each call costs 2: every caller needs completed games, which the vendor returns only when `daysFrom` is set, and without it the endpoint lists live and upcoming games only ([vendor](https://the-odds-api.com/liveapi/guides/v4/)).
- Each key in `ODDS_API_KEYS` has its own plan and quota, spent by the sports that use its index; no setting holds the plan size. Historical endpoints are paid-tier only (inferred from vendor docs, not checked here). A manual `odds fetch current` costs 3 units by default.
- `x-requests-remaining` is the balance of the key the call used and `x-requests-used` the credits it has spent since its last quota reset, so the key's plan size is their sum; the alert percentages divide the balance by it, per key.
- The manifest planner derives a historical call's cost from its markets and bookmakers ([odds-api-backfill](odds-api-backfill.md#planning-calls)); the executor records the cost the header reports ([odds-api-backfill](odds-api-backfill.md#executing-calls)).

## Related

- [Odds API quota accounting](odds-api-quota.md)
- [Odds API ingestion service](odds-ingestion-service.md)
- [Results and status jobs](results-jobs.md), where `fetch-scores` spends 2 units per call
