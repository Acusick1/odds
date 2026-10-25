---
type: reference
title: The Odds API client and key rotation
description: TheOddsAPIClient methods, retry and key rotation with the SSM index, and unit cost per endpoint.
sources:
  - odds_lambda.data_fetcher.TheOddsAPIClient
  - odds_lambda.api_key_manager.APIKeyManager
  - odds_lambda.api_key_manager.AllKeysExhaustedError
  - odds_lambda.api_key_manager.DEFAULT_SSM_PARAMETER_NAME
  - odds_core.config.APIConfig
  - deployment/aws/terraform/lambda.tf
status: stable
---

# The Odds API client and key rotation

`odds_lambda.data_fetcher.TheOddsAPIClient` is an async aiohttp client used as `async with TheOddsAPIClient() as client`; it rewrites response bookmaker keys through `odds_core.bookmakers.ODDS_API_BOOKMAKER_KEY_MAP` ([sport and market keys](sport-and-market-keys.md)). Rotation state lives in `odds_lambda.api_key_manager.APIKeyManager`. Quota is spent only by this client; only the `odds_lambda.data_fetcher.TheOddsAPIClient.get_events` and `odds_lambda.data_fetcher.TheOddsAPIClient.get_historical_events` costs below are stated in method docstrings in `odds_lambda.data_fetcher`; the rest are the vendor's published pricing `(inferred)`, and actual billing is not verifiable from code. Settings are in `odds_core.config.APIConfig` (`ODDS_API_` prefix; [configuration](../foundations/configuration.md)). How usage is recorded and alerted is on [odds-api-quota](odds-api-quota.md).

## Methods

`odds_lambda.data_fetcher.TheOddsAPIClient.get_odds`, `odds_lambda.data_fetcher.TheOddsAPIClient.get_events` (free), `odds_lambda.data_fetcher.TheOddsAPIClient.get_scores` (always sends `daysFrom`, default 1), `odds_lambda.data_fetcher.TheOddsAPIClient.get_historical_odds`, `odds_lambda.data_fetcher.TheOddsAPIClient.get_historical_events`. Odds methods default regions, markets and the sport's bookmakers from `odds_core.config.DataCollectionConfig` ([sport-and-market-keys](sport-and-market-keys.md#odds-api-scope)). Responses carry `quota_remaining` from the `x-requests-remaining` header. `odds_lambda.data_fetcher.TheOddsAPIClient.get_odds` and `odds_lambda.data_fetcher.TheOddsAPIClient.get_historical_odds` relabel soccer three-way `h2h` markets to `1x2`, with canonical team names, in `raw_events_data`, then rename bookmaker keys to pipeline keys with `odds_lambda.data_fetcher.canonicalize_bookmaker_keys` ([market-key dialects](sport-and-market-keys.md#market-key-dialects)).

## Retry and rotation

- `odds_lambda.data_fetcher.TheOddsAPIClient._send` makes one request and is the only retried layer: tenacity, 3 attempts, exponential wait 2-10 s, `reraise=True`, only on connection errors, `TimeoutError` and 5xx (`odds_lambda.data_fetcher._is_transient_error`); 30 s request timeout. Other HTTP errors, including 401 and 429, are not retried because they cannot succeed immediately, and each retry could spend quota `(inferred)`; the vendor's billing rules for failed calls are not checked here. Exhausted retries re-raise the original `aiohttp.ClientError`.
- `odds_lambda.data_fetcher.TheOddsAPIClient._make_request` wraps `odds_lambda.data_fetcher.TheOddsAPIClient._send` in a rotation loop. On a 401 or 429 and a configured key manager it calls `odds_lambda.api_key_manager.APIKeyManager.rotate_key` and sends again, so retries never nest inside rotation: a rotated key costs one call on 401/429, and transient retries apply per key only to 5xx and network failures. `odds_lambda.api_key_manager.AllKeysExhaustedError` propagates unwrapped. A 401/429 with no key manager (or from an explicit `api_key`) raises the `aiohttp.ClientResponseError` after one call and logs `api_request_failed` with the status.
- `odds_lambda.api_key_manager.APIKeyManager` reads the active index from SSM (`SSM_API_KEY_INDEX` env, else `odds_lambda.api_key_manager.DEFAULT_SSM_PARAMETER_NAME`), writes it on rotation, and raises `odds_lambda.api_key_manager.AllKeysExhaustedError` after a full cycle from the start index. If SSM is unreachable it falls back to in-process memory. The index only advances: nothing resets it to 0, so a key that recovers (new billing month) is reused only after a full cycle.
- Rotation is enabled only when `ODDS_API_KEYS` (`odds_core.config.APIConfig.keys`) is set and no explicit `api_key` argument is passed. `ODDS_API_KEYS` takes precedence over `ODDS_API_KEY` (`odds_core.config.APIConfig.key`); both are optional in settings, and `odds_lambda.data_fetcher.TheOddsAPIClient` raises `ValueError` at construction when neither is set.
- Terraform sets `SSM_API_KEY_INDEX` to `/{project_name}/active-api-key-index` for both Lambdas, with IAM read and write on it (`deployment/aws/terraform/lambda.tf`). The local default `/odds/active-api-key-index` differs from that path, so only the env override aligns them. Whether the live parameter exists cannot be verified from the repo; `aws ssm get-parameter --name /<project_name>/active-api-key-index` checks it ([aws-infrastructure](../operations/aws-infrastructure.md)).

## Cost per call

| Endpoint | Client method | Units |
|---|---|---|
| `/odds` | `odds_lambda.data_fetcher.TheOddsAPIClient.get_odds` | 1 per region per market, regardless of game count (default 1 region x 3 markets = 3; every 10 bookmakers count as one region) ([vendor](https://the-odds-api.com/liveapi/guides/v4/#usage-quota-costs)) |
| `/events` | `odds_lambda.data_fetcher.TheOddsAPIClient.get_events` | free (`odds_lambda.data_fetcher.TheOddsAPIClient.get_events` docstring) |
| `/scores` | `odds_lambda.data_fetcher.TheOddsAPIClient.get_scores` | 1 without `daysFrom`, 2 with ([vendor](https://the-odds-api.com/liveapi/guides/v4/#usage-quota-costs)) |
| `/historical/odds` | `odds_lambda.data_fetcher.TheOddsAPIClient.get_historical_odds` | 10 per region per market (`odds_lambda.data_fetcher.TheOddsAPIClient.get_historical_events` docstring) |
| `/historical/events` | `odds_lambda.data_fetcher.TheOddsAPIClient.get_historical_events` | 1 (`odds_lambda.data_fetcher.TheOddsAPIClient.get_historical_events` docstring) |

- `odds_lambda.data_fetcher.TheOddsAPIClient.get_scores` always sends `daysFrom`, so each call costs 2 and the 1-unit case is unreachable through the client (#441).
- Bookmakers are a server-side filter and do not change cost.
- The default quota is 500 requests per key per month (`odds_core.config.APIConfig.quota`, `ODDS_API_QUOTA`); historical endpoints are paid-tier only (inferred from vendor docs, not checked here). Multiple keys (`ODDS_API_KEYS`) multiply the effective quota. A manual `odds fetch current` therefore costs 3 units by default.
- `x-requests-remaining` is the active key's balance, and the alert percentages divide it by `ODDS_API_QUOTA` (default 500 per key). On a larger plan the thresholds never fire until it is set (#441).
- The backfill planner's historical cost estimate is a separate constant ([odds-api-backfill](odds-api-backfill.md)).

## Related

- [Odds API quota accounting](odds-api-quota.md)
- [Odds API ingestion service](odds-ingestion-service.md)
- [Results and status jobs](results-jobs.md), where `fetch-scores` spends 2 units per call
