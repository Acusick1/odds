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

`odds_lambda.data_fetcher.TheOddsAPIClient` is an async aiohttp client used as `async with TheOddsAPIClient() as client`. Rotation state lives in `odds_lambda.api_key_manager.APIKeyManager`. Quota is spent only by this client; the per-call costs below come from the method docstrings in `odds_lambda.data_fetcher`, and the API's actual billing is not verifiable from code. Settings are in `odds_core.config.APIConfig` (`ODDS_API_` prefix; [configuration](../foundations/configuration.md)). How usage is recorded and alerted is on [odds-api-quota](odds-api-quota.md).

## Methods

`get_odds`, `get_events` (free), `get_scores` (always sends `daysFrom`, default 1), `get_historical_odds`, `get_historical_events`. Odds methods default regions, markets and bookmakers from `odds_core.config.DataCollectionConfig` ([sport-and-market-keys](sport-and-market-keys.md#odds-api-scope)). Responses carry `quota_remaining` from the `x-requests-remaining` header.

## Retry and rotation

- `odds_lambda.data_fetcher.TheOddsAPIClient._make_request` is wrapped in tenacity: 3 attempts, exponential wait 2-10 s, on `aiohttp.ClientError` / `TimeoutError`; 30 s request timeout. Exhausted retries raise tenacity `RetryError`, not `aiohttp.ClientError` as the `_make_request` docstring says (no `reraise=True`); `odds_lambda.api_key_manager.AllKeysExhaustedError` propagates unwrapped (#471).
- On a 401 or 429 and a configured key manager, `_make_request` calls `rotate_key()` and recurses into itself. The recursive call re-enters the decorated method, so each rotation gets a fresh retry budget and the worst case is 3 attempts per key times the key count. A 401/429 with no key manager is also retried 3 times, since `ClientResponseError` is a `ClientError` (#471).
- `odds_lambda.api_key_manager.APIKeyManager` reads the active index from SSM (`SSM_API_KEY_INDEX` env, else `odds_lambda.api_key_manager.DEFAULT_SSM_PARAMETER_NAME`), writes it on rotation, and raises `odds_lambda.api_key_manager.AllKeysExhaustedError` after a full cycle from the start index. If SSM is unreachable it falls back to in-process memory. The index only advances: nothing resets it to 0, so a key that recovers (new billing month) is reused only after a full cycle.
- Rotation is enabled only when `ODDS_API_KEYS` (`odds_core.config.APIConfig.keys`) is set and no explicit `api_key` argument is passed. `odds_core.config.APIConfig.key` (`ODDS_API_KEY`) is required even when `ODDS_API_KEYS` is set (#471).
- Terraform sets `SSM_API_KEY_INDEX` to `/{project_name}/active-api-key-index` for both Lambdas, with IAM read/write on it (`deployment/aws/terraform/lambda.tf`). The local default `/odds/active-api-key-index` differs from that path, so only the env override aligns them. Whether the live parameter exists cannot be verified from the repo; `aws ssm get-parameter --name /<project_name>/active-api-key-index` checks it ([aws-infrastructure](../operations/aws-infrastructure.md)).

## Cost per call

| Endpoint | Client method | Units |
|---|---|---|
| `/odds` | `get_odds` | 1 per region per market, regardless of game count (default 1 region x 3 markets = 3) |
| `/events` | `get_events` | free |
| `/scores` | `get_scores` | 1 without `daysFrom`, 2 with |
| `/historical/odds` | `get_historical_odds` | 10 per region per market |
| `/historical/events` | `get_historical_events` | 1 (per its docstring) |

- `get_scores` always sends `daysFrom`, so each call costs 2 and the 1-unit case is unreachable through the client (#441).
- Bookmakers are a server-side filter and do not change cost.
- The default quota is 500 requests per key per month (`odds_core.config.APIConfig.quota`, `ODDS_API_QUOTA`); historical endpoints are paid-tier only (inferred from vendor docs, not checked here). Multiple keys (`ODDS_API_KEYS`) multiply the effective quota. A manual `odds fetch current` therefore costs 3 units by default.
- `x-requests-remaining` is the active key's balance, and the alert percentages divide it by `ODDS_API_QUOTA` (default 500 per key). On a larger plan the thresholds never fire until it is set (#441).
- The backfill planner's historical cost estimate is a separate constant ([odds-api-backfill](odds-api-backfill.md)).

## Related

- [Odds API quota accounting](odds-api-quota.md)
- [Odds API ingestion service](odds-ingestion-service.md)
- [Results and status jobs](results-jobs.md), where `fetch-scores` spends 2 units per call
