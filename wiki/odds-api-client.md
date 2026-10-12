---
type: component
title: The Odds API client and key rotation
summary: TheOddsAPIClient endpoints, tenacity retry, key rotation on 401/429 with the active index in SSM, and rules for changing it.
sources:
  - packages/odds-lambda/odds_lambda/data_fetcher.py::TheOddsAPIClient
  - packages/odds-lambda/odds_lambda/api_key_manager.py::APIKeyManager
  - packages/odds-core/odds_core/config.py::APIConfig
  - docs/ARCHITECTURE.md#api-client
  - docs/DEPLOYMENT.md#api-key-rotation
  - docs/DEVELOPMENT.md#modifying-api-client
  - docs/DEVELOPMENT.md#api-settings-apiconfig
verified_at: 7887039
---

# The Odds API client and key rotation

`TheOddsAPIClient` (`packages/odds-lambda/odds_lambda/data_fetcher.py`) is an async aiohttp client used as `async with TheOddsAPIClient() as client`. Rotation state lives in `APIKeyManager` (`packages/odds-lambda/odds_lambda/api_key_manager.py`).

## Methods

`get_odds`, `get_events` (free), `get_scores` (always sends `daysFrom`, default 1), `get_historical_odds`, `get_historical_events`. Odds methods default regions, markets and bookmakers from `DataCollectionConfig` ([Data sources](data-sources.md)). Responses carry `quota_remaining` from the `x-requests-remaining` header. Costs: [Odds API quota](odds-api-quota.md).

## Retry and rotation

- `_make_request` is wrapped in tenacity: 3 attempts, exponential wait 2-10 s, on `aiohttp.ClientError` / `TimeoutError`; 30 s request timeout.
- On a 401 or 429 and a configured key manager, `_make_request` calls `rotate_key()` and recurses into itself, so each rotation gets its own fresh retry budget.
- `APIKeyManager` reads the active index from SSM (`SSM_API_KEY_INDEX` env, else `DEFAULT_SSM_PARAMETER_NAME`), writes it on rotation, and raises `AllKeysExhaustedError` after a full cycle from the start index. If SSM is unreachable it falls back to in-process memory.
- Rotation is enabled only when `ODDS_API_KEYS` (`APIConfig.keys`) is set and no explicit `api_key` argument is passed.
- Terraform sets `SSM_API_KEY_INDEX` to `/{project_name}/active-api-key-index` for both Lambdas (`deployment/aws/terraform/lambda.tf`). Whether the live parameter exists cannot be verified from the repo; see [AWS infrastructure](aws-infrastructure.md).

## Changing the client

Keep the tenacity decorator, rotation and error handling (`docs/DEVELOPMENT.md#modifying-api-client`); add fixtures under `tests/fixtures/` for new endpoints and update integration tests. Settings are in `APIConfig` (`ODDS_API_` prefix; see [Configuration](configuration.md)).

## Hazards

- **Retry nested inside rotation.** The recursive call re-enters the decorated method, so worst case is 3 attempts per key times the key count. A 401/429 with no key manager is also retried 3 times, since `ClientResponseError` is a `ClientError`.
- **Exhausted retries raise tenacity `RetryError`**, not `aiohttp.ClientError` as the `_make_request` docstring says (no `reraise=True`). `AllKeysExhaustedError` propagates unwrapped.
- **The index only advances.** Nothing resets it to 0; a key that recovers (new billing month) is reused only after a full cycle.
- **`ODDS_API_KEY` is required** (`APIConfig.key`, `Field(...)`) even when `ODDS_API_KEYS` is set.
- The local default SSM path `/odds/active-api-key-index` differs from the Terraform-derived path; only the env override aligns them.

## Related

- [Odds ingestion flow](odds-ingestion-flow.md)
- [Job: fetch-odds](job-fetch-odds.md)
- [Configuration](configuration.md)
