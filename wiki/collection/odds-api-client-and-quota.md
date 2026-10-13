---
type: component
title: The Odds API client, key rotation and quota
description: "TheOddsAPIClient endpoints, retry and key rotation with the SSM index, unit cost per endpoint, and where quota is logged, alerted and checked (and where those checks mislead)."
sources:
  - odds_lambda.data_fetcher.TheOddsAPIClient
  - odds_lambda.api_key_manager.APIKeyManager
  - odds_core.config.APIConfig
  - odds_core.config.AlertConfig
  - odds_core.models.FetchLog
  - odds_cli.commands.status._show_quota
  - odds_analytics.gap_backfill_planner.GapBackfillPlanner
  - deployment/aws/terraform/lambda.tf
status: stable
---

# The Odds API client, key rotation and quota

`odds_lambda.data_fetcher.TheOddsAPIClient` is an async aiohttp client used as `async with TheOddsAPIClient() as client`. Rotation state lives in `odds_lambda.api_key_manager.APIKeyManager`. Quota is spent only by this client; the per-call costs below come from the method docstrings in `odds_lambda.data_fetcher`, and the API's actual billing is not verifiable from code.

## Methods

`get_odds`, `get_events` (free), `get_scores` (always sends `daysFrom`, default 1), `get_historical_odds`, `get_historical_events`. Odds methods default regions, markets and bookmakers from `odds_core.config.DataCollectionConfig` ([sources-and-keys](sources-and-keys.md)). Responses carry `quota_remaining` from the `x-requests-remaining` header.

## Retry and rotation

- `_make_request` is wrapped in tenacity: 3 attempts, exponential wait 2-10 s, on `aiohttp.ClientError` / `TimeoutError`; 30 s request timeout.
- On a 401 or 429 and a configured key manager, `_make_request` calls `rotate_key()` and recurses into itself, so each rotation gets its own fresh retry budget.
- `odds_lambda.api_key_manager.APIKeyManager` reads the active index from SSM (`SSM_API_KEY_INDEX` env, else `odds_lambda.api_key_manager.DEFAULT_SSM_PARAMETER_NAME`), writes it on rotation, and raises `odds_lambda.api_key_manager.AllKeysExhaustedError` after a full cycle from the start index. If SSM is unreachable it falls back to in-process memory.
- Rotation is enabled only when `ODDS_API_KEYS` (`odds_core.config.APIConfig.keys`) is set and no explicit `api_key` argument is passed.
- Terraform sets `SSM_API_KEY_INDEX` to `/{project_name}/active-api-key-index` for both Lambdas, with IAM read/write on it (`deployment/aws/terraform/lambda.tf`). Whether the live parameter exists cannot be verified from the repo; see [aws-infrastructure](../operations/aws-infrastructure.md).

Changing the client: keep the tenacity retry decorator and key rotation on `odds_lambda.data_fetcher.TheOddsAPIClient._make_request`; add fixtures under `tests/fixtures/` for new endpoints and update integration tests; if request patterns change, update quota tracking ([where usage is recorded](#where-usage-is-recorded-and-alerted)). Settings are in `odds_core.config.APIConfig` (`ODDS_API_` prefix; see [configuration](../foundations/configuration.md)).

## Cost per call

| Endpoint | Client method | Units |
|---|---|---|
| `/odds` | `get_odds` | 1 per region per market, regardless of game count (default 1 region x 3 markets = 3) |
| `/events` | `get_events` | free |
| `/scores` | `get_scores` | 1 without `daysFrom`, 2 with |
| `/historical/odds` | `get_historical_odds` | 10 per region per market |
| `/historical/events` | `get_historical_events` | 1 (per its docstring) |

Bookmakers are a server-side filter and do not change cost. The default quota is 500 requests per key per month (`odds_core.config.APIConfig`); historical endpoints are paid-tier only (inferred from vendor docs, not checked here). Multiple keys (`ODDS_API_KEYS`) multiply the effective quota. A manual `odds fetch current` therefore costs 3 units by default.

## Where usage is recorded and alerted

- Every `odds_lambda.ingestion.OddsIngestionService.ingest_sport` writes a `odds_core.models.FetchLog` row with `api_quota_remaining` ([odds-api-ingestion](odds-api-ingestion.md#fetch_logs)). The Betfair job also writes `odds_core.models.FetchLog` rows, with a null quota ([betfair-exchange](betfair-exchange.md)).
- `fetch-odds` compares remaining quota to `odds_core.config.APIConfig.quota` (`ODDS_API_QUOTA`, default 500) and sends a warning below `odds_core.config.AlertConfig.quota_warning_threshold` (0.2) and critical below `quota_critical_threshold` (0.1). These calls use `odds_core.alerts.send_warning`/`odds_core.alerts.send_critical` without an `alert_type`, so they are not rate-limited (every run, up to every 30 minutes) and never recorded in `alert_history`. `odds_lambda.health_monitor.HealthMonitor.check_api_quota` applies the same fraction but rate-limited ([alerts-and-health](../operations/alerts-and-health.md)).
- Check manually: `uv run odds status quota` (`odds_cli.commands.status._show_quota`) prints the latest and last 10 `odds_core.models.FetchLog` quotas ([reporting-and-daily-digest](../operations/reporting-and-daily-digest.md)).

## Hazards

- **Retry nested inside rotation.** The recursive call re-enters the decorated method, so worst case is 3 attempts per key times the key count. A 401/429 with no key manager is also retried 3 times, since `ClientResponseError` is a `ClientError`.
- **Exhausted retries raise tenacity `RetryError`**, not `aiohttp.ClientError` as the `_make_request` docstring says (no `reraise=True`). `odds_lambda.api_key_manager.AllKeysExhaustedError` propagates unwrapped.
- **The index only advances.** Nothing resets it to 0; a key that recovers (new billing month) is reused only after a full cycle.
- **`ODDS_API_KEY` is required** (`odds_core.config.APIConfig.key`, `Field(...)`) even when `ODDS_API_KEYS` is set.
- **SSM path mismatch.** The local default `/odds/active-api-key-index` differs from the Terraform-derived path; only the env override aligns them.
- **`/scores` always costs 2.** `get_scores` always sends `daysFrom` (default 1), so the 1-unit case is unreachable through the client.
- **Quota is per key, thresholds assume one.** `x-requests-remaining` is the active key's balance; percentages divide by `odds_core.config.APIConfig.quota`. On a 20,000-unit paid key with the default 500, `status quota` and alerts show nonsense until `ODDS_API_QUOTA` is set.
- **Historical cost is a hardcoded 30** (10 x 1 region x 3 markets) in `odds_analytics.gap_backfill_planner.GapBackfillPlanner.QUOTA_PER_SNAPSHOT`, and again as literals in `odds_analytics.backfill_executor` and `odds_analytics.game_selector`; none reads the configured regions or markets, so backfill estimates drift if those change ([howto-load-historical-odds](howto-load-historical-odds.md)).
- **`status quota` reads the newest 10 rows unfiltered** (`odds_lambda.storage.readers.OddsReader.get_fetch_logs`) and shows nothing when the newest row has a null quota, as Betfair rows do. `odds_lambda.health_monitor.HealthMonitor.check_api_quota` and `status show` use the latest non-null quota instead (`odds_lambda.storage.readers.OddsReader.get_database_stats`; [odds-reader-and-storage-layer](../storage/odds-reader-and-storage-layer.md)).
- **Failed `/odds` calls write no `odds_core.models.FetchLog`**, so they leave no quota trace.

## Related

- [Odds API ingestion and the fetch-odds job](odds-api-ingestion.md)
- [Event lifecycle and results jobs](event-lifecycle.md), where `fetch-scores` spends 2 units per call
- [Configuration](../foundations/configuration.md)
