---
type: component
title: MCP odds, fixture and prediction tools
summary: "get_upcoming_fixtures, get_current_odds, get_odds_history and get_predictions: inputs, outputs, window semantics and the market vocabulary they need."
sources:
  - packages/odds-mcp/odds_mcp/server.py::get_upcoming_fixtures
  - packages/odds-mcp/odds_mcp/server.py::get_current_odds
  - packages/odds-mcp/odds_mcp/server.py::get_odds_history
  - packages/odds-mcp/odds_mcp/server.py::get_predictions
  - packages/odds-lambda/odds_lambda/storage/readers.py::OddsReader
  - packages/odds-core/odds_core/prediction_models.py::Prediction
  - docs/BETTING_AGENT.md#mcp-tools
  - docs/REPORTING.md#6-mcp-reporting-tools--packagesodds-mcpodds_mcpserverpy
verified_at: 7887039
---

# MCP odds, fixture and prediction tools

Four read-only tools in `packages/odds-mcp/odds_mcp/server.py`, part of the [odds-mcp server](mcp-server.md). Unknown event ids return `{"error": "Event '<id>' not found"}`.

## `get_upcoming_fixtures(league="soccer_epl", days_ahead=7)`

`SCHEDULED` events with `commence_time` between now and now plus `days_ahead` (clamped to 1-30), ordered by kickoff, via `OddsReader.get_events_by_date_range`. Each item: `id`, `sport_key`, `sport_title`, `home_team`, `away_team`, `commence_time`, `status`, scores and `completed_at`. Live events are excluded ([Event lifecycle](event-lifecycle.md)).

## `get_current_odds(event_id, market, include_raw_data=False, lookback_hours=2.0)`

Resolves the latest price per `(bookmaker_key, outcome_name, point)`:

1. Anchor on the newest snapshot that contains `market` (`get_latest_snapshot(market=...)`).
2. Walk snapshots in `[anchor_time - lookback_hours, anchor_time]` newest first; the first price seen for each tuple wins (`OddsReader.get_latest_book_prices`). This lets a Betfair-only snapshot coexist with an older retail snapshot.

Response: `event` plus `snapshot` with `anchor_snapshot_id`, `snapshot_time` (newest contributing snapshot), `anchor_created_at`, `anchor_fetch_tier`, `anchor_hours_until_commence`, `bookmaker_count` (distinct books actually returned), `lookback_hours`, and `odds[]`. Each odds row carries `bookmaker_key`, `outcome_name`, `price` (American), `point`, `last_update` (the book's own timestamp, falling back to snapshot time), `snapshot_time` and `book_age_seconds`. With no snapshot, or no `market` odds in the window, `snapshot` is `null` with a `message`. `include_raw_data` adds the anchor's `raw_data`.

## `get_odds_history(event_id, market)`

Every snapshot for the event in time order, each with the extracted `market` odds instead of raw JSON, plus `snapshot_count`. There is no pagination or limit, so a long-lived event returns a large payload ([Odds reader](odds-reader.md)).

## `get_predictions(event_id, limit=5, since_hours=None)`

Reads stored `predictions` rows newest first; it does not run the model ([Inline CLV scoring](job-score-predictions.md)). `total_matching` counts rows after the `since_hours` filter, `returned` the limited list (`limit` floored at 1). Each row: `id`, `snapshot_id`, `model_name`, `model_version`, `predicted_clv`, `created_at`.

## Market vocabulary

`market` is a `MarketKey`: `h2h`, `1x2`, `totals` or `spreads`, matched exactly against the stored key. EPL OddsPortal and Betfair data is stored as `1x2`, MLB as `h2h` ([Sport and market keys](sport-and-market-keys.md)). A key with no data yields an empty result, not an error.

## Hazards

- `book_age_seconds` and the lookback window are relative to the anchor snapshot, not the wall clock. If scraping stops, ages still look small; compare `snapshot_time` with the current time to judge staleness.
- Per-book prices in one response can come from different snapshots, so they are not co-temporal; each row's `snapshot_time` says which snapshot it came from.
- `get_predictions` returns whatever the scorer wrote; no row exists for events the scoring job never reached.

## Related

- [find_retail_edges](mcp-find-retail-edges.md) — uses the same price resolution
- [CLV model findings](clv-model-findings.md)
