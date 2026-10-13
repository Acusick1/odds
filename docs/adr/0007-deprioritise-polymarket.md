---
status: proposed
---

# Deprioritise Polymarket as a trading venue

## Context

The pipeline was built for NBA: a client, five `polymarket_*` tables, ingestion, fetch and backfill jobs, manual event linking and a 16-column NBA-only feature group ([polymarket](../../wiki/collection/polymarket.md)).

- In the early NBA cross-check, Polymarket probabilities correlated 0.9978 with sportsbook prices and the Polymarket features changed nothing (#134, 228 events; [exp-early-nba-signal-and-lstm](../../wiki/evaluation/exp-early-nba-signal-and-lstm.md)).
- UK trading is blocked across all sports (`AGENTS.md` "Polymarket Integration").
- EPL books are thin ($10K-$100K per match) and AMM-driven, so the book reflects market-maker parameters rather than public sentiment (`AGENTS.md` "Polymarket Integration").
- MLB books are deep (median $152K per game, 1 cent spreads, 32-44 levels per side) and collinear with devigged bet365 through arbitrage, so they are a sharp-reference candidate rather than a mispricing venue (probe scripts `experiments/scripts/polymarket_mlb_*.py`, recorded in dff040d; no results file exists). The observations are on [polymarket-market-research](../../wiki/collection/polymarket-market-research.md).

## Decision

Stop extending the Polymarket pipeline.

- The NBA-targeted code and tables stay in place.
- No Polymarket features go into the deployed model.
- Terraform `enable_polymarket` defaults to false.

Polymarket as an MLB sharp reference is not part of this decision.

## Consequences

- The pipeline stays NBA-only: tickers and the configured series and tag ids are NBA, and events are linked only by `odds polymarket link` (#456).
- `best_bid` and `best_ask` on price snapshots are never populated by the live pipeline (#451). `spread` and `midpoint` are computed in `PolymarketWriter.store_price_snapshot` only when both are present, so they stay null as well. Order-book snapshots compute all four from the book.
- `odds_core.config.PolymarketConfig.enabled` defaults to true while the Terraform gate defaults to false, so a local scheduler runs the jobs unless disabled while AWS does not.
- Price history for resolved markets expires after about 31 days and order books cannot be backfilled, so data not collected live is lost ([polymarket-market-research](../../wiki/collection/polymarket-market-research.md#data-availability)).
- Extending the pipeline to another sport needs its own series and tag ids and ticker parser; the client is sport-agnostic apart from `get_nba_events` ([polymarket](../../wiki/collection/polymarket.md#components)).

## Evidence

- PRs and issues: #134, #451, #456.
- Commit dff040d (MLB probes).
- `AGENTS.md` "Polymarket Integration" section at `9b3ae09^` (removed in 9b3ae09).
- Scripts: `experiments/scripts/polymarket_mlb_discover.py`, `experiments/scripts/polymarket_mlb_orderbooks.py`, `experiments/scripts/polymarket_mlb_volume.py`, `experiments/scripts/polymarket_mlb_sentiment.py`.
