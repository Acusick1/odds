---
status: proposed
---

# Deprioritise Polymarket as a trading venue

Stop extending the Polymarket pipeline. Keep the NBA-targeted code and tables, and keep Polymarket out of the deployed model. The rationale below is only what the wiki already records; items marked `(inferred)` have no source in the repo.

## Context

The pipeline was built for NBA: a client, five `polymarket_*` tables, ingestion, fetch and backfill jobs, manual event linking and a 16-column NBA-only feature group ([polymarket](../../wiki/collection/polymarket.md)).

- In the early NBA cross-check, Polymarket probabilities correlated 0.9978 with sportsbook prices and the Polymarket features changed nothing (#134, 228 events; [exp-early-nba-signal-and-lstm](../../wiki/evaluation/exp-early-nba-signal-and-lstm.md)).
- UK trading is blocked for all sports `(inferred; external)`. The repo holds no evidence either way.
- EPL books are thin ($10K-$100K per match) and AMM-driven, so the book reflects market-maker parameters rather than public sentiment `(inferred)`.
- MLB books are deep (median $152K per game, 1 cent spreads, 32-44 levels per side) and collinear with devigged bet365 through arbitrage, so they are a sharp-reference candidate rather than a mispricing venue (probe scripts `experiments/scripts/polymarket_mlb_*.py`, recorded in dff040d; no results file exists). The observations are on [polymarket-market-research](../../wiki/collection/polymarket-market-research.md).

## Decision

Stop extending the Polymarket pipeline.

- The NBA-targeted code and tables stay in place.
- No Polymarket features go into the deployed model.
- Terraform `enable_polymarket` defaults to false.

Polymarket as an MLB sharp reference is an open question (#456) and not part of this decision.

## Status

proposed. It records a choice that was previously embedded in the Polymarket and agent-data-sources wiki pages. #417 asks that ADR rationale come from the maintainer, because agent-written ADRs tend to invent the why. The market observations behind it are external or from unreproduced probes, so confirm them before accepting.

## Consequences

- The pipeline stays NBA-only: tickers and the configured series and tag ids are NBA, and events are linked only by `odds polymarket link` (#456).
- `best_bid`, `best_ask`, `spread` and `midpoint` on price snapshots are never populated by the live pipeline (#451).
- `odds_core.config.PolymarketConfig.enabled` defaults to true while the Terraform gate defaults to false, so a local scheduler runs the jobs unless disabled while AWS does not.
- Price history for resolved markets expires after about 31 days and order books cannot be backfilled, so data not collected live is lost ([polymarket-market-research](../../wiki/collection/polymarket-market-research.md#data-availability)).
- Extending the pipeline to another sport needs its own series and tag ids and ticker parser; the client is sport-agnostic apart from `get_nba_events` ([polymarket](../../wiki/collection/polymarket.md#components)).

## Evidence

#134, dff040d, `experiments/scripts/polymarket_mlb_discover.py`, `experiments/scripts/polymarket_mlb_orderbooks.py`, `experiments/scripts/polymarket_mlb_volume.py`, `experiments/scripts/polymarket_mlb_sentiment.py`, #417, #451, #456.
