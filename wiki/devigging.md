---
type: concept
title: Devigging
summary: Proportional removal of bookmaker margin to get fair probabilities, where it is implemented, and which consumers deliberately stay vig-inclusive.
sources:
  - packages/odds-core/odds_core/odds_math.py::devig_probabilities
  - packages/odds-analytics/odds_analytics/sequence_loader.py::extract_devigged_h2h_probs
  - packages/odds-analytics/odds_analytics/sequence_loader.py::extract_devigged_totals_probs
  - packages/odds-analytics/odds_analytics/sequence_loader.py::calculate_devigged_bookmaker_target
  - packages/odds-analytics/odds_analytics/feature_groups.py
  - packages/odds-analytics/odds_analytics/training/config.py::FeatureConfig
  - packages/odds-mcp/odds_mcp/server.py::find_retail_edges
  - docs/DATA_MODELS.md#training-target
  - docs/MODELING.md#what-the-model-predicts
verified_at: 7887039
---

# Devigging

Bookmaker prices sum to more than 1 once converted to implied probability; the excess is the vig (hold). Devigging rescales one bookmaker's outcome probabilities for one market so they sum to 1, giving a "fair" estimate.

## Method

Only proportional (multiplicative) devigging exists: `packages/odds-core/odds_core/odds_math.py::devig_probabilities` divides each implied probability by their sum ([Odds arithmetic](odds-math.md)). It works for 2-way and 3-way markets. No power, Shin or additive method is implemented.

## Where it is applied

All production use is in `packages/odds-analytics/odds_analytics/sequence_loader.py`, which owns the training-label math despite its name:

- `extract_devigged_h2h_probs` filters one bookmaker's h2h/1x2 rows, needs home and away, adds draw when present, returns `DeviggedProbs(home, draw, away)`. Default bookmaker `pinnacle`.
- `extract_devigged_totals_probs` does the same for Over/Under (`DeviggedTotalsProbs`).
- `calculate_devigged_bookmaker_target` is `fair_close_home - fair_snapshot_home`; the totals variant picks Over or Under. Both return `None` if the bookmaker is missing on either side.

`packages/odds-analytics/odds_analytics/feature_groups.py` selects these when `FeatureConfig.target_type == "devigged_bookmaker"`, using `FeatureConfig.target_bookmaker`. Events with no closing data for the target bookmaker are dropped. Only h2h and totals are supported; spreads have no devigged target. This is the CLV regression target described in [CLV](clv.md) and [Feature pipeline](feature-pipeline.md).

## Vig-inclusive consumers

These use raw `calculate_implied_probability` and are not devigged:

- `target_type="raw"`: mean implied-probability delta across books, snapshot to close.
- `packages/odds-mcp/odds_mcp/server.py::find_retail_edges`: `divergence = retail_implied_prob - sharp_implied_prob`, both vig-inclusive, so a sharp book's own margin is part of every divergence. See [find_retail_edges](mcp-find-retail-edges.md).
- Hold measurement (`per_book_market_holds`) measures the vig rather than removing it.

## Hazards

- `docs/DATA_MODELS.md#training-target` states the target is the devigged bookmaker CLV delta and bet365 is the EPL target. Code defaults are `target_type="raw"` and `target_bookmaker="pinnacle"` (`FeatureConfig`); devigging and bet365 apply only when the config sets them (e.g. `experiments/configs/xgboost_epl_production_all_features.yaml`).
- For a 3-way market the target is the home-win probability only; draw is computed but unused.
- Devigging is per bookmaker; the target does not use a cross-book consensus.
- The label definition lives in `sequence_loader.py` rather than a target module; grep there first.

## Related

- [Sharp benchmark](sharp-benchmark.md)
- [Retail-sharp divergence](retail-sharp-divergence.md)
- [Odds arithmetic](odds-math.md)
