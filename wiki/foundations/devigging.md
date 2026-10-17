---
type: explanation
title: Devigging and vig-inclusive prices
description: What vig and proportional devigging are here, which consumers devig and which stay vig-inclusive, and what that means for comparing prices.
sources:
  - odds_core.odds_math.devig_probabilities
  - odds_core.odds_math.calculate_implied_probability
  - odds_analytics.utils.per_book_market_holds
  - odds_analytics.sequence_loader.extract_devigged_h2h_probs
  - odds_analytics.sequence_loader.extract_devigged_totals_probs
  - odds_analytics.sequence_loader.calculate_devigged_bookmaker_target
  - odds_analytics.sequence_loader.DeviggedProbs
  - odds_analytics.sequence_loader.DeviggedTotalsProbs
  - odds_analytics.feature_groups
  - odds_analytics.training.config.FeatureConfig
  - odds_mcp.server.find_retail_edges
status: stable
---

# Devigging and vig-inclusive prices

Bookmaker prices sum to more than 1 once converted to implied probability (`odds_core.odds_math.calculate_implied_probability`); the excess is the vig (hold). Devigging rescales one bookmaker's outcome probabilities for one market so they sum to 1, giving a "fair" estimate. The helper functions are on [Odds arithmetic helpers](odds-math.md).

## What exists

Only proportional (multiplicative) devigging exists: `odds_core.odds_math.devig_probabilities` divides each implied probability by their sum, for 2-way and 3-way markets. No power, Shin or additive method is implemented. Devigging is per bookmaker, with no cross-book consensus.

All production use is in `odds_analytics.sequence_loader`, which holds the training-label maths despite its name ([Workspace packages](workspace-packages.md#names-that-mislead)): `odds_analytics.sequence_loader.extract_devigged_h2h_probs` (returns `odds_analytics.sequence_loader.DeviggedProbs`), `odds_analytics.sequence_loader.extract_devigged_totals_probs` and `odds_analytics.sequence_loader.calculate_devigged_bookmaker_target`. The functions, closing-snapshot selection and dispatch are on [Target functions](../modelling/clv-target-functions.md); why the target is a devigged delta is on [CLV and the training target](../modelling/clv-target.md).

`odds_analytics.feature_groups` selects the devigged functions only when `FeatureConfig.target_type == "devigged_bookmaker"`, using `odds_analytics.training.config.FeatureConfig.target_bookmaker`; events with no closing data for that bookmaker are dropped. Devigging therefore applies only when the training YAML sets it (for example `experiments/configs/xgboost_epl_production_tabular_match_stats.yaml`); the defaults are on [Training config](../modelling/training-config.md). Only h2h and totals are supported; spreads have no devigged target. For a 3-way market the target uses the home-win probability only (draw is computed but unused).

## Vig-inclusive consumers

These use raw `odds_core.odds_math.calculate_implied_probability` and are not devigged:

- `target_type="raw"`: mean implied-probability delta across books, snapshot to close.
- `odds_mcp.server.find_retail_edges`: `divergence = retail_implied_prob - sharp_implied_prob`, both vig-inclusive, so a sharp book's own margin is part of every divergence ([retail edges](../agent/mcp-market-tools.md#retail-edges)).
- The `sharp_prob` and `retail_sharp_diff` model features ([Sharp benchmark](../modelling/sharp-benchmark.md)).
- `basic_ev` backtests use the vig-inclusive sharp probability as truth ([Backtest strategies](../evaluation/backtest-strategies.md)).
- Hold measurement (`odds_analytics.utils.per_book_market_holds`) measures the vig rather than removing it.

Comparing a vig-inclusive probability with a devigged one mixes two scales; the vig-inclusive consumers above are the places where that can happen `(inferred)`.

## Related

- [Odds arithmetic helpers](odds-math.md)
- [Look-ahead bias](../modelling/look-ahead-bias.md)
