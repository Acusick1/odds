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

`odds_analytics.feature_groups._compute_target` selects the devigged functions only when `odds_analytics.training.config.FeatureConfig.target_type` is `devigged_bookmaker`, using `odds_analytics.training.config.FeatureConfig.target_bookmaker`, which only devigged configs set, and the primary market; `odds_analytics.feature_groups.prepare_training_data` drops events with no closing data for that bookmaker, whichever sampler picks the decision snapshots. Devigging therefore applies only when the training YAML sets it (for example the production config `experiments/configs/xgboost_epl_production_tabular_match_stats.yaml`); the fields are on [Training config](../modelling/training-config.md). Only h2h and totals are supported; spreads have no devigged target. The target uses the probability of the configured `outcome`. `odds_analytics.training.config.FeatureConfig.outcome` cannot take `draw`, so only home and away are reachable from a config.

## Vig-inclusive consumers

These use raw `odds_core.odds_math.calculate_implied_probability` and are not devigged:

- `target_type="raw"`: mean implied-probability delta across books, snapshot to close.
- `odds_mcp.server.find_retail_edges`: `divergence = retail_implied_prob - sharp_implied_prob`, both vig-inclusive: the sharp probability comes from a book in the sharp list, and only the rows are retail books (the sharp list, Pinnacle and Polymarket are never rows), so a sharp book's own margin is part of every divergence ([retail edges](../agent/mcp-market-tools.md#retail-edges)).
- The `sharp_prob` and `retail_sharp_diff` model features ([Sharp benchmark](../modelling/sharp-benchmark.md)).
- `basic_ev` backtests use the vig-inclusive sharp probability as truth ([Backtest strategies](../evaluation/backtest-strategies.md)).
- Hold measurement (`odds_analytics.utils.per_book_market_holds`) measures the vig rather than removing it.

Comparing a vig-inclusive probability with a devigged one mixes two scales; the vig-inclusive consumers above are the places where that can happen `(inferred)`.

## Related

- [Odds arithmetic helpers](odds-math.md)
- [Look-ahead bias](../modelling/look-ahead-bias.md)
