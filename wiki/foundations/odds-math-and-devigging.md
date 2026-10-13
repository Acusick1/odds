---
type: concept
title: Odds arithmetic and devigging
description: The shared price-conversion, implied-probability, proportional-devig and profit helpers, which consumers devig and which stay vig-inclusive, and the diverging converter copies that store prices one point apart.
sources:
  - odds_core.odds_math
  - odds_core.odds_math.devig_probabilities
  - odds_analytics.utils.per_book_market_holds
  - odds_analytics.utils.calculate_market_hold
  - odds_analytics.sequence_loader.extract_devigged_h2h_probs
  - odds_analytics.sequence_loader.extract_devigged_totals_probs
  - odds_analytics.sequence_loader.calculate_devigged_bookmaker_target
  - odds_analytics.feature_groups
  - odds_analytics.training.config.FeatureConfig
  - odds_analytics.sequence_loader.calculate_devigged_bookmaker_target
  - odds_lambda.oddsportal_common.decimal_to_american
  - odds_lambda.storage.validators.OddsValidator.american_to_probability
  - odds_lambda.betfair.adapter
  - odds_lambda.oddsportal_adapter
  - odds_mcp.server.find_retail_edges
status: stable
---

# Odds arithmetic and devigging

`odds_core.odds_math` is the intended single home for pure odds arithmetic. It has no third-party imports so the scraper Lambda (which bundles odds-lambda but not odds-analytics) can use it.

## Functions

- `odds_core.odds_math.american_to_decimal(american_odds)`: positive `a/100 + 1`, negative `100/|a| + 1`.
- `decimal_to_american(decimal_odds)`: `>= 2.0` gives `int((d-1)*100)`, otherwise `int(-100/(d-1))`. `int()` truncates toward zero.
- `odds_core.odds_math.calculate_implied_probability(american_odds)`: `1 / american_to_decimal(...)`. Includes vig.
- `odds_core.odds_math.devig_probabilities(*probs)`: proportional normalization to sum 1 for any outcome count; uniform if the sum is `<= 0`; `()` for no input.
- `determine_h2h_winner(home_score, away_score)` returns `"home"`, `"away"` or `"draw"`.
- `calculate_profit_from_odds(stake, american_odds, won)`.

Consumers include `odds_lambda.storage.readers` ([OddsReader](../storage/odds-reader-and-storage-layer.md)), `odds_lambda.paper_trading`, backtesting services, feature extraction and the MCP server. Analytics-side hold helpers (`odds_analytics.utils.calculate_market_hold`, `odds_analytics.utils.per_book_market_holds` in `odds_analytics.utils`) are built on `odds_core.odds_math.calculate_implied_probability`; `odds_analytics.utils.per_book_market_holds` drops books missing any required outcome rather than returning a partial hold.

## Devigging

Bookmaker prices sum to more than 1 once converted to implied probability; the excess is the vig (hold). Devigging rescales one bookmaker's outcome probabilities for one market so they sum to 1, giving a "fair" estimate.

Only proportional (multiplicative) devigging exists: `odds_core.odds_math.devig_probabilities` divides each implied probability by their sum, for 2-way and 3-way markets. No power, Shin or additive method is implemented.

All production use is in `odds_analytics.sequence_loader`, which owns the training-label maths despite its name:

- `odds_analytics.sequence_loader.extract_devigged_h2h_probs` filters one bookmaker's h2h/1x2 rows, needs home and away, adds draw when present, returns `DeviggedProbs(home, draw, away)`. Default bookmaker `pinnacle`.
- `odds_analytics.sequence_loader.extract_devigged_totals_probs` does the same for Over/Under (`odds_analytics.sequence_loader.DeviggedTotalsProbs`).
- `odds_analytics.sequence_loader.calculate_devigged_bookmaker_target` is `fair_close_home - fair_snapshot_home`; the totals variant picks Over or Under. Both return `None` if the bookmaker is missing on either side.

`odds_analytics.feature_groups` selects these only when `FeatureConfig.target_type == "devigged_bookmaker"`, using `odds_analytics.training.config.FeatureConfig.target_bookmaker`; events with no closing data for that bookmaker are dropped. The code defaults are `target_type="raw"` and `target_bookmaker="pinnacle"`, so devigging applies only when the training YAML sets it (for example `experiments/configs/xgboost_epl_production_all_features.yaml`). Only h2h and totals are supported; spreads have no devigged target. For a 3-way market the target uses the home-win probability only (draw is computed but unused), and devigging is per bookmaker, with no cross-book consensus. The resulting target is described in [CLV and the training target](../modelling/clv-target.md) and [Feature pipeline](../modelling/feature-pipeline.md).

### Vig-inclusive consumers

These use raw `odds_core.odds_math.calculate_implied_probability` and are not devigged:

- `target_type="raw"`: mean implied-probability delta across books, snapshot to close.
- `odds_mcp.server.find_retail_edges`: `divergence = retail_implied_prob - sharp_implied_prob`, both vig-inclusive, so a sharp book's own margin is part of every divergence ([MCP market tools](../agent/mcp-market-tools.md)).
- The `sharp_prob` and `retail_sharp_diff` model features ([Sharp benchmark](../modelling/sharp-benchmark.md)).
- `basic_ev` backtests use the vig-inclusive sharp probability as truth ([Backtest strategies](../evaluation/backtest-strategies.md)).
- Hold measurement (`odds_analytics.utils.per_book_market_holds`) measures the vig rather than removing it.

## Hazards

- **Two implementations of decimal-to-American conversion.** `odds_lambda.oddsportal_common.decimal_to_american` rounds, and returns `-10000` for `d <= 1.0`. The core one (`odds_core.odds_math.decimal_to_american`) truncates and raises `ZeroDivisionError` at `d == 1.0`. For `d = 1.91` the core copy gives `-109`, the OddsPortal copy `-110`.
- **Which stored prices come from which copy.** [Betfair](../collection/betfair-exchange.md) snapshots use the core function (`odds_lambda.betfair.adapter.build_raw_data`); OddsPortal snapshots (`odds_lambda.oddsportal_adapter`, `odds_lambda.oddsportal_common.build_raw_data`) and `scripts/ingest_football_data_uk.py` use the `oddsportal_common` rounding copy ([OddsPortal](../collection/oddsportal.md)). Identical decimal prices can therefore be stored one point apart by source.
- **Implied probability is reimplemented** in `odds_lambda.storage.validators.OddsValidator.american_to_probability` with the same formula as core, not calling it ([Odds validation](../storage/data-validation.md)).
- `odds_core.odds_math.calculate_implied_probability` divides by `odds_core.odds_math.american_to_decimal`, so `american_odds == 0` raises.
- Docstring examples in `odds_core.odds_math` are rounded (`1.909`, `0.524`) and would not pass a doctest runner against the exact returns.
- `odds_core.odds_math.devig_probabilities` accepts probabilities only; it does not validate they are in `[0, 1]`.
- The label definition lives in `odds_analytics.sequence_loader` rather than a target module; grep there first.

## Related

- [Sharp benchmark and retail-sharp divergence](../modelling/sharp-benchmark.md)
- [Odds snapshots and OddsWriter](../storage/odds-storage.md) — raw price format
- [Decision time and look-ahead bias](../modelling/decision-time-and-look-ahead.md)
