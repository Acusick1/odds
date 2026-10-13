---
type: concept
title: Odds arithmetic and devigging
description: The shared price-conversion, implied-probability, proportional-devig and profit helpers, which consumers devig and which stay vig-inclusive, and the diverging converter copies that store prices one point apart.
sources:
  - packages/odds-core/odds_core/odds_math.py
  - packages/odds-core/odds_core/odds_math.py::devig_probabilities
  - packages/odds-analytics/odds_analytics/utils.py::per_book_market_holds
  - packages/odds-analytics/odds_analytics/utils.py::calculate_market_hold
  - packages/odds-analytics/odds_analytics/sequence_loader.py::extract_devigged_h2h_probs
  - packages/odds-analytics/odds_analytics/sequence_loader.py::extract_devigged_totals_probs
  - packages/odds-analytics/odds_analytics/sequence_loader.py::calculate_devigged_bookmaker_target
  - packages/odds-analytics/odds_analytics/feature_groups.py
  - packages/odds-analytics/odds_analytics/training/config.py::FeatureConfig
  - packages/odds-analytics/odds_analytics/sequence_loader.py::calculate_devigged_bookmaker_target
  - packages/odds-lambda/odds_lambda/oddsportal_common.py::decimal_to_american
  - packages/odds-lambda/odds_lambda/storage/validators.py::american_to_probability
  - packages/odds-lambda/odds_lambda/betfair/adapter.py
  - packages/odds-lambda/odds_lambda/oddsportal_adapter.py
  - packages/odds-mcp/odds_mcp/server.py::find_retail_edges
status: stable
---

# Odds arithmetic and devigging

`packages/odds-core/odds_core/odds_math.py` is the intended single home for pure odds arithmetic. It has no third-party imports so the scraper Lambda (which bundles odds-lambda but not odds-analytics) can use it.

## Functions

- `packages/odds-core/odds_core/odds_math.py::american_to_decimal(american_odds)`: positive `a/100 + 1`, negative `100/|a| + 1`.
- `decimal_to_american(decimal_odds)`: `>= 2.0` gives `int((d-1)*100)`, otherwise `int(-100/(d-1))`. `int()` truncates toward zero.
- `packages/odds-core/odds_core/odds_math.py::calculate_implied_probability(american_odds)`: `1 / american_to_decimal(...)`. Includes vig.
- `packages/odds-core/odds_core/odds_math.py::devig_probabilities(*probs)`: proportional normalization to sum 1 for any outcome count; uniform if the sum is `<= 0`; `()` for no input.
- `determine_h2h_winner(home_score, away_score)` returns `"home"`, `"away"` or `"draw"`.
- `calculate_profit_from_odds(stake, american_odds, won)`.

Consumers include `packages/odds-lambda/odds_lambda/storage/readers.py` ([OddsReader](../storage/odds-reader-and-storage-layer.md)), `packages/odds-lambda/odds_lambda/paper_trading.py`, backtesting services, feature extraction and the MCP server. Analytics-side hold helpers (`packages/odds-analytics/odds_analytics/utils.py::calculate_market_hold`, `packages/odds-analytics/odds_analytics/utils.py::per_book_market_holds` in `packages/odds-analytics/odds_analytics/utils.py`) are built on `packages/odds-core/odds_core/odds_math.py::calculate_implied_probability`; `packages/odds-analytics/odds_analytics/utils.py::per_book_market_holds` drops books missing any required outcome rather than returning a partial hold.

## Devigging

Bookmaker prices sum to more than 1 once converted to implied probability; the excess is the vig (hold). Devigging rescales one bookmaker's outcome probabilities for one market so they sum to 1, giving a "fair" estimate.

Only proportional (multiplicative) devigging exists: `packages/odds-core/odds_core/odds_math.py::devig_probabilities` divides each implied probability by their sum, for 2-way and 3-way markets. No power, Shin or additive method is implemented.

All production use is in `packages/odds-analytics/odds_analytics/sequence_loader.py`, which owns the training-label maths despite its name:

- `packages/odds-analytics/odds_analytics/sequence_loader.py::extract_devigged_h2h_probs` filters one bookmaker's h2h/1x2 rows, needs home and away, adds draw when present, returns `DeviggedProbs(home, draw, away)`. Default bookmaker `pinnacle`.
- `packages/odds-analytics/odds_analytics/sequence_loader.py::extract_devigged_totals_probs` does the same for Over/Under (`packages/odds-analytics/odds_analytics/sequence_loader.py::DeviggedTotalsProbs`).
- `packages/odds-analytics/odds_analytics/sequence_loader.py::calculate_devigged_bookmaker_target` is `fair_close_home - fair_snapshot_home`; the totals variant picks Over or Under. Both return `None` if the bookmaker is missing on either side.

`packages/odds-analytics/odds_analytics/feature_groups.py` selects these only when `FeatureConfig.target_type == "devigged_bookmaker"`, using `FeatureConfig.target_bookmaker`; events with no closing data for that bookmaker are dropped. The code defaults are `target_type="raw"` and `target_bookmaker="pinnacle"`, so devigging applies only when the training YAML sets it (for example `experiments/configs/xgboost_epl_production_all_features.yaml`). Only h2h and totals are supported; spreads have no devigged target. For a 3-way market the target uses the home-win probability only (draw is computed but unused), and devigging is per bookmaker, with no cross-book consensus. The resulting target is described in [CLV and the training target](../modelling/clv-target.md) and [Feature pipeline](../modelling/feature-pipeline.md).

### Vig-inclusive consumers

These use raw `packages/odds-core/odds_core/odds_math.py::calculate_implied_probability` and are not devigged:

- `target_type="raw"`: mean implied-probability delta across books, snapshot to close.
- `packages/odds-mcp/odds_mcp/server.py::find_retail_edges`: `divergence = retail_implied_prob - sharp_implied_prob`, both vig-inclusive, so a sharp book's own margin is part of every divergence ([MCP market tools](../agent/mcp-market-tools.md)).
- The `sharp_prob` and `retail_sharp_diff` model features ([Sharp benchmark](../modelling/sharp-benchmark.md)).
- `basic_ev` backtests use the vig-inclusive sharp probability as truth ([Backtest strategies](../evaluation/backtest-strategies.md)).
- Hold measurement (`packages/odds-analytics/odds_analytics/utils.py::per_book_market_holds`) measures the vig rather than removing it.

## Hazards

- **Two `decimal_to_american` implementations.** `packages/odds-lambda/odds_lambda/oddsportal_common.py::decimal_to_american` rounds, and returns `-10000` for `d <= 1.0`. The core one truncates and raises `ZeroDivisionError` at `d == 1.0`. For `d = 1.91` the core copy gives `-109`, the OddsPortal copy `-110`.
- **Which stored prices come from which copy.** [Betfair](../collection/betfair-exchange.md) snapshots use the core function (`packages/odds-lambda/odds_lambda/betfair/adapter.py`); OddsPortal snapshots (`packages/odds-lambda/odds_lambda/oddsportal_adapter.py`, `build_raw_data`) and `scripts/ingest_football_data_uk.py` use the `oddsportal_common` rounding copy ([OddsPortal](../collection/oddsportal.md)). Identical decimal prices can therefore be stored one point apart by source.
- **Implied probability is reimplemented** in `OddsValidator.american_to_probability` (`packages/odds-lambda/odds_lambda/storage/validators.py`) with the same formula as core, not calling it ([Odds validation](../storage/data-validation.md)).
- `packages/odds-core/odds_core/odds_math.py::calculate_implied_probability` divides by `packages/odds-core/odds_core/odds_math.py::american_to_decimal`, so `american_odds == 0` raises.
- Docstring examples in `packages/odds-core/odds_core/odds_math.py` are rounded (`1.909`, `0.524`) and would not pass a doctest runner against the exact returns.
- `packages/odds-core/odds_core/odds_math.py::devig_probabilities` accepts probabilities only; it does not validate they are in `[0, 1]`.
- The label definition lives in `packages/odds-analytics/odds_analytics/sequence_loader.py` rather than a target module; grep there first.

## Related

- [Sharp benchmark and retail-sharp divergence](../modelling/sharp-benchmark.md)
- [Odds snapshots and OddsWriter](../storage/odds-storage.md) — raw price format
- [Decision time and look-ahead bias](../modelling/decision-time-and-look-ahead.md)
