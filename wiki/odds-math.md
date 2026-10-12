---
type: component
title: Odds arithmetic (odds_math)
summary: Shared American/decimal conversion, implied probability, proportional devig and profit helpers in odds-core, plus the diverging converter copies elsewhere.
sources:
  - packages/odds-core/odds_core/odds_math.py
  - packages/odds-analytics/odds_analytics/utils.py::per_book_market_holds
  - packages/odds-analytics/odds_analytics/utils.py::calculate_market_hold
  - packages/odds-lambda/odds_lambda/oddsportal_common.py::decimal_to_american
  - packages/odds-lambda/odds_lambda/storage/validators.py::american_to_probability
  - packages/odds-lambda/odds_lambda/betfair/adapter.py
  - packages/odds-lambda/odds_lambda/oddsportal_adapter.py
verified_at: 7887039
---

# Odds arithmetic (odds_math)

`packages/odds-core/odds_core/odds_math.py` is the intended single home for pure odds arithmetic. It has no third-party imports so the scraper Lambda (which bundles odds-lambda but not odds-analytics) can use it.

## Functions

- `american_to_decimal(american_odds)`: positive `a/100 + 1`, negative `100/|a| + 1`.
- `decimal_to_american(decimal_odds)`: `>= 2.0` gives `int((d-1)*100)`, otherwise `int(-100/(d-1))`. `int()` truncates toward zero.
- `calculate_implied_probability(american_odds)`: `1 / american_to_decimal(...)`. Includes vig; see [Devigging](devigging.md).
- `devig_probabilities(*probs)`: proportional normalization to sum 1 for any outcome count; uniform if the sum is `<= 0`; `()` for no input.
- `determine_h2h_winner(home_score, away_score)` returns `"home"`, `"away"` or `"draw"`.
- `calculate_profit_from_odds(stake, american_odds, won)`.

Consumers include `packages/odds-lambda/odds_lambda/storage/readers.py` ([OddsReader](odds-reader.md)), `packages/odds-lambda/odds_lambda/paper_trading.py`, backtesting services, feature extraction and the MCP server. Analytics-side hold helpers (`calculate_market_hold`, `per_book_market_holds` in `packages/odds-analytics/odds_analytics/utils.py`) are built on `calculate_implied_probability`. `per_book_market_holds` drops books missing any required outcome rather than returning a partial hold.

## Hazards

- **Two `decimal_to_american` implementations.** `packages/odds-lambda/odds_lambda/oddsportal_common.py::decimal_to_american` rounds, and returns `-10000` for `d <= 1.0`. The core one truncates and raises `ZeroDivisionError` at `d == 1.0`. For `d = 1.91` the core copy gives `-109`, the OddsPortal copy `-110`.
- Which stored prices come from which copy: [Betfair](betfair-exchange-client.md) snapshots use the core function (`packages/odds-lambda/odds_lambda/betfair/adapter.py`); OddsPortal snapshots (`oddsportal_adapter.py`, `build_raw_data`) and `scripts/ingest_football_data_uk.py` use the `oddsportal_common` copy. Identical decimal prices can therefore be stored one point apart by source.
- **Implied probability is reimplemented** in `OddsValidator.american_to_probability` (`packages/odds-lambda/odds_lambda/storage/validators.py`) with the same formula as core, not calling it.
- `calculate_implied_probability` divides by `american_to_decimal`, so `american_odds == 0` raises.
- Docstring examples in `odds_math.py` are rounded (`1.909`, `0.524`) and would not pass a doctest runner against the exact returns.
- `devig_probabilities` accepts probabilities only; it does not validate they are in `[0, 1]`.

## Related

- [Devigging](devigging.md)
- [Odds snapshots table](table-odds-snapshots.md) (raw price format)
- [Data quality validation](data-quality-validation.md)
- [Look-ahead bias](look-ahead-bias.md)
