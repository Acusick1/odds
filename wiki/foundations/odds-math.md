---
type: reference
title: Odds arithmetic helpers
description: The odds_core.odds_math functions, their edge cases, their consumers, and the other copies of the same conversions.
sources:
  - odds_core.odds_math
  - odds_core.odds_math.devig_probabilities
  - odds_analytics.utils.per_book_market_holds
  - odds_analytics.utils.calculate_market_hold
  - odds_lambda.oddsportal_common.decimal_to_american
  - odds_lambda.oddsportal_common.build_raw_data
  - odds_lambda.betfair.adapter
  - odds_lambda.oddsportal_adapter
  - scripts/ingest_football_data_uk.py
status: stable
---

# Odds arithmetic helpers

`odds_core.odds_math` is the intended single home for pure odds arithmetic. It has no third-party imports so the scraper Lambda (which bundles odds-lambda but not odds-analytics) can use it.

## Functions

- `odds_core.odds_math.american_to_decimal(american_odds)`: positive `a/100 + 1`, negative `100/|a| + 1`.
- `odds_core.odds_math.decimal_to_american(decimal_odds)`: `>= 2.0` gives `int((d-1)*100)`, otherwise `int(-100/(d-1))`. `int()` truncates toward zero; `d == 1.0` raises `ZeroDivisionError`.
- `odds_core.odds_math.calculate_implied_probability(american_odds)`: `1 / american_to_decimal(...)`. Includes vig. It divides by `odds_core.odds_math.american_to_decimal`, so `american_odds == 0` raises.
- `odds_core.odds_math.devig_probabilities(*probs)`: proportional normalization to sum 1 for any outcome count; uniform if the sum is `<= 0`; `()` for no input. It does not validate that inputs are in `[0, 1]` ([Devigging](devigging.md)).
- `odds_core.odds_math.determine_h2h_winner(home_score, away_score)` returns `"home"`, `"away"` or `"draw"`.
- `odds_core.odds_math.calculate_profit_from_odds(stake, american_odds, won)`.

## Consumers

`odds_lambda.storage.readers` ([OddsReader](../storage/odds-reader.md)), `odds_lambda.paper_trading`, backtesting services, feature extraction and the MCP server. Analytics-side hold helpers (`odds_analytics.utils.calculate_market_hold`, `odds_analytics.utils.per_book_market_holds`) are built on `odds_core.odds_math.calculate_implied_probability`; `odds_analytics.utils.per_book_market_holds` drops books missing any required outcome rather than returning a partial hold. The training target maths built on these helpers is on [Target functions](../modelling/clv-target-functions.md).

## Conversion copies

Two implementations of decimal-to-American conversion exist (#446): `odds_lambda.oddsportal_common.decimal_to_american` rounds and returns `-10000` for `d <= 1.0`; `odds_core.odds_math.decimal_to_american` truncates and raises `ZeroDivisionError` at `d == 1.0`. For `d = 1.91` the core copy gives `-109` and the OddsPortal copy `-110`.

Which stored prices come from which copy:

- [Betfair](../collection/betfair-exchange.md) snapshots use the core function (`odds_lambda.betfair.adapter.build_raw_data`).
- OddsPortal snapshots (`odds_lambda.oddsportal_adapter`, `odds_lambda.oddsportal_common.build_raw_data`) and `scripts/ingest_football_data_uk.py` use the `odds_lambda.oddsportal_common` rounding copy ([OddsPortal scraper](../collection/oddsportal-scraper.md)).

Identical decimal prices can therefore be stored one point apart by source (#446). `odds_lambda.storage.validators.OddsValidator.american_to_probability` is a third copy of the implied-probability conversion ([Odds validation](../storage/odds-validator.md)).

## Related

- [Devigging and vig-inclusive prices](devigging.md)
- [Odds snapshot and odds tables](../storage/odds-tables.md) — raw price format
- [Sharp benchmark and retail-sharp divergence](../modelling/sharp-benchmark.md)
