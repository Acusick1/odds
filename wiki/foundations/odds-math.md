---
type: reference
title: Odds arithmetic helpers
description: The odds_core.odds_math functions, their edge cases, their consumers, and and the two `raw_data` price encodings.
sources:
  - odds_core.odds_math
  - odds_core.odds_math.devig_probabilities
  - odds_analytics.utils.per_book_market_holds
  - odds_analytics.utils.calculate_market_hold
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
- `odds_core.odds_math.decimal_to_american(decimal_odds)`: `>= 2.0` gives `round((d-1)*100)`, otherwise `round(-100/(d-1))`, the nearest integer American quote. `d <= 1.0` raises `ValueError`. Integer American odds cannot represent every decimal price, so this is for display and bet records.
- `odds_core.odds_math.decimal_to_implied_probability(decimal_odds)`: `1 / d`. Includes vig.
- `odds_core.odds_math.outcome_price_decimal(outcome)`: the decimal price of a `raw_data` outcome, reading `price_decimal` if present, else converting the integer `price`; `KeyError` if neither exists.
- `odds_core.odds_math.calculate_implied_probability(american_odds)`: `decimal_to_implied_probability(american_to_decimal(...))`. Includes vig. `american_odds == 0` raises.
- `odds_core.odds_math.devig_probabilities(*probs)`: proportional normalization to sum 1 for any outcome count; uniform if the sum is `<= 0`; `()` for no input. It does not validate that inputs are in `[0, 1]` ([Devigging](devigging.md)).
- `odds_core.odds_math.determine_h2h_winner(home_score, away_score)` returns `"home"`, `"away"` or `"draw"`.
- `odds_core.odds_math.calculate_profit_from_odds(stake, american_odds, won)`.

The docstring examples show rounded values (`1.909`, `0.524`), so a doctest runner would fail them against the exact returns.

## Consumers

`odds_lambda.storage.readers.OddsReader` ([OddsReader](../storage/odds-reader.md)), `odds_lambda.paper_trading`, backtesting services, feature extraction and the MCP server. Analytics-side hold helpers (`odds_analytics.utils.calculate_market_hold`, `odds_analytics.utils.per_book_market_holds`) are built on `odds_core.odds_math.decimal_to_implied_probability` and take decimal prices; `odds_analytics.utils.per_book_market_holds` drops books missing any required outcome rather than returning a partial hold. The training target maths built on these helpers is on [Target functions](../modelling/clv-target-functions.md).

## Price encodings

`odds_core.models.Odds.price_decimal` stores the source's decimal price; `odds_core.models.Odds.implied_probability` and `odds_core.models.Odds.american_price` derive from it through `odds_core.odds_math.decimal_to_implied_probability` and `odds_core.odds_math.decimal_to_american` ([Odds snapshot and odds tables](../storage/odds-tables.md)). `odds_core.odds_math.decimal_to_american` is the only decimal-to-American conversion.

`raw_data` outcomes carry one of two encodings, named by `odds_core.odds_math.PRICE_DECIMAL_KEY` (`price_decimal`) and `odds_core.odds_math.PRICE_AMERICAN_KEY` (`price`):

- Snapshots written by the current ingest code store the source decimal under `price_decimal` and have no `price` key: Betfair (`odds_lambda.betfair.adapter`), OddsPortal live (`odds_lambda.oddsportal_adapter`), OddsPortal historical (`odds_lambda.oddsportal_common.build_raw_data`) and `scripts/ingest_football_data_uk.py`.
- The Odds API's responses keep their integer American `price` unchanged, which `odds_core.odds_math.american_to_decimal` converts exactly.

Snapshots stored before the decimal price migration keep `price` where the migration could not recover the original: `betfair_api` outcomes whose `betfair_meta` runner did not truncate to the stored price, the OddsPortal `-10000` outcomes in a market the migration kept (decoded as 1.01 in `odds.price_decimal` only), and every `football_data_uk` snapshot until `scripts/ingest_football_data_uk.py` is re-run. That `price` is the old integer American quote (truncated for Betfair, rounded for OddsPortal and FDUK), so its decimal is exact only for the Odds API. The migration also removed OddsPortal markets that contain `-10000` and whose implied probabilities sum above 1.3 (source garbage), from `raw_data` and from `odds`.

`odds_core.odds_math.outcome_price_decimal` reads either encoding; `odds_core.snapshot_utils.extract_odds_from_snapshot`, `odds_lambda.storage.writers.OddsWriter._parse_and_store_odds` and `odds_lambda.storage.validators.OddsValidator` all go through it.

## Related

- [Devigging and vig-inclusive prices](devigging.md)
- [Odds snapshot and odds tables](../storage/odds-tables.md) — raw price format
- [Sharp benchmark and retail-sharp divergence](../modelling/sharp-benchmark.md)
