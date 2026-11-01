---
type: reference
title: Odds arithmetic helpers
description: The odds_core.odds_math functions, their edge cases, their consumers, and the two envelope price encodings.
sources:
  - odds_core.odds_math
  - odds_core.odds_math.devig_probabilities
  - odds_analytics.utils.per_book_market_holds
  - odds_analytics.utils.calculate_market_hold
  - odds_lambda.oddsportal_common.build_envelope
  - odds_lambda.betfair.adapter.betfair_book_to_bookmaker_entry
  - odds_lambda.oddsportal_adapter.convert_market
  - scripts/ingest_football_data_uk.py
mentions:
  - odds_lambda.storage.readers.OddsReader
status: stable
---

# Odds arithmetic helpers

`odds_core.odds_math` is the intended single home for pure odds arithmetic. It has no third-party imports so `odds-lambda` code that cannot import `odds-analytics` can use it.

## Functions

- `odds_core.odds_math.american_to_decimal(american_odds)`: positive `a/100 + 1`, negative `100/|a| + 1`.
- `odds_core.odds_math.decimal_to_american(decimal_odds)`: `>= 2.0` gives `round((d-1)*100)`, otherwise `round(-100/(d-1))`, the nearest integer American quote. `d <= 1.0` raises `ValueError`. Integer American odds cannot represent every decimal price, so this is for display and bet records.
- `odds_core.odds_math.decimal_to_implied_probability(decimal_odds)`: `1 / d`. Includes vig.
- `odds_core.odds_math.outcome_price_decimal(outcome)`: the decimal price of an envelope outcome, reading `price_decimal` if present, else converting the integer `price`; `KeyError` if neither exists.
- `odds_core.odds_math.calculate_implied_probability(american_odds)`: `decimal_to_implied_probability(american_to_decimal(...))`. Includes vig. `american_odds == 0` raises.
- `odds_core.odds_math.devig_probabilities(*probs)`: proportional normalization to sum 1 for any outcome count; uniform if the sum is `<= 0`; `()` for no input. It does not validate that inputs are in `[0, 1]` ([Devigging](devigging.md)).
- `odds_core.odds_math.determine_h2h_winner(home_score, away_score)` returns `"home"`, `"away"` or `"draw"`.
- `odds_core.odds_math.selection_won(selection, line, home_score, away_score)`: whether a selection won on the final score. `home`, `draw` or `away` is the result of `odds_core.odds_math.determine_h2h_winner` and takes no line; `over` or `under` compares the total score with a half line. A totals selection without a half line, or a line on another selection, raises `ValueError`. Paper-trade settlement and the replay backtest use it.
- `odds_core.odds_math.calculate_profit_from_odds(stake, american_odds, won)`.
- `odds_core.odds_math.calculate_profit_from_decimal(stake, price_decimal, won)`: `stake * (d - 1)` on a win, `-stake` on a loss. Paper-trade settlement uses it.

The docstring examples show rounded values (`1.909`, `0.524`), so a doctest runner would fail them against the exact returns.

## Consumers

`odds_lambda.storage.readers.OddsReader` ([OddsReader](../storage/odds-reader.md); it reads events (keyed by integer id), snapshots and logs, never prices, which come from `odds_core.quotes.quotes`), `odds_lambda.paper_trading` (settlement grades with `odds_core.odds_math.selection_won`, takes profit from `odds_core.odds_math.calculate_profit_from_decimal` and voids trades on cancelled events, with no bankroll arithmetic for a flat-stake trade; placement rejects a draw bet on a two-way sport and an `h2h` candidate trade on a three-way sport), backtesting services, the feature extractors and the MCP server tools. Analytics-side hold helpers (`odds_analytics.utils.calculate_market_hold`, `odds_analytics.utils.per_book_market_holds`) are built on `odds_core.odds_math.decimal_to_implied_probability` and take decimal prices; `odds_analytics.utils.per_book_market_holds` drops books missing any required outcome rather than returning a partial hold. The training target maths built on these helpers is on [Target functions](../modelling/clv-target-functions.md).

## Price encodings

`odds_core.models.Price.price` stores the decimal price of one quote (`price > 1.0` by check constraint), and `odds_core.quotes.Quote.price` reads it back. Probabilities derive from it through `odds_core.odds_math.decimal_to_implied_probability` ([Price tables](../storage/price-tables.md)). `odds_core.odds_math.decimal_to_american` is the only decimal-to-American conversion.

The envelope a source returns (archived in `odds_core.models.SnapshotPayload`, [envelope format](../storage/odds-tables.md#envelope-format)) carries each outcome's price in one of two encodings, named by `odds_core.odds_math.PRICE_DECIMAL_KEY` (`price_decimal`) and `odds_core.odds_math.PRICE_AMERICAN_KEY` (`price`):

- The envelopes built by this pipeline hold the source decimal under `price_decimal` and have no `price` key: Betfair (`odds_lambda.betfair.adapter.betfair_book_to_bookmaker_entry`, every market it collects, each totals runner on its spec's line), OddsPortal live (`odds_lambda.oddsportal_adapter.convert_market`, whose spread points are signed per outcome and whose ladder lines each carry their own volumes, [spreads line sign](../storage/price-tables.md#spreads-line-sign)), OddsPortal historical (`odds_lambda.oddsportal_common.build_envelope`, whose `last_update` and `_snapshot_time` come from UTC-aware times) and `scripts/ingest_football_data_uk.py` (kick-off read as UK local time, stored as UTC), which skips a game whose event match is ambiguous.
- The Odds API's responses keep their integer American `price` unchanged, which `odds_core.odds_math.american_to_decimal` converts exactly.

`odds_core.odds_math.outcome_price_decimal` reads either encoding; `odds_lambda.price_rows.expand_envelope` (which turns an envelope into price rows) and `odds_lambda.storage.validators.OddsValidator`, which validates one event's envelope per call, both go through it.

## Related

- [Devigging and vig-inclusive prices](devigging.md)
- [Odds snapshots table](../storage/odds-tables.md) — envelope format
- [Price tables](../storage/price-tables.md)
- [Sharp benchmark and retail-sharp divergence](../modelling/sharp-benchmark.md)
