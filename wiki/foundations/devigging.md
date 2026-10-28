---
type: explanation
title: Devigging and vig-inclusive prices
description: What vig and proportional devigging are here, which consumers devig and which stay vig-inclusive, and what that means for comparing prices.
sources:
  - odds_core.odds_math.devig_probabilities
  - odds_core.odds_math.decimal_to_implied_probability
  - odds_core.models.Odds.implied_probability
  - odds_analytics.utils.per_book_market_holds
  - odds_analytics.retail_ev
  - odds_analytics.feature_groups._compute_target
  - odds_mcp.server.find_retail_edges
status: stable
---

# Devigging and vig-inclusive prices

Bookmaker prices sum to more than 1 once converted to implied probability (`odds_core.odds_math.decimal_to_implied_probability` on the stored decimal price, exposed as `odds_core.models.Odds.implied_probability`); the excess is the vig (hold). Devigging rescales one bookmaker's outcome probabilities for one market so they sum to 1, giving a "fair" estimate. The helper functions are on [Odds arithmetic helpers](odds-math.md).

## What exists

Only proportional (multiplicative) devigging exists: `odds_core.odds_math.devig_probabilities` divides each implied probability by their sum, for 2-way and 3-way markets. No power, Shin or additive method is implemented. Devigging is per bookmaker, with no cross-book consensus.

The only consumer is the `retail_ev` target, and the [retail-EV model rule](../evaluation/candidate-rules.md#the-retail-ev-model-rule) that trains on it. `odds_analytics.retail_ev` devigs every outcome of the exchange (with `odds_core.exchange_price.devig`, which `odds_core.exchange_price.exchange_price` also applies to the Betfair Exchange backs on one line of a retail snapshot, the over/under pair of a totals line, for the [candidate rules' market state](../evaluation/candidate-rules.md#market-state)), Polymarket and book prices, proportionally and per snapshot, for the events that have kicked off and have a close (a fully priced live Betfair snapshot or FDUK), reading their price rows through `odds_core.quotes.quotes` (`odds_analytics.retail_ev.load_retail_ev_dataset`). `odds_analytics.feature_groups.prepare_training_data`, whose `raw` target compares a decision snapshot, sampled only of the sources whose close has a sharp book, with its own source's close, hands `retail_ev` configs on to `odds_analytics.retail_ev.prepare_retail_ev_training_data`, so they never reach `odds_analytics.feature_groups._compute_target`, and `odds_analytics.training.config.FeatureConfig.retail_ev_arm` does not change how they devig ([CLV target](../modelling/clv-target.md#the-retail-ev-target), [target functions](../modelling/clv-target-functions.md#retail-ev-rows)).

## Vig-inclusive consumers

These use raw implied probabilities (1 over a price row's decimal price) and are not devigged:

- `target_type="raw"`: mean implied-probability delta across books, snapshot to close (`odds_analytics.feature_groups._compute_target`).
- `odds_mcp.server.find_retail_edges`: `divergence = retail_implied_prob - sharp_implied_prob`, both vig-inclusive and taken from back prices at the row's own line: the sharp probability comes from a book in the sharp list, and only the rows are retail books (the sharp list, Pinnacle and Polymarket are never rows), so a sharp book's own margin is part of every divergence ([retail edges](../agent/mcp-market-tools.md#retail-edges)).
- The `sharp_prob` and `retail_sharp_diff` model features ([Sharp benchmark](../modelling/sharp-benchmark.md)).
- Hold measurement (`odds_analytics.utils.per_book_market_holds`, over decimal prices) measures the vig rather than removing it.

Comparing a vig-inclusive probability with a devigged one mixes two scales; the vig-inclusive consumers above are the places where that can happen `(inferred)`.

## Related

- [Odds arithmetic helpers](odds-math.md)
- [Look-ahead bias](../modelling/look-ahead-bias.md)
