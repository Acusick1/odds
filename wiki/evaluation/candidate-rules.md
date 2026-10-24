---
type: reference
title: Candidate rules
description: The market state a mechanical rule reads at decision time, the bets it returns, the rule registry, decision times, params validation, source pinning and the exchange-gap rule.
sources:
  - odds_lambda.candidate_rules
  - odds_core.as_of
mentions:
  - odds_core.betfair_close.ExchangePrice
  - odds_core.bookmakers.UK_RETAIL_BOOKMAKERS
  - odds_core.candidates.MechanicalCandidate
status: stable
---

# Candidate rules

A mechanical [candidate](candidate-registry.md)'s `rule` names a function in the rule registry. The live runner and the replay backtest are meant to call that one function, so a backtest and a forward test of one version run the same code ([goal and strategy](../foundations/goal-and-strategy.md#what-exists-today) lists what exists). Neither exists yet.

## Market state

`odds_lambda.candidate_rules.MarketState` is a frozen dataclass: what was known about one event at its decision time. `odds_lambda.candidate_rules.build_market_state` builds it from the event and its snapshots, and ignores every snapshot stamped after the decision time. `odds_lambda.candidate_rules.load_market_state` fetches the snapshots with one query, for the event, stamped at or before the decision time and within `odds_lambda.candidate_rules.market_state.SNAPSHOT_LOOKBACK` of it, and calls the builder.

| Field | Meaning |
|---|---|
| `odds_lambda.candidate_rules.MarketState.event_id`, `odds_lambda.candidate_rules.MarketState.sport_key`, `odds_lambda.candidate_rules.MarketState.kickoff`, `odds_lambda.candidate_rules.MarketState.decision_time` | The event and the time of the state |
| `market` | `odds_lambda.candidate_rules.market_state.sport_market`: the market key of the sport's `odds_lambda.betfair.constants.SPORT_CONFIG` entry, `1x2` for EPL and `h2h` for MLB. Other sports raise `ValueError` ([market keys](../collection/sport-and-market-keys.md)) |
| `retail` | Per UK retail book (`odds_core.bookmakers.UK_RETAIL_BOOKMAKERS`), an `odds_lambda.candidate_rules.RetailQuote`: decimal prices per outcome and the snapshot time |
| `exchange` | The devigged Betfair Exchange price (`odds_core.betfair_close.ExchangePrice`, with its time and overround), or `None` |

As-of-t rules:

- **Retail.** Each book keeps the prices of the latest snapshot at or before t that quotes it, from the kinds in `odds_core.as_of.RETAIL_KINDS`. A book whose latest quote is older than `odds_lambda.candidate_rules.MAX_RETAIL_AGE` (2 hours) is absent. Book keys are canonical (`odds_core.bookmakers.canonical_bookmaker_key`), and books outside the roster are dropped.
- **Exchange.** The latest complete price from a live Betfair snapshot that is not in-play, or an FDUK opening snapshot (`odds_core.as_of.EXCHANGE_AT_T_KINDS`), skipping prices whose back overround exceeds `odds_core.betfair_close.MAX_EXCHANGE_OVERROUND`. It is `None` when that price is older than `odds_core.as_of.MAX_EXCHANGE_AGE` (2 hours), with no fall back to an earlier one. This is the exchange price at t of the retail-EV baseline ([CLV target functions](../modelling/clv-target-functions.md)), and `odds_analytics.retail_ev` shares the constants and `odds_core.as_of.latest_at_or_before` with it.
- **No Polymarket price.** The state carries none.

An FDUK opening snapshot is stamped at the latest documented collection time, so counting it at or before t is safe: the price was taken no later ([CLV target](../modelling/clv-target.md#the-retail-ev-target)).

## Rule interface

A rule is a function `(MarketState, params) -> list[BetIntent]`. `odds_lambda.candidate_rules.BetIntent` is a frozen dataclass of `outcome` (`odds_core.market_prices.Outcome`), `bookmaker`, `price_decimal` and `stake`.

`odds_lambda.candidate_rules.rules.RULES` maps each rule name to an `odds_lambda.candidate_rules.RuleSpec`: the function and its params model. A params model subclasses `odds_lambda.candidate_rules.RuleParams`, which is frozen, rejects unknown fields and requires `minutes_before_kickoff`, a positive integer.

## Decision times

`odds_lambda.candidate_rules.rules.decision_time` returns the kick-off minus the params' `minutes_before_kickoff`. The runner and the backtest both use it.

## Params validation

`odds_lambda.candidate_rules.validate_candidate_rules` checks every `odds_core.candidates.MechanicalCandidate`: the rule name is in the registry and the candidate's raw parameters validate against the rule's params model. It raises a `ValueError` naming the candidate otherwise. `odds candidates list` and `odds candidates report` call it after loading the registry, so an invalid rule fails there with exit code 1, and a unit test runs it over the `candidates/` directory.

## Source pinning

`odds_lambda.candidate_rules.RULE_SOURCE_HASHES` holds the sha256 of each rule function's source (`odds_lambda.candidate_rules.rule_source_hash`). `odds_lambda.candidate_rules.changed_pinned_rules` lists the rules whose source differs from, and a unit test fails when it lists a rule that a candidate in `candidates/` references. A changed rule needs a new rule name, and a new candidate version points at it. A rule no candidate references is not checked, so its pinned hash is updated with its source until a candidate references it.

## Rules

## The exchange-gap rule

Params `odds_lambda.candidate_rules.ExchangeGapParams`: `threshold` (> 0), `minutes_before_kickoff` and `stake` (> 0, default 1.0).

`odds_lambda.candidate_rules.rules.exchange_gap` returns no bets when the state has no exchange price. For each outcome it takes the longest decimal price among the retail books (ties go to the alphabetically first book key) and bets that book at that price when `exchange.fair[outcome] × price − 1` is greater than `threshold`. It places at most one bet per outcome, and each outcome is decided on its own.
