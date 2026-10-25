---
type: reference
title: Candidate rules
description: The market state a mechanical rule reads at decision time, the bets it returns, the rule registry, decision times, params validation, rule pinning and the exchange-gap rule.
sources:
  - odds_lambda.candidate_rules
  - odds_core.as_of
mentions:
  - odds_core.betfair_close.ExchangePrice
  - odds_core.bookmakers.UK_RETAIL_BOOKMAKERS
  - odds_core.candidates.MechanicalCandidate
  - odds_core.as_of.UPPER_BOUND_TIME_KINDS
status: stable
---

# Candidate rules

A mechanical [candidate](candidate-registry.md)'s `rule` names a function in the rule registry. The [run-candidates job](run-candidates-job.md), the [replay backtest](candidate-backtest.md) and the [parity check](candidate-parity.md) decide through `odds_lambda.candidate_rules.rules.run_rule`, so a backtest and a forward test of one version run the same code.

A mechanical candidate records the hash of its rule's code, so the YAML version check forces a new version when that code changes ([rule pinning](#rule-pinning)).

## Market state

`odds_lambda.candidate_rules.MarketState` is a frozen dataclass: what was known about one event at its decision time t. `odds_lambda.candidate_rules.build_market_state` builds it from the event and its snapshots, and ignores every snapshot stamped after t. `odds_lambda.candidate_rules.load_market_state` fetches the snapshots with one query, for the event, stamped at or before t and within `odds_lambda.candidate_rules.market_state.SNAPSHOT_LOOKBACK` of it, and calls the builder.

The state holds only sources a live run sees at t, at the exact time they were taken. It does not claim to match the baseline choice of `odds_analytics.retail_ev.build_event_rows`: that function takes each retail book's latest quote separately, and lets an FDUK opening row supply its own exchange price, while the state takes retail by whole snapshot and has no FDUK opening data.

| Field | Meaning |
|---|---|
| `odds_lambda.candidate_rules.MarketState.event_id`, `odds_lambda.candidate_rules.MarketState.sport_key`, `odds_lambda.candidate_rules.MarketState.kickoff`, `odds_lambda.candidate_rules.MarketState.decision_time` | The event and the time of the state |
| `market` | `odds_lambda.candidate_rules.market_state.sport_market`: the market key of the sport's `odds_lambda.betfair.constants.SPORT_CONFIG` entry, `1x2` for EPL and `h2h` for MLB. Other sports raise `ValueError` ([market keys](../collection/sport-and-market-keys.md)) |
| `retail` | Per UK retail book (`odds_core.bookmakers.UK_RETAIL_BOOKMAKERS`), an `odds_lambda.candidate_rules.RetailQuote`: decimal prices per outcome and the snapshot time |
| `exchange` | The devigged Betfair Exchange price (`odds_core.betfair_close.ExchangePrice`, with its time and overround), or `None` |

As-of-t rules:

- **Retail, by whole snapshot.** For each kind in `odds_core.as_of.RETAIL_KINDS`, only its latest snapshot at or before t that quotes the state's market (by any book) counts (OddsPortal writes one snapshot per market per scrape, so a later `totals` snapshot does not hide the `1x2` one), and only when it is at most `odds_lambda.candidate_rules.MAX_RETAIL_AGE` (2 hours) old, the slowest OddsPortal cadence at 24 hours out (`odds_lambda.jobs.fetch_oddsportal.CADENCE`). A book absent from that snapshot is absent from the state, however recently an earlier snapshot quoted it. When two kinds both have a fresh latest snapshot and quote one book, the more recent snapshot's price is used. Book keys are canonical (`odds_core.bookmakers.canonical_bookmaker_key`), and books outside the roster are dropped.
- **Exchange.** The latest complete price from a live Betfair snapshot that is not in-play, skipping prices whose back overround exceeds `odds_core.betfair_close.MAX_EXCHANGE_OVERROUND`. It is `None` when that price is older than `odds_core.as_of.MAX_EXCHANGE_AGE` (2 hours), with no fall back to an earlier one. `odds_analytics.retail_ev` shares the constants and `odds_core.as_of.latest_at_or_before` with it.
- **No FDUK opening data.** An FDUK opening snapshot is stamped at FDUK's latest documented collection time, an upper bound on when its prices were taken ([CLV target](../modelling/clv-target.md#the-retail-ev-target)), so the kinds in `odds_core.as_of.UPPER_BOUND_TIME_KINDS` supply neither retail quotes nor an exchange price.
- **No Polymarket price.** The state carries none.

### known_at

`odds_lambda.candidate_rules.market_state.load_market_state` takes an optional `known_at`. When given, it also drops snapshots whose `created_at` is later than `known_at`: the data the system had stored by then. The [run-candidates job](run-candidates-job.md#decisions) passes its run time and the parity check's data comparison passes the decision's recorded `known_at`, so neither sees a snapshot stamped before t but stored afterwards. The replay backtest passes `None`, because backfilled rows were created long after their snapshot times.

## Rule interface

A rule is a function `(MarketState, params) -> list[BetIntent]`. `odds_lambda.candidate_rules.BetIntent` is a frozen dataclass of `outcome` (`odds_core.market_prices.Outcome`), `bookmaker`, `price_decimal` and `stake`.

`odds_lambda.candidate_rules.rules.RULES` maps each rule name to an `odds_lambda.candidate_rules.RuleSpec`: the function and its params model. A params model subclasses `odds_lambda.candidate_rules.RuleParams`, which is frozen, rejects unknown fields and requires `minutes_before_kickoff`, a positive integer.

## Running a rule

`odds_lambda.candidate_rules.rules.run_rule` loads the state of an event at a time as known at a `known_at` and returns the state and the rule's bets. Its optional `kickoff` replaces the event's kick-off in the state, for a replay of a decision that saw an earlier one.

## Decision times

`odds_lambda.candidate_rules.rules.decision_time` returns the kick-off minus the params' `minutes_before_kickoff`. The [run-candidates job](run-candidates-job.md#decisions) and the [replay backtest](candidate-backtest.md#replay) both use it.

## Params validation

`odds_lambda.candidate_rules.resolve_rule` checks an `odds_core.candidates.MechanicalCandidate`: the sport has a market (`odds_lambda.candidate_rules.market_state.sport_market`), the rule name is in the registry and the candidate's raw parameters validate against the rule's params model. It returns the rule and its validated params, and raises a `ValueError` naming the candidate otherwise. A draft backtest uses it alone. `odds_lambda.candidate_rules.validate_candidate_rules` does that for every mechanical candidate and also checks that the recorded `odds_core.candidates.MechanicalCandidate.rule_sha256` is current ([rule pinning](#rule-pinning)). `odds candidates list` and `odds candidates report` call it after loading the registry, so an invalid rule fails there with exit code 1, and a unit test runs it over the `candidates/` directory.

## Rule pinning

A `odds_core.candidates.MechanicalCandidate` records `odds_core.candidates.MechanicalCandidate.rule_sha256`, 64 lowercase hex characters. `odds_lambda.candidate_rules.rules.rule_sha256` computes the current value: the sha256 of `inspect.getsource` for the rule function, its params model class, the `odds_lambda.candidate_rules.market_state` module and the `odds_core.as_of` module, in that order. `odds_lambda.candidate_rules.validate_candidate_rules` fails a candidate whose recorded value differs, saying the rule's code changed and that a new candidate version must record the new hash. Changing the hash in the file is a change to the YAML, so [the version check](candidate-registry.md#version-check) then requires a version bump. `odds candidates rule-hash <rule>` prints the current hash.

## Rules

### The exchange-gap rule

Params `odds_lambda.candidate_rules.ExchangeGapParams`: `threshold` (> 0), `minutes_before_kickoff` and `stake` (> 0, default 1.0).

`odds_lambda.candidate_rules.rules.exchange_gap` returns no bets when the state has no exchange price. For each outcome it takes the longest decimal price among the retail books (ties go to the alphabetically first book key) and bets that book at that price when `exchange.fair[outcome] × price − 1` is greater than `threshold`. It places at most one bet per outcome, and each outcome is decided on its own.
