---
type: reference
title: Candidate rules
description: The market state a mechanical rule reads at decision time, the bets it returns, the rule registry, decision times, maximum input age, params validation, rule pinning and the exchange-gap rule.
sources:
  - odds_lambda.candidate_rules
  - odds_core.as_of
mentions:
  - odds_core.betfair_close.ExchangePrice
  - odds_core.betfair_close.exchange_price
  - odds_core.bookmakers.UK_RETAIL_BOOKMAKERS
  - odds_core.candidates.MechanicalCandidate
status: stable
---

# Candidate rules

A mechanical [candidate](candidate-registry.md)'s `rule` names a function in the rule registry. The [run-candidates job](run-candidates-job.md), the [replay backtest](candidate-backtest.md) and the [parity check](candidate-parity.md) decide through `odds_lambda.candidate_rules.rules.run_rule`, so a backtest and a forward test of one version run the same code.

A mechanical candidate records the hash of its rule's code, so the YAML version check forces a new version when that code changes ([rule pinning](#rule-pinning)).

## Market state

`odds_lambda.candidate_rules.MarketState` is a frozen dataclass: what was known about one event at its decision time t, from one snapshot. `odds_lambda.candidate_rules.build_market_state` builds it from the event, its snapshots, t and a maximum input age, and ignores every snapshot stamped after t. `odds_lambda.candidate_rules.load_market_state` fetches the snapshots with one query, for the event, stamped at or before t and within the maximum input age of it, and calls the builder.

| Field | Meaning |
|---|---|
| `odds_lambda.candidate_rules.MarketState.event_id`, `odds_lambda.candidate_rules.MarketState.sport_key`, `odds_lambda.candidate_rules.MarketState.kickoff`, `odds_lambda.candidate_rules.MarketState.decision_time` | The event and the time of the state |
| `market` | `odds_lambda.candidate_rules.market_state.sport_market`: the market key of the sport's `odds_lambda.betfair.constants.SPORT_CONFIG` entry, `1x2` for EPL and `h2h` for MLB. Other sports raise `ValueError` ([market keys](../collection/sport-and-market-keys.md)) |
| `odds_lambda.candidate_rules.MarketState.snapshot_id`, `odds_lambda.candidate_rules.MarketState.snapshot_time` | The snapshot the state was built from, or `None` when it has no inputs (`odds_lambda.candidate_rules.MarketState.has_input` is false) |
| `retail` | Decimal prices per outcome for each UK retail book (`odds_core.bookmakers.UK_RETAIL_BOOKMAKERS`) in that snapshot |
| `exchange` | The devigged Betfair Exchange price in that snapshot (`odds_core.betfair_close.ExchangePrice`, with its time and overround), or `None` |

As-of-t rules:

- **One co-timed snapshot.** The input is the latest snapshot at or before t of a kind in `odds_core.as_of.RETAIL_KINDS` that quotes the state's market by at least one UK retail book and by the Betfair Exchange (`odds_core.betfair_close.EXCHANGE_BOOKMAKER`, after `odds_core.bookmakers.canonical_bookmaker_key`, so the Odds API's `betfair_ex_uk` counts). The retail prices and the exchange price a rule compares were therefore taken together (#667). Live OddsPortal snapshots carry both, and so do live Odds API snapshots that quote the exchange; a snapshot of another market (OddsPortal writes one per market per scrape), one without an exchange row or one without a UK retail book is skipped for an earlier one. Live Betfair snapshots are not a retail kind, and FDUK opening snapshots (`odds_core.as_of.UPPER_BOUND_TIME_KINDS`) are excluded because their stamp is only an upper bound on when the prices were taken ([CLV target](../modelling/clv-target.md#the-retail-ev-target)).
- **Maximum input age.** When that snapshot is older than the rule's maximum input age at t ([params](#rule-interface)), or there is none, the state has no inputs: no retail and no exchange price. There is no fall back to an earlier snapshot.
- **Retail.** The snapshot's books in the UK retail roster. Book keys are canonical (`odds_core.bookmakers.canonical_bookmaker_key`), and books outside the roster are dropped.
- **Exchange.** The snapshot's own Betfair Exchange row through `odds_core.betfair_close.exchange_price`, devigged proportionally. It is `None` when the row does not price every outcome or its back overround exceeds `odds_core.betfair_close.MAX_EXCHANGE_OVERROUND` ([ADR-0014](../../docs/adr/0014-ignore-wide-exchange-prices.md)). The close the report scores against is still the live Betfair API price ([ADR-0012](../../docs/adr/0012-retail-ev-against-betfair-exchange-close.md)).

The state does not match the rows of `odds_analytics.retail_ev.build_event_rows`, which take the exchange price at t from live Betfair snapshots, separately from the retail snapshot ([target functions](../modelling/clv-target-functions.md#retail-ev-rows)).

### known_at

`odds_lambda.candidate_rules.market_state.load_market_state` takes an optional `known_at`. When given, it also drops snapshots whose `created_at` is later than `known_at`: the data the system had stored by then. The [run-candidates job](run-candidates-job.md#decisions) passes its run time and the parity check's data comparison passes the decision's recorded `known_at`, so neither sees a snapshot stamped before t but stored afterwards. The replay backtest passes `None`, because backfilled rows were created long after their snapshot times.

## Rule interface

A rule is a function `(MarketState, params) -> list[BetIntent]`. `odds_lambda.candidate_rules.BetIntent` is a frozen dataclass of `outcome` (`odds_core.market_prices.Outcome`), `bookmaker`, `price_decimal` and `stake`.

`odds_lambda.candidate_rules.rules.RULES` maps each rule name to an `odds_lambda.candidate_rules.RuleSpec`: the function and its params model. A params model subclasses `odds_lambda.candidate_rules.RuleParams`, which is frozen, rejects unknown fields and requires two positive integers: `minutes_before_kickoff` ([decision times](#decision-times)) and `max_input_age_minutes`, the oldest snapshot the state may be built from (`odds_lambda.candidate_rules.max_input_age`). Registered versions use 10 minutes, which the [OddsPortal runs before each decision](../collection/fetch-oddsportal-job.md#runs-before-candidate-decisions) meet. A draft backtest may widen it to screen an idea on older data, and its output names the age it used ([coverage](candidate-backtest.md#coverage)).

## Running a rule

`odds_lambda.candidate_rules.rules.run_rule` loads the state of an event at a time, within the params' maximum input age, as known at a `known_at`, and returns the state and the rule's bets through `odds_lambda.candidate_rules.apply_rule`. Its optional `kickoff` replaces the event's kick-off in the state, for a replay of a decision that saw an earlier one. A state without inputs gives no bets, whatever the rule, and the [run-candidates job](run-candidates-job.md#decisions) records such a decision as missed.

## Decision times

`odds_lambda.candidate_rules.rules.decision_time` returns the kick-off minus the params' `minutes_before_kickoff`. The [run-candidates job](run-candidates-job.md#decisions) and the [replay backtest](candidate-backtest.md#replay) both use it. `odds_lambda.candidate_rules.decision_leads` lists the `minutes_before_kickoff` of a sport's active mechanical candidates whose params resolve, for the [fetch-oddsportal job](../collection/fetch-oddsportal-job.md#runs-before-candidate-decisions) to scrape before each decision.

## Params validation

`odds_lambda.candidate_rules.resolve_rule` checks an `odds_core.candidates.MechanicalCandidate`: the sport has a market (`odds_lambda.candidate_rules.market_state.sport_market`), the rule name is in the registry and the candidate's raw parameters validate against the rule's params model. It returns the rule and its validated params, and raises a `ValueError` naming the candidate otherwise. A draft backtest uses it alone. `odds_lambda.candidate_rules.validate_candidate_rules` does that for every mechanical candidate and also checks that the recorded `odds_core.candidates.MechanicalCandidate.rule_sha256` is current ([rule pinning](#rule-pinning)). `odds candidates list` and `odds candidates report` call it after loading the registry, so an invalid rule fails there with exit code 1, and a unit test runs it over the `candidates/` directory.

## Rule pinning

A `odds_core.candidates.MechanicalCandidate` records `odds_core.candidates.MechanicalCandidate.rule_sha256`, 64 lowercase hex characters. `odds_lambda.candidate_rules.rules.rule_sha256` computes the current value: the sha256 of `inspect.getsource` for the rule function, its params model class, `odds_lambda.candidate_rules.RuleParams`, the `odds_lambda.candidate_rules.market_state` module and the `odds_core.as_of` module, in that order. Other code they import, such as `odds_core.betfair_close.exchange_price`, is not hashed. `odds_lambda.candidate_rules.validate_candidate_rules` fails a candidate whose recorded value differs, saying the rule's code changed and that a new candidate version must record the new hash. Changing the hash in the file is a change to the YAML, so [the version check](candidate-registry.md#version-check) then requires a version bump. `odds candidates rule-hash <rule>` prints the current hash.

## Rules

### The exchange-gap rule

Params `odds_lambda.candidate_rules.ExchangeGapParams`: `threshold` (> 0), `minutes_before_kickoff`, `max_input_age_minutes` and `stake` (> 0, default 1.0).

`odds_lambda.candidate_rules.rules.exchange_gap` returns no bets when the state has no exchange price. For each outcome it takes the longest decimal price among the retail books (ties go to the alphabetically first book key) and bets that book at that price when `exchange.fair[outcome] × price − 1` is greater than `threshold`. It places at most one bet per outcome, and each outcome is decided on its own.
