---
type: reference
title: Candidate rules
description: The market state a mechanical rule reads at decision time, the bets it returns, the rule registry, decision times and the once-per-outcome rule, maximum input age, params validation, rule pinning, the exchange-gap rule and the retail-EV model rule.
sources:
  - odds_lambda.candidate_rules
  - odds_core.as_of
  - odds_analytics.candidate_model
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

- **One co-timed snapshot.** The input is the latest snapshot at or before t of a kind in `odds_core.as_of.RETAIL_KINDS` that quotes the state's market by at least one UK retail book and by the Betfair Exchange (`odds_core.betfair_close.EXCHANGE_BOOKMAKER`, after `odds_core.bookmakers.canonical_bookmaker_key`, so the Odds API's `betfair_ex_uk` counts). The retail prices and the exchange price a rule compares were therefore taken together (#682). Live OddsPortal snapshots carry both, and so do live Odds API snapshots that quote the exchange; a snapshot of another market (OddsPortal writes one per market per scrape), one without an exchange row or one without a UK retail book is skipped for an earlier one. Live Betfair snapshots are not a retail kind, and FDUK opening snapshots (`odds_core.as_of.UPPER_BOUND_TIME_KINDS`) are excluded because their stamp is only an upper bound on when the prices were taken ([CLV target](../modelling/clv-target.md#the-retail-ev-target)).
- **Maximum input age.** When that snapshot is older than the rule's maximum input age at t ([params](#rule-interface)), or there is none, the state has no inputs: no retail and no exchange price. There is no fall back to an earlier snapshot.
- **Retail.** The snapshot's books in the UK retail roster. Book keys are canonical (`odds_core.bookmakers.canonical_bookmaker_key`), and books outside the roster are dropped.
- **Exchange.** The snapshot's own Betfair Exchange row through `odds_core.betfair_close.exchange_price`, devigged proportionally. It is `None` when the row does not price every outcome or its back overround exceeds `odds_core.betfair_close.MAX_EXCHANGE_OVERROUND` ([ADR-0014](../../docs/adr/0014-ignore-wide-exchange-prices.md)). The close the report scores against is still the live Betfair API price ([ADR-0012](../../docs/adr/0012-retail-ev-against-betfair-exchange-close.md)).

The state does not match the rows of `odds_analytics.retail_ev.build_event_rows`, which take the exchange price at t from live Betfair snapshots, separately from the retail snapshot ([target functions](../modelling/clv-target-functions.md#retail-ev-rows)). The [retail-EV model rule](#the-retail-ev-model-rule) therefore trains on rows built from market states, not on those rows.

### known_at

`odds_lambda.candidate_rules.market_state.load_market_state` takes an optional `known_at`. When given, it also drops snapshots whose `created_at` is later than `known_at`: the data the system had stored by then. The [run-candidates job](run-candidates-job.md#decisions) passes its run time and the parity check's data comparison passes the decision's recorded `known_at`, so neither sees a snapshot stamped before t but stored afterwards. The replay backtest passes `None`, because backfilled rows were created long after their snapshot times.

## Rule interface

A rule is a function `(MarketState, params) -> list[BetIntent]`. `odds_lambda.candidate_rules.BetIntent` is a frozen dataclass of `outcome` (`odds_core.market_prices.Outcome`), `bookmaker`, `price_decimal` and `stake`.

`odds_lambda.candidate_rules.rules.RULES` maps each rule name to an `odds_lambda.candidate_rules.RuleSpec`, the function and its params model, or to an `odds_lambda.candidate_rules.ModelRuleSpec` for a [model rule](#model-rules). A params model subclasses `odds_lambda.candidate_rules.RuleParams`, which is frozen, rejects unknown fields and requires `minutes_before_kickoff`, a non-empty list of distinct positive integers ([decision times](#decision-times)), and `max_input_age_minutes`, a positive integer: the oldest snapshot the state may be built from (`odds_lambda.candidate_rules.max_input_age`). Registered versions use 10 minutes, which the [OddsPortal runs before each decision](../collection/fetch-oddsportal-job.md#runs-before-candidate-decisions) meet. A draft backtest may widen it to screen an idea on older data, and its output names the age it used ([coverage](candidate-backtest.md#coverage)).

## Running a rule

`odds_lambda.candidate_rules.rules.run_rule` loads the state of an event at a time, within the params' maximum input age, as known at a `known_at`, and returns the state and the rule's bets through `odds_lambda.candidate_rules.apply_rule`. Its optional `kickoff` replaces the event's kick-off in the state, for a replay of a decision that saw an earlier one. Its optional `bet_outcomes` are the outcomes the candidate version already bet on the match at an earlier decision time; the rule's bets on them are dropped ([once per outcome](#once-per-outcome)). A state without inputs gives no bets, whatever the rule, and the [run-candidates job](run-candidates-job.md#decisions) records such a decision as missed.

## Decision times

A candidate decides each match at every time in its `minutes_before_kickoff` list, for example `[60]` for one decision an hour before kick-off or `[60, 30]` for two. The params model holds the list earliest decision first (largest number of minutes first). `odds_lambda.candidate_rules.rules.decision_times` returns the kick-off minus each entry, earliest first. The [run-candidates job](run-candidates-job.md#decisions), the [replay backtest](candidate-backtest.md#replay) and the [parity check](candidate-parity.md#replay-of-a-decision) decide each time as its own decision. `odds_lambda.candidate_rules.decision_leads` lists every entry of the `minutes_before_kickoff` of a sport's active mechanical candidates whose params resolve, for the [fetch-oddsportal job](../collection/fetch-oddsportal-job.md#runs-before-candidate-decisions) to scrape before each decision.

### Once per outcome

A candidate version bets each outcome of a match at most once, at the first of its decision times where the rule bets it. A later time may bet a different outcome of the same match. The live runner, the replay backtest and the parity check all pass the outcomes already bet on the match as `bet_outcomes` to `odds_lambda.candidate_rules.apply_rule`, which drops the rule's bets on them, and decide a match's times earliest first. Repeat bets on one outcome were rejected because they double exposure on a persistent gap and count near-duplicate bets toward a checkpoint. A version's verdict covers its set of decision times, not any one of them ([#656](https://github.com/Acusick1/odds/issues/656)).

## Params validation

`odds_lambda.candidate_rules.resolve_rule` checks an `odds_core.candidates.MechanicalCandidate`: the sport has a market (`odds_lambda.candidate_rules.market_state.sport_market`), the rule name is in the registry and the candidate's raw parameters validate against the rule's params model. It returns the rule and its validated params, and raises a `ValueError` naming the candidate otherwise. A draft backtest uses it alone. `odds_lambda.candidate_rules.validate_candidate_rules` does that for every mechanical candidate and also checks that the recorded `odds_core.candidates.MechanicalCandidate.rule_sha256` is current ([rule pinning](#rule-pinning)). `odds candidates list` and `odds candidates report` call it after loading the registry, so an invalid rule fails there with exit code 1, and a unit test runs it over the `candidates/` directory.

## Rule pinning

A `odds_core.candidates.MechanicalCandidate` records `odds_core.candidates.MechanicalCandidate.rule_sha256`, 64 lowercase hex characters. `odds_lambda.candidate_rules.rules.rule_sha256` computes the current value: the sha256, in a fixed order, of `inspect.getsource` for the rule function, its params model class, `odds_lambda.candidate_rules.RuleParams`, `odds_lambda.candidate_rules.apply_rule` (the [once-per-outcome](#once-per-outcome) filter), the `odds_lambda.candidate_rules.market_state` module, the `odds_core.as_of` module and the functions outside the package that pick a state's prices (`odds_lambda.candidate_rules.rules.RULE_INPUT_FUNCTIONS`: `odds_core.betfair_close.exchange_price` and the helpers it calls, `odds_core.market_prices.book_prices` and `odds_core.bookmakers.canonical_bookmaker_key`), then of the values in `odds_lambda.candidate_rules.rules.rule_input_constants`: `odds_core.bookmakers.UK_RETAIL_BOOKMAKERS`, the bookmaker alias maps, `odds_core.betfair_close.EXCHANGE_BOOKMAKER` and `odds_core.betfair_close.MAX_EXCHANGE_OVERROUND`. A [model rule](#model-rules) then adds the source of each module in its `model_modules`. Code those functions call in turn, such as the devig arithmetic in `odds_core.odds_math`, is not hashed. `odds_lambda.candidate_rules.validate_candidate_rules` fails a candidate whose recorded value differs, saying the rule's code changed and that a new candidate version must record the new hash. Changing the hash in the file is a change to the YAML, so [the version check](candidate-registry.md#version-check) applies to it. `odds candidates rule-hash <rule>` prints the current hash.

## Rules

### The exchange-gap rule

Params `odds_lambda.candidate_rules.ExchangeGapParams`: `threshold` (> 0), `minutes_before_kickoff` (the list of decision times), `max_input_age_minutes` and `stake` (> 0, default 1.0).

`odds_lambda.candidate_rules.rules.exchange_gap` returns no bets when the state has no exchange price. For each outcome it takes the longest decimal price among the retail books (ties go to the alphabetically first book key) and bets that book at that price when `exchange.fair[outcome] × price − 1` is greater than `threshold`. It places at most one bet per outcome in a decision, and each outcome is decided on its own.

## Model rules

An `odds_lambda.candidate_rules.ModelRuleSpec` is a rule whose function also takes a predictor (`odds_lambda.candidate_rules.EvPredictor`): the predicted EV of each UK retail book and outcome quoted in a state. `odds_lambda.candidate_rules.ModelRuleSpec.bind` fixes a fitted predictor into an ordinary `odds_lambda.candidate_rules.RuleSpec`, so a bound model rule decides through `odds_lambda.candidate_rules.apply_rule` like any other rule, with the same once-per-outcome filter.

Only the [replay backtest](candidate-backtest.md#model-rules) fits and binds a model. The [run-candidates job](run-candidates-job.md#one-run) skips a model-rule candidate with a `candidate_invalid` alert, and the [parity check](candidate-parity.md#command) refuses one. Pinning a fitted model for a live forward test does not exist yet: it waits on the model passing its re-test ([#657](https://github.com/Acusick1/odds/issues/657)).

`model_modules` names the modules that build, tune, fit and query the model, including `odds_core.betfair_close`, which defines the close every training target is measured against. `odds_lambda` does not depend on `odds_analytics`, so `odds_lambda.candidate_rules.rules.rule_sha256` imports them by name and hashes their source ([rule pinning](#rule-pinning)).

### The retail-EV model rule

Params `odds_lambda.candidate_rules.RetailEvModelParams`: `model` (`retail_ev_price_only`), `threshold` (> 0), `training_matches` and `refit_every_matches` (the [replay's](candidate-backtest.md#model-rules) training-only prefix and refit cadence), `tuning_trials`, `tuning_seed`, `tuning_min_train_matches` and `tuning_step_matches` (the hyperparameter search on that prefix, below; the minimum must leave a training match to validate), `minutes_before_kickoff`, `max_input_age_minutes` and `stake` (> 0, default 1.0). Every count is a positive integer. The tuning settings are params, so the YAML version check covers them.

`odds_lambda.candidate_rules.rules.retail_ev_model` returns no bets when the state has no exchange price. For each outcome it takes the longest decimal price among the retail books (ties go to the alphabetically first book key), as [the exchange-gap rule](#the-exchange-gap-rule) does, and bets that book at that price when the predicted EV of that quote is greater than `threshold`. The two rules differ only in the EV estimate: the exchange gap's `exchange.fair[outcome] × price − 1` against the model's prediction.

`odds_analytics.candidate_model` holds the model. The rule's `model_modules` are it, `odds_analytics.retail_ev`, `odds_analytics.xgboost_line_movement`, `odds_core.betfair_close`, `odds_analytics.retail_ev_metrics` (the tuning objective), `odds_analytics.training.tuner` (the search-space draws) and `odds_analytics.training.config`. The installed XGBoost, Optuna and NumPy versions are not hashed.

- **Model.** `retail_ev_price_only` is the residual arm on the price features of `odds_analytics.retail_ev.RetailEvFeatures`, with no optional groups, fitted by `odds_analytics.candidate_model.fit_model` through `odds_analytics.retail_ev.fit_retail_ev` with the target and loss weights of the [retail-EV target](../modelling/clv-target.md#the-retail-ev-target). XGBoost's `random_state` is `tuning_seed`.
- **Hyperparameters.** `odds_analytics.candidate_model.choose_hyperparameters` runs `tuning_trials` Optuna TPE trials, its sampler seeded with `tuning_seed`, over `odds_analytics.candidate_model.SEARCH_SPACES`: the bounds of the price-only retail-EV experiment's space (`experiments/configs/xgboost_epl_retail_ev_price_only.yaml`), with `gamma` and `reg_alpha` locked at zero in `odds_analytics.candidate_model.FIXED_XGBOOST_PARAMS`. There is no early stopping, because a refit has no validation set to stop on. A trial's score is its mean fold loss-weighted MSE, the objective the [retail-EV evaluation](retail-ev-evaluation.md) was tuned on, over the folds of `odds_analytics.candidate_model.tuning_folds`: expanding walk-forward over the training matches in kick-off order, each validating `tuning_step_matches` matches after the first `tuning_min_train_matches`, trained on the matches that kicked off at least `odds_analytics.retail_ev.DECISION_WINDOW` before the fold's first validation match, so no training close falls after a validation row. A fold without training or validation rows is skipped, and with none left `odds_analytics.candidate_model.choose_hyperparameters` raises `ValueError`. The best trial's hyperparameters then hold for every refit.
- **Inputs.** `odds_analytics.candidate_model.state_inputs` turns a state's quotes into `odds_analytics.retail_ev.RetailEvInput` with `odds_analytics.retail_ev.quote_inputs`, stamped at the state's snapshot time, with its own exchange price and no Polymarket price or dynamics. A state without an exchange price has none. `odds_analytics.candidate_model.RetailEvPredictor` predicts every input.
- **Training rows.** `odds_analytics.candidate_model.training_rows` builds, for each snapshot of a match from `odds_analytics.retail_ev.DECISION_WINDOW` to `odds_analytics.retail_ev.DECISION_CUTOFF` before kick-off, the state `odds_lambda.candidate_rules.build_market_state` gives at that snapshot's own time, and takes its inputs against the match's close (`odds_core.betfair_close.event_close`) when that is the live Betfair close. Only co-timed snapshots give rows, the exchange price is always the row's own snapshot's, and a training row is the input the rule would have predicted from had it decided at that time. A match closed only by FDUK has no rows: FDUK publishes its closing prices days after the match, so a refit at the next matchday would not have the label. A match without a close has none either.
