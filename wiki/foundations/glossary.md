---
type: reference
title: Glossary
description: The definition of each domain term, the words to avoid for it, and the page that describes the machinery behind it.
sources: []
status: stable
---

# Glossary

This page defines the project's domain terms, and other pages link here instead of defining them again. Each term lists the words to avoid in prose; identifiers in code keep their own names. The linked page under each term describes the machinery, not the term.

## Leagues, events and prices

### Sport

Football, baseball or basketball. A sport's leagues share its markets.

### League

One competition a candidate bets in or a collector covers, such as the EPL, La Liga or MLB. The Odds API names each with a sport key ([sport and market keys](../collection/sport-and-market-keys.md#sport-keys)).
_Avoid_: sport (for a league), competition

### Game

A real-world contest between two teams, as a source lists it. Each game resolves to an event.
_Avoid_: match, fixture, listing

### Event

The stored record of one game, which bets, briefs and decisions refer to. Several events stored for one game are twins, and are merged ([events table](../storage/events.md), [event merge](../storage/event-merge.md)).
_Avoid_: match, fixture, game (for the stored record)

### Kick-off

The scheduled start of an event ([kick-off authority](../storage/kickoff-authority.md)).
_Avoid_: kickoff, commence time, start time

### Market

A set of outcomes of an event priced together, such as the three-way result, the two-way result or totals ([sport and market keys](../collection/sport-and-market-keys.md#market-key-dialects)).

### Selection

One outcome of a market on one line, or the outcome alone in a market without lines.
_Avoid_: pick, outcome (for an outcome on a line)

### Book

A bookmaker, exchange or prediction market that prices outcomes, such as bet365, Betfair Exchange or Polymarket.
_Avoid_: bookmaker, bookie

### Retail book

A UK sportsbook the user can bet at. Exchanges, prediction markets and Pinnacle are not retail books ([sharp references](../modelling/sharp-benchmark.md#priority-semantics)).
_Avoid_: soft book, bettable book

### Source

A feed the pipeline collects prices or results from, such as The Odds API, OddsPortal, Betfair or football-data.co.uk (FDUK) ([data sources](../collection/data-sources.md)).
_Avoid_: provider

### Snapshot

One source's prices for one market of one event at one time ([how odds are stored](../storage/odds-storage-model.md)).

### Price

One book's decimal odds for one selection in a snapshot ([price tables](../storage/price-tables.md)).
_Avoid_: quote, odds (for a single price)

### Fetch tier

How long before kick-off a snapshot's prices held, from opening to in-play, derived from its time, time basis and source, and never stored ([fetch tier](../storage/fetch-tier.md)).
_Avoid_: sharp (for the tier 12 to 24 hours out; say the day-before tier)

## Value and scoring

### Vig

The amount by which one book's implied probabilities over a market sum to more than 1 ([devigging](devigging.md)).
_Avoid_: hold, margin, overround, juice

### Devig

To rescale one book's implied probabilities over a market so they sum to 1 ([devigging](devigging.md)).

### Fair probability

A selection's devigged implied probability.
_Avoid_: true probability, no-vig probability

### Sharp reference

The book whose fair probability a given use treats as the truth. Scoring uses Betfair Exchange; the agent's tools and the tabular features use Pinnacle first ([sharp references](../modelling/sharp-benchmark.md#which-reference-is-used-for-what)).
_Avoid_: sharp benchmark, sharp book (for the reference a use reads)

### Close

The fair probability at kick-off that a bet is scored against, from Betfair Exchange where it exists. Its kinds are the live close, from the Betfair feed just before kick-off; the fallback close, FDUK's last Betfair Exchange price for football or Polymarket for NBA two-way markets; and the substitute close, The Odds API's Betfair price, in EPL three-way backtests only ([retail-EV target](../modelling/clv-target.md#the-retail-ev-target)).
_Avoid_: closing line, closing price (for the scoring close)

### Last snapshot

A source's latest closing-tier snapshot of a market before kick-off, which the raw training target measures to ([target functions](../modelling/clv-target-functions.md#last-snapshot-of-a-raw-target)).
_Avoid_: close (for this snapshot)

### Retail EV

The expected value of a bet at a retail price against the close: the close's fair probability times the decimal price, minus 1. It scores every candidate in a league with a close ([retail-EV target](../modelling/clv-target.md#the-retail-ev-target)).
_Avoid_: CLV, closing line value, EV at the price taken

### Look-ahead bias

Using data stamped after the decision time it is meant to inform ([look-ahead bias](../modelling/look-ahead-bias.md)).

## Candidates

### Idea

A proposed way to profit, held as a GitHub issue labelled `candidate` until it is registered or killed ([take an idea to a verdict](../evaluation/howto-take-an-idea-to-a-verdict.md)).

### Candidate

A registered way of deciding bets, judged by its forward test. It lists the leagues it bets in, and a candidate listing several is judged on one series of bets ([goal and strategy](goal-and-strategy.md#candidates)).
_Avoid_: strategy, system

### Rule candidate

A candidate whose bets a rule decides ([candidate registry](../evaluation/candidate-registry.md)).
_Avoid_: mechanical candidate

### Agent candidate

A candidate whose bets an LLM betting agent decides, defined by its prompts and model ([betting agent](../agent/betting-agent.md)).

### Rule

The function a rule candidate runs on the market state at each decision time, returning the bets to place. A model rule is a rule that wraps a trained model ([candidate rules](../evaluation/candidate-rules.md#rule-interface)).
_Avoid_: rule family, family (for the module holding a rule; say rule module)

### Version

A candidate's whole-number version. Anything that could change its bets is a new version: its rule, parameters, model, prompts, leagues or book set, or a fix to prices its earlier decisions saw ([candidate registry](../evaluation/candidate-registry.md)).

### Book set

A named set of retail books a rule candidate prices from. The `uk_retail` set is every retail book ([candidate rules](../evaluation/candidate-rules.md#book-set)).
_Avoid_: roster, UK retail books (for a set)

### Decision time

The instant a bet or prediction is taken to be made. A candidate's are fixed offsets before kick-off ([candidate rules](../evaluation/candidate-rules.md#decision-times)).
_Avoid_: as-of instant

### Market state

Everything known about one event's market at a decision time: the retail prices and the sharp prices a rule reads ([candidate rules](../evaluation/candidate-rules.md#market-state)).

### Decision

One rule candidate version's record at one event and decision time, with the inputs it read and the bets it placed, including a decision that placed none ([run-candidates job](../evaluation/run-candidates-job.md#the-decision-record)).

### Bet

A bet placed by a candidate or a betting agent, recorded without real money ([bets](../agent/paper-trading.md)).
_Avoid_: paper trade, paper bet, trade

### Cell

One combination of parameter values from an idea's pre-registered grid ([cell selection](../evaluation/cell-selection.md)).

### Backtest

A replay of a rule over stored data, scored the way a forward test is ([candidate backtest](../evaluation/candidate-backtest.md)).
_Avoid_: simulation

### Development data

Stored history that ideas are generated and killed on. It never promotes a candidate ([goal and strategy](goal-and-strategy.md#how-a-candidate-is-judged)).
_Avoid_: dev data, training data (for this set)

### Sealed holdout

Seasons kept out of development data that an idea's selected cell may be backtested on once per version ([candidate backtest](../evaluation/candidate-backtest.md#sealed-holdout)).
_Avoid_: test set, holdout (alone)

### Forward test

A candidate's live paper betting from its start date at a fixed version, the only evidence that promotes it ([goal and strategy](goal-and-strategy.md#how-a-candidate-is-judged)).
_Avoid_: live test, trial, paper-trading test

### Checkpoint

A fixed count of a candidate's first bets with a close, the only points at which it is judged ([goal and strategy](goal-and-strategy.md#how-a-candidate-is-judged)).

### Verdict

The judgement at a checkpoint: promote, kill, continue or not yet. The first promote or kill stands ([candidate registry](../evaluation/candidate-registry.md#odds-candidates-report)).

### Family

The versions sharing a candidate's rule and market whose leagues overlap its own. The more of them have decided, the stricter the bar to promote ([goal and strategy](goal-and-strategy.md#how-a-candidate-is-judged)).

## Betting agent

### Brief

The agent's note on one event at one wake-up, written once and never updated ([match briefs](../agent/match-briefs.md)).
_Avoid_: match brief

### Wake-up

One run of a betting agent. The agent may book its next one ([agent run job](../agent/agent-run-job.md#wake-up-requests)).
_Avoid_: session, run (for an agent run)
