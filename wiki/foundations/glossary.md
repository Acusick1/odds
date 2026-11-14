---
type: reference
title: Glossary
description: The project's domain terms, one line each, with the page that defines each one and the words to avoid for it.
sources: []
status: stable
---

# Glossary

One line per domain term, grouped by area. Each term links to the page that defines it, and that page is authoritative: when the two disagree, fix this line. Words under _Avoid_ are synonyms the wiki or code also uses; write the term instead. General programming terms are left out.

## Candidates and verdicts

**Candidate**: any rule that decides bets, whether a mechanical rule on prices, a model or an LLM agent ([goal and strategy](goal-and-strategy.md#candidates)).
_Avoid_: strategy, system

**Idea**: a GitHub issue labelled `candidate`, before it is registered as a candidate ([goal and strategy](goal-and-strategy.md#candidates)).

**Mechanical candidate**: a candidate whose rule, or the model a model rule binds, runs on the market state at its decision times ([candidate registry](../evaluation/candidate-registry.md)).

**Agent candidate**: a candidate whose bets an LLM betting agent places, defined by its prompts and model ([candidate registry](../evaluation/candidate-registry.md)).

**Model rule**: a mechanical candidate's rule that wraps a trained model ([candidate rules](../evaluation/candidate-rules.md#model-rules)).

**Candidate registry**: the YAML files in `candidates/`, one per candidate, holding its definition and its status: active, promoted or killed ([candidate registry](../evaluation/candidate-registry.md)).

**Version**: a candidate's whole-number version, bumped whenever its rule, parameters, model, book set or league set changes, or a collection fix changes the prices its earlier decisions saw ([goal and strategy](goal-and-strategy.md#candidates)).

**Rule**: the function a mechanical candidate runs on the market state at a decision time, returning the bets it wants ([candidate rules](../evaluation/candidate-rules.md#rule-interface)).

**Bet intent**: one bet a rule returns: outcome, line, bookmaker, price and stake ([candidate rules](../evaluation/candidate-rules.md#rule-interface)).

**Selection**: an outcome on a line, or the outcome alone in a market without lines ([candidate rules](../evaluation/candidate-rules.md#once-per-selection)).
_Avoid_: pick

**Book set**: the named group of roster books a candidate prices from, part of its definition ([candidate rules](../evaluation/candidate-rules.md#book-set)).

**UK retail roster**: the UK retail books the user bets with, the largest book set ([candidate rules](../evaluation/candidate-rules.md#book-set)).
_Avoid_: retail books (for the roster as a set), UK books

**Decision time**: the instant a bet or prediction is taken to be made; for a candidate, a fixed number of minutes before kick-off ([candidate rules](../evaluation/candidate-rules.md#decision-times), [look-ahead bias](../modelling/look-ahead-bias.md)).

**Market state**: what was known about one event's market at a decision time, built from one retail snapshot plus the sharp prices the rule reads ([candidate rules](../evaluation/candidate-rules.md#market-state)).

**Max input age**: the oldest a snapshot may be at the decision time and still be an input ([candidate rules](../evaluation/candidate-rules.md#market-state)).

**Lookback point**: the market state as of a declared number of minutes before the decision time ([candidate rules](../evaluation/candidate-rules.md#lookback-points)).

**Extra market**: a second market a rule reads beside its own, as read-only input ([candidate rules](../evaluation/candidate-rules.md#extra-markets)).

**Fixture context**: each team's previous league result and the number of matches kicking off together, as rule inputs ([candidate rules](../evaluation/candidate-rules.md#fixture-context)).

**Rule pinning**: the hash of the first-party modules a rule imports, recorded in the candidate file ([candidate rules](../evaluation/candidate-rules.md#rule-pinning)).

**Rehash**: a pinned-hash change accepted at the same version because replays show every bet unchanged ([candidate registry](../evaluation/candidate-registry.md#rehash-ledger)).

**Decision**: the record of one candidate version at one event and decision time, with its inputs and bets, including a missed decision that placed none ([run-candidates job](../evaluation/run-candidates-job.md#the-decision-record)).

**Forward test**: a candidate's live paper trading from its start date at a fixed version, the only evidence that promotes ([goal and strategy](goal-and-strategy.md#how-a-candidate-is-judged)).
_Avoid_: live test, trial

**Development data**: stored history used to generate and kill ideas, never to promote one ([goal and strategy](goal-and-strategy.md#how-a-candidate-is-judged)).

**Pre-registration**: the committed grid of parameter values, data range and selection rule, fixed before any backtest of an idea ([take an idea to a verdict](../evaluation/howto-take-an-idea-to-a-verdict.md)).

**Cell**: one parameter combination of a pre-registered grid ([cell selection](../evaluation/cell-selection.md)).

**Cell selection**: picking the one eligible cell of a grid from its development replays ([cell selection](../evaluation/cell-selection.md#score-and-pick)).

**Sealed holdout**: a season of the non-EPL leagues kept out of development data, replayable once per idea and version as a one-shot gate ([candidate backtest](../evaluation/candidate-backtest.md#sealed-holdout)).

**Pooled candidate**: a candidate betting in several leagues whose bets form one series and one verdict ([goal and strategy](goal-and-strategy.md#how-a-candidate-is-judged)).

**Replay backtest**: scoring a rule on stored data with the report's code ([candidate backtest](../evaluation/candidate-backtest.md)).
_Avoid_: simulation

**Draft**: a candidate definition outside the registry, replayed by a backtest and counted as one more version of its family ([candidate backtest](../evaluation/candidate-backtest.md#command)).

**Parity check**: rebuilding each live decision from its recorded inputs and comparing it with what the runner did ([parity check](../evaluation/candidate-parity.md)).

**Book replay**: an information-only replay of a version's recorded decisions under other book sets ([book replays](../evaluation/book-replays.md)).

**Checkpoint**: a fixed count of a candidate's first bets with a close, the only points at which it is scored ([goal and strategy](goal-and-strategy.md#how-a-candidate-is-judged)).

**Verdict**: the report's judgement of a candidate: promote, kill, continue (checkpoint reached, neither bar met) or not yet (no checkpoint reached); the first promote or kill stands for good ([candidate registry](../evaluation/candidate-registry.md#odds-candidates-report)).

**Family**: the versions sharing a candidate's rule and market whose league sets intersect its own; those that have decided set k in the promote bar ([goal and strategy](goal-and-strategy.md#how-a-candidate-is-judged)).
_Avoid_: family (alone, for a rule module)

**Promote bar**: the checkpoint test that the lower CI bound of mean retail EV, tightened for the family's size, is above zero ([goal and strategy](goal-and-strategy.md#how-a-candidate-is-judged)).

**Kill bar**: the checkpoint test that the upper CI bound of mean retail EV is at or below zero, or that a fixed bet count passed without promotion ([goal and strategy](goal-and-strategy.md#how-a-candidate-is-judged)).

## Prices and scoring

**Retail EV**: the EV of a bet at a retail price against the devigged Betfair Exchange close, the score of every candidate ([retail-EV target](../modelling/clv-target.md#the-retail-ev-target)).
_Avoid_: retail-EV (as a noun)

**CLV**: closing line value, how much better a price is than the fair price at the close; retail EV is the measure of it that scores candidates ([CLV and the training target](../modelling/clv-target.md)).

**Raw target**: the vig-inclusive move in implied probability or point from a decision snapshot to the close, the other training target ([CLV and the training target](../modelling/clv-target.md#the-raw-close-minus-snapshot-delta)).
_Avoid_: CLV target, line movement target

**Close**: the devigged Betfair Exchange price a bet is scored against: the live close, else the sport's fallback close. The raw target's close is a different thing, the last snapshot before kick-off ([retail-EV target](../modelling/clv-target.md#the-retail-ev-target)).
_Avoid_: closing line

**Live close**: the last fully priced, pre-play Betfair Exchange snapshot shortly before kick-off ([retail-EV target](../modelling/clv-target.md#the-retail-ev-target)).

**Fallback close**: the close a sport takes when an event has no live close: FDUK's Betfair closing price for football, Polymarket for NBA h2h ([retail-EV target](../modelling/clv-target.md#the-retail-ev-target)).

**Substitute close**: The Odds API's Betfair Exchange price, used as the close in EPL backtests only ([substitute close comparison](../evaluation/substitute-close-comparison.md)).

**Sharp reference**: the sharp price one use reads; scoring uses Betfair Exchange, the agent tools and tabular features a Pinnacle-first list ([sharp references](../modelling/sharp-benchmark.md#which-reference-is-used-for-what)).
_Avoid_: sharp benchmark

**Retail book**: a bookmaker that takes bets and is not a sharp reference; Pinnacle and Polymarket never are ([sharp references](../modelling/sharp-benchmark.md#priority-semantics)).
_Avoid_: soft book

**Vig**: the amount by which one book's implied probabilities over a market sum to more than 1 ([devigging](devigging.md)).
_Avoid_: margin, juice (in prose; code names keep hold and overround)

**Devig**: rescaling one book's implied probabilities so they sum to 1 ([devigging](devigging.md)).

**Fair probability**: a book's devigged implied probability for an outcome ([devigging](devigging.md)).

**Vig-inclusive**: an implied probability taken as 1/price without devigging ([devigging](devigging.md#vig-inclusive-consumers)).

**Retail-sharp divergence**: a retail book's vig-inclusive implied probability minus the sharp reference's ([sharp references](../modelling/sharp-benchmark.md#retail-sharp-divergence)).

**Look-ahead bias**: reading data stamped after the decision time ([look-ahead bias](../modelling/look-ahead-bias.md)).

**Paper trade**: a simulated bet, flat-stake for a candidate and bankroll-sized for an agent ([paper trading](../agent/paper-trading.md)).
_Avoid_: paper bet (in prose; the MCP tool keeps the name)

## Storage and collection

**Event**: one match, held as one row of the events table ([events table](../storage/events.md)).
_Avoid_: game, fixture (for a stored event)

**Kick-off**: the start time of an event ([kick-off authority](../storage/kickoff-authority.md)).
_Avoid_: kickoff, commence time, start (in prose; code names keep kickoff)

**Kick-off authority**: the per-sport ranked list of sources allowed to move an event's kick-off ([kick-off authority](../storage/kickoff-authority.md#authority-order)).

**Twin**: an event duplicating another event of the same match, written from a different source ([event merge](../storage/event-merge.md#detection)).
_Avoid_: duplicate event

**Kept event**: the event a twin is merged into ([event merge](../storage/event-merge.md#kept-event)).

**External id**: a source's own id for an event, stored beside the event's integer id ([events table](../storage/events.md#external-ids)).

**Event status**: where an event is in its lifecycle, from scheduled to final, cancelled, postponed or unmatched ([event status lifecycle](../collection/event-status-lifecycle.md)).

**Source**: the feed that wrote a snapshot, such as The Odds API, OddsPortal, Betfair or FDUK ([data sources](../collection/data-sources.md)).
_Avoid_: provider

**FDUK**: football-data.co.uk, the historical source of football opening and closing prices ([data sources](../collection/data-sources.md)).

**Snapshot**: one source's prices for one market of one event at one time ([odds storage model](../storage/odds-storage-model.md)).

**Envelope**: the Odds-API-shaped JSON a source returns, archived in a snapshot payload ([odds tables](../storage/odds-tables.md#envelope-format)).

**Snapshot payload**: the stored row archiving one source run's envelope, shared by the snapshots of its markets ([price tables](../storage/price-tables.md#snapshot_payloads)).

**Price row**: one quote of a snapshot: bookmaker, outcome, line, side and decimal price ([price tables](../storage/price-tables.md)).

**Side**: whether a price row is a back, lay, mid or traded price ([price tables](../storage/price-tables.md#closed-sets)).

**Time basis**: what a snapshot's time means: observed, source-stamped, an upper bound, or the close ([odds tables](../storage/odds-tables.md)).

**Known at**: a storage-time bound on a read, excluding snapshots stored after it ([candidate rules](../evaluation/candidate-rules.md#known_at-and-recorded-inputs)).

**Fetch tier**: a label for how long before kick-off a snapshot was taken, from opening to in-play, derived and never stored ([fetch tier](../storage/fetch-tier.md)).

**Market key**: the string naming a market, such as `1x2` for football's three-way market and `h2h` for a two-way one ([sport and market keys](../collection/sport-and-market-keys.md#market-key-dialects)).

**Sport key**: The Odds API key naming a league or sport, such as `soccer_epl` ([sport and market keys](../collection/sport-and-market-keys.md#sport-keys)).

**Quota**: the per-key monthly balance of The Odds API units ([Odds API quota](../collection/odds-api-quota.md)).

**Acquisition manifest**: a YAML file fixing what one historical Odds API purchase requests ([Odds API backfill](../collection/odds-api-backfill.md#acquisition-manifests)).

## Scheduling and operations

**Self-scheduling**: a job booking its own next run before it works, in place of a fixed cron ([self-scheduling](../scheduling/self-scheduling.md)).

**Cadence**: the hours between a collector's runs in each fetch tier ([decision engine](../scheduling/decision-engine.md#cadence)).

**Lead run**: a run at a fixed time before kick-off, whatever the cadence ([decision engine](../scheduling/decision-engine.md#lead-runs)).

**Overnight window**: the UTC hours in which a sport's runs are suppressed ([decision engine](../scheduling/decision-engine.md#overnight-windows)).

**Dormancy**: a collector's state with no fixture within its horizon, booking only a check-in ([decision engine](../scheduling/decision-engine.md#dormancy)).

**Heartbeat**: the record a job writes on a clean exit, which the health monitor checks ([health monitor](../operations/health-monitor.md#heartbeats)).

## Betting agent

**Match brief**: the agent's note on one match at one wake-up, written once and never updated ([match briefs](../agent/match-briefs.md)).

**Wake-up**: one agent run; the agent may request when the next one happens ([agent run job](../agent/agent-run-job.md#wake-up-requests)).

**Edge type**: one of the bases the agent names for a bet ([agent prompts](../agent/agent-prompts.md#edge-types)).

**Conviction tier**: the agent's stake level, a share of bankroll ([agent prompts](../agent/agent-prompts.md#conviction-tiers)).
_Avoid_: tier (alone, which means fetch tier)
