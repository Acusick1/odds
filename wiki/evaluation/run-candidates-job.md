---
type: reference
title: The run-candidates job
description: How the scheduled run-candidates job takes each active mechanical candidate's decisions at their decision times, what it records, when it records a decision as missed, and how it books its next run.
sources:
  - odds_lambda.jobs.run_candidates
  - odds_core.paper_trade_models.CandidateDecision
mentions:
  - odds_lambda.paper_trading.place_trade
  - odds_lambda.paper_trading.settle_trades
  - odds_core.alerts.job_alert_context
  - odds_lambda.scheduling.jobs._CANDIDATE_JOBS
  - odds_lambda.storage.event_merge.canonical_events
  - odds_lambda.candidate_rules.rules.run_rule
  - odds_lambda.candidate_rules.market_state.MarketState.snapshot_id
  - odds_lambda.candidate_rules.market_state.MarketState.has_input
  - odds_lambda.candidate_rules.market_state.build_market_state
status: stable
---

# The run-candidates job

`odds_lambda.jobs.run_candidates.main` places the paper trades of every active [mechanical candidate](candidate-registry.md) of one sport. It runs per sport as `run-candidates-epl` and `run-candidates-mlb`, self-schedules ([self-scheduling](../scheduling/self-scheduling.md)) and is bootstrapped by default ([where each job runs](../scheduling/job-deployment-matrix.md#where-each-job-runs)). Smoke checks skip it, because it places bets, unless `--include-candidates` (`odds_lambda.scheduling.jobs._CANDIDATE_JOBS`).

## One run

At its run time `now`:

1. **Select candidates.** `odds_lambda.jobs.run_candidates.runnable_candidates` keeps the registry's mechanical candidates whose sport is the job's, whose status is active and whose `start_date` is not after today (UTC), and that pass `odds_lambda.candidate_rules.validate_candidate_rules` one at a time. An invalid candidate (bad params, an `odds_core.candidates.MechanicalCandidate.rule_sha256` that does not match the rule's code, an unknown rule or an unsupported sport) is skipped and raises a `candidate_invalid:<name>` warning, sent at most once per `odds_core.alerts.stale_reminder_interval`; the other candidates still run. A candidate of a [model rule](candidate-rules.md#model-rules) is skipped and alerted the same way before its hash is checked, because the job cannot load a fitted model. A registry file that cannot be read skips every candidate and raises `candidate_invalid:registry`.
2. **Select events** ([decisions](#decisions)).
3. **Book the next run** ([next run](#next-run)), before any decision, so a failure later leaves the chain intact. A failure in steps 1 to 3 (a database error, a failed booking) raises before anything is booked, and the scheduler re-books the chain ([self-scheduling](../scheduling/local-scheduler.md#scheduling-primitives)).
4. **Take the due decisions** ([decisions](#decisions)) inside `odds_core.alerts.job_alert_context`.
5. **Settle.** `odds_lambda.paper_trading.settle_trades` runs in a session of its own and commits, after the decisions so a settlement failure cannot affect them. A failure is logged as `candidate_settlement_failed` and alerted as `candidate_settlement_failed:<sport>` (critical, at most once per `odds_core.alerts.stale_reminder_interval`), and the run goes on. Nothing else settles paper trades on a schedule ([settlement](../agent/paper-trading.md#settlement)).
6. **Retry.** When a decision failed, the next run is re-booked at the earlier of the booked time and `odds_lambda.jobs.run_candidates.RETRY_DELAY` (2 minutes) after `now`, and the run raises so `odds_core.alerts.job_alert_context` sends its failure alert. A clean run writes the job's heartbeat.

## Decisions

`odds_lambda.jobs.run_candidates.candidate_events` selects the events of the sport that are scheduled with a kick-off after `now`, or whose kick-off is at most `odds_lambda.jobs.run_candidates.MISSED_LOOKBACK` (7 days) before `now`. Twins of one match ([event merge](../storage/event-merge.md#canonical-event)) are collapsed to the canonical event by `odds_lambda.storage.event_merge.canonical_events`, so a match is decided once per decision time whichever twin was selected. The [replay backtest](candidate-backtest.md#replay) selects its events through the same function. For each candidate and event the decision times are `odds_lambda.candidate_rules.rules.decision_times` of the event's kick-off and the candidate's params, one per entry of its `minutes_before_kickoff` ([decision times](candidate-rules.md#decision-times)). A decision is an event and one entry of that list. It is due when its time is at or before `now` and not before the candidate's `start_date`, and the candidate version has no `odds_core.paper_trade_models.CandidateDecision` row for the event and entry. `odds_lambda.jobs.run_candidates.due_decisions` lists the due decisions earliest first, so a match's earlier decision is taken before a later one in the same run. Each due decision commits on its own:

- **On time.** When `now` is at most `odds_lambda.jobs.run_candidates.MAX_LATENESS` (10 minutes) after the decision time and the event has not kicked off, `odds_lambda.candidate_rules.rules.run_rule` builds the state at the decision time with `known_at=now`, so only snapshots stored by the run are visible. The outcomes the candidate version's earlier decisions already bet on the event (its trades with a decision) are passed as `bet_outcomes`, so each outcome of a match is bet at most once ([once per outcome](candidate-rules.md#once-per-outcome)). The rule's other bets are placed with `odds_lambda.paper_trading.place_trade`: flat stake, no bankroll, the candidate's name and version, and the decision's id.
- **Missed, late.** Beyond `odds_lambda.jobs.run_candidates.MAX_LATENESS`, or once the event has kicked off, nothing is placed. The decision is recorded as missed and logged as `candidate_decision_missed` (a warning with the candidate, event, minutes before kick-off, decision time and reason `late`). Recording it stops later runs from taking, or logging, it again, and an event that kicked off within `odds_lambda.jobs.run_candidates.MISSED_LOOKBACK` without a decision gets its missed record this way.
- **Missed, no input.** When the state has no inputs (`odds_lambda.candidate_rules.market_state.MarketState.has_input` is false: no snapshot with UK retail and Betfair Exchange prices together, stored by the run, within the rule's maximum input age before the decision time, [market state](candidate-rules.md#market-state)), nothing is placed and the decision is recorded and logged as missed with reason `no_input` (`odds_lambda.jobs.run_candidates.MissReason`). After the decision commits, a `candidate_no_input:<name>:<minutes>` warning is alerted, because the parity check skips missed decisions and a collection gap (no Betfair Exchange row in the snapshots, or no run before the decision) would otherwise go unseen. `<minutes>` is the decision's lead before kick-off, and the alert is sent at most once per `odds_core.alerts.stale_reminder_interval` per candidate and lead, so a gap before one decision time is not silenced by an alert for another. Collection failed the decision, so it is not counted as a decision not to bet. A later run does not retake it, even if a snapshot stamped before the decision time is stored afterwards.
- **Already decided.** A unique violation on the decision row means a concurrent run committed it first. The decision is rolled back and counted as already decided, with no failure and no alert. A unique violation on a trade of this decision, with no decision row behind it, is a failure.
- **Failed.** A decision that raises (a rejected bet, a duplicate bet from one rule call) is rolled back whole, with no decision row, and logged as `candidate_decision_failed`. The run skips the event's later decisions, so they are not taken before it, goes on with the other events and raises at the end. The next run retries it only while within `odds_lambda.jobs.run_candidates.MAX_LATENESS`; once it is recorded, as taken or missed, the later decisions follow.

A decision whose rule returns no bet from a state with inputs, such as one whose snapshot has no usable exchange price, or only bets on outcomes already bet, is recorded with zero bets and not missed, so it is not retaken. A missed earlier decision does not stop a later one: each time is decided on its own. A kick-off that moves after a decision was taken does not reopen it, because the record is keyed by the entry and not by the time; the event's later decision times follow the kick-off as it then stands.

### The decision record

`odds_core.paper_trade_models.CandidateDecision` is table `candidate_decisions`, unique on candidate name, version, event and `minutes_before_kickoff`, the entry of the candidate's list the decision is for. It holds the decision time and the kick-off the decision used, `known_at` (the run time), `input_snapshot_ids`, `bets_placed` and `missed`. `input_snapshot_ids` holds the one snapshot `odds_lambda.candidate_rules.market_state.build_market_state` built the decision's state from (`odds_lambda.candidate_rules.market_state.MarketState.snapshot_id`); a missed decision records none. A trade placed by the job references its decision through `odds_core.paper_trade_models.PaperTrade.decision_id`. A parity check replays the decision from those snapshots and the recorded kick-off ([parity check](candidate-parity.md#replay-of-a-decision)), so neither a kick-off rescheduled afterwards nor a later twin merge changes the replay.

A unique index on a runner trade's candidate, version, event and selection, for trades with a decision, rejects a second bet on an outcome of a match, from one rule call or from two decisions. The agent's trades have no decision and may repeat a bet ([paper trading](../agent/paper-trading.md#candidate-fields)).

## Next run

`odds_lambda.jobs.run_candidates.next_decision_wakeup` is the earliest decision time after `now` across the runnable candidates, each of their decision times, and the selected events, at most `odds_lambda.jobs.run_candidates.MAX_WAIT` (30 minutes) after `now`. The cap picks up new fixtures and newly registered candidates. With no candidate or no fixture the next run is `odds_lambda.jobs.run_candidates.MAX_WAIT` away. The time is booked with `odds_lambda.scheduling.helpers.self_schedule` under the compound job name.
