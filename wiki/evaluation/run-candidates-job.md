---
type: reference
title: The run-candidates job
description: How the scheduled run-candidates job takes each active mechanical candidate's decisions at their decision times, what it records, when it skips a decision, and how it books its next run.
sources:
  - odds_lambda.jobs.run_candidates
  - odds_core.paper_trade_models.CandidateDecision
mentions:
  - odds_lambda.paper_trading.place_trade
  - odds_lambda.paper_trading.settle_trades
  - odds_core.alerts.job_alert_context
status: stable
---

# The run-candidates job

`odds_lambda.jobs.run_candidates.main` places the paper trades of every active [mechanical candidate](candidate-registry.md) of one sport. It runs per sport as `run-candidates-epl` and `run-candidates-mlb`, self-schedules ([self-scheduling](../scheduling/self-scheduling.md)) and is bootstrapped by default ([where each job runs](../scheduling/job-deployment-matrix.md#where-each-job-runs)). Smoke checks skip it, because it places bets (`odds_lambda.scheduling.jobs._SMOKE_UNSAFE_JOBS`).

## One run

At its run time `now`:

1. **Book the next run** ([next run](#next-run)), before any work, so a failure later leaves the chain intact.
2. **Settle.** `odds_lambda.paper_trading.settle_trades` runs and commits. Nothing else settles paper trades on a schedule ([settlement](../agent/paper-trading.md#settlement)).
3. **Select candidates.** `odds_lambda.jobs.run_candidates.runnable_candidates` keeps the registry's mechanical candidates whose sport is the job's, whose status is active and whose `start_date` is not after today (UTC), and that pass `odds_lambda.candidate_rules.validate_candidate_rules` one at a time. An invalid candidate (bad params, an `odds_core.candidates.MechanicalCandidate.rule_sha256` that does not match the rule's code, an unknown rule or an unsupported sport) is skipped and raises a `candidate_invalid:<name>` warning, sent at most once per `odds_core.alerts.stale_reminder_interval`; the other candidates still run. A registry file that cannot be read skips every candidate and raises `candidate_invalid:registry`.
4. **Take the due decisions** ([decisions](#decisions)).
5. The run raises when a decision failed, so `odds_core.alerts.job_alert_context` sends its failure alert. A clean run writes the job's heartbeat.

## Decisions

The events considered are those of the sport with status `scheduled` and a kick-off after `now` (`odds_lambda.jobs.run_candidates.upcoming_events`). For each candidate and event the decision time is `odds_lambda.candidate_rules.rules.decision_time` of the event's kick-off and the candidate's params. A decision is due when that time is at or before `now` and the candidate version has no `odds_core.paper_trade_models.CandidateDecision` row for the event. Each due decision commits on its own:

- **On time.** When `now` is at most `odds_lambda.jobs.run_candidates.MAX_LATENESS` (10 minutes) after the decision time, `odds_lambda.candidate_rules.load_market_state` builds the state at the decision time with `known_at=now`, so only snapshots stored by the run are visible. The rule's bets are placed with `odds_lambda.paper_trading.place_trade`: flat stake, no bankroll, the candidate's name and version, and the decision's id.
- **Late.** Beyond `odds_lambda.jobs.run_candidates.MAX_LATENESS` nothing is placed. The decision is recorded as missed and logged as `candidate_decision_missed` (a warning with the candidate, event and decision time). Recording it stops later runs from taking, or logging, it again.
- **Failed.** A decision that raises (a rejected bet, a duplicate bet from one rule call) is rolled back whole, with no decision row, and logged as `candidate_decision_failed`. The run goes on with the other decisions and raises at the end. The next run retries it only while within `odds_lambda.jobs.run_candidates.MAX_LATENESS`.

A decision whose rule returns no bet is recorded with zero bets, so it is not retaken. A kick-off that moves after the decision was taken does not reopen it.

### The decision record

`odds_core.paper_trade_models.CandidateDecision` is table `candidate_decisions`, unique on candidate name, version and event. It holds the decision time and the kick-off the decision used, `known_at` (the run time), `bets_placed` and `missed`. A trade placed by the job references its decision through `odds_core.paper_trade_models.PaperTrade.decision_id`. A parity check replays the decision with the recorded kick-off and `known_at`, so a kick-off rescheduled afterwards does not change the replay's decision time.

A unique index on a runner trade's candidate, version, event, selection and bookmaker, for trades with a decision, rejects a rule that returns the same bet twice. The agent's trades have no decision and may repeat a bet ([paper trading](../agent/paper-trading.md#candidate-fields)).

## Next run

`odds_lambda.jobs.run_candidates.next_decision_wakeup` is the earliest decision time after `now` across the runnable candidates and the upcoming events, at most `odds_lambda.jobs.run_candidates.MAX_WAIT` (6 hours) after `now`. The cap picks up new fixtures and newly registered candidates. With no candidate or no fixture the next run is `odds_lambda.jobs.run_candidates.MAX_WAIT` away. The time is booked with `odds_lambda.scheduling.helpers.self_schedule` under the compound job name.
