"""The run-candidates job: decision times, lateness, idempotency and trade fields."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
import structlog
import yaml
from odds_core.market_prices import Outcome
from odds_core.models import Event, EventStatus
from odds_core.paper_trade_models import CandidateDecision, PaperTrade
from odds_lambda.candidate_rules import RULES, BetIntent, MarketState, RuleParams, RuleSpec
from odds_lambda.candidate_rules import rules as rules_module
from odds_lambda.candidate_rules.rules import rule_sha256
from odds_lambda.jobs import run_candidates
from odds_lambda.jobs.run_candidates import (
    MAX_LATENESS,
    MAX_WAIT,
    candidate_events,
    next_decision_wakeup,
    run_decisions,
    runnable_candidates,
)
from odds_lambda.paper_trading import get_portfolio_summary, place_trade, settle_trades
from odds_lambda.scheduling.jobs import JobContext
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from tests.test_helpers import make_epl_event, make_tagged_snapshot, three_way_bookmaker

pytestmark = pytest.mark.integration

NOW = datetime.now(UTC).replace(microsecond=0)
EXCHANGE_ROW = three_way_bookmaker("betfair_exchange", (2.0, 4.0, 4.0))
EPL = "soccer_epl"


class _Params(RuleParams):
    pass


def _bet_home(state: MarketState, params: _Params) -> list[BetIntent]:
    return [BetIntent(Outcome.HOME, "bet365", state.retail["bet365"][Outcome.HOME], 1.0)]


def _no_bet(state: MarketState, params: _Params) -> list[BetIntent]:
    return []


def _twice(state: MarketState, params: _Params) -> list[BetIntent]:
    return _bet_home(state, params) * 2


def _bet_longest(state: MarketState, params: _Params) -> list[BetIntent]:
    prices = state.retail["bet365"]
    outcome = max(prices, key=lambda o: prices[o])
    return [BetIntent(outcome, "bet365", prices[outcome], 1.0)]


@pytest.fixture(autouse=True)
def rules(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, function in (
        ("bet_home", _bet_home),
        ("no_bet", _no_bet),
        ("twice", _twice),
        ("bet_longest", _bet_longest),
    ):
        monkeypatch.setitem(RULES, name, RuleSpec(function, _Params))


@pytest.fixture
def registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr("odds_lambda.jobs.run_candidates.candidates_dir", lambda: tmp_path)
    monkeypatch.setattr("odds_lambda.paper_trading.candidates_dir", lambda: tmp_path)
    return tmp_path


def write_candidate(directory: Path, name: str = "cand", **fields: Any) -> None:
    rule = fields.get("rule", "bet_home")
    data = {
        "name": name,
        "description": "d",
        "kind": "mechanical",
        "sport": EPL,
        "version": 1,
        "issue": 1,
        "start_date": (NOW - timedelta(days=1)).date().isoformat(),
        "status": "active",
        "rule": rule,
        "rule_sha256": rule_sha256(rule) if rule in RULES else "0" * 64,
        "params": {"minutes_before_kickoff": [60], "max_input_age_minutes": 120},
        **fields,
    }
    (directory / f"{name}.yaml").write_text(yaml.safe_dump(data))


async def add_event(
    session: Any, event_id: str, kickoff: datetime, away_team: str = "Chelsea"
) -> None:
    session.add(make_epl_event(event_id, kickoff, away_team=away_team))
    snapshot = make_tagged_snapshot(
        event_id,
        NOW - timedelta(hours=1),
        [three_way_bookmaker("bet365", (2.0, 3.4, 4.0)), EXCHANGE_ROW],
        "oddsportal_live",
    )
    snapshot.created_at = NOW - timedelta(hours=1)
    session.add(snapshot)
    await session.commit()


async def run(session: Any, now: datetime = NOW) -> run_candidates.RunSummary:
    runnable = await runnable_candidates(EPL, now)
    events = await candidate_events(session, EPL, now)
    return await run_decisions(session, runnable, events, now)


async def all_of(session: Any, model: type) -> list[Any]:
    return list((await session.execute(select(model))).scalars().all())


# Kick-off 59 minutes out with a 60-minute rule: decided one minute ago.
DUE = NOW + timedelta(minutes=59)
# Decided 15 minutes ago, beyond MAX_LATENESS.
LATE = NOW + timedelta(minutes=45)
NOT_DUE = NOW + timedelta(hours=3)


class TestDecisions:
    async def test_due_decision_places_flat_stake_bets_and_records_it(
        self, pglite_async_session, registry
    ) -> None:
        write_candidate(registry)
        await add_event(pglite_async_session, "e1", DUE)

        summary = await run(pglite_async_session)

        assert summary.placed == 1
        [decision] = await all_of(pglite_async_session, CandidateDecision)
        assert (decision.candidate_name, decision.candidate_version, decision.event_id) == (
            "cand",
            1,
            "e1",
        )
        assert decision.decision_time == DUE - timedelta(minutes=60)
        assert decision.kickoff_at_decision == DUE
        assert decision.known_at == NOW
        assert (decision.bets_placed, decision.missed) == (1, False)
        assert len(decision.input_snapshot_ids) == 1
        [trade] = await all_of(pglite_async_session, PaperTrade)
        assert (trade.candidate_name, trade.candidate_version) == ("cand", 1)
        assert trade.decision_id == decision.id
        assert (trade.selection, trade.bookmaker, trade.price_decimal) == ("home", "bet365", 2.0)
        assert trade.market == "1x2"
        assert trade.bankroll_before is None and trade.stake == 1.0

    async def test_late_decision_is_recorded_as_missed_and_logged(
        self, pglite_async_session, registry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        alert = AsyncMock()
        monkeypatch.setattr(run_candidates, "send_recorded_alert", alert)
        write_candidate(registry)
        await add_event(pglite_async_session, "e1", LATE)
        assert NOW - (LATE - timedelta(minutes=60)) > MAX_LATENESS

        with structlog.testing.capture_logs() as logs:
            summary = await run(pglite_async_session)

        assert summary.missed == 1
        assert await all_of(pglite_async_session, PaperTrade) == []
        [decision] = await all_of(pglite_async_session, CandidateDecision)
        assert (decision.missed, decision.bets_placed) == (True, 0)
        assert decision.input_snapshot_ids == []
        [missed] = [log for log in logs if log["event"] == "candidate_decision_missed"]
        assert missed["log_level"] == "warning"
        assert (missed["candidate"], missed["event_id"], missed["reason"]) == ("cand", "e1", "late")
        alert.assert_not_awaited()

    async def test_decision_without_a_recent_snapshot_is_recorded_as_missed(
        self, pglite_async_session, registry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        alert = AsyncMock()
        monkeypatch.setattr(run_candidates, "send_recorded_alert", alert)
        write_candidate(
            registry, params={"minutes_before_kickoff": [60], "max_input_age_minutes": 10}
        )
        await add_event(pglite_async_session, "e1", DUE)

        with structlog.testing.capture_logs() as logs:
            summary = await run(pglite_async_session)

        assert (summary.missed, summary.placed) == (1, 0)
        [decision] = await all_of(pglite_async_session, CandidateDecision)
        assert (decision.missed, decision.bets_placed, decision.input_snapshot_ids) == (
            True,
            0,
            [],
        )
        [missed] = [log for log in logs if log["event"] == "candidate_decision_missed"]
        assert missed["reason"] == "no_input"
        [call] = alert.await_args_list
        assert call.args[:2] == ("candidate_no_input:cand:60", "warning")
        assert "e1" in call.args[2] and "60 minutes before kick-off" in call.args[2]
        assert call.kwargs["every"] == run_candidates.stale_reminder_interval()

    async def test_rerun_places_nothing_new(self, pglite_async_session, registry) -> None:
        write_candidate(registry)
        await add_event(pglite_async_session, "e1", DUE)
        await add_event(pglite_async_session, "e2", LATE, away_team="Spurs")

        await run(pglite_async_session)
        again = await run(pglite_async_session, NOW + timedelta(minutes=1))

        assert (again.placed, again.missed, again.already_decided) == (0, 0, 0)
        assert len(await all_of(pglite_async_session, PaperTrade)) == 1
        assert len(await all_of(pglite_async_session, CandidateDecision)) == 2

    async def test_zero_bet_decision_is_recorded_once(self, pglite_async_session, registry) -> None:
        write_candidate(registry, rule="no_bet")
        await add_event(pglite_async_session, "e1", DUE)

        first = await run(pglite_async_session)
        second = await run(pglite_async_session)

        assert (first.skipped_zero_bet, second.skipped_zero_bet) == (1, 0)
        [decision] = await all_of(pglite_async_session, CandidateDecision)
        assert (decision.bets_placed, decision.missed) == (0, False)
        assert await all_of(pglite_async_session, PaperTrade) == []

    async def test_event_not_yet_due_is_left_alone(self, pglite_async_session, registry) -> None:
        write_candidate(registry)
        await add_event(pglite_async_session, "e1", NOT_DUE)

        await run(pglite_async_session)

        assert await all_of(pglite_async_session, CandidateDecision) == []

    async def test_started_event_is_recorded_as_missed(
        self, pglite_async_session, registry
    ) -> None:
        write_candidate(registry)
        await add_event(pglite_async_session, "e1", NOW - timedelta(minutes=5))
        await add_event(pglite_async_session, "e2", NOW - timedelta(days=8))

        summary = await run(pglite_async_session)

        assert (summary.missed, summary.placed) == (1, 0)
        [decision] = await all_of(pglite_async_session, CandidateDecision)
        assert (decision.event_id, decision.missed, decision.bets_placed) == ("e1", True, 0)
        assert await all_of(pglite_async_session, PaperTrade) == []

    async def test_decision_time_before_the_start_date_is_not_decided(
        self, pglite_async_session, registry
    ) -> None:
        write_candidate(registry, start_date=NOW.date().isoformat())
        before_start = datetime.combine(NOW.date(), datetime.min.time(), tzinfo=UTC)
        await add_event(pglite_async_session, "e1", before_start + timedelta(minutes=30))

        await run(pglite_async_session, before_start + timedelta(hours=2))

        assert await all_of(pglite_async_session, CandidateDecision) == []

    async def test_twins_are_decided_once_on_the_canonical_event(
        self, pglite_async_session, registry
    ) -> None:
        write_candidate(registry)
        await add_event(pglite_async_session, "op_live_e1", DUE)
        await add_event(pglite_async_session, "api1", DUE - timedelta(minutes=5))
        for event_id in ("op_live_e1", "api1"):
            pglite_async_session.add(
                make_tagged_snapshot(
                    event_id,
                    NOW - timedelta(minutes=30),
                    [three_way_bookmaker("bet365", (2.0, 3.4, 4.0)), EXCHANGE_ROW],
                    "oddsportal_live",
                )
            )
        await pglite_async_session.commit()

        summary = await run(pglite_async_session)

        assert summary.placed == 1
        [decision] = await all_of(pglite_async_session, CandidateDecision)
        assert decision.event_id == "api1"

    async def test_a_decision_committed_by_a_concurrent_run_is_not_a_failure(
        self, pglite_async_session, registry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        write_candidate(registry)
        await add_event(pglite_async_session, "e1", DUE)
        real_decide = run_candidates._decide

        async def racing(session: Any, item: Any, event: Any, minutes: int, now: datetime) -> Any:
            # The other run's decision is committed between our check and our insert.
            session.add(
                CandidateDecision(
                    candidate_name="cand",
                    candidate_version=1,
                    event_id="e1",
                    minutes_before_kickoff=60,
                    decision_time=DUE - timedelta(minutes=60),
                    kickoff_at_decision=DUE,
                    known_at=now,
                )
            )
            await session.commit()
            return await real_decide(session, item, event, minutes, now)

        monkeypatch.setattr(run_candidates, "_decide", racing)

        summary = await run(pglite_async_session)

        assert (summary.failed, summary.already_decided, summary.placed) == (0, 1, 0)
        assert len(await all_of(pglite_async_session, CandidateDecision)) == 1

    async def test_snapshots_are_loaded_as_known_at_run_time(
        self, pglite_async_session, registry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        write_candidate(registry)
        await add_event(pglite_async_session, "e1", DUE)
        spy = AsyncMock(side_effect=rules_module.load_market_state)
        monkeypatch.setattr(rules_module, "load_market_state", spy)

        await run(pglite_async_session)

        assert spy.await_args.args[2] == DUE - timedelta(minutes=60)
        assert spy.await_args.args[3] == timedelta(minutes=120)
        assert spy.await_args.args[4] == NOW

    async def test_failed_decision_is_not_recorded_and_others_run(
        self, pglite_async_session, registry
    ) -> None:
        write_candidate(registry, "a-twice", rule="twice")
        write_candidate(registry, "b-good")
        await add_event(pglite_async_session, "e1", DUE)

        summary = await run(pglite_async_session)

        assert (summary.failed, summary.placed) == (1, 1)
        [decision] = await all_of(pglite_async_session, CandidateDecision)
        assert decision.candidate_name == "b-good"


class TestSeveralDecisionTimes:
    """A candidate deciding at 90 and 60 minutes before a kick-off 59 minutes away."""

    TIMES = {"minutes_before_kickoff": [60, 90], "max_input_age_minutes": 120}
    EARLY_RUN = DUE - timedelta(minutes=89)

    async def test_each_time_is_decided_and_an_outcome_is_bet_once(
        self, pglite_async_session, registry
    ) -> None:
        write_candidate(registry, params=self.TIMES)
        await add_event(pglite_async_session, "e1", DUE)

        early = await run(pglite_async_session, self.EARLY_RUN)
        late = await run(pglite_async_session)

        assert (early.placed, late.placed, late.skipped_zero_bet) == (1, 0, 1)
        decisions = sorted(
            await all_of(pglite_async_session, CandidateDecision), key=lambda d: d.decision_time
        )
        assert [d.decision_time for d in decisions] == [
            DUE - timedelta(minutes=90),
            DUE - timedelta(minutes=60),
        ]
        assert [(d.bets_placed, d.missed) for d in decisions] == [(1, False), (0, False)]
        assert len(decisions[1].input_snapshot_ids) == 1
        [trade] = await all_of(pglite_async_session, PaperTrade)
        assert (trade.selection, trade.decision_id) == ("home", decisions[0].id)

    async def test_a_later_time_may_bet_a_different_outcome(
        self, pglite_async_session, registry
    ) -> None:
        write_candidate(registry, rule="bet_longest", params=self.TIMES)
        await add_event(pglite_async_session, "e1", DUE)
        # Away is longest at the early time; this later snapshot makes the draw longest.
        later = make_tagged_snapshot(
            "e1",
            NOW - timedelta(minutes=5),
            [three_way_bookmaker("bet365", (1.5, 9.0, 3.0)), EXCHANGE_ROW],
            "oddsportal_live",
        )
        later.created_at = NOW - timedelta(minutes=5)
        pglite_async_session.add(later)
        await pglite_async_session.commit()

        await run(pglite_async_session, self.EARLY_RUN)
        await run(pglite_async_session)

        trades = await all_of(pglite_async_session, PaperTrade)
        assert sorted(t.selection for t in trades) == ["away", "draw"]

    async def test_a_kickoff_moved_after_a_decision_does_not_reopen_it(
        self, pglite_async_session, registry
    ) -> None:
        write_candidate(registry)
        await add_event(pglite_async_session, "e1", DUE)
        await run(pglite_async_session)
        event = await pglite_async_session.get(Event, "e1")
        assert event is not None
        event.commence_time = DUE + timedelta(minutes=20)
        await pglite_async_session.commit()

        again = await run(pglite_async_session, NOW + timedelta(minutes=20))

        assert (again.placed, again.missed, again.skipped_zero_bet) == (0, 0, 0)
        [decision] = await all_of(pglite_async_session, CandidateDecision)
        assert decision.minutes_before_kickoff == 60

    async def test_a_failed_earlier_decision_holds_back_the_later_one(
        self, pglite_async_session, registry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Both times due and on time in one run: 12 and 5 minutes before a kick-off 4 minutes
        # away.
        write_candidate(
            registry, params={"minutes_before_kickoff": [12, 5], "max_input_age_minutes": 120}
        )
        await add_event(pglite_async_session, "e1", NOW + timedelta(minutes=4))
        real_decide = run_candidates._decide

        async def fail_early(session: Any, item: Any, event: Any, minutes: int, now: Any) -> Any:
            if minutes == 12:
                raise RuntimeError("boom")
            return await real_decide(session, item, event, minutes, now)

        monkeypatch.setattr(run_candidates, "_decide", fail_early)

        summary = await run(pglite_async_session)

        assert (summary.failed, summary.placed) == (1, 0)
        assert await all_of(pglite_async_session, CandidateDecision) == []

    async def test_a_missed_earlier_time_leaves_the_later_one_to_decide(
        self, pglite_async_session, registry
    ) -> None:
        write_candidate(registry, params=self.TIMES)
        await add_event(pglite_async_session, "e1", DUE)

        summary = await run(pglite_async_session)

        assert (summary.missed, summary.placed) == (1, 1)
        decisions = sorted(
            await all_of(pglite_async_session, CandidateDecision), key=lambda d: d.decision_time
        )
        assert [(d.missed, d.bets_placed) for d in decisions] == [(True, 0), (False, 1)]


class TestCandidateSelection:
    async def test_inactive_unstarted_and_other_sport_candidates_are_ignored(
        self, registry
    ) -> None:
        write_candidate(registry, "active")
        write_candidate(registry, "killed", status="killed")
        write_candidate(registry, "promoted", status="promoted")
        write_candidate(registry, "future", start_date=(NOW + timedelta(days=1)).date().isoformat())
        write_candidate(registry, "mlb", sport="baseball_mlb")

        runnable = await runnable_candidates(EPL, NOW)

        assert [r.candidate.name for r in runnable] == ["active"]

    async def test_invalid_candidates_alert_once_each_and_others_still_run(
        self, registry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        alert = AsyncMock()
        monkeypatch.setattr(run_candidates, "send_recorded_alert", alert)
        write_candidate(registry, "good")
        write_candidate(registry, "bad-params", params={"minutes_before_kickoff": [-1]})
        write_candidate(registry, "bad-hash", rule_sha256="0" * 64)
        write_candidate(registry, "bad-rule", rule="nonexistent", rule_sha256="0" * 64)

        runnable = await runnable_candidates(EPL, NOW)

        assert [r.candidate.name for r in runnable] == ["good"]
        types = sorted(call.args[0] for call in alert.await_args_list)
        assert types == [
            "candidate_invalid:bad-hash",
            "candidate_invalid:bad-params",
            "candidate_invalid:bad-rule",
        ]

    async def test_a_model_rule_candidate_is_skipped_with_an_alert(
        self, registry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        alert = AsyncMock()
        monkeypatch.setattr(run_candidates, "send_recorded_alert", alert)
        write_candidate(registry, "good")
        write_candidate(
            registry,
            "model",
            rule="retail_ev_model",
            params={
                "model": "retail_ev_price_only",
                "threshold": 0.01,
                "training_matches": 50,
                "refit_every_matches": 10,
                "minutes_before_kickoff": [60],
                "max_input_age_minutes": 10,
            },
        )

        runnable = await runnable_candidates(EPL, NOW)

        assert [r.candidate.name for r in runnable] == ["good"]
        [call] = alert.await_args_list
        assert call.args[0] == "candidate_invalid:model"
        assert "needs a fitted model" in call.args[2]

    async def test_unreadable_registry_alerts_and_runs_nothing(
        self, registry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        alert = AsyncMock()
        monkeypatch.setattr(run_candidates, "send_recorded_alert", alert)
        (registry / "broken.yaml").write_text("name: [")

        assert await runnable_candidates(EPL, NOW) == []
        assert alert.await_args.args[0] == "candidate_invalid:registry"


def _event(kickoff: datetime) -> Any:
    return SimpleNamespace(commence_time=kickoff)


def _runnable(*minutes: int) -> Any:
    return SimpleNamespace(params=_Params(minutes_before_kickoff=minutes, max_input_age_minutes=10))


class TestNextRunTime:
    def test_is_the_earliest_future_decision_time(self) -> None:
        events = [_event(NOW + timedelta(minutes=200)), _event(NOW + timedelta(minutes=80))]
        runnable = [_runnable(60), _runnable(30)]

        # 80 minutes to the nearer kick-off, minus 60 minutes
        assert next_decision_wakeup(runnable, events, NOW) == NOW + timedelta(minutes=20)

    def test_past_decision_times_do_not_count(self) -> None:
        events = [_event(NOW + timedelta(minutes=30)), _event(NOW + timedelta(minutes=80))]

        assert next_decision_wakeup([_runnable(60)], events, NOW) == NOW + timedelta(minutes=20)

    def test_counts_every_decision_time_of_a_candidate(self) -> None:
        events = [_event(NOW + timedelta(minutes=80))]

        assert next_decision_wakeup([_runnable(120, 60)], events, NOW) == NOW + timedelta(
            minutes=20
        )
        assert next_decision_wakeup([_runnable(70, 30)], events, NOW) == NOW + timedelta(minutes=10)

    def test_is_capped(self) -> None:
        events = [_event(NOW + timedelta(days=2))]

        assert next_decision_wakeup([_runnable(60)], events, NOW) == NOW + MAX_WAIT

    def test_without_candidates_or_events_waits_the_cap(self) -> None:
        assert next_decision_wakeup([], [_event(NOW + timedelta(hours=1))], NOW) == NOW + MAX_WAIT
        assert next_decision_wakeup([_runnable(60)], [], NOW) == NOW + MAX_WAIT


class TestMain:
    async def test_books_before_settling_and_decides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[str] = []

        @asynccontextmanager
        async def session_maker() -> Iterator[Any]:
            yield AsyncMock()

        async def settle(session: Any) -> list[Any]:
            calls.append("settle")
            return []

        async def decide(*args: Any, **kwargs: Any) -> run_candidates.RunSummary:
            calls.append("decide")
            return run_candidates.RunSummary()

        async def book(**kwargs: Any) -> None:
            calls.append("schedule")

        schedule = AsyncMock(side_effect=book)

        @asynccontextmanager
        async def alert_context(name: str) -> Iterator[None]:
            yield

        monkeypatch.setattr(run_candidates, "async_session_maker", session_maker)
        monkeypatch.setattr(run_candidates, "runnable_candidates", AsyncMock(return_value=[]))
        monkeypatch.setattr(run_candidates, "candidate_events", AsyncMock(return_value=[]))
        monkeypatch.setattr(run_candidates, "self_schedule", schedule)
        monkeypatch.setattr(run_candidates, "settle_trades", settle)
        monkeypatch.setattr(run_candidates, "run_decisions", decide)
        monkeypatch.setattr(run_candidates, "job_alert_context", alert_context)

        await run_candidates.main(JobContext(sport=EPL))

        assert calls == ["schedule", "decide", "settle"]
        kwargs = schedule.await_args.kwargs
        assert kwargs["job_name"] == "run-candidates-epl"
        assert kwargs["sport"] == EPL

    async def test_a_failed_decision_rebooks_soon_and_fails_the_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        schedule = self._patch(monkeypatch, run_candidates.RunSummary(failed=1))

        with pytest.raises(RuntimeError, match="1 candidate decision"):
            await run_candidates.main(JobContext(sport=EPL))

        first, retry = (call.kwargs["next_time"] for call in schedule.await_args_list)
        assert retry < first
        assert retry - datetime.now(UTC) <= run_candidates.RETRY_DELAY

    async def test_a_settlement_failure_is_alerted_and_does_not_fail_the_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        schedule = self._patch(monkeypatch, run_candidates.RunSummary(placed=1))
        monkeypatch.setattr(run_candidates, "settle_trades", AsyncMock(side_effect=ValueError("x")))
        alert = AsyncMock()
        monkeypatch.setattr(run_candidates, "send_recorded_alert", alert)

        await run_candidates.main(JobContext(sport=EPL))

        assert alert.await_args.args[0] == "candidate_settlement_failed:soccer_epl"
        assert schedule.await_count == 1

    @staticmethod
    def _patch(monkeypatch: pytest.MonkeyPatch, summary: run_candidates.RunSummary) -> AsyncMock:
        @asynccontextmanager
        async def session_maker() -> Iterator[Any]:
            yield AsyncMock()

        @asynccontextmanager
        async def alert_context(name: str) -> Iterator[None]:
            yield

        schedule = AsyncMock()
        monkeypatch.setattr(run_candidates, "async_session_maker", session_maker)
        monkeypatch.setattr(run_candidates, "runnable_candidates", AsyncMock(return_value=[]))
        monkeypatch.setattr(run_candidates, "candidate_events", AsyncMock(return_value=[]))
        monkeypatch.setattr(run_candidates, "self_schedule", schedule)
        monkeypatch.setattr(run_candidates, "settle_trades", AsyncMock(return_value=[]))
        monkeypatch.setattr(run_candidates, "run_decisions", AsyncMock(return_value=summary))
        monkeypatch.setattr(run_candidates, "job_alert_context", alert_context)
        return schedule


class TestFlatStakeTrades:
    async def test_portfolio_summary_excludes_flat_stake_trades(
        self, pglite_async_session, registry
    ) -> None:
        session = pglite_async_session
        write_candidate(registry)
        await add_event(session, "e1", DUE)
        await place_trade(
            session,
            event_id="e1",
            market="1x2",
            selection="away",
            bookmaker="bet365",
            price_decimal=3.0,
            stake=10.0,
            bankroll=1000.0,
        )
        await run(session)
        await session.commit()

        summary = await get_portfolio_summary(session)

        assert (summary.total_trades, summary.open_trades) == (1, 1)
        assert summary.current_bankroll == 990.0

    async def test_settlement_leaves_the_bankroll_empty(
        self, pglite_async_session, registry
    ) -> None:
        session = pglite_async_session
        write_candidate(registry)
        await add_event(session, "e1", DUE)
        await run(session)
        event = await session.get(Event, "e1")
        assert event is not None
        event.status = EventStatus.FINAL
        event.home_score, event.away_score = 2, 0
        await session.commit()

        [settled] = await settle_trades(session)

        assert settled.pnl == 1.0
        assert settled.bankroll_before is None and settled.bankroll_after is None

    async def test_a_trade_without_a_candidate_needs_a_bankroll(self, pglite_async_session) -> None:
        await add_event(pglite_async_session, "e1", DUE)

        with pytest.raises(ValueError, match="needs a bankroll"):
            await place_trade(
                pglite_async_session,
                event_id="e1",
                market="1x2",
                selection="home",
                bookmaker="bet365",
                price_decimal=2.0,
                stake=1.0,
                bankroll=None,
            )

    async def test_unique_index_rejects_a_second_runner_bet_on_an_outcome(
        self, pglite_async_session
    ) -> None:
        session = pglite_async_session
        await add_event(session, "e1", DUE)
        decision = CandidateDecision(
            candidate_name="cand",
            candidate_version=1,
            event_id="e1",
            minutes_before_kickoff=60,
            decision_time=NOW,
            kickoff_at_decision=DUE,
            known_at=NOW,
        )
        session.add(decision)
        await session.flush()

        def trade(decision_id: int | None, bookmaker: str = "bet365") -> PaperTrade:
            return PaperTrade(
                event_id="e1",
                market="1x2",
                selection="home",
                bookmaker=bookmaker,
                price_decimal=2.0,
                stake=1.0,
                bankroll_before=None if decision_id else 1000.0,
                candidate_name="cand",
                candidate_version=1,
                decision_id=decision_id,
            )

        # The agent may take the same price at the same book twice.
        session.add_all([trade(None), trade(None), trade(decision.id)])
        await session.commit()

        session.add(trade(decision.id, bookmaker="betway"))
        with pytest.raises(IntegrityError):
            await session.commit()
