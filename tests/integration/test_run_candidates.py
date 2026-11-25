"""The run-candidates job: decision times, lateness, idempotency and bet fields."""

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
from odds_analytics.candidate_parity import ParityStatus, check_parity
from odds_core.bet_models import Bet, CandidateDecision
from odds_core.bookmakers import BOOK_SETS, BookSet
from odds_core.candidate_registry_models import CandidateRegistryCheck
from odds_core.candidates import RuleCandidate
from odds_core.models import Event, EventStatus
from odds_lambda.betting import get_portfolio_summary, place_bet, settle_bets
from odds_lambda.candidate_rules import EstimateBasis, RuleSpec, SharpInput
from odds_lambda.candidate_rules import rules as rules_module
from odds_lambda.candidate_rules.pinning import rule_sha256
from odds_lambda.candidate_rules.registry import RULES
from odds_lambda.candidate_rules.snapshot import registry_snapshot
from odds_lambda.jobs import run_candidates
from odds_lambda.jobs.run_candidates import (
    MAX_LATENESS,
    MAX_WAIT,
    candidate_events,
    next_decision_wakeup,
    record_selection,
    run_decisions,
    select_candidates,
)
from odds_lambda.scheduling.jobs import JobContext
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from tests.candidate_rule_fixtures import (
    ESTIMATED_EV,
    FixtureRuleParams,
    bet_home,
    bet_longest,
    no_bet,
    twice,
)
from tests.test_helpers import (
    make_epl_event,
    make_tagged_snapshot,
    polymarket_bookmaker,
    store_tagged,
    three_way_bookmaker,
    totals_book,
)

pytestmark = pytest.mark.integration

NOW = datetime.now(UTC).replace(microsecond=0)
EXCHANGE_ROW = three_way_bookmaker("betfair_exchange", (2.0, 4.0, 4.0))
EPL = "soccer_epl"


@pytest.fixture(autouse=True)
def rules(monkeypatch: pytest.MonkeyPatch) -> None:
    for function in (bet_home, no_bet, twice, bet_longest):
        monkeypatch.setitem(
            RULES,
            function.__name__,
            RuleSpec(
                function,
                FixtureRuleParams,
                frozenset({SharpInput.EXCHANGE}),
                estimate_basis=EstimateBasis.EXCHANGE,
            ),
        )


def write_candidate(directory: Path, name: str = "cand", **fields: Any) -> None:
    rule = fields.get("rule", "bet_home")
    data = {
        "name": name,
        "description": "d",
        "kind": "rule",
        "books": "uk_retail",
        "market": "1x2",
        "leagues": [EPL],
        "version": 1,
        "issue": 1,
        "start_date": (NOW - timedelta(days=1)).date().isoformat(),
        "status": "active",
        "rule": rule,
        "rule_sha256": rule_sha256(rule) if isinstance(RULES.get(rule), RuleSpec) else "0" * 64,
        "params": {"minutes_before_kickoff": [60], "max_input_age_minutes": 120},
        **fields,
    }
    (directory / f"{name}.yaml").write_text(yaml.safe_dump(data))


async def add_event(
    session: Any, event_id: int, kickoff: datetime, away_team: str = "Chelsea"
) -> None:
    session.add(make_epl_event(event_id, kickoff, away_team=away_team))
    snapshot = make_tagged_snapshot(
        event_id,
        NOW - timedelta(hours=1),
        [three_way_bookmaker("bet365", (2.0, 3.4, 4.0)), EXCHANGE_ROW],
        "oddsportal_live",
    )
    snapshot.created_at = NOW - timedelta(hours=1)
    await store_tagged(session, snapshot)
    await session.commit()


async def run(session: Any, now: datetime = NOW) -> run_candidates.RunSummary:
    runnable = select_candidates(EPL, now).runnable
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
        await add_event(pglite_async_session, 1, DUE)

        summary = await run(pglite_async_session)

        assert summary.placed == 1
        [decision] = await all_of(pglite_async_session, CandidateDecision)
        assert (decision.candidate_name, decision.candidate_version, decision.event_id) == (
            "cand",
            1,
            1,
        )
        assert decision.decision_time == DUE - timedelta(minutes=60)
        assert decision.kickoff_at_decision == DUE
        assert decision.known_at == NOW
        assert (decision.bets_placed, decision.missed) == (1, False)
        assert len(decision.input_snapshot_ids) == 1
        [bet] = await all_of(pglite_async_session, Bet)
        assert (bet.candidate_name, bet.candidate_version) == ("cand", 1)
        assert bet.decision_id == decision.id
        assert bet.estimated_ev == ESTIMATED_EV
        assert (bet.selection, bet.bookmaker, bet.price_decimal) == ("home", "bet365", 2.0)
        assert bet.market == "1x2"
        assert bet.bankroll_before is None and bet.stake == 1.0

    async def test_polymarket_decision_records_both_inputs_and_replays_in_parity(
        self, pglite_async_session, registry
    ) -> None:
        params = {"threshold": 0.02, "minutes_before_kickoff": [60], "max_input_age_minutes": 120}
        write_candidate(registry, rule="polymarket_gap", params=params)
        pglite_async_session.add(make_epl_event(1, DUE))
        retail = make_tagged_snapshot(
            1,
            NOW - timedelta(hours=1),
            [three_way_bookmaker("bet365", (2.2, 3.4, 4.0))],
            "oddsportal_live",
        )
        polymarket = make_tagged_snapshot(
            1,
            NOW - timedelta(minutes=30),
            [polymarket_bookmaker((0.5, 0.25, 0.25))],
            "polymarket",
        )
        retail.created_at = NOW - timedelta(hours=1)
        polymarket.created_at = NOW - timedelta(minutes=30)
        await store_tagged(pglite_async_session, retail)
        await store_tagged(pglite_async_session, polymarket)
        await pglite_async_session.commit()

        summary = await run(pglite_async_session)

        assert summary.placed == 1
        [decision] = await all_of(pglite_async_session, CandidateDecision)
        assert decision.input_snapshot_ids == [retail.id, polymarket.id]
        [bet] = await all_of(pglite_async_session, Bet)
        assert (bet.selection, bet.bookmaker) == ("home", "bet365")
        cand = RuleCandidate.model_validate(
            {
                "name": "cand",
                "description": "d",
                "kind": "rule",
                "books": "uk_retail",
                "market": "1x2",
                "leagues": [EPL],
                "version": 1,
                "issue": 1,
                "start_date": "2026-01-01",
                "status": "active",
                "rule": "polymarket_gap",
                "rule_sha256": rule_sha256("polymarket_gap"),
                "params": params,
            }
        )
        report = await check_parity(pglite_async_session, cand)
        assert [d.status for d in report.decisions] == [ParityStatus.MATCH]
        assert report.ok

    async def test_late_decision_is_recorded_as_missed_and_logged(
        self, pglite_async_session, registry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        alert = AsyncMock()
        monkeypatch.setattr(run_candidates, "send_recorded_alert", alert)
        write_candidate(registry)
        await add_event(pglite_async_session, 1, LATE)
        assert NOW - (LATE - timedelta(minutes=60)) > MAX_LATENESS

        with structlog.testing.capture_logs() as logs:
            summary = await run(pglite_async_session)

        assert summary.missed == 1
        assert await all_of(pglite_async_session, Bet) == []
        [decision] = await all_of(pglite_async_session, CandidateDecision)
        assert (decision.missed, decision.bets_placed) == (True, 0)
        assert decision.input_snapshot_ids == []
        [missed] = [log for log in logs if log["event"] == "candidate_decision_missed"]
        assert missed["log_level"] == "warning"
        assert (missed["candidate"], missed["event_id"], missed["reason"]) == ("cand", 1, "late")
        alert.assert_not_awaited()

    async def test_decision_without_a_recent_snapshot_is_recorded_as_missed(
        self, pglite_async_session, registry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        alert = AsyncMock()
        monkeypatch.setattr(run_candidates, "send_recorded_alert", alert)
        write_candidate(
            registry, params={"minutes_before_kickoff": [60], "max_input_age_minutes": 10}
        )
        await add_event(pglite_async_session, 1, DUE)

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
        assert "decision on 1 at" in call.args[2] and "60 minutes before kick-off" in call.args[2]
        assert call.kwargs["every"] == run_candidates.stale_reminder_interval()

    async def test_rerun_places_nothing_new(self, pglite_async_session, registry) -> None:
        write_candidate(registry)
        await add_event(pglite_async_session, 1, DUE)
        await add_event(pglite_async_session, 2, LATE, away_team="Spurs")

        await run(pglite_async_session)
        again = await run(pglite_async_session, NOW + timedelta(minutes=1))

        assert (again.placed, again.missed, again.already_decided) == (0, 0, 0)
        assert len(await all_of(pglite_async_session, Bet)) == 1
        assert len(await all_of(pglite_async_session, CandidateDecision)) == 2

    async def test_zero_bet_decision_is_recorded_once(self, pglite_async_session, registry) -> None:
        write_candidate(registry, rule="no_bet")
        await add_event(pglite_async_session, 1, DUE)

        first = await run(pglite_async_session)
        second = await run(pglite_async_session)

        assert (first.skipped_zero_bet, second.skipped_zero_bet) == (1, 0)
        [decision] = await all_of(pglite_async_session, CandidateDecision)
        assert (decision.bets_placed, decision.missed) == (0, False)
        assert await all_of(pglite_async_session, Bet) == []

    async def test_event_not_yet_due_is_left_alone(self, pglite_async_session, registry) -> None:
        write_candidate(registry)
        await add_event(pglite_async_session, 1, NOT_DUE)

        await run(pglite_async_session)

        assert await all_of(pglite_async_session, CandidateDecision) == []

    async def test_started_event_is_recorded_as_missed(
        self, pglite_async_session, registry
    ) -> None:
        write_candidate(registry)
        await add_event(pglite_async_session, 1, NOW - timedelta(minutes=5))
        await add_event(pglite_async_session, 2, NOW - timedelta(days=8))

        summary = await run(pglite_async_session)

        assert (summary.missed, summary.placed) == (1, 0)
        [decision] = await all_of(pglite_async_session, CandidateDecision)
        assert (decision.event_id, decision.missed, decision.bets_placed) == (1, True, 0)
        assert await all_of(pglite_async_session, Bet) == []

    async def test_decision_time_before_the_start_date_is_not_decided(
        self, pglite_async_session, registry
    ) -> None:
        write_candidate(registry, start_date=NOW.date().isoformat())
        before_start = datetime.combine(NOW.date(), datetime.min.time(), tzinfo=UTC)
        await add_event(pglite_async_session, 1, before_start + timedelta(minutes=30))

        await run(pglite_async_session, before_start + timedelta(hours=2))

        assert await all_of(pglite_async_session, CandidateDecision) == []

    async def test_a_decision_committed_by_a_concurrent_run_is_not_a_failure(
        self, pglite_async_session, registry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        write_candidate(registry)
        await add_event(pglite_async_session, 1, DUE)
        real_decide = run_candidates._decide

        async def racing(session: Any, item: Any, event: Any, minutes: int, now: datetime) -> Any:
            # The other run's decision is committed between our check and our insert.
            session.add(
                CandidateDecision(
                    candidate_name="cand",
                    candidate_version=1,
                    leagues=["soccer_epl"],
                    event_id=1,
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
        await add_event(pglite_async_session, 1, DUE)
        spy = AsyncMock(side_effect=rules_module.load_market_state)
        monkeypatch.setattr(rules_module, "load_market_state", spy)

        await run(pglite_async_session)

        assert spy.await_args.args[2] == "1x2"
        assert spy.await_args.args[3] == DUE - timedelta(minutes=60)
        assert spy.await_args.args[4] == timedelta(minutes=120)
        assert spy.await_args.args[6] == BOOK_SETS[BookSet.UK_RETAIL]
        assert spy.await_args.args[7] == NOW

    async def test_failed_decision_is_not_recorded_and_others_run(
        self, pglite_async_session, registry
    ) -> None:
        write_candidate(registry, "a-twice", rule="twice")
        write_candidate(registry, "b-good")
        await add_event(pglite_async_session, 1, DUE)

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
        await add_event(pglite_async_session, 1, DUE)

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
        [bet] = await all_of(pglite_async_session, Bet)
        assert (bet.selection, bet.decision_id) == ("home", decisions[0].id)

    async def test_a_later_time_may_bet_a_different_outcome(
        self, pglite_async_session, registry
    ) -> None:
        write_candidate(registry, rule="bet_longest", params=self.TIMES)
        await add_event(pglite_async_session, 1, DUE)
        # Away is longest at the early time; this later snapshot makes the draw longest.
        later = make_tagged_snapshot(
            1,
            NOW - timedelta(minutes=5),
            [three_way_bookmaker("bet365", (1.5, 9.0, 3.0)), EXCHANGE_ROW],
            "oddsportal_live",
        )
        later.created_at = NOW - timedelta(minutes=5)
        await store_tagged(pglite_async_session, later)
        await pglite_async_session.commit()

        await run(pglite_async_session, self.EARLY_RUN)
        await run(pglite_async_session)

        bets = await all_of(pglite_async_session, Bet)
        assert sorted(t.selection for t in bets) == ["away", "draw"]

    async def test_a_kickoff_moved_after_a_decision_does_not_reopen_it(
        self, pglite_async_session, registry
    ) -> None:
        write_candidate(registry)
        await add_event(pglite_async_session, 1, DUE)
        await run(pglite_async_session)
        event = await pglite_async_session.get(Event, 1)
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
        await add_event(pglite_async_session, 1, NOW + timedelta(minutes=4))
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
        await add_event(pglite_async_session, 1, DUE)

        summary = await run(pglite_async_session)

        assert (summary.missed, summary.placed) == (1, 1)
        decisions = sorted(
            await all_of(pglite_async_session, CandidateDecision), key=lambda d: d.decision_time
        )
        assert [(d.missed, d.bets_placed) for d in decisions] == [(True, 0), (False, 1)]


class TestTotals:
    """An exchange-gap candidate on totals, deciding at 90 and 60 minutes before a kick-off
    59 minutes away."""

    PARAMS = {"threshold": 0.01, "minutes_before_kickoff": [90, 60], "max_input_age_minutes": 120}
    EARLY_RUN = DUE - timedelta(minutes=89)
    # Over 2.5 at 2.3 against a fair 0.5, and under 3.5 at 1.6 against a fair 0.69, clear the
    # threshold; over 3.5 at 3.0 and under 2.5 at 1.6 do not.
    RETAIL = {2.5: (2.3, 1.6), 3.5: (3.0, 1.6)}
    EXCHANGE = {2.5: (2.0, 2.0), 3.5: (3.2, 1.45)}

    @staticmethod
    async def add_snapshot(
        session: Any, retail: dict[float, tuple[float, float]], minutes_ago: int
    ) -> None:
        snapshot = make_tagged_snapshot(
            1,
            NOW - timedelta(minutes=minutes_ago),
            [totals_book("bet365", retail), totals_book("betfair_exchange", TestTotals.EXCHANGE)],
            "oddsportal_live",
        )
        snapshot.created_at = NOW - timedelta(minutes=minutes_ago)
        await store_tagged(session, snapshot)
        await session.commit()

    @staticmethod
    def candidate() -> RuleCandidate:
        return RuleCandidate.model_validate(
            {
                "name": "cand",
                "description": "d",
                "kind": "rule",
                "books": "uk_retail",
                "market": "totals",
                "leagues": [EPL],
                "version": 1,
                "issue": 1,
                "start_date": "2026-01-01",
                "status": "active",
                "rule": "exchange_gap",
                "rule_sha256": rule_sha256("exchange_gap"),
                "params": TestTotals.PARAMS,
            }
        )

    async def test_bets_each_selection_on_its_line_once_and_replays_in_parity(
        self, pglite_async_session, registry
    ) -> None:
        session = pglite_async_session
        write_candidate(registry, market="totals", rule="exchange_gap", params=self.PARAMS)
        session.add(make_epl_event(1, DUE))
        await self.add_snapshot(session, self.RETAIL, 60)

        early = await run(session, self.EARLY_RUN)
        # Over 3.5 lengthens to 3.5 before the later time: a new selection on the same outcome.
        await self.add_snapshot(session, {**self.RETAIL, 3.5: (3.5, 1.6)}, 5)
        late = await run(session)

        assert (early.placed, late.placed) == (2, 1)
        decisions = sorted(await all_of(session, CandidateDecision), key=lambda d: d.decision_time)
        bets = await all_of(session, Bet)
        assert sorted((t.selection, t.line, t.decision_id) for t in bets) == [
            ("over", 2.5, decisions[0].id),
            ("over", 3.5, decisions[1].id),
            ("under", 3.5, decisions[0].id),
        ]
        assert {(t.market, t.bookmaker) for t in bets} == {("totals", "bet365")}

        report = await check_parity(session, self.candidate())
        assert [d.status for d in report.decisions] == [ParityStatus.MATCH, ParityStatus.MATCH]
        assert report.decisions[1].live == (("over", 3.5, "bet365", 3.5, 1.0),)

    async def test_a_live_bet_on_another_line_is_a_parity_mismatch(
        self, pglite_async_session, registry
    ) -> None:
        session = pglite_async_session
        write_candidate(registry, market="totals", rule="exchange_gap", params=self.PARAMS)
        session.add(make_epl_event(1, DUE))
        await self.add_snapshot(session, self.RETAIL, 60)
        await run(session, self.EARLY_RUN)
        bet = next(t for t in await all_of(session, Bet) if t.selection == "over")
        bet.line = 4.5
        await session.commit()

        report = await check_parity(session, self.candidate())

        assert [d.status for d in report.decisions] == [ParityStatus.MISMATCH]

    async def test_bets_settle_on_the_total_score(self, pglite_async_session, registry) -> None:
        session = pglite_async_session
        write_candidate(registry, market="totals", rule="exchange_gap", params=self.PARAMS)
        session.add(make_epl_event(1, DUE))
        await self.add_snapshot(session, self.RETAIL, 60)
        await run(session, self.EARLY_RUN)
        event = await session.get(Event, 1)
        assert event is not None
        event.status, event.home_score, event.away_score = EventStatus.FINAL, 4, 0
        await session.commit()

        settled = await settle_bets(session)

        # Four goals: over 2.5 wins, under 3.5 loses.
        assert sorted((t.selection, t.result.value, t.pnl) for t in settled) == [
            ("over", "win", pytest.approx(1.3)),
            ("under", "loss", -1.0),
        ]


class TestCandidateSelection:
    async def test_inactive_unstarted_and_other_sport_candidates_are_ignored(
        self, registry
    ) -> None:
        write_candidate(registry, "active")
        write_candidate(registry, "killed", status="killed")
        write_candidate(registry, "promoted", status="promoted")
        write_candidate(registry, "future", start_date=(NOW + timedelta(days=1)).date().isoformat())
        write_candidate(registry, "mlb", leagues=["baseball_mlb"], market="h2h")

        runnable = select_candidates(EPL, NOW).runnable

        assert [r.candidate.name for r in runnable] == ["active"]

    async def test_a_pooled_candidate_runs_in_the_job_of_each_league_it_lists(
        self, registry
    ) -> None:
        write_candidate(registry, "pooled", leagues=[EPL, "soccer_italy_serie_a"])

        for sport, expected in ((EPL, ["pooled"]), ("soccer_italy_serie_a", ["pooled"])):
            runnable = select_candidates(sport, NOW).runnable
            assert [r.candidate.name for r in runnable] == expected
        assert select_candidates("soccer_efl_champ", NOW).runnable == []

    async def test_invalid_candidates_are_listed_and_others_still_run(self, registry) -> None:
        write_candidate(registry, "good")
        write_candidate(registry, "bad-params", params={"minutes_before_kickoff": [-1]})
        write_candidate(registry, "bad-hash", rule_sha256="0" * 64)
        write_candidate(registry, "bad-rule", rule="nonexistent", rule_sha256="0" * 64)

        selection = select_candidates(EPL, NOW)

        assert [r.candidate.name for r in selection.runnable] == ["good"]
        assert sorted(selection.invalid) == ["bad-hash", "bad-params", "bad-rule"]
        assert selection.registry_error is None

    async def test_a_model_rule_candidate_is_skipped_as_invalid(self, registry) -> None:
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
                "tuning_trials": 50,
                "tuning_seed": 42,
                "tuning_min_train_matches": 20,
                "tuning_step_matches": 10,
                "minutes_before_kickoff": [60],
                "max_input_age_minutes": 10,
            },
        )

        selection = select_candidates(EPL, NOW)

        assert [r.candidate.name for r in selection.runnable] == ["good"]
        assert "needs a fitted model" in selection.invalid["model"]

    async def test_unreadable_registry_runs_nothing(self, registry) -> None:
        (registry / "broken.yaml").write_text("name: [")

        selection = select_candidates(EPL, NOW)

        assert selection.runnable == []
        assert selection.registry_error


class TestRecordSelection:
    async def test_several_invalid_candidates_send_one_alert_naming_all(
        self, pglite_async_session, registry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        alert = AsyncMock()
        monkeypatch.setattr(run_candidates, "send_recorded_alert", alert)
        write_candidate(registry, "good")
        write_candidate(registry, "bad-params", params={"minutes_before_kickoff": [-1]})
        write_candidate(registry, "bad-hash", rule_sha256="0" * 64)
        write_candidate(registry, "bad-rule", rule="nonexistent", rule_sha256="0" * 64)

        await record_selection(pglite_async_session, EPL, select_candidates(EPL, NOW), NOW)

        [call] = alert.await_args_list
        assert call.args[0] == "candidates_invalid:bad-hash,bad-params,bad-rule"
        message = call.args[2]
        assert EPL in message
        for name in ("bad-hash", "bad-params", "bad-rule"):
            assert name in message
        assert "good" not in message
        [check] = await all_of(pglite_async_session, CandidateRegistryCheck)
        assert check.sport_key == EPL
        assert sorted(check.invalid_candidates) == ["bad-hash", "bad-params", "bad-rule"]
        assert check.registry_error is None

    async def test_a_registry_error_is_recorded_and_alerted(
        self, pglite_async_session, registry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        alert = AsyncMock()
        monkeypatch.setattr(run_candidates, "send_recorded_alert", alert)
        (registry / "broken.yaml").write_text("name: [")

        await record_selection(pglite_async_session, EPL, select_candidates(EPL, NOW), NOW)

        assert alert.await_args.args[0] == "candidates_invalid:registry"
        [check] = await all_of(pglite_async_session, CandidateRegistryCheck)
        assert check.registry_error

    async def test_a_clean_run_replaces_the_failed_check_and_sends_nothing(
        self, pglite_async_session, registry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        alert = AsyncMock()
        monkeypatch.setattr(run_candidates, "send_recorded_alert", alert)
        write_candidate(registry, "bad-hash", rule_sha256="0" * 64)
        await record_selection(pglite_async_session, EPL, select_candidates(EPL, NOW), NOW)
        (registry / "bad-hash.yaml").unlink()
        write_candidate(registry, "good")
        # The scheduler reads the fixed registry when it restarts.
        registry_snapshot.cache_clear()
        later = NOW + timedelta(minutes=30)

        await record_selection(pglite_async_session, EPL, select_candidates(EPL, later), later)

        assert alert.await_count == 1
        [check] = await all_of(pglite_async_session, CandidateRegistryCheck)
        assert check.invalid_candidates == {}
        assert check.checked_at == later


def _event(kickoff: datetime) -> Any:
    return SimpleNamespace(commence_time=kickoff)


def _runnable(*minutes: int) -> Any:
    return SimpleNamespace(
        params=FixtureRuleParams(minutes_before_kickoff=minutes, max_input_age_minutes=10)
    )


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
        monkeypatch.setattr(
            run_candidates,
            "select_candidates",
            lambda *a: run_candidates.CandidateSelection([], {}),
        )
        monkeypatch.setattr(run_candidates, "record_selection", AsyncMock())
        monkeypatch.setattr(run_candidates, "candidate_events", AsyncMock(return_value=[]))
        monkeypatch.setattr(run_candidates, "self_schedule", schedule)
        monkeypatch.setattr(run_candidates, "settle_bets", settle)
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
        monkeypatch.setattr(run_candidates, "settle_bets", AsyncMock(side_effect=ValueError("x")))
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
        monkeypatch.setattr(
            run_candidates,
            "select_candidates",
            lambda *a: run_candidates.CandidateSelection([], {}),
        )
        monkeypatch.setattr(run_candidates, "record_selection", AsyncMock())
        monkeypatch.setattr(run_candidates, "candidate_events", AsyncMock(return_value=[]))
        monkeypatch.setattr(run_candidates, "self_schedule", schedule)
        monkeypatch.setattr(run_candidates, "settle_bets", AsyncMock(return_value=[]))
        monkeypatch.setattr(run_candidates, "run_decisions", AsyncMock(return_value=summary))
        monkeypatch.setattr(run_candidates, "job_alert_context", alert_context)
        return schedule


class TestFlatStakeBets:
    async def test_portfolio_summary_excludes_flat_stake_bets(
        self, pglite_async_session, registry
    ) -> None:
        session = pglite_async_session
        write_candidate(registry)
        await add_event(session, 1, DUE)
        await place_bet(
            session,
            event_id=1,
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

        assert (summary.total_bets, summary.open_bets) == (1, 1)
        assert summary.current_bankroll == 990.0

    async def test_settlement_leaves_the_bankroll_empty(
        self, pglite_async_session, registry
    ) -> None:
        session = pglite_async_session
        write_candidate(registry)
        await add_event(session, 1, DUE)
        await run(session)
        event = await session.get(Event, 1)
        assert event is not None
        event.status = EventStatus.FINAL
        event.home_score, event.away_score = 2, 0
        await session.commit()

        [settled] = await settle_bets(session)

        assert settled.pnl == 1.0
        assert settled.bankroll_before is None and settled.bankroll_after is None

    async def test_a_bet_without_a_candidate_needs_a_bankroll(self, pglite_async_session) -> None:
        await add_event(pglite_async_session, 1, DUE)

        with pytest.raises(ValueError, match="needs a bankroll"):
            await place_bet(
                pglite_async_session,
                event_id=1,
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
        await add_event(session, 1, DUE)
        decision = CandidateDecision(
            candidate_name="cand",
            candidate_version=1,
            leagues=["soccer_epl"],
            event_id=1,
            minutes_before_kickoff=60,
            decision_time=NOW,
            kickoff_at_decision=DUE,
            known_at=NOW,
        )
        session.add(decision)
        await session.flush()

        def bet(decision_id: int | None, bookmaker: str = "bet365") -> Bet:
            return Bet(
                event_id=1,
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
        session.add_all([bet(None), bet(None), bet(decision.id)])
        await session.commit()

        session.add(bet(decision.id, bookmaker="betway"))
        with pytest.raises(IntegrityError):
            await session.commit()

    async def test_unique_index_takes_the_line_into_a_selection(self, pglite_async_session) -> None:
        session = pglite_async_session
        await add_event(session, 1, DUE)
        decision = CandidateDecision(
            candidate_name="cand",
            candidate_version=1,
            leagues=["soccer_epl"],
            event_id=1,
            minutes_before_kickoff=60,
            decision_time=NOW,
            kickoff_at_decision=DUE,
            known_at=NOW,
        )
        session.add(decision)
        await session.flush()

        def bet(line: float) -> Bet:
            return Bet(
                event_id=1,
                market="totals",
                selection="over",
                line=line,
                bookmaker="bet365",
                price_decimal=2.0,
                stake=1.0,
                candidate_name="cand",
                candidate_version=1,
                decision_id=decision.id,
            )

        session.add_all([bet(2.5), bet(3.5)])
        await session.commit()

        session.add(bet(2.5))
        with pytest.raises(IntegrityError):
            await session.commit()
