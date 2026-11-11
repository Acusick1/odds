"""Fixture context: each team's previous result and the concurrent kick-offs, read from stored
events by one loader the runner, the replay backtest and the parity check share."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
import yaml
from odds_analytics.candidate_backtest import BacktestResult, replay
from odds_analytics.candidate_parity import ParityStatus, check_parity
from odds_core.candidates import load_candidate_file
from odds_core.market_prices import Outcome
from odds_core.models import Event, EventStatus
from odds_core.paper_trade_models import CandidateDecision
from odds_lambda.candidate_rules import (
    FixtureContext,
    FixtureInput,
    PreviousResult,
    TeamResult,
    load_fixture_context,
    run_rule,
)
from odds_lambda.candidate_rules.fixture_context import RESULT_STORE_LAG
from odds_lambda.candidate_rules.registry import RULES, resolve_rule
from odds_lambda.jobs.run_candidates import candidate_events, run_decisions, runnable_candidates
from odds_lambda.storage.event_merge import EventMerger
from odds_lambda.storage.writers import OddsWriter
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from tests.integration.conftest import AddMatch
from tests.integration.test_candidate_backtest import (
    KICKOFF,
    N_BOOT,
    T,
    candidate,
)
from tests.integration.test_candidate_backtest import (
    NOW as REPLAY_NOW,
)
from tests.integration.test_candidate_end_to_end import NOW, decide, seed
from tests.integration.test_candidate_parity import add_decision

pytestmark = pytest.mark.integration

BOTH = frozenset({FixtureInput.PREVIOUS_RESULT, FixtureInput.CONCURRENT_KICKOFFS})
PREVIOUS = frozenset({FixtureInput.PREVIOUS_RESULT})
CONCURRENT = frozenset({FixtureInput.CONCURRENT_KICKOFFS})
EPL = ("soccer_epl",)
SIX_DAYS_BEFORE = KICKOFF - timedelta(days=6)
FIVE_DAYS_BEFORE = KICKOFF - timedelta(days=5)


async def load(
    session: AsyncSession,
    inputs: frozenset[FixtureInput] = BOTH,
    *,
    leagues: Collection[str] = EPL,
    known_at: datetime | None = None,
    kickoff: datetime = KICKOFF,
    at: datetime = T,
) -> FixtureContext:
    event = await session.get(Event, 1)
    assert event is not None
    return await load_fixture_context(session, event, kickoff, at, inputs, leagues, known_at)


@pytest.fixture
async def match(add_match: AddMatch) -> Event:
    """Arsenal v Chelsea at ``KICKOFF``, the match the tests decide."""
    return await add_match(1, KICKOFF)


class TestPreviousResult:
    async def test_each_team_gets_its_previous_league_result(
        self, add_match: AddMatch, match: Event, pglite_async_session: AsyncSession
    ) -> None:
        await add_match(2, SIX_DAYS_BEFORE, "Arsenal", "Everton", score=(3, 0))
        await add_match(3, FIVE_DAYS_BEFORE, "Brentford", "Chelsea", score=(2, 1))

        fixture = await load(pglite_async_session, PREVIOUS)

        assert fixture.previous[Outcome.HOME] == PreviousResult(2, SIX_DAYS_BEFORE, 3, 0)
        assert fixture.previous[Outcome.AWAY] == PreviousResult(3, FIVE_DAYS_BEFORE, 1, 2)
        home, away = fixture.previous[Outcome.HOME], fixture.previous[Outcome.AWAY]
        assert home is not None and away is not None
        assert (home.goal_difference, home.result) == (3, TeamResult.WIN)
        assert (away.goal_difference, away.result) == (-1, TeamResult.LOSS)
        assert fixture.unknown_results == () and fixture.event_ids == (2, 3)

    async def test_the_latest_earlier_match_is_the_previous_one(
        self, add_match: AddMatch, match: Event, pglite_async_session: AsyncSession
    ) -> None:
        await add_match(2, SIX_DAYS_BEFORE, "Arsenal", "Everton", score=(3, 0))
        await add_match(3, FIVE_DAYS_BEFORE, "Arsenal", "Spurs", score=(0, 1))
        await add_match(4, KICKOFF + timedelta(days=5), "Arsenal", "Villa", score=(1, 0))

        previous = (await load(pglite_async_session, PREVIOUS)).previous[Outcome.HOME]

        assert previous is not None and (previous.event_id, previous.goal_difference) == (3, -1)

    async def test_a_cancelled_or_postponed_match_is_not_a_previous_one(
        self, add_match: AddMatch, match: Event, pglite_async_session: AsyncSession
    ) -> None:
        await add_match(2, SIX_DAYS_BEFORE, "Arsenal", "Everton", score=(3, 0))
        await add_match(3, FIVE_DAYS_BEFORE, "Arsenal", "Spurs", status="cancelled")
        await add_match(4, FIVE_DAYS_BEFORE, "Arsenal", "Villa", status="postponed")

        previous = (await load(pglite_async_session, PREVIOUS)).previous[Outcome.HOME]

        assert previous is not None and previous.event_id == 2

    async def test_another_leagues_match_is_not_a_previous_one(
        self, add_match: AddMatch, match: Event, pglite_async_session: AsyncSession
    ) -> None:
        await add_match(2, SIX_DAYS_BEFORE, "Arsenal", "Everton", score=(3, 0))
        await add_match(
            3, FIVE_DAYS_BEFORE, "Arsenal", "Spurs", score=(0, 1), sport_key="soccer_efl_champ"
        )

        previous = (await load(pglite_async_session, PREVIOUS)).previous[Outcome.HOME]

        assert previous is not None and previous.event_id == 2


class TestFirstMatchOfTheSeason:
    async def test_a_team_with_no_earlier_match_that_season_has_no_previous_result(
        self, add_match: AddMatch, match: Event, pglite_async_session: AsyncSession
    ) -> None:
        await add_match(2, SIX_DAYS_BEFORE, "Arsenal", "Everton", score=(3, 0))

        fixture = await load(pglite_async_session, PREVIOUS)

        assert fixture.previous[Outcome.AWAY] is None
        assert fixture.unknown_results == ()

    async def test_last_seasons_final_match_is_not_the_previous_result(
        self, add_match: AddMatch, pglite_async_session: AsyncSession
    ) -> None:
        opener = datetime(2025, 8, 16, 15, 0, tzinfo=UTC)
        await add_match(1, opener)
        await add_match(2, datetime(2025, 5, 25, 15, 0, tzinfo=UTC), score=(2, 2))

        fixture = await load(pglite_async_session, PREVIOUS, kickoff=opener, at=opener)

        assert dict(fixture.previous) == {Outcome.HOME: None, Outcome.AWAY: None}
        assert fixture.unknown_results == () and fixture.event_ids == ()

    async def test_the_season_opens_on_the_first_of_august(
        self, add_match: AddMatch, pglite_async_session: AsyncSession
    ) -> None:
        kickoff = datetime(2025, 8, 2, 15, 0, tzinfo=UTC)
        await add_match(1, kickoff)
        await add_match(2, datetime(2025, 7, 31, 15, 0, tzinfo=UTC), score=(1, 0))
        await add_match(3, datetime(2025, 8, 1, 15, 0, tzinfo=UTC), score=(0, 0))

        fixture = await load(pglite_async_session, PREVIOUS, kickoff=kickoff, at=kickoff)

        previous = fixture.previous[Outcome.HOME]
        assert previous is not None and previous.event_id == 3


class TestUnknownResult:
    async def test_a_previous_match_without_a_result_is_unknown_not_skipped(
        self, add_match: AddMatch, match: Event, pglite_async_session: AsyncSession
    ) -> None:
        await add_match(2, SIX_DAYS_BEFORE, "Arsenal", "Everton", score=(3, 0))
        await add_match(3, FIVE_DAYS_BEFORE, "Arsenal", "Spurs")
        await add_match(4, FIVE_DAYS_BEFORE, "Brentford", "Chelsea", score=(2, 1))

        fixture = await load(pglite_async_session, PREVIOUS)

        assert fixture.unknown_results == (Outcome.HOME,)
        assert fixture.previous[Outcome.HOME] is None and fixture.previous[Outcome.AWAY] is not None

    async def test_a_final_match_without_scores_is_unknown(
        self, add_match: AddMatch, match: Event, pglite_async_session: AsyncSession
    ) -> None:
        await add_match(2, FIVE_DAYS_BEFORE, "Arsenal", "Spurs", status="final")

        assert (await load(pglite_async_session, PREVIOUS)).unknown_results == (Outcome.HOME,)


class TestPendingOrOverdue:
    async def test_a_match_played_since_the_decision_is_pending(
        self, add_match: AddMatch, match: Event, pglite_async_session: AsyncSession
    ) -> None:
        await add_match(2, KICKOFF - timedelta(minutes=30), "Arsenal", "Everton")

        fixture = await load(pglite_async_session, PREVIOUS)

        assert fixture.pending_results == (Outcome.HOME,) and fixture.overdue_results == ()

    async def test_a_result_not_stored_within_the_store_lag_is_overdue(
        self, add_match: AddMatch, match: Event, pglite_async_session: AsyncSession
    ) -> None:
        await add_match(2, T - RESULT_STORE_LAG, "Arsenal", "Everton")
        await add_match(3, T - RESULT_STORE_LAG + timedelta(minutes=1), "Chelsea", "Spurs")

        fixture = await load(pglite_async_session, PREVIOUS)

        assert fixture.overdue_results == (Outcome.HOME,)
        assert fixture.pending_results == (Outcome.AWAY,)


class TestNoLookAhead:
    """T is 60 minutes before the kick-off and the previous match's result lands in the past."""

    async def test_a_result_stored_after_the_decision_is_not_known(
        self, add_match: AddMatch, match: Event, pglite_async_session: AsyncSession
    ) -> None:
        await add_match(
            2,
            SIX_DAYS_BEFORE,
            "Arsenal",
            "Everton",
            score=(3, 0),
            completed_at=T + timedelta(minutes=5),
        )
        known_at = T + timedelta(minutes=1)

        before = await load(pglite_async_session, PREVIOUS, known_at=known_at)
        after = await load(
            pglite_async_session,
            PREVIOUS,
            known_at=T + timedelta(minutes=5),
            at=T + timedelta(minutes=5),
        )

        assert before.unknown_results == (Outcome.HOME,)
        assert after.unknown_results == () and after.previous[Outcome.HOME] is not None

    async def test_a_result_with_no_completion_time_is_not_known_by_a_run(
        self, add_match: AddMatch, match: Event, pglite_async_session: AsyncSession
    ) -> None:
        event = await add_match(2, SIX_DAYS_BEFORE, "Arsenal", "Everton", score=(3, 0))
        event.completed_at = None

        run = await load(pglite_async_session, PREVIOUS, known_at=T)
        replay = await load(pglite_async_session, PREVIOUS)

        assert run.unknown_results == (Outcome.HOME,)
        assert replay.unknown_results == () and replay.previous[Outcome.HOME] is not None

    async def test_a_replay_counts_a_result_stored_days_after_the_decision(
        self, add_match: AddMatch, match: Event, pglite_async_session: AsyncSession
    ) -> None:
        # A backfill stores results long after their matches.
        await add_match(
            2,
            SIX_DAYS_BEFORE,
            "Arsenal",
            "Everton",
            score=(0, 3),
            completed_at=T + timedelta(days=30),
        )

        run = await load(pglite_async_session, PREVIOUS, known_at=T)
        replay = await load(pglite_async_session, PREVIOUS, known_at=None)

        assert run.unknown_results == (Outcome.HOME,)
        assert replay.unknown_results == ()
        assert replay.previous[Outcome.HOME] == PreviousResult(2, SIX_DAYS_BEFORE, 0, 3)

    async def test_a_match_finishing_after_the_decision_is_not_known(
        self, add_match: AddMatch, match: Event, pglite_async_session: AsyncSession
    ) -> None:
        # Kicked off 2 hours before the decision: still short of the 3 hours that cover full
        # time, though it is FINAL now.
        await add_match(
            2, T - timedelta(hours=2), "Arsenal", "Everton", score=(3, 0), completed_at=T
        )

        assert (await load(pglite_async_session, PREVIOUS)).unknown_results == (Outcome.HOME,)

    async def test_a_match_kicked_off_three_hours_before_is_known(
        self, add_match: AddMatch, match: Event, pglite_async_session: AsyncSession
    ) -> None:
        await add_match(2, T - timedelta(hours=3), "Arsenal", "Everton", score=(3, 0))
        await add_match(3, T - timedelta(hours=3, minutes=-1), "Chelsea", "Spurs", score=(3, 0))

        fixture = await load(pglite_async_session, PREVIOUS)

        assert fixture.unknown_results == (Outcome.AWAY,)
        assert fixture.previous[Outcome.HOME] is not None

    async def test_an_event_stored_after_the_run_is_not_a_fixture(
        self, add_match: AddMatch, match: Event, pglite_async_session: AsyncSession
    ) -> None:
        await add_match(2, SIX_DAYS_BEFORE, "Arsenal", "Everton", score=(3, 0))
        await add_match(
            3,
            FIVE_DAYS_BEFORE,
            "Arsenal",
            "Spurs",
            score=(0, 1),
            stored_at=T + timedelta(minutes=5),
        )

        fixture = await load(pglite_async_session, known_at=T + timedelta(minutes=1))

        previous = fixture.previous[Outcome.HOME]
        assert previous is not None and previous.event_id == 2
        assert fixture.event_ids == (2,)


class TestConcurrentKickoffs:
    async def test_the_count_includes_the_match_itself(
        self, match: Event, pglite_async_session: AsyncSession
    ) -> None:
        assert (await load(pglite_async_session, CONCURRENT)).concurrent_kickoffs == 1

    @pytest.mark.parametrize(
        ("offset", "counted"),
        [
            (timedelta(minutes=10), True),
            (timedelta(minutes=-10), True),
            (timedelta(minutes=10, seconds=1), False),
            (timedelta(minutes=-10, seconds=-1), False),
            (timedelta(0), True),
        ],
    )
    async def test_the_boundary_is_ten_minutes_inclusive(
        self,
        offset: timedelta,
        counted: bool,
        add_match: AddMatch,
        match: Event,
        pglite_async_session: AsyncSession,
    ) -> None:
        await add_match(2, KICKOFF + offset, "Villa", "Everton")

        fixture = await load(pglite_async_session, CONCURRENT)

        assert fixture.concurrent_kickoffs == (2 if counted else 1)
        assert fixture.event_ids == ((2,) if counted else ())

    async def test_the_league_set_is_a_parameter(
        self, add_match: AddMatch, match: Event, pglite_async_session: AsyncSession
    ) -> None:
        await add_match(2, KICKOFF, "Villa", "Everton")
        await add_match(3, KICKOFF, "Real Madrid", "Girona", sport_key="soccer_spain_la_liga")
        await add_match(4, KICKOFF, "Inter", "Roma", sport_key="soccer_italy_serie_a")

        own = await load(pglite_async_session, CONCURRENT)
        pooled = await load(
            pglite_async_session,
            CONCURRENT,
            leagues=("soccer_epl", "soccer_spain_la_liga", "soccer_italy_serie_a"),
        )

        assert own.concurrent_kickoffs == 2
        assert pooled.concurrent_kickoffs == 4 and pooled.event_ids == (2, 3, 4)

    async def test_a_cancelled_match_is_not_concurrent(
        self, add_match: AddMatch, match: Event, pglite_async_session: AsyncSession
    ) -> None:
        await add_match(2, KICKOFF, "Villa", "Everton", status="cancelled")

        assert (await load(pglite_async_session, CONCURRENT)).concurrent_kickoffs == 1

    async def test_the_recorded_kickoff_places_the_window(
        self, add_match: AddMatch, match: Event, pglite_async_session: AsyncSession
    ) -> None:
        # The event has since moved; the decision saw it at 17:00.
        later = KICKOFF + timedelta(hours=2)
        await add_match(2, later, "Villa", "Everton")

        fixture = await load(pglite_async_session, CONCURRENT, kickoff=later)

        assert fixture.concurrent_kickoffs == 2


class TestReadOnly:
    async def test_the_context_cannot_be_changed(
        self, add_match: AddMatch, match: Event, pglite_async_session: AsyncSession
    ) -> None:
        await add_match(2, SIX_DAYS_BEFORE, "Arsenal", "Everton", score=(3, 0))
        fixture = await load(pglite_async_session)

        with pytest.raises(FrozenInstanceError):
            fixture.concurrent_kickoffs = 9  # type: ignore[misc]
        with pytest.raises(TypeError):
            fixture.previous[Outcome.HOME] = None  # type: ignore[index]
        home = fixture.previous[Outcome.HOME]
        assert home is not None
        with pytest.raises(FrozenInstanceError):
            home.goals_for = 9  # type: ignore[misc]

    async def test_loading_writes_nothing(
        self, add_match: AddMatch, match: Event, pglite_async_session: AsyncSession
    ) -> None:
        await add_match(2, SIX_DAYS_BEFORE, "Arsenal", "Everton", score=(3, 0))
        await pglite_async_session.commit()
        before = (await pglite_async_session.execute(select(Event))).scalars().all()
        snapshot = [(e.id, e.status, e.home_score, e.updated_at) for e in before]

        await load(pglite_async_session)

        assert not pglite_async_session.dirty and not pglite_async_session.new
        after = (await pglite_async_session.execute(select(Event))).scalars().all()
        assert [(e.id, e.status, e.home_score, e.updated_at) for e in after] == snapshot


@pytest.fixture
def declares_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(RULES, "exchange_gap", replace(RULES["exchange_gap"], fixture_inputs=BOTH))


@pytest.fixture
def registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, declares_fixture: None) -> Path:
    monkeypatch.setattr("odds_lambda.jobs.run_candidates.candidates_dir", lambda: tmp_path)
    monkeypatch.setattr("odds_lambda.paper_trading.candidates_dir", lambda: tmp_path)
    (tmp_path / "gap.yaml").write_text(yaml.safe_dump(candidate().model_dump(mode="json")))
    assert load_candidate_file(tmp_path / "gap.yaml").name == "gap"
    return tmp_path


async def seed_matches(add_match: AddMatch, *, home_result: bool = True) -> None:
    """The previous matches of event 1's teams and a concurrent kick-off, all stored and
    completed before the run."""
    if home_result:
        await add_match(2, SIX_DAYS_BEFORE, "Arsenal", "Everton", score=(3, 0))
    await add_match(3, FIVE_DAYS_BEFORE, "Brentford", "Chelsea", score=(2, 1))
    await add_match(4, KICKOFF, "Villa", "Everton")


async def decisions_of_event_one(session: AsyncSession) -> list[CandidateDecision]:
    rows = await session.execute(select(CandidateDecision).where(CandidateDecision.event_id == 1))
    return list(rows.scalars())


class TestDecidingRule:
    async def test_a_declaring_rule_receives_both_inputs(
        self,
        pglite_async_session: AsyncSession,
        add_match: AddMatch,
        registry: Path,
    ) -> None:
        await seed(pglite_async_session, 1, (2.2, 3.0, 3.0))
        await seed_matches(add_match)
        await pglite_async_session.commit()
        item = (await runnable_candidates("soccer_epl", NOW))[0]
        event = await pglite_async_session.get(Event, 1)
        assert event is not None

        state, intents = await run_rule(
            pglite_async_session, item.spec, item.params, event, item.candidate.market, T, NOW
        )

        assert state.has_input and len(intents) == 1
        assert state.fixture.concurrent_kickoffs == 2
        assert state.fixture.previous[Outcome.HOME] is not None
        assert state.input_event_ids == [2, 3, 4]

    async def test_a_rule_declaring_nothing_gets_an_empty_context(
        self, pglite_async_session: AsyncSession, add_match: AddMatch
    ) -> None:
        await seed(pglite_async_session, 1, (2.2, 3.0, 3.0))
        await seed_matches(add_match)
        await pglite_async_session.commit()
        spec, params = resolve_rule(candidate())
        event = await pglite_async_session.get(Event, 1)
        assert event is not None

        state, intents = await run_rule(
            pglite_async_session,
            spec,
            params,
            event,
            candidate().market,
            T,
            NOW,  # type: ignore[arg-type]
        )

        assert state.has_input and len(intents) == 1
        assert state.fixture == FixtureContext() and state.input_event_ids == []

    async def test_the_runner_records_the_events_it_read(
        self, pglite_async_session: AsyncSession, add_match: AddMatch, registry: Path
    ) -> None:
        await seed(pglite_async_session, 1, (2.2, 3.0, 3.0))
        await seed_matches(add_match)
        await pglite_async_session.commit()

        await decide(pglite_async_session)

        [decision] = await decisions_of_event_one(pglite_async_session)
        assert not decision.missed and decision.bets_placed == 1
        assert decision.input_event_ids == [2, 3, 4]
        assert decision.input_fixture == {
            "previous": {
                "home": {
                    "event_id": 2,
                    "kickoff": SIX_DAYS_BEFORE.isoformat(),
                    "goals_for": 3,
                    "goals_against": 0,
                },
                "away": {
                    "event_id": 3,
                    "kickoff": FIVE_DAYS_BEFORE.isoformat(),
                    "goals_for": 1,
                    "goals_against": 2,
                },
            },
            "concurrent_kickoffs": 2,
        }

    async def _decide_with_home_match(
        self, session: AsyncSession, add_match: AddMatch, kickoff: datetime
    ) -> Any:
        await seed(session, 1, (2.2, 3.0, 3.0))
        await add_match(2, kickoff, "Arsenal", "Everton")
        await add_match(3, FIVE_DAYS_BEFORE, "Brentford", "Chelsea", score=(2, 1))
        await session.commit()
        runnable = await runnable_candidates("soccer_epl", NOW)
        events = await candidate_events(session, "soccer_epl", NOW)
        return await run_decisions(session, runnable, events, NOW)

    async def test_an_overdue_result_misses_the_decision_and_alerts(
        self,
        pglite_async_session: AsyncSession,
        add_match: AddMatch,
        registry: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        alert = AsyncMock()
        monkeypatch.setattr("odds_lambda.jobs.run_candidates.send_recorded_alert", alert)

        await self._decide_with_home_match(pglite_async_session, add_match, SIX_DAYS_BEFORE)

        [decision] = await decisions_of_event_one(pglite_async_session)
        assert decision.missed and decision.bets_placed == 0
        assert decision.input_event_ids == [] and decision.input_snapshot_ids == []
        assert decision.input_fixture is None
        assert [c.args[0] for c in alert.await_args_list if "fixture" in c.args[0]] == [
            "candidate_no_fixture_context:gap:60"
        ]

    async def test_a_pending_result_misses_the_decision_without_an_alert(
        self,
        pglite_async_session: AsyncSession,
        add_match: AddMatch,
        registry: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        alert = AsyncMock()
        monkeypatch.setattr("odds_lambda.jobs.run_candidates.send_recorded_alert", alert)

        await self._decide_with_home_match(pglite_async_session, add_match, T - timedelta(hours=10))

        [decision] = await decisions_of_event_one(pglite_async_session)
        assert decision.missed and decision.bets_placed == 0
        assert not [c for c in alert.await_args_list if "fixture" in c.args[0]]

    async def test_a_first_match_of_the_season_is_decided(
        self, pglite_async_session: AsyncSession, add_match: AddMatch, registry: Path
    ) -> None:
        await seed(pglite_async_session, 1, (2.2, 3.0, 3.0))
        await seed_matches(add_match, home_result=False)
        await pglite_async_session.commit()

        await decide(pglite_async_session)

        [decision] = await decisions_of_event_one(pglite_async_session)
        assert not decision.missed and decision.bets_placed == 1


class TestReplayAndParity:
    async def _replay_event_one(
        self, session: AsyncSession, add_match: AddMatch, *, arsenal_result: bool
    ) -> BacktestResult:
        from tests.integration.test_candidate_backtest import add_event

        await add_event(session, 1, KICKOFF)
        event = await session.get(Event, 1)
        assert event is not None
        event.status, event.home_score, event.away_score = "final", 2, 1  # type: ignore[assignment]
        await add_match(
            2, SIX_DAYS_BEFORE, "Arsenal", "Everton", score=(3, 0) if arsenal_result else None
        )
        await session.commit()
        return await replay(
            session,
            candidate(),
            KICKOFF.date() - timedelta(days=9),
            KICKOFF.date() + timedelta(days=20),
            1,
            now=REPLAY_NOW,
            n_boot=N_BOOT,
            only_events=[1],
        )

    async def test_the_replay_decides_through_the_same_loader(
        self, pglite_async_session: AsyncSession, add_match: AddMatch, declares_fixture: None
    ) -> None:
        result = await self._replay_event_one(pglite_async_session, add_match, arsenal_result=True)

        assert (result.coverage.decisions, result.coverage.with_input) == (1, 1)

    async def test_the_replay_misses_a_decision_whose_previous_result_is_unknown(
        self, pglite_async_session: AsyncSession, add_match: AddMatch, declares_fixture: None
    ) -> None:
        result = await self._replay_event_one(pglite_async_session, add_match, arsenal_result=False)

        assert (result.coverage.decisions, result.coverage.with_input) == (1, 0)

    async def test_every_decision_matches(
        self, pglite_async_session: AsyncSession, add_match: AddMatch, registry: Path
    ) -> None:
        await seed(pglite_async_session, 1, (2.2, 3.0, 3.0))
        await seed_matches(add_match)
        await pglite_async_session.commit()
        await decide(pglite_async_session)

        report = await check_parity(pglite_async_session, candidate())

        decisions = [d for d in report.decisions if d.event_id == 1]
        assert [d.status for d in decisions] == [ParityStatus.MATCH]
        assert report.data_differences == []

    async def test_parity_replays_from_the_recorded_events(
        self, pglite_async_session: AsyncSession, add_match: AddMatch, registry: Path
    ) -> None:
        await seed(pglite_async_session, 1, (2.2, 3.0, 3.0))
        await seed_matches(add_match)
        await pglite_async_session.commit()
        await decide(pglite_async_session)
        # A fixture stored later, which the data replay reads and the recorded one does not.
        await add_match(9, KICKOFF - timedelta(days=1), "Arsenal", "Spurs", score=(0, 4))
        await pglite_async_session.commit()

        [d] = [
            x
            for x in (await check_parity(pglite_async_session, candidate())).decisions
            if x.event_id == 1
        ]

        assert d.status is ParityStatus.MATCH

    async def test_parity_reports_a_recorded_event_that_is_gone(
        self, pglite_async_session: AsyncSession, add_match: AddMatch, registry: Path
    ) -> None:
        await seed(pglite_async_session, 1, (2.2, 3.0, 3.0))
        await seed_matches(add_match)
        await pglite_async_session.commit()
        await decide(pglite_async_session)
        [recorded] = await decisions_of_event_one(pglite_async_session)
        recorded.input_event_ids = [*recorded.input_event_ids, 999]
        await pglite_async_session.commit()

        [d] = [
            x
            for x in (await check_parity(pglite_async_session, candidate())).decisions
            if x.event_id == 1
        ]

        assert d.status is ParityStatus.MISMATCH and d.missing_event_ids == (999,)


@pytest.fixture
def home_won_big(monkeypatch: pytest.MonkeyPatch) -> None:
    """A rule that bets ``exchange_gap``'s bets only when the home team's previous match was
    won by 3 goals and two kick-offs share the match's slot."""
    from odds_lambda.candidate_rules.gap import GapParams, exchange_gap

    def rule(state: Any, params: GapParams) -> Any:
        previous = state.fixture.previous[Outcome.HOME]
        if previous is None or previous.goal_difference != 3:
            return []
        return exchange_gap(state, params) if state.fixture.concurrent_kickoffs == 2 else []

    monkeypatch.setitem(
        RULES,
        "exchange_gap",
        replace(RULES["exchange_gap"], function=rule, fixture_inputs=BOTH),
    )
    # The rule lives in this module, which the code hash cannot follow.
    monkeypatch.setattr("tests.integration.test_candidate_backtest.rule_sha256", lambda _: "a" * 64)


RECORDED_FIXTURE = {
    "previous": {
        "home": {
            "event_id": 2,
            "kickoff": SIX_DAYS_BEFORE.isoformat(),
            "goals_for": 3,
            "goals_against": 0,
        },
        "away": None,
    },
    "concurrent_kickoffs": 2,
}


class TestParityReplaysRecordedValues:
    async def _decision(self, session: AsyncSession, add_match: AddMatch, **fixture: Any) -> Any:
        from tests.integration.test_candidate_parity import add_event

        ids = await add_event(session)
        await add_match(2, SIX_DAYS_BEFORE, "Arsenal", "Everton", score=(3, 0))
        await add_match(4, KICKOFF, "Villa", "Everton")
        await add_decision(
            session,
            ids,
            [("home", "bet365", 2.2)],
            event_ids=[2, 4],
            fixture=fixture.get("record", RECORDED_FIXTURE),
        )
        await session.commit()

    async def _parity(self, session: AsyncSession) -> Any:
        report = await check_parity(session, candidate())
        [d] = report.decisions
        return d

    async def test_the_recorded_values_match(
        self, pglite_async_session: AsyncSession, add_match: AddMatch, home_won_big: None
    ) -> None:
        await self._decision(pglite_async_session, add_match)

        assert (await self._parity(pglite_async_session)).status is ParityStatus.MATCH

    async def test_a_corrected_score_leaves_the_replay_unchanged(
        self, pglite_async_session: AsyncSession, add_match: AddMatch, home_won_big: None
    ) -> None:
        await self._decision(pglite_async_session, add_match)
        await OddsWriter(pglite_async_session).update_event_status(
            2, EventStatus.FINAL, home_score=0, away_score=3
        )
        await pglite_async_session.commit()

        d = await self._parity(pglite_async_session)

        assert d.status is ParityStatus.MATCH and d.data_difference

    async def test_a_rescheduled_concurrent_match_leaves_the_replay_unchanged(
        self, pglite_async_session: AsyncSession, add_match: AddMatch, home_won_big: None
    ) -> None:
        await self._decision(pglite_async_session, add_match)
        other = await pglite_async_session.get(Event, 4)
        assert other is not None
        other.commence_time += timedelta(hours=3)
        await pglite_async_session.commit()

        assert (await self._parity(pglite_async_session)).status is ParityStatus.MATCH

    async def test_a_merge_into_a_newer_survivor_leaves_the_replay_unchanged(
        self, pglite_async_session: AsyncSession, add_match: AddMatch, home_won_big: None
    ) -> None:
        await self._decision(pglite_async_session, add_match)
        old = await pglite_async_session.get(Event, 4)
        newer = await add_match(5, KICKOFF, "Villa", "Everton", stored_at=T + timedelta(hours=2))
        assert old is not None
        await EventMerger(pglite_async_session).merge(newer, [old])
        await pglite_async_session.commit()

        d = await self._parity(pglite_async_session)

        assert d.status is ParityStatus.MATCH and d.missing_event_ids == ()

    async def test_a_different_record_replays_different_bets(
        self, pglite_async_session: AsyncSession, add_match: AddMatch, home_won_big: None
    ) -> None:
        record = {**RECORDED_FIXTURE, "concurrent_kickoffs": 1}
        await self._decision(pglite_async_session, add_match, record=record)

        d = await self._parity(pglite_async_session)

        assert d.status is ParityStatus.MISMATCH and d.replayed == ()


class TestRecordRoundTrip:
    async def test_a_context_survives_its_record(self) -> None:
        record = FixtureContext.from_record(RECORDED_FIXTURE, [4, 2])
        previous = record.previous[Outcome.HOME]
        assert previous == PreviousResult(2, SIX_DAYS_BEFORE, 3, 0)
        assert record.previous[Outcome.AWAY] is None and record.concurrent_kickoffs == 2
        assert record.event_ids == (2, 4) and record.to_record() == RECORDED_FIXTURE

    async def test_nothing_declared_records_nothing(self) -> None:
        assert FixtureContext().to_record() is None
