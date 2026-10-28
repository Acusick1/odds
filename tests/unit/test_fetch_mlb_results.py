"""Tests for the fetch-mlb-results job: settling MLB events by gamePk (#612)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from odds_core.models import Event, EventStatus
from odds_core.time import ensure_utc
from odds_lambda.jobs import fetch_mlb_results
from odds_lambda.jobs.fetch_mlb_results import process_results
from odds_lambda.mlb_game_resolver import mlb_date
from odds_lambda.mlb_stats_fetcher import MlbGameResult, MlbGameState
from odds_lambda.scheduling.jobs import JobContext
from sqlalchemy import select

from tests.test_helpers import hold_game_pks

HOME, AWAY = "Pittsburgh Pirates", "Milwaukee Brewers"
START = datetime.now(UTC).replace(hour=16, minute=5, second=0, microsecond=0) - timedelta(days=3)


def _event(
    event_id: str,
    game_pk: int | None,
    start: datetime = START,
    status: EventStatus = EventStatus.SCHEDULED,
) -> Event:
    return Event(
        id=event_id,
        sport_key="baseball_mlb",
        sport_title="MLB",
        commence_time=start,
        home_team=HOME,
        away_team=AWAY,
        status=status,
        mlb_game_pk=game_pk,
    )


def _slot(
    game_pk: int,
    start: datetime = START,
    state: MlbGameState = MlbGameState.FINAL,
    score: tuple[int, int] | None = (7, 6),
    game_number: int = 1,
    tbd: bool = False,
) -> MlbGameResult:
    final = state is MlbGameState.FINAL and score is not None
    return MlbGameResult(
        game_pk,
        start,
        HOME,
        AWAY,
        final,
        score[0] if final else None,
        score[1] if final else None,
        game_number=game_number,
        start_time_tbd=tbd,
        official_date=mlb_date(start),
        state=state,
        schedule_date=mlb_date(start),
    )


def _fetcher(schedule: list[MlbGameResult], by_pk: dict[int, list[MlbGameResult]] | None = None):
    fetcher = MagicMock()
    fetcher.fetch_results = AsyncMock(return_value=schedule)
    fetcher.fetch_game = AsyncMock(side_effect=lambda pk: (by_pk or {}).get(pk, []))
    return fetcher


def _session_maker(session) -> MagicMock:
    maker = MagicMock()
    maker.return_value.__aenter__ = AsyncMock(return_value=session)
    maker.return_value.__aexit__ = AsyncMock(return_value=None)
    return maker


async def _run(session, fetcher):
    with patch.object(fetch_mlb_results, "async_session_maker", _session_maker(session)):
        return await process_results(fetcher=fetcher)


async def _stored(session, event_id: str) -> Event:
    await session.rollback()
    return (await session.execute(select(Event).where(Event.id == event_id))).scalar_one()


class TestProcessResults:
    @pytest.mark.asyncio
    async def test_final_game_settles_the_event_with_mlb_score(self, test_session) -> None:
        test_session.add(_event("e", 823357, status=EventStatus.LIVE))
        await hold_game_pks(test_session)
        await test_session.commit()

        stats = await _run(test_session, _fetcher([_slot(823357, score=(7, 6))]))

        stored = await _stored(test_session, "e")
        assert (stored.status, stored.home_score, stored.away_score) == (EventStatus.FINAL, 7, 6)
        assert stored.completed_at is not None
        assert (stats.events_updated, stats.events_unresolved) == (1, 0)

    @pytest.mark.asyncio
    async def test_postponed_game_without_a_replay_marks_the_event_postponed(
        self, test_session
    ) -> None:
        test_session.add(_event("e", 5))
        await hold_game_pks(test_session)
        await test_session.commit()
        postponed = _slot(5, state=MlbGameState.POSTPONED, score=None)
        fetcher = _fetcher([postponed], {5: [postponed]})

        stats = await _run(test_session, fetcher)

        assert (await _stored(test_session, "e")).status == EventStatus.POSTPONED
        fetcher.fetch_game.assert_awaited_once_with(5)
        assert (stats.events_postponed, stats.events_unresolved) == (1, 0)

    @pytest.mark.asyncio
    async def test_postponed_slot_event_moves_to_the_played_slot_and_settles(
        self, test_session
    ) -> None:
        played = START + timedelta(days=1)
        test_session.add(_event("slot", 823357, status=EventStatus.POSTPONED))
        await hold_game_pks(test_session)
        await test_session.commit()
        schedule = [
            _slot(823357, state=MlbGameState.POSTPONED, score=None),
            _slot(823357, start=played, score=(7, 6)),
        ]

        stats = await _run(test_session, _fetcher(schedule))

        stored = await _stored(test_session, "slot")
        assert ensure_utc(stored.commence_time) == played
        assert (stored.status, stored.home_score, stored.away_score) == (EventStatus.FINAL, 7, 6)
        assert stats.events_updated == 1

    @pytest.mark.asyncio
    async def test_replay_booked_later_moves_the_event_there(self, test_session) -> None:
        replay = datetime.now(UTC).replace(microsecond=0) + timedelta(days=10)
        test_session.add(_event("slot", 9, status=EventStatus.POSTPONED))
        await hold_game_pks(test_session)
        await test_session.commit()
        postponed = _slot(9, state=MlbGameState.POSTPONED, score=None)
        fetcher = _fetcher(
            [postponed],
            {9: [postponed, _slot(9, start=replay, state=MlbGameState.SCHEDULED, score=None)]},
        )

        stats = await _run(test_session, fetcher)

        stored = await _stored(test_session, "slot")
        assert (ensure_utc(stored.commence_time), stored.status) == (replay, EventStatus.SCHEDULED)
        assert (stats.events_moved, stats.events_unresolved) == (1, 0)

    @pytest.mark.asyncio
    async def test_cancelled_game_cancels_the_event(self, test_session) -> None:
        test_session.add(_event("e", 6))
        await hold_game_pks(test_session)
        await test_session.commit()
        cancelled = _slot(6, state=MlbGameState.CANCELLED, score=None)

        stats = await _run(test_session, _fetcher([cancelled], {6: [cancelled]}))

        stored = await _stored(test_session, "e")
        assert (stored.status, stored.home_score) == (EventStatus.CANCELLED, None)
        assert stats.events_cancelled == 1

    @pytest.mark.asyncio
    async def test_event_without_a_game_pk_is_untouched(self, test_session) -> None:
        test_session.add(_event("allstar", None))
        await hold_game_pks(test_session)
        await test_session.commit()
        fetcher = _fetcher([_slot(1)])

        stats = await _run(test_session, fetcher)

        stored = await _stored(test_session, "allstar")
        assert (stored.status, stored.home_score) == (EventStatus.SCHEDULED, None)
        fetcher.fetch_results.assert_not_awaited()
        assert (stats.events_unresolved, stats.unresolved_kickoffs) == (1, [START])

    @pytest.mark.asyncio
    async def test_doubleheader_games_settle_their_own_events(self, test_session) -> None:
        game_two = START + timedelta(hours=4)
        test_session.add_all([_event("g1", 101), _event("g2", 102, start=game_two)])
        await hold_game_pks(test_session)
        await test_session.commit()
        schedule = [
            _slot(102, start=START + timedelta(minutes=6), score=(7, 6), game_number=2, tbd=True),
            _slot(101, score=(3, 2), game_number=1),
        ]

        stats = await _run(test_session, _fetcher(schedule))

        g1 = await _stored(test_session, "g1")
        assert (g1.home_score, g1.away_score) == (3, 2)
        g2 = await _stored(test_session, "g2")
        assert (g2.home_score, g2.away_score) == (7, 6)
        assert ensure_utc(g2.commence_time) == game_two
        assert stats.events_updated == 2

    @pytest.mark.asyncio
    async def test_game_in_progress_leaves_the_event_pending(self, test_session) -> None:
        test_session.add(_event("e", 7, status=EventStatus.LIVE))
        await hold_game_pks(test_session)
        await test_session.commit()

        stats = await _run(test_session, _fetcher([_slot(7, state=MlbGameState.LIVE)]))

        assert (await _stored(test_session, "e")).status == EventStatus.LIVE
        assert (stats.events_updated, stats.events_unresolved) == (0, 1)

    @pytest.mark.asyncio
    async def test_no_pending_events_skips_the_fetch(self, test_session) -> None:
        fetcher = _fetcher([])

        await _run(test_session, fetcher)

        fetcher.fetch_results.assert_not_awaited()


class TestMain:
    @pytest.mark.asyncio
    async def test_books_tomorrow_under_its_own_name_without_a_sport(self) -> None:
        from odds_lambda import results_run

        with (
            patch.object(
                fetch_mlb_results,
                "process_results",
                AsyncMock(return_value=results_run.ResultsStats()),
            ),
            patch.object(results_run, "record_run_outcome", AsyncMock()) as record,
            patch.object(results_run, "self_schedule", AsyncMock()) as book,
        ):
            await fetch_mlb_results.main(JobContext())

        assert record.await_args.args[:2] == ("fetch-mlb-results", "baseball_mlb")
        assert book.await_args.kwargs["job_name"] == "fetch-mlb-results"
        assert book.await_args.kwargs["sport"] is None
