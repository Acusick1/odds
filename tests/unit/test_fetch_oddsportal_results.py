"""Tests for fetch_oddsportal_results job."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from odds_core.models import Event, EventStatus
from odds_lambda.jobs.fetch_oddsportal_results import (
    DEFAULT_SPORT_KEY,
    ResultsStats,
    get_pending_events,
    process_results,
    record_run_outcome,
)
from odds_lambda.oddsportal_client import (
    CancelledMatch,
    FetchedResults,
    FinishedMatch,
    OddsPortalError,
)
from sqlalchemy import select

KICKOFF = datetime.now(UTC).replace(minute=0, second=0, microsecond=0) - timedelta(days=3)


def _make_event(
    *,
    event_id: str = "test_event_1",
    home_team: str = "Arsenal",
    away_team: str = "Chelsea",
    commence_time: datetime = KICKOFF,
    status: EventStatus = EventStatus.SCHEDULED,
    scores: tuple[int, int] | None = None,
    sport_key: str = "soccer_epl",
) -> Event:
    home_score, away_score = scores if scores is not None else (None, None)
    return Event(
        id=event_id,
        sport_key=sport_key,
        sport_title="EPL",
        commence_time=commence_time,
        home_team=home_team,
        away_team=away_team,
        status=status,
        home_score=home_score,
        away_score=away_score,
    )


def _make_match(
    *,
    home_team: str = "Arsenal",
    away_team: str = "Chelsea",
    start_time: datetime = KICKOFF,
    home_score: int = 2,
    away_score: int = 1,
) -> FinishedMatch:
    return FinishedMatch(
        home_team=home_team,
        away_team=away_team,
        start_time=start_time,
        home_score=home_score,
        away_score=away_score,
    )


class TestGetPendingEvents:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", [EventStatus.SCHEDULED, EventStatus.LIVE])
    async def test_includes_unsettled_events(self, test_session, status: EventStatus) -> None:
        event = _make_event(status=status)
        test_session.add(event)
        await test_session.commit()

        result = await get_pending_events(test_session)
        assert [p.id for p in result] == [event.id]

    @pytest.mark.asyncio
    async def test_includes_final_events_without_scores(self, test_session) -> None:
        event = _make_event(status=EventStatus.FINAL)
        test_session.add(event)
        await test_session.commit()

        result = await get_pending_events(test_session)
        assert [p.id for p in result] == [event.id]

    @pytest.mark.asyncio
    async def test_excludes_final_events_with_scores(self, test_session) -> None:
        test_session.add(_make_event(status=EventStatus.FINAL, scores=(1, 0)))
        await test_session.commit()

        assert await get_pending_events(test_session) == []

    @pytest.mark.asyncio
    async def test_excludes_future_events(self, test_session) -> None:
        test_session.add(_make_event(commence_time=datetime.now(UTC) + timedelta(days=30)))
        await test_session.commit()

        assert await get_pending_events(test_session) == []

    @pytest.mark.asyncio
    async def test_excludes_other_sports(self, test_session) -> None:
        test_session.add(_make_event(sport_key="baseball_mlb"))
        await test_session.commit()

        assert await get_pending_events(test_session, sport_key="soccer_epl") == []


def _session_maker(session) -> MagicMock:
    maker = MagicMock()
    maker.return_value.__aenter__ = AsyncMock(return_value=session)
    maker.return_value.__aexit__ = AsyncMock(return_value=None)
    return maker


def _fetched(
    *matches: FinishedMatch,
    rows_seen: int | None = None,
    cancelled: Sequence[CancelledMatch] = (),
) -> FetchedResults:
    return FetchedResults(
        list(matches),
        rows_seen or len(matches) + len(cancelled),
        complete=True,
        cancelled=list(cancelled),
    )


async def _stored(session, event_id: str) -> Event:
    await session.rollback()
    result = await session.execute(select(Event).where(Event.id == event_id))
    return result.scalar_one()


class TestProcessResults:
    @pytest.mark.asyncio
    async def test_settles_matched_events_and_counts_the_rest(self, test_session) -> None:
        test_session.add(_make_event(event_id="matched"))
        test_session.add(_make_event(event_id="no_result", home_team="Fulham", away_team="Burnley"))
        test_session.add(
            _make_event(
                event_id="scoreless_final",
                home_team="Leeds",
                away_team="Everton",
                status=EventStatus.FINAL,
            )
        )
        await test_session.commit()

        matches = [
            _make_match(),
            _make_match(home_team="Leeds", away_team="Everton", home_score=0, away_score=0),
        ]
        with patch(
            "odds_lambda.jobs.fetch_oddsportal_results.async_session_maker",
            _session_maker(test_session),
        ):
            stats = await process_results(fetched=_fetched(*matches))

        assert (stats.matches_fetched, stats.events_updated, stats.events_unresolved) == (2, 2, 1)
        matched = await _stored(test_session, "matched")
        assert (matched.status, matched.home_score, matched.away_score) == (
            EventStatus.FINAL,
            2,
            1,
        )
        assert matched.completed_at is not None
        scoreless = await _stored(test_session, "scoreless_final")
        assert (scoreless.home_score, scoreless.away_score) == (0, 0)
        unresolved = await _stored(test_session, "no_result")
        assert (unresolved.status, unresolved.home_score) == (EventStatus.SCHEDULED, None)

    @pytest.mark.asyncio
    async def test_postponed_event_that_is_played_reaches_final(self, test_session) -> None:
        test_session.add(_make_event(event_id="late", status=EventStatus.POSTPONED))
        await test_session.commit()

        with patch(
            "odds_lambda.jobs.fetch_oddsportal_results.async_session_maker",
            _session_maker(test_session),
        ):
            stats = await process_results(fetched=_fetched(_make_match()))

        assert stats.events_updated == 1
        stored = await _stored(test_session, "late")
        assert (stored.status, stored.home_score) == (EventStatus.FINAL, 2)

    @pytest.mark.asyncio
    async def test_cancelled_match_cancels_the_pending_event(self, test_session) -> None:
        test_session.add(_make_event(event_id="called_off"))
        test_session.add(_make_event(event_id="other", home_team="Fulham", away_team="Burnley"))
        await test_session.commit()

        cancelled = CancelledMatch(home_team="Arsenal", away_team="Chelsea", start_time=KICKOFF)
        with patch(
            "odds_lambda.jobs.fetch_oddsportal_results.async_session_maker",
            _session_maker(test_session),
        ):
            stats = await process_results(fetched=_fetched(cancelled=[cancelled]))

        assert (stats.events_cancelled, stats.events_updated, stats.events_unresolved) == (1, 0, 1)
        stored = await _stored(test_session, "called_off")
        assert (stored.status, stored.home_score, stored.completed_at) == (
            EventStatus.CANCELLED,
            None,
            None,
        )
        assert (await _stored(test_session, "other")).status == EventStatus.SCHEDULED

    @pytest.mark.asyncio
    async def test_cancelled_match_leaves_a_scoreless_final_event_final(self, test_session) -> None:
        test_session.add(_make_event(event_id="played", status=EventStatus.FINAL))
        await test_session.commit()

        cancelled = CancelledMatch(home_team="Arsenal", away_team="Chelsea", start_time=KICKOFF)
        with patch(
            "odds_lambda.jobs.fetch_oddsportal_results.async_session_maker",
            _session_maker(test_session),
        ):
            stats = await process_results(fetched=_fetched(cancelled=[cancelled]))

        assert (stats.events_cancelled, stats.errors) == (0, [])
        assert (await _stored(test_session, "played")).status == EventStatus.FINAL

    @pytest.mark.asyncio
    async def test_fetches_results_back_to_the_oldest_pending_kickoff(self, test_session) -> None:
        test_session.add(_make_event())
        await test_session.commit()
        client = MagicMock()
        client.fetch_results = AsyncMock(return_value=_fetched(_make_match()))
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)

        with (
            patch(
                "odds_lambda.jobs.fetch_oddsportal_results.async_session_maker",
                _session_maker(test_session),
            ),
            patch(
                "odds_lambda.jobs.fetch_oddsportal_results.OddsPortalClient", return_value=client
            ),
        ):
            stats = await process_results(sport="soccer_epl")

        client.fetch_results.assert_awaited_once_with(
            "/football/england/premier-league/results/", since=KICKOFF - timedelta(hours=2)
        )
        assert stats.events_updated == 1

    @pytest.mark.asyncio
    async def test_no_pending_events_skips_the_fetch(self, test_session) -> None:
        with (
            patch(
                "odds_lambda.jobs.fetch_oddsportal_results.async_session_maker",
                _session_maker(test_session),
            ),
            patch("odds_lambda.jobs.fetch_oddsportal_results.OddsPortalClient") as client_cls,
        ):
            stats = await process_results()

        client_cls.assert_not_called()
        assert stats.events_updated == 0

    @pytest.mark.asyncio
    async def test_a_failed_update_is_recorded_and_the_rest_continue(self, test_session) -> None:
        test_session.add(_make_event(event_id="first", home_team="Fulham", away_team="Burnley"))
        test_session.add(_make_event(event_id="second"))
        await test_session.commit()

        real_update = __import__(
            "odds_lambda.storage.writers", fromlist=["OddsWriter"]
        ).OddsWriter.update_event_status

        async def flaky(self, event_id, **kwargs):
            if event_id == "first":
                raise RuntimeError("simulated update failure")
            return await real_update(self, event_id, **kwargs)

        matches = [
            _make_match(home_team="Fulham", away_team="Burnley", start_time=KICKOFF),
            _make_match(start_time=KICKOFF + timedelta(minutes=1)),
        ]
        with (
            patch(
                "odds_lambda.jobs.fetch_oddsportal_results.async_session_maker",
                _session_maker(test_session),
            ),
            patch("odds_lambda.storage.writers.OddsWriter.update_event_status", flaky),
        ):
            stats = await process_results(fetched=_fetched(*matches))

        assert len(stats.errors) == 1
        assert stats.events_updated == 1
        assert (await _stored(test_session, "second")).status == EventStatus.FINAL

    @pytest.mark.asyncio
    async def test_series_games_hours_apart_each_get_their_own_result(self, test_session) -> None:
        test_session.add(
            _make_event(
                event_id="game_one",
                status=EventStatus.FINAL,
                scores=(5, 4),
                sport_key="baseball_mlb",
            )
        )
        test_session.add(
            _make_event(
                event_id="game_two",
                commence_time=KICKOFF + timedelta(hours=20),
                sport_key="baseball_mlb",
            )
        )
        await test_session.commit()
        matches = [
            _make_match(start_time=KICKOFF, home_score=5, away_score=4),
            _make_match(start_time=KICKOFF + timedelta(hours=20), home_score=1, away_score=0),
        ]

        with patch(
            "odds_lambda.jobs.fetch_oddsportal_results.async_session_maker",
            _session_maker(test_session),
        ):
            await process_results(sport="baseball_mlb", fetched=_fetched(*matches))

        game_two = await _stored(test_session, "game_two")
        assert (game_two.home_score, game_two.away_score) == (1, 0)

    @pytest.mark.asyncio
    async def test_a_result_more_than_the_match_window_away_is_not_applied(
        self, test_session
    ) -> None:
        test_session.add(_make_event())
        await test_session.commit()

        with patch(
            "odds_lambda.jobs.fetch_oddsportal_results.async_session_maker",
            _session_maker(test_session),
        ):
            stats = await process_results(
                fetched=_fetched(_make_match(start_time=KICKOFF + timedelta(hours=5)))
            )

        assert stats.events_updated == 0

    @pytest.mark.asyncio
    async def test_feed_team_names_are_normalised(self, test_session) -> None:
        test_session.add(_make_event(home_team="Manchester Utd", away_team="Wolves"))
        await test_session.commit()

        with patch(
            "odds_lambda.jobs.fetch_oddsportal_results.async_session_maker",
            _session_maker(test_session),
        ):
            stats = await process_results(
                fetched=_fetched(
                    _make_match(home_team="Manchester United", away_team="Wolverhampton Wanderers")
                )
            )

        assert stats.events_updated == 1

    @pytest.mark.asyncio
    async def test_raises_when_rows_came_back_but_none_parsed_as_finished(
        self, test_session
    ) -> None:
        test_session.add(_make_event())
        await test_session.commit()

        with (
            patch(
                "odds_lambda.jobs.fetch_oddsportal_results.async_session_maker",
                _session_maker(test_session),
            ),
            pytest.raises(OddsPortalError, match="no finished or cancelled match"),
        ):
            await process_results(fetched=_fetched(rows_seen=50))

    @pytest.mark.asyncio
    async def test_an_empty_feed_does_not_raise(self, test_session) -> None:
        test_session.add(_make_event())
        await test_session.commit()

        with patch(
            "odds_lambda.jobs.fetch_oddsportal_results.async_session_maker",
            _session_maker(test_session),
        ):
            stats = await process_results(fetched=_fetched())

        assert stats.events_unresolved == 1

    @pytest.mark.asyncio
    async def test_events_older_than_the_lookback_are_left_unless_it_is_lifted(
        self, test_session
    ) -> None:
        test_session.add(_make_event(commence_time=KICKOFF - timedelta(days=30)))
        await test_session.commit()
        old = _make_match(start_time=KICKOFF - timedelta(days=30))

        maker = _session_maker(test_session)
        with patch("odds_lambda.jobs.fetch_oddsportal_results.async_session_maker", maker):
            within = await process_results(fetched=_fetched(old))
            lifted = await process_results(lookback=None, fetched=_fetched(old))

        assert (within.events_updated, lifted.events_updated) == (0, 1)

    @pytest.mark.asyncio
    async def test_unknown_sport_returns_empty_stats(self) -> None:
        stats = await process_results(sport="unknown_sport", fetched=_fetched())
        assert stats.matches_fetched == 0
        assert stats.events_updated == 0

    def test_default_sport_key_is_epl(self) -> None:
        assert DEFAULT_SPORT_KEY == "soccer_epl"


class TestMainSelfSchedules:
    """Self-schedule must run even when the results phase raises."""

    @pytest.mark.asyncio
    async def test_self_schedules_when_process_results_raises(self) -> None:
        from odds_lambda.jobs import fetch_oddsportal_results
        from odds_lambda.scheduling.jobs import JobContext

        scheduled_calls: list[dict] = []

        async def mock_schedule_next(
            job_name: str,
            next_time: datetime,
            payload: dict[str, object] | None = None,
        ) -> None:
            scheduled_calls.append(
                {"job_name": job_name, "next_time": next_time, "payload": payload}
            )

        mock_backend = AsyncMock()
        mock_backend.schedule_next_execution = mock_schedule_next
        mock_backend.get_backend_name = lambda: "mock_backend"

        async def failing_process_results(**_kwargs) -> None:
            raise RuntimeError("simulated per-record failure")

        with (
            patch.object(
                fetch_oddsportal_results, "process_results", side_effect=failing_process_results
            ),
            patch(
                "odds_lambda.scheduling.helpers.get_scheduler_backend",
                return_value=mock_backend,
            ),
        ):
            with pytest.raises(RuntimeError, match="simulated per-record failure"):
                await fetch_oddsportal_results.main(JobContext(sport="soccer_epl"))

        assert len(scheduled_calls) == 1, "self-schedule must run even when process_results raises"
        assert scheduled_calls[0]["job_name"] == "fetch-oddsportal-results-epl"
        assert scheduled_calls[0]["payload"] == {"sport": "soccer_epl"}


class TestRecordRunOutcome:
    NOW = datetime(2026, 10, 20, 8, 0, tzinfo=UTC)

    @staticmethod
    async def _run(
        stats: ResultsStats, last_settled: datetime | None
    ) -> tuple[AsyncMock, AsyncMock]:
        record = AsyncMock()
        warn = AsyncMock(return_value=True)
        with (
            patch("odds_core.alerts.record_to_alert_history", record),
            patch("odds_core.alerts.send_job_warning", warn),
            patch(
                "odds_lambda.jobs.fetch_oddsportal_results._last_settled_at",
                AsyncMock(return_value=last_settled),
            ),
        ):
            await record_run_outcome("soccer_epl", stats, TestRecordRunOutcome.NOW)
        return record, warn

    @pytest.mark.asyncio
    async def test_a_settling_run_records_the_marker_and_does_not_warn(self) -> None:
        stats = ResultsStats(events_updated=2, unresolved_kickoffs=[self.NOW - timedelta(days=3)])

        record, warn = await self._run(stats, None)

        assert record.await_args.args[0] == "results_settled:soccer_epl"
        warn.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_warns_when_nothing_settles_and_events_kicked_off_after_the_last_settle(
        self,
    ) -> None:
        stats = ResultsStats(unresolved_kickoffs=[self.NOW - timedelta(days=2)])

        record, warn = await self._run(stats, self.NOW - timedelta(days=5))

        record.assert_not_awaited()
        assert warn.await_args.args[0] == "results_stalled:soccer_epl"

    @pytest.mark.asyncio
    async def test_quiet_when_no_settling_run_is_on_record(self) -> None:
        stats = ResultsStats(unresolved_kickoffs=[self.NOW - timedelta(days=2)])

        _, warn = await self._run(stats, None)

        warn.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_quiet_for_unmatched_events_older_than_the_last_settle(self) -> None:
        stats = ResultsStats(unresolved_kickoffs=[self.NOW - timedelta(days=10)])

        _, warn = await self._run(stats, self.NOW - timedelta(days=1))

        warn.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_quiet_for_events_under_a_day_old(self) -> None:
        stats = ResultsStats(unresolved_kickoffs=[self.NOW - timedelta(hours=5)])

        _, warn = await self._run(stats, self.NOW - timedelta(days=5))

        warn.assert_not_awaited()
