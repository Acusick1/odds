"""Tests for the MLB score repair (#578)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from odds_cli.commands import events as events_cli
from odds_core.models import Event, EventStatus
from odds_lambda.mlb_stats_fetcher import MlbGameResult, MlbStatsFetcher
from odds_lambda.storage.mlb_score_repair import ScoreOutcome, check_scores, repair_scores

T0 = datetime(2026, 5, 1, 23, 10, tzinfo=UTC)
HOME, AWAY = "Boston Red Sox", "St.Louis Cardinals"


def _event(
    event_id: str,
    start: datetime = T0,
    scores: tuple[int | None, int | None] = (3, 2),
    status: EventStatus = EventStatus.FINAL,
    sport_key: str = "baseball_mlb",
) -> Event:
    return Event(
        id=event_id,
        sport_key=sport_key,
        sport_title="MLB",
        commence_time=start,
        home_team=HOME,
        away_team=AWAY,
        status=status,
        home_score=scores[0],
        away_score=scores[1],
    )


def _game(
    pk: int,
    start: datetime = T0,
    scores: tuple[int, int] = (3, 2),
    is_final: bool = True,
    home: str = HOME,
    away: str = "St. Louis Cardinals",
    **extra: Any,
) -> MlbGameResult:
    return MlbGameResult(pk, start, home, away, is_final, scores[0], scores[1], **extra)


def _outcomes(checks) -> dict[str, ScoreOutcome]:
    return {c.event.id: c.outcome for c in checks}


class TestCheckScores:
    def test_detects_mismatch_and_accepts_correct_score(self) -> None:
        events = [_event("wrong", T0, (1, 0)), _event("right", T0 + timedelta(days=1))]
        games = [_game(1, T0), _game(2, T0 + timedelta(days=1))]

        assert _outcomes(check_scores(events, games)) == {
            "wrong": ScoreOutcome.MISMATCH,
            "right": ScoreOutcome.CORRECT,
        }

    def test_scoreless_final_event_is_a_mismatch(self) -> None:
        checks = check_scores([_event("e", scores=(None, None))], [_game(1)])

        assert _outcomes(checks) == {"e": ScoreOutcome.MISMATCH}

    def test_doubleheader_pairs_by_order_despite_placeholder_start_of_game_two(self) -> None:
        # Mets-Rockies 2026-04-26: game 2 carries startTimeTBD and a start 5 min after game 1.
        day = date(2026, 4, 26)
        g1 = datetime(2026, 4, 26, 17, 40, tzinfo=UTC)
        games = [
            _game(823635, g1, (1, 3), game_number=1, official_date=day),
            _game(
                823637,
                g1 + timedelta(minutes=5),
                (0, 3),
                game_number=2,
                start_time_tbd=True,
                official_date=day,
            ),
        ]
        events = [
            _event("late", g1 + timedelta(hours=4), (1, 3)),
            _event("early", g1, (0, 3)),
        ]

        checks = {c.event.id: c for c in check_scores(events, games)}

        assert checks["early"].game.game_pk == 823635
        assert checks["late"].game.game_pk == 823637
        assert (checks["early"].outcome, checks["late"].outcome) == (
            ScoreOutcome.MISMATCH,
            ScoreOutcome.MISMATCH,
        )

    def test_doubleheader_with_unequal_counts_is_ambiguous(self) -> None:
        day = date(2026, 4, 26)
        games = [
            _game(1, T0, game_number=1, official_date=day),
            _game(2, T0 + timedelta(minutes=5), game_number=2, official_date=day),
        ]

        checks = check_scores([_event("only", T0)], games)

        assert _outcomes(checks) == {"only": ScoreOutcome.AMBIGUOUS}

    def test_postponed_entry_of_a_rescheduled_game_is_ignored(self) -> None:
        postponed = MlbGameResult(
            9,
            T0 - timedelta(days=1),
            HOME,
            AWAY,
            False,
            None,
            None,
            official_date=date(2026, 4, 30),
        )

        checks = check_scores([_event("e", scores=(1, 1))], [postponed, _game(9)])

        assert _outcomes(checks) == {"e": ScoreOutcome.MISMATCH}

    def test_start_gap_over_six_hours_is_ambiguous(self) -> None:
        checks = check_scores([_event("e", T0 + timedelta(hours=7), (1, 0))], [_game(1, T0)])

        assert _outcomes(checks) == {"e": ScoreOutcome.AMBIGUOUS}

    def test_two_events_claiming_one_game_are_ambiguous(self) -> None:
        events = [_event("a", T0), _event("b", T0 + timedelta(minutes=30))]

        assert set(_outcomes(check_scores(events, [_game(1)])).values()) == {ScoreOutcome.AMBIGUOUS}

    def test_unmatched_when_no_game_in_window_or_teams_differ_or_not_final(self) -> None:
        events = [
            _event("far", T0 + timedelta(days=3)),
            _event("allstar", T0),
            _event("pending", T0 + timedelta(days=1)),
        ]
        events[1].home_team = "National League All-Stars"
        games = [_game(1, T0), _game(2, T0 + timedelta(days=1), is_final=False)]

        assert set(_outcomes(check_scores(events, games)).values()) == {ScoreOutcome.UNMATCHED}


def _fetcher(games: list[MlbGameResult]) -> MagicMock:
    fetcher = MagicMock()
    fetcher.fetch_results = AsyncMock(return_value=games)
    fetcher.__aenter__ = AsyncMock(return_value=fetcher)
    fetcher.__aexit__ = AsyncMock(return_value=None)
    return fetcher


def _session_maker(session) -> MagicMock:
    maker = MagicMock()
    maker.return_value.__aenter__ = AsyncMock(return_value=session)
    maker.return_value.__aexit__ = AsyncMock(return_value=None)
    return maker


class TestRepairScores:
    @pytest.mark.asyncio
    async def test_apply_writes_scores_only_and_is_idempotent(self, test_session) -> None:
        event = _event("wrong", scores=(1, 0))
        event.completed_at = T0
        test_session.add_all(
            [
                event,
                _event("scheduled", status=EventStatus.SCHEDULED, scores=(None, None)),
                _event("epl", sport_key="soccer_epl", scores=(9, 9)),
            ]
        )
        await test_session.commit()
        fetcher = _fetcher([_game(1, T0, (3, 2))])

        first = await repair_scores(test_session, fetcher, apply=True)
        await test_session.commit()
        second = await repair_scores(test_session, fetcher, apply=True)

        assert _outcomes(first) == {"wrong": ScoreOutcome.MISMATCH}
        assert _outcomes(second) == {"wrong": ScoreOutcome.CORRECT}
        await test_session.refresh(event)
        assert (event.home_score, event.away_score) == (3, 2)
        assert (event.status, event.completed_at) == (EventStatus.FINAL, T0)
        assert (await test_session.get(Event, "epl")).home_score == 9
        assert (await test_session.get(Event, "scheduled")).home_score is None

    @pytest.mark.asyncio
    async def test_dry_run_command_writes_nothing(self, test_session, monkeypatch, capsys) -> None:
        test_session.add(_event("wrong", scores=(1, 0)))
        await test_session.commit()
        monkeypatch.setattr(events_cli, "async_session_maker", _session_maker(test_session))
        monkeypatch.setattr(events_cli, "MlbStatsFetcher", lambda: _fetcher([_game(1)]))

        await events_cli._repair_scores(apply=False)

        assert "MISMATCH wrong" in capsys.readouterr().out
        assert (await test_session.get(Event, "wrong")).home_score == 1


class TestFetchResults:
    @pytest.mark.asyncio
    async def test_parses_linescore_schedule_in_date_chunks(self) -> None:
        def game(pk: int, state: str, home: int | None) -> dict:
            return {
                "gamePk": pk,
                "gameDate": "2026-05-01T23:10:00Z",
                "status": {"abstractGameState": state},
                "teams": {
                    "home": {"team": {"name": HOME}, "score": home},
                    "away": {"team": {"name": "St. Louis Cardinals"}, "score": 2},
                },
            }

        fetcher = MlbStatsFetcher(client=MagicMock(), request_delay_seconds=0)
        fetch = AsyncMock(
            side_effect=[
                {"dates": [{"games": [game(1, "Final", 3), game(2, "Preview", None)]}]},
                {"dates": []},
            ]
        )
        fetcher._fetch_json = fetch

        results = await fetcher.fetch_results(date(2026, 5, 1), date(2026, 6, 15))

        assert [(r.game_pk, r.is_final, r.home_score) for r in results] == [
            (1, True, 3),
            (2, False, None),
        ]
        assert [c.args[1]["startDate"] for c in fetch.await_args_list] == [
            "2026-05-01",
            "2026-05-31",
        ]
        assert fetch.await_args_list[1].args[1]["endDate"] == "2026-06-15"
