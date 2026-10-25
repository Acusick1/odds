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
    game_pk: int | None = 1,
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
        mlb_game_pk=game_pk,
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
        events = [
            _event("wrong", T0, (1, 0), game_pk=1),
            _event("right", T0 + timedelta(days=1), game_pk=2),
        ]
        games = [_game(1, T0), _game(2, T0 + timedelta(days=1))]

        assert _outcomes(check_scores(events, games)) == {
            "wrong": ScoreOutcome.MISMATCH,
            "right": ScoreOutcome.CORRECT,
        }

    def test_scoreless_final_event_is_a_mismatch(self) -> None:
        checks = check_scores([_event("e", scores=(None, None))], [_game(1)])

        assert _outcomes(checks) == {"e": ScoreOutcome.MISMATCH}

    def test_swapped_doubleheader_scores_pair_by_game_pk(self) -> None:
        # op_live_PITPIR_MILBRE_2026-07-11T1605 stored 3-2 while its game, 823357, ended 7-6.
        g1 = datetime(2026, 7, 11, 16, 5, tzinfo=UTC)
        games = [
            _game(823357, g1, (7, 6), game_number=1),
            _game(823358, g1 + timedelta(minutes=6), (3, 2), game_number=2, start_time_tbd=True),
        ]
        events = [
            _event("op_live_PITPIR_MILBRE_2026-07-11T1605", g1, (3, 2), game_pk=823357),
            _event("game_two", g1 + timedelta(hours=4), (7, 6), game_pk=823358),
        ]

        checks = {c.event.id: c for c in check_scores(events, games)}

        assert checks["op_live_PITPIR_MILBRE_2026-07-11T1605"].mlb_score == (7, 6)
        assert checks["game_two"].mlb_score == (3, 2)
        assert {c.outcome for c in checks.values()} == {ScoreOutcome.MISMATCH}

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

        checks = check_scores([_event("e", scores=(1, 1), game_pk=9)], [postponed, _game(9)])

        assert _outcomes(checks) == {"e": ScoreOutcome.MISMATCH}

    def test_unmatched_without_a_game_pk_or_a_final_game(self) -> None:
        events = [_event("allstar", game_pk=None), _event("pending", game_pk=2)]
        games = [_game(2, is_final=False)]

        checks = {c.event.id: c for c in check_scores(events, games)}

        assert {c.outcome for c in checks.values()} == {ScoreOutcome.UNMATCHED}
        assert (checks["allstar"].reason, checks["pending"].reason) == (
            "no gamePk",
            "game not final",
        )


def _fetcher(
    games: list[MlbGameResult], by_pk: dict[int, list[MlbGameResult]] | None = None
) -> MagicMock:
    fetcher = MagicMock()
    fetcher.fetch_results = AsyncMock(return_value=list(games))
    fetcher.fetch_game = AsyncMock(side_effect=lambda pk: (by_pk or {}).get(pk, []))
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
                _event("scheduled", status=EventStatus.SCHEDULED, scores=(None, None), game_pk=2),
                _event("epl", sport_key="soccer_epl", scores=(9, 9), game_pk=None),
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
    async def test_game_played_away_from_the_event_date_is_fetched_by_game_pk(
        self, test_session
    ) -> None:
        test_session.add(_event("moved", scores=(0, 0), game_pk=5))
        await test_session.commit()
        fetcher = _fetcher([], {5: [_game(5, T0 + timedelta(days=20), (4, 1))]})

        checks = await repair_scores(test_session, fetcher, apply=False)

        fetcher.fetch_game.assert_awaited_once_with(5)
        assert checks[0].mlb_score == (4, 1)

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
