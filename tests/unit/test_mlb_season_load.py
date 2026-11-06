"""Final MLB games from a recorded 2024 MLB Stats API schedule."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import pytest
from odds_lambda.mlb_season_load import FinalGame, fetch_season, final_games


def at(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


@pytest.fixture
async def games(recorded_mlb_fetcher: Any) -> dict[int, FinalGame]:
    async with recorded_mlb_fetcher as fetcher:
        return {game.game_pk: game for game in await fetch_season(fetcher, 2024)}


class TestFinalGames:
    def test_only_final_regular_season_and_postseason_games(
        self, games: dict[int, FinalGame]
    ) -> None:
        assert sorted(games) == [
            745843,
            745844,
            745848,
            746656,
            746658,
            746816,
            746942,
            775300,
        ]

    def test_a_postponed_game_is_placed_at_its_played_slot(
        self, games: dict[int, FinalGame]
    ) -> None:
        game = games[746816]

        assert (game.commence_time, game.start) == (at("2024-06-27T20:10"), at("2024-06-27T20:10"))
        assert (game.home_team, game.away_team, game.home_score, game.away_score) == (
            "Chicago White Sox",
            "Atlanta Braves",
            1,
            0,
        )

    def test_a_resumed_game_is_placed_where_it_began(self, games: dict[int, FinalGame]) -> None:
        game = games[746942]

        assert (game.commence_time, game.home_score, game.away_score) == (
            at("2024-06-26T23:10"),
            1,
            4,
        )

    def test_a_tbd_doubleheader_game_starts_at_its_first_pitch(
        self, games: dict[int, FinalGame]
    ) -> None:
        first, second = games[745844], games[745843]

        assert (second.commence_time, second.start) == (at("2024-04-04T20:21"), None)
        assert first.sibling_starts == (at("2024-04-04T20:21"),)
        assert second.sibling_starts == (at("2024-04-04T16:10"),)

    def test_split_doubleheader_games_are_each_others_siblings(
        self, games: dict[int, FinalGame]
    ) -> None:
        assert games[746656].sibling_starts == (at("2024-04-13T22:10"),)
        assert games[746658].sibling_starts == (at("2024-04-13T16:10"),)
        assert games[746816].sibling_starts == ()

    def test_games_come_in_start_order(self, games: dict[int, FinalGame]) -> None:
        starts = [game.commence_time for game in games.values()]

        assert starts == sorted(starts)
        assert games[775300].season == 2024

    def test_a_tbd_game_without_a_first_pitch_keeps_the_placeholder(
        self, recorded_mlb_schedule: dict[str, Any]
    ) -> None:
        from odds_lambda.mlb_stats_fetcher import _results_from_schedule

        for day in recorded_mlb_schedule["dates"]:
            for game in day["games"]:
                game.pop("gameInfo", None)

        placed = {g.game_pk: g for g in final_games(_results_from_schedule(recorded_mlb_schedule))}

        assert placed[745843].commence_time == at("2024-04-04T16:15")
        assert placed[745843].start is None
        assert date(2024, 4, 4) == placed[745843].commence_time.date()
