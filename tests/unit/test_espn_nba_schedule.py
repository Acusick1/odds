"""ESPN NBA scoreboard parsing."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from odds_core.vocabulary import Source
from odds_lambda.espn_nba_schedule import parse_scoreboard


def scoreboard(
    status: str = "STATUS_FINAL", home_score: str | None = "125", away_score: str | None = "132"
) -> dict[str, Any]:
    return {
        "events": [
            {
                "id": "401909842",
                "date": "2026-10-22T02:30Z",
                "competitions": [
                    {
                        "status": {"type": {"name": status}},
                        "competitors": [
                            {
                                "homeAway": "home",
                                "score": home_score,
                                "team": {"displayName": "LA Clippers"},
                            },
                            {
                                "homeAway": "away",
                                "score": away_score,
                                "team": {"displayName": "Sacramento Kings"},
                            },
                        ],
                    }
                ],
            },
            {"id": "1", "date": "2026-10-22T02:30Z", "competitions": [{"competitors": []}]},
        ]
    }


class TestParseScoreboard:
    def test_final_game_carries_score_and_canonical_names(self) -> None:
        [game] = parse_scoreboard(scoreboard())

        assert game.espn_id == "401909842"
        assert game.commence_time == datetime(2026, 10, 22, 2, 30, tzinfo=UTC)
        assert (game.home_team, game.away_team) == ("Los Angeles Clippers", "Sacramento Kings")
        assert (game.final, game.home_score, game.away_score) == (True, 125, 132)

    @pytest.mark.parametrize("status", ["STATUS_SCHEDULED", "STATUS_IN_PROGRESS"])
    def test_unfinished_game_has_no_score(self, status: str) -> None:
        [game] = parse_scoreboard(scoreboard(status=status))

        assert (game.final, game.home_score, game.away_score) == (False, None, None)

    def test_listed_under_the_espn_source_and_id(self) -> None:
        [game] = parse_scoreboard(scoreboard())

        listed = game.listed()

        assert (listed.source, listed.external_id, listed.sport_key) == (
            Source.ESPN,
            "401909842",
            "basketball_nba",
        )
