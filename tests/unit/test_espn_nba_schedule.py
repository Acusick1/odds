"""ESPN NBA scoreboard parsing."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from odds_core.models import EventStatus
from odds_core.vocabulary import Source
from odds_lambda.espn_nba_schedule import parse_scoreboard
from odds_lambda.polymarket_matching import canonical_nba_team


def event(
    espn_id: str = "401909842",
    home: str = "LA Clippers",
    away: str = "Sacramento Kings",
    status: str = "STATUS_FINAL",
    home_score: str | None = "125",
    away_score: str | None = "132",
    season_type: int = 2,
    time_valid: bool | None = None,
) -> dict[str, Any]:
    competition: dict[str, Any] = {
        "status": {"type": {"name": status}},
        "competitors": [
            {"homeAway": "home", "score": home_score, "team": {"displayName": home}},
            {"homeAway": "away", "score": away_score, "team": {"displayName": away}},
        ],
    }
    if time_valid is not None:
        competition["timeValid"] = time_valid
    return {
        "id": espn_id,
        "date": "2026-10-22T02:30Z",
        "season": {"type": season_type},
        "competitions": [competition],
    }


def scoreboard(*events: dict[str, Any]) -> dict[str, Any]:
    return {"events": list(events)}


class TestParseScoreboard:
    def test_final_game_carries_score_and_canonical_names(self) -> None:
        [game] = parse_scoreboard(scoreboard(event()))

        assert game.espn_id == "401909842"
        assert game.commence_time == datetime(2026, 10, 22, 2, 30, tzinfo=UTC)
        assert (game.home_team, game.away_team) == ("Los Angeles Clippers", "Sacramento Kings")
        assert (game.status, game.home_score, game.away_score) == (EventStatus.FINAL, 125, 132)

    @pytest.mark.parametrize("status", ["STATUS_SCHEDULED", "STATUS_IN_PROGRESS"])
    def test_unfinished_game_has_no_status_or_score(self, status: str) -> None:
        [game] = parse_scoreboard(scoreboard(event(status=status)))

        assert (game.status, game.home_score, game.away_score) == (None, None, None)

    def test_a_final_game_without_a_score_is_not_final(self) -> None:
        [game] = parse_scoreboard(scoreboard(event(home_score=None)))

        assert game.status is None

    @pytest.mark.parametrize(
        ("espn_status", "status"),
        [
            ("STATUS_POSTPONED", EventStatus.POSTPONED),
            ("STATUS_CANCELED", EventStatus.CANCELLED),
        ],
    )
    def test_postponed_and_cancelled_games_map_to_event_statuses(
        self, espn_status: str, status: EventStatus
    ) -> None:
        [game] = parse_scoreboard(scoreboard(event(status=espn_status, home_score=None)))

        assert (game.status, game.home_score) == (status, None)

    def test_listed_under_the_espn_source_and_id(self) -> None:
        [game] = parse_scoreboard(scoreboard(event()))

        listed = game.listed()

        assert (listed.source, listed.external_id, listed.sport_key) == (
            Source.ESPN,
            "401909842",
            "basketball_nba",
        )


class TestPlaceholderGames:
    def test_event_without_two_competitors_is_skipped(self) -> None:
        bare = {"id": "1", "date": "2026-10-22T02:30Z", "competitions": [{"competitors": []}]}

        assert parse_scoreboard(scoreboard(bare)) == []

    @pytest.mark.parametrize(
        ("home", "away"), [("TBD", "TBD"), ("Team Stars", "Team Stripes"), ("Team World", "Lakers")]
    )
    def test_a_side_that_is_not_one_of_the_30_teams_is_skipped(self, home: str, away: str) -> None:
        assert parse_scoreboard(scoreboard(event(home=home, away=away))) == []

    def test_a_start_espn_marks_invalid_is_skipped(self) -> None:
        assert parse_scoreboard(scoreboard(event(time_valid=False))) == []

    def test_a_valid_start_is_kept(self) -> None:
        assert len(parse_scoreboard(scoreboard(event(time_valid=True)))) == 1

    def test_preseason_games_are_skipped(self) -> None:
        assert parse_scoreboard(scoreboard(event(season_type=1))) == []

    def test_playoff_games_are_kept(self) -> None:
        assert len(parse_scoreboard(scoreboard(event(season_type=3)))) == 1

    def test_only_the_placeholder_is_dropped_from_a_mixed_day(self) -> None:
        games = parse_scoreboard(
            scoreboard(event(espn_id="1", home="TBD", away="TBD"), event(espn_id="2"))
        )

        assert [g.espn_id for g in games] == ["2"]


class TestCanonicalNbaTeam:
    @pytest.mark.parametrize(
        ("name", "canonical"),
        [
            ("LA Clippers", "Los Angeles Clippers"),
            ("Boston Celtics", "Boston Celtics"),
            ("L.A. Lakers", "Los Angeles Lakers"),
        ],
    )
    def test_nba_spellings_resolve(self, name: str, canonical: str) -> None:
        assert canonical_nba_team(name) == canonical

    @pytest.mark.parametrize("name", ["TBD", "Team Stripes", "Wolves", "Tottenham", ""])
    def test_everything_else_is_none(self, name: str) -> None:
        assert canonical_nba_team(name) is None
