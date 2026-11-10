"""Parsing of ESPN's football scoreboards into the matches of a league's schedule."""

from __future__ import annotations

import copy
from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from odds_core.models import EventStatus
from odds_core.team import normalize_team_name
from odds_lambda.espn_football_schedule import LEAGUES, fetch_games, parse_scoreboard
from odds_lambda.live_events import SkipReason

Scoreboard = Callable[[str], dict[str, Any]]


class TestRosters:
    @pytest.mark.parametrize("league", LEAGUES.values(), ids=lambda league: league.sport_key)
    def test_every_club_is_its_own_canonical_name(self, league: Any) -> None:
        assert {c for c in league.clubs if normalize_team_name(c) != c} == set()

    @pytest.mark.parametrize("league", LEAGUES.values(), ids=lambda league: league.sport_key)
    def test_the_recorded_scoreboard_names_only_the_leagues_clubs(
        self, league: Any, espn_football_scoreboard: Scoreboard
    ) -> None:
        parsed = parse_scoreboard(espn_football_scoreboard(league.slug), league)

        assert parsed.dropped == []
        assert len(parsed.games) == len(espn_football_scoreboard(league.slug)["events"])

    def test_the_clubs_of_two_leagues_do_not_overlap_except_in_the_championship(self) -> None:
        # Relegated EPL clubs are in the Championship; no other club plays in two leagues.
        overlap = [
            c
            for a in LEAGUES.values()
            for b in LEAGUES.values()
            if a.sport_key < b.sport_key
            for c in a.clubs & b.clubs
        ]
        assert overlap == []

    @pytest.mark.parametrize(
        ("espn", "canonical"),
        [
            ("Borussia Mönchengladbach", "Borussia Monchengladbach"),
            ("Mgladbach", "Borussia Monchengladbach"),
            ("Schalke 04", "Schalke"),
            ("Paris Saint-Germain", "Paris Saint Germain"),
            ("Paris St-G", "Paris Saint Germain"),
            ("ESTAC Troyes", "Troyes"),
            ("Deportivo", "Deportivo La Coruña"),
            ("Dep. A Coruna", "Deportivo La Coruña"),
            ("Internazionale", "Inter Milan"),
        ],
    )
    def test_each_spelling_maps_to_the_oddsportal_canonical_name(
        self, espn: str, canonical: str
    ) -> None:
        assert normalize_team_name(espn) == canonical


class TestParseScoreboard:
    def test_a_scheduled_match_is_upcoming_with_its_espn_kickoff(
        self, espn_football_scoreboard: Scoreboard
    ) -> None:
        parsed = parse_scoreboard(
            espn_football_scoreboard("esp.1"), LEAGUES["soccer_spain_la_liga"]
        )

        game = next(g for g in parsed.games if g.home_team == "Alavés")
        assert (game.away_team, game.commence_time, game.upcoming, game.status) == (
            "Deportivo La Coruña",
            datetime(2026, 11, 21, 13, 0, tzinfo=UTC),
            True,
            None,
        )

    def test_a_finished_match_is_not_upcoming(self, espn_football_scoreboard: Scoreboard) -> None:
        parsed = parse_scoreboard(
            espn_football_scoreboard("ita.1"), LEAGUES["soccer_italy_serie_a"]
        )

        assert any(not g.upcoming for g in parsed.games)

    def test_a_postponed_match_carries_its_status(
        self, espn_football_scoreboard: Scoreboard
    ) -> None:
        data = copy.deepcopy(espn_football_scoreboard("fra.1"))
        data["events"][0]["competitions"][0]["status"]["type"] = {
            "name": "STATUS_POSTPONED",
            "state": "post",
        }

        parsed = parse_scoreboard(data, LEAGUES["soccer_france_ligue_one"])

        assert parsed.games[0].status is EventStatus.POSTPONED

    def test_a_team_outside_the_roster_drops_the_match_and_is_named(
        self, espn_football_scoreboard: Scoreboard
    ) -> None:
        data = copy.deepcopy(espn_football_scoreboard("ger.1"))
        away = next(
            c
            for c in data["events"][0]["competitions"][0]["competitors"]
            if c["homeAway"] == "away"
        )
        away["team"]["displayName"] = "FC Neustadt"

        parsed = parse_scoreboard(data, LEAGUES["soccer_germany_bundesliga"])

        assert len(parsed.games) == len(data["events"]) - 1
        [dropped] = parsed.dropped
        assert dropped.reason is SkipReason.UNKNOWN_TEAM
        assert "'FC Neustadt'" in dropped.detail
        assert "FC Neustadt" in dropped.describe()

    def test_a_match_with_an_invalid_start_is_skipped(
        self, espn_football_scoreboard: Scoreboard
    ) -> None:
        data = copy.deepcopy(espn_football_scoreboard("eng.2"))
        data["events"][0]["competitions"][0]["timeValid"] = False

        parsed = parse_scoreboard(data, LEAGUES["soccer_efl_champ"])

        assert len(parsed.games) == len(data["events"]) - 1


class TestFetchGames:
    async def test_reads_each_month_once_and_keeps_the_days_asked_for(
        self, espn_football_scoreboard: Scoreboard
    ) -> None:
        fetcher = MagicMock()
        fetcher.fetch_month_scoreboard_json = AsyncMock(
            return_value=espn_football_scoreboard("esp.1")
        )

        parsed = await fetch_games(
            LEAGUES["soccer_spain_la_liga"], date(2026, 11, 20), date(2026, 11, 22), fetcher
        )

        assert [c.args for c in fetcher.fetch_month_scoreboard_json.await_args_list] == [
            ("esp.1", 2026, 11)
        ]
        assert {g.commence_time.date() for g in parsed.games} <= {
            date(2026, 11, 20),
            date(2026, 11, 21),
            date(2026, 11, 22),
        }
        assert parsed.games

    async def test_a_window_over_a_month_end_reads_both_months_without_repeating_a_match(
        self, espn_football_scoreboard: Scoreboard
    ) -> None:
        fetcher = MagicMock()
        fetcher.fetch_month_scoreboard_json = AsyncMock(
            return_value=espn_football_scoreboard("esp.1")
        )

        parsed = await fetch_games(
            LEAGUES["soccer_spain_la_liga"], date(2026, 11, 20), date(2026, 12, 2), fetcher
        )

        assert fetcher.fetch_month_scoreboard_json.await_count == 2
        ids = [g.espn_id for g in parsed.games]
        assert len(ids) == len(set(ids))
