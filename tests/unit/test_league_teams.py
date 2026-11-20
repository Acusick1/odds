"""Each league's canonical teams, and the refusal of any other name."""

from __future__ import annotations

import pytest
from odds_core.league_teams import LEAGUE_TEAMS, UnknownTeam, league_team
from odds_core.sports import SUPPORTED_SPORTS
from odds_core.team import _TEAM_ALIASES, normalize_team_name

# A relegated EPL club plays in the Championship; no other club is in two leagues.
SHARED_LEAGUES = {"soccer_epl", "soccer_efl_champ"}


class TestLeagueTeams:
    def test_lists_only_supported_leagues(self) -> None:
        assert set(LEAGUE_TEAMS) <= set(SUPPORTED_SPORTS)

    @pytest.mark.parametrize("sport_key", sorted(LEAGUE_TEAMS))
    def test_every_team_is_its_own_canonical_name(self, sport_key: str) -> None:
        assert {t for t in LEAGUE_TEAMS[sport_key] if normalize_team_name(t) != t} == set()

    def test_every_alias_names_a_league_team(self) -> None:
        teams = set().union(*LEAGUE_TEAMS.values())
        assert {alias for alias, canonical in _TEAM_ALIASES.items() if canonical not in teams} == (
            set()
        )

    def test_only_the_epl_and_championship_share_clubs(self) -> None:
        overlap = {
            (a, b, team)
            for a in LEAGUE_TEAMS
            for b in LEAGUE_TEAMS
            if a < b and {a, b} != SHARED_LEAGUES
            for team in LEAGUE_TEAMS[a] & LEAGUE_TEAMS[b]
        }
        assert overlap == set()


class TestLeagueTeam:
    @pytest.mark.parametrize(
        ("sport_key", "name", "canonical"),
        [
            ("soccer_spain_la_liga", "Ath. Bilbao", "Athletic Bilbao"),
            ("soccer_spain_la_liga", "A Coruna", "Deportivo La Coruña"),
            ("soccer_spain_la_liga", "Celta de Vigo", "Celta Vigo"),
            ("soccer_italy_serie_a", "AC Monza", "Monza"),
            ("soccer_epl", " Manchester  United ", "Manchester Utd"),
            ("basketball_nba", "LA Lakers", "Los Angeles Lakers"),
        ],
    )
    def test_a_known_spelling_gives_the_canonical_name(
        self, sport_key: str, name: str, canonical: str
    ) -> None:
        assert league_team(sport_key, name) == canonical

    def test_an_unknown_name_is_refused_naming_the_league_and_raw_name(self) -> None:
        with pytest.raises(UnknownTeam) as refused:
            league_team("soccer_spain_la_liga", "Ath Bilbao FC")

        assert refused.value.sport_key == "soccer_spain_la_liga"
        assert refused.value.name == "Ath Bilbao FC"
        assert "soccer_spain_la_liga" in str(refused.value)
        assert "'Ath Bilbao FC'" in str(refused.value)

    def test_another_leagues_team_is_refused(self) -> None:
        with pytest.raises(UnknownTeam):
            league_team("soccer_spain_la_liga", "Arsenal")

    def test_a_league_with_no_list_refuses_every_name(self) -> None:
        with pytest.raises(UnknownTeam, match="americanfootball_nfl"):
            league_team("americanfootball_nfl", "New England Patriots")
