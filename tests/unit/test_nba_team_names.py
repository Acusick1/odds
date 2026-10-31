"""Unit tests for the NBA abbreviation lookup and full-name normalization."""

import pytest
from odds_core.team import normalize_team_name
from odds_lambda.polymarket_matching import CANONICAL_TO_ABBREV, NBA_ABBREV_MAP


class TestAbbrevMap:
    def test_covers_all_30_teams(self):
        assert len(NBA_ABBREV_MAP) == 30
        assert len(set(NBA_ABBREV_MAP.values())) == 30

    def test_inverse_maps_each_team_to_its_upper_case_abbreviation(self):
        assert CANONICAL_TO_ABBREV["Los Angeles Lakers"] == "LAL"
        assert len(CANONICAL_TO_ABBREV) == 30

    @pytest.mark.parametrize("team", sorted(NBA_ABBREV_MAP.values()))
    def test_every_team_is_its_own_canonical_name(self, team: str):
        assert normalize_team_name(team) == team

    @pytest.mark.parametrize(
        ("spelling", "canonical"),
        [("LA Clippers", "Los Angeles Clippers"), ("LA Lakers", "Los Angeles Lakers")],
    )
    def test_polymarket_and_espn_spellings_resolve_to_odds_api_names(
        self, spelling: str, canonical: str
    ):
        assert normalize_team_name(spelling) == canonical

    def test_a_shared_nickname_stays_the_epl_team(self):
        assert normalize_team_name("Wolves") == "Wolves"

    def test_los_angeles_dotted_spellings(self):
        assert normalize_team_name("L.A. Lakers") == "Los Angeles Lakers"
