"""Unit tests for NBA team-name normalization."""

from odds_core.team import normalize_team_name
from odds_lambda.polymarket_matching import NBA_ABBREV_MAP, TEAM_ALIASES, normalize_team


class TestTeamAliases:
    """Tests for TEAM_ALIASES coverage and consistency."""

    def test_all_30_teams_covered(self):
        """All 30 NBA teams have alias entries."""
        assert len(TEAM_ALIASES) == 30

    def test_canonical_name_in_own_aliases(self):
        """Each canonical name appears in its own alias set."""
        for canonical, aliases in TEAM_ALIASES.items():
            assert canonical in aliases, f"{canonical} not in its own alias set"

    def test_no_alias_maps_to_two_teams(self):
        """No alias resolves to more than one canonical name."""
        seen: dict[str, str] = {}
        for canonical, aliases in TEAM_ALIASES.items():
            for alias in aliases:
                key = alias.lower()
                assert key not in seen, (
                    f"Alias '{alias}' appears in both '{seen[key]}' and '{canonical}'"
                )
                seen[key] = canonical

    def test_abbrev_map_covers_all_teams(self):
        """NBA_ABBREV_MAP covers all 30 teams."""
        assert len(NBA_ABBREV_MAP) == 30

    def test_abbrev_map_values_are_canonical(self):
        """Every value in NBA_ABBREV_MAP is a key in TEAM_ALIASES."""
        for abbrev, team in NBA_ABBREV_MAP.items():
            assert team in TEAM_ALIASES, f"Abbrev '{abbrev}' maps to unknown team '{team}'"


class TestNormalizeTeam:
    """Tests for normalize_team()."""

    def test_canonical_name_returns_self(self):
        assert normalize_team("Dallas Mavericks") == "Dallas Mavericks"
        assert normalize_team("Los Angeles Lakers") == "Los Angeles Lakers"
        assert normalize_team("Golden State Warriors") == "Golden State Warriors"

    def test_known_aliases(self):
        assert normalize_team("Mavericks") == "Dallas Mavericks"
        assert normalize_team("Mavs") == "Dallas Mavericks"
        assert normalize_team("Lakers") == "Los Angeles Lakers"
        assert normalize_team("LA Lakers") == "Los Angeles Lakers"
        assert normalize_team("Celtics") == "Boston Celtics"
        assert normalize_team("76ers") == "Philadelphia 76ers"
        assert normalize_team("Sixers") == "Philadelphia 76ers"
        assert normalize_team("Blazers") == "Portland Trail Blazers"
        assert normalize_team("Trail Blazers") == "Portland Trail Blazers"
        assert normalize_team("Cavs") == "Cleveland Cavaliers"
        assert normalize_team("Wolves") == "Minnesota Timberwolves"
        assert normalize_team("OKC") == "Oklahoma City Thunder"

    def test_case_insensitive(self):
        assert normalize_team("LAKERS") == "Los Angeles Lakers"
        assert normalize_team("lakers") == "Los Angeles Lakers"
        assert normalize_team("LaKeRs") == "Los Angeles Lakers"

    def test_leading_trailing_whitespace(self):
        assert normalize_team("  Lakers  ") == "Los Angeles Lakers"

    def test_unknown_name_returns_none(self):
        assert normalize_team("Unknown Team") is None
        assert normalize_team("") is None
        assert normalize_team("NBA") is None

    def test_all_canonical_names_normalize(self):
        """Every canonical team name normalizes to itself."""
        for canonical in TEAM_ALIASES:
            assert normalize_team(canonical) == canonical


class TestSharedTeamNormalization:
    """Tests for odds_core.team.normalize_team_name on NBA names."""

    def test_every_canonical_name_passes_through(self):
        for canonical in TEAM_ALIASES:
            assert normalize_team_name(canonical) == canonical

    def test_los_angeles_spellings(self):
        assert normalize_team_name("LA Clippers") == "Los Angeles Clippers"
        assert normalize_team_name("Los Angeles Clippers") == "Los Angeles Clippers"
        assert normalize_team_name("LA Lakers") == "Los Angeles Lakers"
        assert normalize_team_name("L.A. Lakers") == "Los Angeles Lakers"

    def test_a_shared_nickname_stays_the_epl_team(self):
        assert normalize_team_name("Wolves") == "Wolves"
