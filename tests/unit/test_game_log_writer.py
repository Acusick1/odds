"""Unit tests for game log writer."""

from odds_lambda.storage.game_log_writer import _ABBREV_TO_CANONICAL


class TestAbbrevToCanonical:
    """Verify the hardcoded abbreviation mapping covers all 30 NBA teams."""

    def test_all_30_teams(self):
        assert len(_ABBREV_TO_CANONICAL) == 30

    def test_known_abbreviations(self):
        assert _ABBREV_TO_CANONICAL["BOS"] == "Boston Celtics"
        assert _ABBREV_TO_CANONICAL["LAL"] == "Los Angeles Lakers"
        assert _ABBREV_TO_CANONICAL["GSW"] == "Golden State Warriors"
        assert _ABBREV_TO_CANONICAL["NYK"] == "New York Knicks"
        assert _ABBREV_TO_CANONICAL["PHX"] == "Phoenix Suns"

    def test_matches_nba_api_teams(self):
        """Verify abbreviations match nba_api static team data."""
        from nba_api.stats.static import teams

        nba_teams = {t["abbreviation"]: t["full_name"] for t in teams.get_teams()}
        for abbrev, canonical in _ABBREV_TO_CANONICAL.items():
            assert abbrev in nba_teams, f"Unknown abbreviation: {abbrev}"
            assert nba_teams[abbrev] == canonical, (
                f"Mismatch for {abbrev}: expected {nba_teams[abbrev]}, got {canonical}"
            )

    def test_matches_polymarket_aliases(self):
        """Verify all canonical names exist in TEAM_ALIASES."""
        from odds_lambda.polymarket_matching import TEAM_ALIASES

        for canonical in _ABBREV_TO_CANONICAL.values():
            assert canonical in TEAM_ALIASES, f"Canonical name {canonical!r} not in TEAM_ALIASES"
