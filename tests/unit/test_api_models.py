"""Unit tests for API models and conversion functions."""

from datetime import UTC

from odds_core.api_models import ApiEvent, parse_api_event, parse_scores_from_api_dict


class TestParseScoresFromApiDict:
    """Test score parsing helper function."""

    def test_valid_scores(self):
        """Test parsing valid home and away scores."""
        score_data = {
            "home_team": "Lakers",
            "away_team": "Celtics",
            "scores": [
                {"name": "Lakers", "score": "108"},
                {"name": "Celtics", "score": "105"},
            ],
        }

        home_score, away_score = parse_scores_from_api_dict(score_data)

        assert home_score == 108
        assert away_score == 105

    def test_scores_in_reverse_order(self):
        """Test parsing when away team score comes first."""
        score_data = {
            "home_team": "Lakers",
            "away_team": "Celtics",
            "scores": [
                {"name": "Celtics", "score": "105"},
                {"name": "Lakers", "score": "108"},
            ],
        }

        home_score, away_score = parse_scores_from_api_dict(score_data)

        assert home_score == 108
        assert away_score == 105

    def test_missing_home_score(self):
        """Test when home score is missing."""
        score_data = {
            "home_team": "Lakers",
            "away_team": "Celtics",
            "scores": [{"name": "Celtics", "score": "105"}],
        }

        home_score, away_score = parse_scores_from_api_dict(score_data)

        assert home_score is None
        assert away_score == 105

    def test_missing_away_score(self):
        """Test when away score is missing."""
        score_data = {
            "home_team": "Lakers",
            "away_team": "Celtics",
            "scores": [{"name": "Lakers", "score": "108"}],
        }

        home_score, away_score = parse_scores_from_api_dict(score_data)

        assert home_score == 108
        assert away_score is None

    def test_empty_scores_list(self):
        """Test when scores list is empty."""
        score_data = {
            "home_team": "Lakers",
            "away_team": "Celtics",
            "scores": [],
        }

        home_score, away_score = parse_scores_from_api_dict(score_data)

        assert home_score is None
        assert away_score is None

    def test_missing_scores_key(self):
        """Test when scores key is missing."""
        score_data = {
            "home_team": "Lakers",
            "away_team": "Celtics",
        }

        home_score, away_score = parse_scores_from_api_dict(score_data)

        assert home_score is None
        assert away_score is None

    def test_invalid_score_value_string(self):
        """Test when score value is not a valid integer string."""
        score_data = {
            "home_team": "Lakers",
            "away_team": "Celtics",
            "scores": [
                {"name": "Lakers", "score": "invalid"},
                {"name": "Celtics", "score": "105"},
            ],
        }

        home_score, away_score = parse_scores_from_api_dict(score_data)

        assert home_score is None  # Invalid conversion should return None
        assert away_score == 105

    def test_null_score_value(self):
        """Test when score value is null."""
        score_data = {
            "home_team": "Lakers",
            "away_team": "Celtics",
            "scores": [
                {"name": "Lakers", "score": None},
                {"name": "Celtics", "score": "105"},
            ],
        }

        home_score, away_score = parse_scores_from_api_dict(score_data)

        assert home_score is None
        assert away_score == 105

    def test_integer_scores(self):
        """Test when scores are already integers."""
        score_data = {
            "home_team": "Lakers",
            "away_team": "Celtics",
            "scores": [
                {"name": "Lakers", "score": 108},
                {"name": "Celtics", "score": 105},
            ],
        }

        home_score, away_score = parse_scores_from_api_dict(score_data)

        assert home_score == 108
        assert away_score == 105

    def test_zero_scores(self):
        """Test when scores are zero (valid edge case)."""
        score_data = {
            "home_team": "Lakers",
            "away_team": "Celtics",
            "scores": [
                {"name": "Lakers", "score": "0"},
                {"name": "Celtics", "score": "0"},
            ],
        }

        home_score, away_score = parse_scores_from_api_dict(score_data)

        assert home_score == 0
        assert away_score == 0


class TestParseApiEvent:
    """Test parsing a listed game."""

    def test_reads_the_listed_game(self):
        """Test that the API's id and the game's fields are carried over."""
        event_data = {
            "id": "abc123",
            "sport_key": "basketball_nba",
            "sport_title": "NBA",
            "commence_time": "2024-10-20T00:00:00Z",
            "home_team": "Lakers",
            "away_team": "Celtics",
        }

        event = parse_api_event(event_data)

        assert isinstance(event, ApiEvent)
        assert event.odds_api_id == "abc123"
        assert event.sport_key == "basketball_nba"
        assert event.sport_title == "NBA"
        assert event.home_team == "Lakers"
        assert event.away_team == "Celtics"

    def test_parses_commence_time_as_utc(self):
        """Test that commence_time is parsed as timezone-aware UTC."""
        event_data = {
            "id": "abc123",
            "sport_key": "basketball_nba",
            "commence_time": "2024-10-20T23:30:00Z",
            "home_team": "Lakers",
            "away_team": "Celtics",
        }

        event = parse_api_event(event_data)

        assert event.commence_time.tzinfo == UTC
        assert event.commence_time.year == 2024
        assert event.commence_time.month == 10
        assert event.commence_time.day == 20
        assert event.commence_time.hour == 23
        assert event.commence_time.minute == 30

    def test_uses_sport_key_as_fallback_title(self):
        """Test that sport_key is used when sport_title is missing."""
        event_data = {
            "id": "abc123",
            "sport_key": "basketball_nba",
            "commence_time": "2024-10-20T00:00:00Z",
            "home_team": "Lakers",
            "away_team": "Celtics",
            # No sport_title provided
        }

        event = parse_api_event(event_data)

        assert event.sport_title == "basketball_nba"

    def test_team_names_are_canonical(self):
        """Test that the API's spelling of a team is normalized."""
        event_data = {
            "id": "abc123",
            "sport_key": "soccer_epl",
            "commence_time": "2024-10-20T00:00:00Z",
            "home_team": "Man City",
            "away_team": "Tottenham Hotspur",
        }

        event = parse_api_event(event_data)

        assert (event.home_team, event.away_team) == ("Manchester City", "Tottenham")
