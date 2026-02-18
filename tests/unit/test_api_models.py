"""Unit tests for API models and conversion functions."""

from datetime import UTC

import pytest
from odds_core.api_models import (
    api_dict_to_event,
    create_completed_event,
    create_scheduled_event,
    parse_scores_from_api_dict,
)
from odds_core.models import EventStatus


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


class TestCreateScheduledEvent:
    """Test scheduled event creation function."""

    def test_creates_event_with_scheduled_status(self):
        """Test that event is created with SCHEDULED status."""
        event_data = {
            "id": "abc123",
            "sport_key": "basketball_nba",
            "sport_title": "NBA",
            "commence_time": "2024-10-20T00:00:00Z",
            "home_team": "Lakers",
            "away_team": "Celtics",
        }

        event = create_scheduled_event(event_data)

        assert event.id == "abc123"
        assert event.sport_key == "basketball_nba"
        assert event.sport_title == "NBA"
        assert event.home_team == "Lakers"
        assert event.away_team == "Celtics"
        assert event.status == EventStatus.SCHEDULED
        assert event.home_score is None
        assert event.away_score is None

    def test_parses_commence_time_as_utc(self):
        """Test that commence_time is parsed as timezone-aware UTC."""
        event_data = {
            "id": "abc123",
            "sport_key": "basketball_nba",
            "commence_time": "2024-10-20T23:30:00Z",
            "home_team": "Lakers",
            "away_team": "Celtics",
        }

        event = create_scheduled_event(event_data)

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

        event = create_scheduled_event(event_data)

        assert event.sport_title == "basketball_nba"


class TestCreateCompletedEvent:
    """Test completed event creation function."""

    def test_creates_event_with_final_status(self):
        """Test that event is created with FINAL status and scores."""
        event_data = {
            "id": "abc123",
            "sport_key": "basketball_nba",
            "sport_title": "NBA",
            "commence_time": "2024-10-20T00:00:00Z",
            "home_team": "Lakers",
            "away_team": "Celtics",
            "scores": [
                {"name": "Lakers", "score": "108"},
                {"name": "Celtics", "score": "105"},
            ],
        }

        event = create_completed_event(event_data)

        assert event.id == "abc123"
        assert event.sport_key == "basketball_nba"
        assert event.sport_title == "NBA"
        assert event.home_team == "Lakers"
        assert event.away_team == "Celtics"
        assert event.status == EventStatus.FINAL
        assert event.home_score == 108
        assert event.away_score == 105

    def test_raises_error_when_home_score_missing(self):
        """Test that ValueError is raised when home score is missing."""
        event_data = {
            "id": "abc123",
            "sport_key": "basketball_nba",
            "commence_time": "2024-10-20T00:00:00Z",
            "home_team": "Lakers",
            "away_team": "Celtics",
            "scores": [
                {"name": "Celtics", "score": "105"},
                # No Lakers score
            ],
        }

        with pytest.raises(ValueError, match="Missing or invalid scores"):
            create_completed_event(event_data)

    def test_raises_error_when_away_score_missing(self):
        """Test that ValueError is raised when away score is missing."""
        event_data = {
            "id": "abc123",
            "sport_key": "basketball_nba",
            "commence_time": "2024-10-20T00:00:00Z",
            "home_team": "Lakers",
            "away_team": "Celtics",
            "scores": [
                {"name": "Lakers", "score": "108"},
                # No Celtics score
            ],
        }

        with pytest.raises(ValueError, match="Missing or invalid scores"):
            create_completed_event(event_data)

    def test_raises_error_when_scores_empty(self):
        """Test that ValueError is raised when scores list is empty."""
        event_data = {
            "id": "abc123",
            "sport_key": "basketball_nba",
            "commence_time": "2024-10-20T00:00:00Z",
            "home_team": "Lakers",
            "away_team": "Celtics",
            "scores": [],
        }

        with pytest.raises(ValueError, match="Missing or invalid scores"):
            create_completed_event(event_data)

    def test_raises_error_when_scores_invalid(self):
        """Test that ValueError is raised when scores are invalid."""
        event_data = {
            "id": "abc123",
            "sport_key": "basketball_nba",
            "commence_time": "2024-10-20T00:00:00Z",
            "home_team": "Lakers",
            "away_team": "Celtics",
            "scores": [
                {"name": "Lakers", "score": "invalid"},
                {"name": "Celtics", "score": "105"},
            ],
        }

        with pytest.raises(ValueError, match="Missing or invalid scores"):
            create_completed_event(event_data)

    def test_handles_zero_scores(self):
        """Test that zero scores are handled correctly (edge case)."""
        event_data = {
            "id": "abc123",
            "sport_key": "basketball_nba",
            "commence_time": "2024-10-20T00:00:00Z",
            "home_team": "Lakers",
            "away_team": "Celtics",
            "scores": [
                {"name": "Lakers", "score": "0"},
                {"name": "Celtics", "score": "0"},
            ],
        }

        event = create_completed_event(event_data)

        assert event.home_score == 0
        assert event.away_score == 0
        assert event.status == EventStatus.FINAL


class TestApiDictToEvent:
    """Test backward compatibility of deprecated api_dict_to_event function."""

    def test_delegates_to_create_scheduled_event(self):
        """Test that api_dict_to_event delegates to create_scheduled_event."""
        event_data = {
            "id": "abc123",
            "sport_key": "basketball_nba",
            "commence_time": "2024-10-20T00:00:00Z",
            "home_team": "Lakers",
            "away_team": "Celtics",
        }

        event = api_dict_to_event(event_data)

        # Should behave identically to create_scheduled_event
        assert event.status == EventStatus.SCHEDULED
        assert event.home_score is None
        assert event.away_score is None
        assert event.id == "abc123"
