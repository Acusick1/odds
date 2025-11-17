"""Unit tests for NBA score fetcher."""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest
from odds_lambda.nba_score_fetcher import NBAScoreFetcher


class TestNBAScoreFetcher:
    """Tests for NBAScoreFetcher class."""

    def test_teams_match_exact(self):
        """Test exact team name matching."""
        fetcher = NBAScoreFetcher()

        # Exact match
        assert fetcher._teams_match("Lakers", "Lakers")
        assert fetcher._teams_match("LAL", "LAL")

    def test_teams_match_case_insensitive(self):
        """Test case-insensitive team matching."""
        fetcher = NBAScoreFetcher()

        assert fetcher._teams_match("Lakers", "lakers")
        assert fetcher._teams_match("LAKERS", "Lakers")
        assert fetcher._teams_match("lal", "LAL")

    def test_teams_match_contains(self):
        """Test team matching with partial names."""
        fetcher = NBAScoreFetcher()

        # Team name contained in full name
        assert fetcher._teams_match("Lakers", "Los Angeles Lakers")
        assert fetcher._teams_match("Celtics", "Boston Celtics")
        assert fetcher._teams_match("Warriors", "Golden State Warriors")

    def test_teams_match_no_match(self):
        """Test team matching returns false for non-matching teams."""
        fetcher = NBAScoreFetcher()

        assert not fetcher._teams_match("Lakers", "Celtics")
        assert not fetcher._teams_match("LAL", "BOS")
        assert not fetcher._teams_match("Warriors", "Lakers")

    @patch("odds_lambda.nba_score_fetcher.scoreboard.ScoreBoard")
    def test_get_live_scores(self, mock_scoreboard):
        """Test fetching live scores."""
        # Mock NBA API response
        mock_board = MagicMock()
        mock_board.get_dict.return_value = {
            "scoreboard": {
                "games": [
                    {
                        "gameId": "0022400123",
                        "homeTeam": {
                            "teamTricode": "LAL",
                            "score": 110,
                        },
                        "awayTeam": {
                            "teamTricode": "BOS",
                            "score": 105,
                        },
                        "gameStatusText": "Final",
                        "gameTimeUTC": "2024-01-15T02:00:00Z",
                    }
                ]
            }
        }
        mock_scoreboard.return_value = mock_board

        fetcher = NBAScoreFetcher()
        results = fetcher.get_live_scores()

        assert len(results) == 1
        game = results[0]
        assert game["game_id"] == "0022400123"
        assert game["home_team"] == "LAL"
        assert game["away_team"] == "BOS"
        assert game["home_score"] == 110
        assert game["away_score"] == 105
        assert game["game_status"] == "Final"
        assert isinstance(game["game_date"], datetime)

    @patch("odds_lambda.nba_score_fetcher.leaguegamefinder.LeagueGameFinder")
    def test_get_historical_scores(self, mock_game_finder):
        """Test fetching historical scores."""
        # Mock NBA API response (pandas DataFrame)
        import pandas as pd

        mock_finder = MagicMock()
        mock_df = pd.DataFrame(
            {
                "GAME_ID": ["0022300456", "0022300456"],
                "GAME_DATE": ["2024-01-15", "2024-01-15"],
                "MATCHUP": ["LAL vs. BOS", "BOS @ LAL"],
                "TEAM_ABBREVIATION": ["LAL", "BOS"],
                "PTS": [110, 105],
            }
        )
        mock_finder.get_data_frames.return_value = [mock_df]
        mock_game_finder.return_value = mock_finder

        fetcher = NBAScoreFetcher()
        start_date = datetime(2024, 1, 1, tzinfo=UTC)
        end_date = datetime(2024, 1, 31, tzinfo=UTC)

        results = fetcher.get_historical_scores(start_date, end_date)

        assert len(results) == 1
        game = results[0]
        assert game["game_id"] == "0022300456"
        assert game["home_team"] == "LAL"
        assert game["away_team"] == "BOS"
        assert game["home_score"] == 110
        assert game["away_score"] == 105
        assert isinstance(game["game_date"], datetime)

        # Verify API was called with correct date format
        mock_game_finder.assert_called_once()
        call_kwargs = mock_game_finder.call_args.kwargs
        assert call_kwargs["date_from_nullable"] == "2024-01-01"
        assert call_kwargs["date_to_nullable"] == "2024-01-31"

    @patch("odds_lambda.nba_score_fetcher.leaguegamefinder.LeagueGameFinder")
    def test_match_game_by_teams_and_date(self, mock_game_finder):
        """Test matching a game by teams and date."""
        # Mock NBA API response
        import pandas as pd

        mock_finder = MagicMock()
        mock_df = pd.DataFrame(
            {
                "GAME_ID": ["0022300456", "0022300456"],
                "GAME_DATE": ["2024-01-15", "2024-01-15"],
                "MATCHUP": ["LAL vs. BOS", "BOS @ LAL"],
                "TEAM_ABBREVIATION": ["LAL", "BOS"],
                "PTS": [110, 105],
            }
        )
        mock_finder.get_data_frames.return_value = [mock_df]
        mock_game_finder.return_value = mock_finder

        fetcher = NBAScoreFetcher()
        game_date = datetime(2024, 1, 15, 12, 0, tzinfo=UTC)

        # Match with exact team abbreviations
        matched = fetcher.match_game_by_teams_and_date("LAL", "BOS", game_date)

        assert matched is not None
        assert matched["game_id"] == "0022300456"
        assert matched["home_team"] == "LAL"
        assert matched["away_team"] == "BOS"
        assert matched["home_score"] == 110
        assert matched["away_score"] == 105

    @patch("odds_lambda.nba_score_fetcher.leaguegamefinder.LeagueGameFinder")
    def test_match_game_by_teams_and_date_not_found(self, mock_game_finder):
        """Test matching returns None when no game found."""
        # Mock empty NBA API response
        import pandas as pd

        mock_finder = MagicMock()
        mock_df = pd.DataFrame(
            {
                "GAME_ID": [],
                "GAME_DATE": [],
                "MATCHUP": [],
                "TEAM_ABBREVIATION": [],
                "PTS": [],
            }
        )
        mock_finder.get_data_frames.return_value = [mock_df]
        mock_game_finder.return_value = mock_finder

        fetcher = NBAScoreFetcher()
        game_date = datetime(2024, 1, 15, 12, 0, tzinfo=UTC)

        # Try to match with no results
        matched = fetcher.match_game_by_teams_and_date("LAL", "BOS", game_date)

        assert matched is None

    @patch("odds_lambda.nba_score_fetcher.leaguegamefinder.LeagueGameFinder")
    def test_get_historical_scores_with_team_filter(self, mock_game_finder):
        """Test fetching historical scores with team filter."""
        # Mock NBA API response
        import pandas as pd

        mock_finder = MagicMock()
        mock_df = pd.DataFrame(
            {
                "GAME_ID": ["0022300456", "0022300456"],
                "GAME_DATE": ["2024-01-15", "2024-01-15"],
                "MATCHUP": ["LAL vs. BOS", "BOS @ LAL"],
                "TEAM_ABBREVIATION": ["LAL", "BOS"],
                "PTS": [110, 105],
            }
        )
        mock_finder.get_data_frames.return_value = [mock_df]
        mock_game_finder.return_value = mock_finder

        fetcher = NBAScoreFetcher()
        start_date = datetime(2024, 1, 1, tzinfo=UTC)
        end_date = datetime(2024, 1, 31, tzinfo=UTC)

        results = fetcher.get_historical_scores(start_date, end_date, team="LAL")

        assert len(results) == 1

        # Verify team filter was passed to API
        call_kwargs = mock_game_finder.call_args.kwargs
        assert call_kwargs["team_id_nullable"] == "LAL"

    @patch("odds_lambda.nba_score_fetcher.scoreboard.ScoreBoard")
    def test_get_live_scores_error_handling(self, mock_scoreboard):
        """Test error handling in get_live_scores."""
        # Mock API error
        mock_scoreboard.side_effect = Exception("API Error")

        fetcher = NBAScoreFetcher()

        with pytest.raises(Exception, match="API Error"):
            fetcher.get_live_scores()

    @patch("odds_lambda.nba_score_fetcher.leaguegamefinder.LeagueGameFinder")
    def test_get_historical_scores_error_handling(self, mock_game_finder):
        """Test error handling in get_historical_scores."""
        # Mock API error
        mock_game_finder.side_effect = Exception("API Error")

        fetcher = NBAScoreFetcher()
        start_date = datetime(2024, 1, 1, tzinfo=UTC)
        end_date = datetime(2024, 1, 31, tzinfo=UTC)

        with pytest.raises(Exception, match="API Error"):
            fetcher.get_historical_scores(start_date, end_date)
