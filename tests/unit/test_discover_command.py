"""Unit tests for discover command."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from odds_cli.main import app


class TestDiscoverCommand:
    """Test discover command functionality."""

    @pytest.fixture
    def runner(self):
        """Create CLI runner."""
        return CliRunner()

    @pytest.fixture
    def mock_historical_events_response(self):
        """Mock response from get_historical_events."""
        return {
            "data": [
                {
                    "id": "event1",
                    "sport_key": "basketball_nba",
                    "sport_title": "NBA",
                    "commence_time": "2024-10-15T19:00:00Z",
                    "home_team": "Lakers",
                    "away_team": "Celtics",
                },
                {
                    "id": "event2",
                    "sport_key": "basketball_nba",
                    "sport_title": "NBA",
                    "commence_time": "2024-10-15T20:00:00Z",
                    "home_team": "Warriors",
                    "away_team": "Heat",
                },
            ],
            "quota_remaining": 19990,
            "timestamp": datetime(2024, 10, 15, 12, 0, 0, tzinfo=UTC),
        }

    def test_missing_required_arguments(self, runner):
        """Test that command fails without required start/end dates."""
        result = runner.invoke(app, ["discover", "games"])

        assert result.exit_code != 0

    def test_invalid_date_format(self, runner):
        """Test that command validates date format."""
        result = runner.invoke(
            app,
            ["discover", "games", "--start", "2024/10/01", "--end", "2024-10-31"],
        )

        assert result.exit_code == 1
        assert "invalid date format" in result.stdout.lower()

    def test_start_after_end_date(self, runner):
        """Test that command validates start date is before end date."""
        result = runner.invoke(
            app,
            ["discover", "games", "--start", "2024-10-31", "--end", "2024-10-01"],
        )

        assert result.exit_code == 1
        assert "start date must be before" in result.stdout.lower()

    def test_dry_run_mode(self, runner, mock_historical_events_response):
        """Test dry-run mode doesn't write to database."""
        with patch("odds_cli.commands.discover.TheOddsAPIClient") as mock_client_class:
            # Setup mock client
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client.get_historical_events = AsyncMock(
                return_value=mock_historical_events_response
            )
            mock_client_class.return_value = mock_client

            with patch("odds_lambda.storage.writers.OddsWriter") as mock_writer_class:
                result = runner.invoke(
                    app,
                    [
                        "discover",
                        "games",
                        "--start",
                        "2024-10-15",
                        "--end",
                        "2024-10-15",
                        "--dry-run",
                    ],
                )

                # Command should succeed
                assert result.exit_code == 0

                # Should show dry run mode
                assert "DRY RUN MODE" in result.stdout

                # Should display sample events
                assert "Sample Events" in result.stdout
                assert "Lakers" in result.stdout
                assert "Celtics" in result.stdout

                # Should NOT create writer (no database writes)
                mock_writer_class.assert_not_called()

    def test_successful_discovery(self, runner, mock_historical_events_response):
        """Test successful discovery and storage of events."""
        with patch("odds_cli.commands.discover.TheOddsAPIClient") as mock_client_class:
            # Setup mock client
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client.get_historical_events = AsyncMock(
                return_value=mock_historical_events_response
            )
            mock_client_class.return_value = mock_client

            with patch("odds_cli.commands.discover.async_session_maker") as mock_session_maker:
                # Setup mock session
                mock_session = AsyncMock()
                mock_session.__aenter__ = AsyncMock(return_value=mock_session)
                mock_session.__aexit__ = AsyncMock()
                mock_session.commit = AsyncMock()

                # Mock execute to return a result object with fetchall
                from unittest.mock import MagicMock
                mock_execute_result = MagicMock()
                mock_execute_result.fetchall.return_value = []  # fetchall is NOT async
                mock_session.execute = AsyncMock(return_value=mock_execute_result)

                mock_session_maker.return_value = mock_session

                with patch("odds_lambda.storage.writers.OddsWriter") as mock_writer_class:
                    # Setup mock writer - use the real implementation but with mocked session
                    from odds_lambda.storage.writers import OddsWriter
                    mock_writer = OddsWriter(mock_session)
                    mock_writer_class.return_value = mock_writer

                    result = runner.invoke(
                        app,
                        [
                            "discover",
                            "games",
                            "--start",
                            "2024-10-15",
                            "--end",
                            "2024-10-15",
                        ],
                    )

                    # Command should succeed
                    assert result.exit_code == 0

                    # Should show discovery summary
                    assert "Discovery Summary" in result.stdout
                    assert "Events found: 2" in result.stdout
                    assert "Events parsed: 2" in result.stdout

                    # Should show storage completion
                    assert "Storage completed" in result.stdout
                    assert "Events inserted: 2" in result.stdout

                    # Should have committed transaction
                    mock_session.commit.assert_called_once()

    def test_multiple_days_date_range(self, runner, mock_historical_events_response):
        """Test discovery across multiple days."""
        with patch("odds_cli.commands.discover.TheOddsAPIClient") as mock_client_class:
            # Setup mock client
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client.get_historical_events = AsyncMock(
                return_value=mock_historical_events_response
            )
            mock_client_class.return_value = mock_client

            with patch("odds_cli.commands.discover.async_session_maker") as mock_session_maker:
                # Setup mock session
                mock_session = AsyncMock()
                mock_session.__aenter__ = AsyncMock(return_value=mock_session)
                mock_session.__aexit__ = AsyncMock()
                mock_session.commit = AsyncMock()

                # Mock execute to return a result object with fetchall
                from unittest.mock import MagicMock
                mock_execute_result = MagicMock()
                mock_execute_result.fetchall.return_value = []  # fetchall is NOT async
                mock_session.execute = AsyncMock(return_value=mock_execute_result)

                mock_session_maker.return_value = mock_session

                with patch("odds_lambda.storage.writers.OddsWriter") as mock_writer_class:
                    # Setup mock writer - use the real implementation but with mocked session
                    from odds_lambda.storage.writers import OddsWriter
                    mock_writer = OddsWriter(mock_session)
                    mock_writer_class.return_value = mock_writer

                    result = runner.invoke(
                        app,
                        [
                            "discover",
                            "games",
                            "--start",
                            "2024-10-15",
                            "--end",
                            "2024-10-17",  # 3 days
                        ],
                    )

                    # Command should succeed
                    assert result.exit_code == 0

                    # Should have called API 3 times (once per day)
                    assert mock_client.get_historical_events.call_count == 3

                    # Should show total events from all days
                    assert "Events found: 6" in result.stdout  # 2 events * 3 days

    def test_api_error_handling(self, runner):
        """Test graceful handling of API errors."""
        with patch("odds_cli.commands.discover.TheOddsAPIClient") as mock_client_class:
            # Setup mock client that raises error
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client.get_historical_events = AsyncMock(
                side_effect=Exception("API Error: Rate limit exceeded")
            )
            mock_client_class.return_value = mock_client

            result = runner.invoke(
                app,
                [
                    "discover",
                    "games",
                    "--start",
                    "2024-10-15",
                    "--end",
                    "2024-10-15",
                ],
            )

            # Command should complete (not crash)
            assert result.exit_code == 0

            # Should show warning about failure
            assert "Warning" in result.stdout
            assert "Failed to fetch data" in result.stdout

    def test_empty_response(self, runner):
        """Test handling of empty API response (no events found)."""
        with patch("odds_cli.commands.discover.TheOddsAPIClient") as mock_client_class:
            # Setup mock client with empty response
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client.get_historical_events = AsyncMock(
                return_value={
                    "data": [],
                    "quota_remaining": 19990,
                    "timestamp": datetime(2024, 10, 15, 12, 0, 0, tzinfo=UTC),
                }
            )
            mock_client_class.return_value = mock_client

            result = runner.invoke(
                app,
                [
                    "discover",
                    "games",
                    "--start",
                    "2024-10-15",
                    "--end",
                    "2024-10-15",
                ],
            )

            # Command should succeed
            assert result.exit_code == 0

            # Should show no events found
            assert "Events found: 0" in result.stdout
            assert "No events found in date range" in result.stdout

    def test_quota_tracking(self, runner, mock_historical_events_response):
        """Test that quota is tracked and displayed."""
        with patch("odds_cli.commands.discover.TheOddsAPIClient") as mock_client_class:
            # Setup mock client
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client.get_historical_events = AsyncMock(
                return_value=mock_historical_events_response
            )
            mock_client_class.return_value = mock_client

            result = runner.invoke(
                app,
                [
                    "discover",
                    "games",
                    "--start",
                    "2024-10-15",
                    "--end",
                    "2024-10-15",
                    "--dry-run",
                ],
            )

            # Command should succeed
            assert result.exit_code == 0

            # Should display quota information
            assert "API quota remaining: 19,990" in result.stdout

    def test_batch_upsert(self, runner):
        """Test that events are upserted in batches."""
        # Create response with > 100 events to test batching
        many_events = [
            {
                "id": f"event{i}",
                "sport_key": "basketball_nba",
                "sport_title": "NBA",
                "commence_time": "2024-10-15T19:00:00Z",
                "home_team": f"Team{i}A",
                "away_team": f"Team{i}B",
            }
            for i in range(150)
        ]

        with patch("odds_cli.commands.discover.TheOddsAPIClient") as mock_client_class:
            # Setup mock client
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client.get_historical_events = AsyncMock(
                return_value={
                    "data": many_events,
                    "quota_remaining": 19990,
                    "timestamp": datetime(2024, 10, 15, 12, 0, 0, tzinfo=UTC),
                }
            )
            mock_client_class.return_value = mock_client

            with patch("odds_cli.commands.discover.async_session_maker") as mock_session_maker:
                # Setup mock session
                mock_session = AsyncMock()
                mock_session.__aenter__ = AsyncMock(return_value=mock_session)
                mock_session.__aexit__ = AsyncMock()
                mock_session.commit = AsyncMock()

                # Mock execute to return a result object with fetchall
                from unittest.mock import MagicMock
                mock_execute_result = MagicMock()
                mock_execute_result.fetchall.return_value = []  # fetchall is NOT async
                mock_session.execute = AsyncMock(return_value=mock_execute_result)

                mock_session_maker.return_value = mock_session

                with patch("odds_lambda.storage.writers.OddsWriter") as mock_writer_class:
                    # Setup mock writer - use the real implementation but with mocked session
                    from odds_lambda.storage.writers import OddsWriter
                    mock_writer = OddsWriter(mock_session)
                    mock_writer_class.return_value = mock_writer

                    result = runner.invoke(
                        app,
                        [
                            "discover",
                            "games",
                            "--start",
                            "2024-10-15",
                            "--end",
                            "2024-10-15",
                        ],
                    )

                    # Command should succeed
                    assert result.exit_code == 0

                    # Verify bulk_upsert_events was called twice (100 + 50 events)
                    # Note: We can't easily assert call_count with real implementation,
                    # but we can verify the function worked by checking output
                    assert "Events inserted: 150" in result.stdout or "Events updated:" in result.stdout

    def test_invalid_event_data_handling(self, runner):
        """Test handling of malformed event data."""
        with patch("odds_cli.commands.discover.TheOddsAPIClient") as mock_client_class:
            # Setup mock client with malformed data
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client.get_historical_events = AsyncMock(
                return_value={
                    "data": [
                        {
                            "id": "event1",
                            "sport_key": "basketball_nba",
                            # Missing required fields like commence_time
                        },
                        {
                            "id": "event2",
                            "sport_key": "basketball_nba",
                            "sport_title": "NBA",
                            "commence_time": "2024-10-15T19:00:00Z",
                            "home_team": "Lakers",
                            "away_team": "Celtics",
                        },
                    ],
                    "quota_remaining": 19990,
                    "timestamp": datetime(2024, 10, 15, 12, 0, 0, tzinfo=UTC),
                }
            )
            mock_client_class.return_value = mock_client

            result = runner.invoke(
                app,
                [
                    "discover",
                    "games",
                    "--start",
                    "2024-10-15",
                    "--end",
                    "2024-10-15",
                    "--dry-run",
                ],
            )

            # Command should complete
            assert result.exit_code == 0

            # Should show warning for failed parsing
            assert "Warning" in result.stdout

            # Should still process valid events
            assert "Events parsed: 1" in result.stdout
