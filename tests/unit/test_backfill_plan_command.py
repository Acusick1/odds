"""Unit tests for backfill plan command with database mode."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest
from odds_core.models import Event, EventStatus
from odds_lambda.storage.readers import OddsReader


@pytest.fixture
def sample_db_events():
    """Sample Event models representing games in database."""
    return [
        Event(
            id="test_event_1",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime(2024, 1, 15, 19, 0, 0, tzinfo=UTC),
            home_team="Los Angeles Lakers",
            away_team="Boston Celtics",
            status=EventStatus.FINAL,
            home_score=110,
            away_score=105,
        ),
        Event(
            id="test_event_2",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime(2024, 1, 16, 20, 0, 0, tzinfo=UTC),
            home_team="Golden State Warriors",
            away_team="Miami Heat",
            status=EventStatus.FINAL,
            home_score=98,
            away_score=102,
        ),
        Event(
            id="test_event_3",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime(2024, 1, 17, 19, 30, 0, tzinfo=UTC),
            home_team="Chicago Bulls",
            away_team="Phoenix Suns",
            status=EventStatus.FINAL,
            home_score=115,
            away_score=108,
        ),
    ]


class TestBackfillPlanFromDatabase:
    """Test database-based backfill plan generation."""

    @pytest.mark.asyncio
    async def test_queries_database(self, sample_db_events):
        """Test that command queries the database for events."""
        from odds_cli.commands.backfill import _create_plan_async
        from rich.console import Console

        # Mock the database session and reader
        mock_session = AsyncMock()
        mock_reader = AsyncMock(spec=OddsReader)
        mock_reader.get_events_by_date_range = AsyncMock(return_value=sample_db_events)

        # Mock GameSelector
        mock_selector = MagicMock()
        mock_selector.generate_backfill_plan = MagicMock(
            return_value={
                "total_games": 3,
                "total_snapshots": 15,
                "estimated_quota_usage": 450,
                "games": [],
                "start_date": "2024-01-15",
                "end_date": "2024-01-17",
            }
        )

        m_open = mock_open()

        with (
            patch("odds_cli.commands.backfill.async_session_maker") as mock_session_maker,
            patch("odds_lambda.storage.readers.OddsReader", return_value=mock_reader),
            patch("odds_cli.commands.backfill.GameSelector", return_value=mock_selector),
            patch("odds_cli.commands.backfill.console", Console()),
            patch("builtins.open", m_open),
            patch("odds_cli.commands.backfill.Path", MagicMock()),
        ):
            mock_session_maker.return_value.__aenter__.return_value = mock_session

            # Execute plan creation
            await _create_plan_async(
                start_date_str="2024-01-15",
                end_date_str="2024-01-17",
                target_games=3,
                output_file="test_plan.json",
            )

            # Verify database was queried
            mock_reader.get_events_by_date_range.assert_called_once()
            call_args = mock_reader.get_events_by_date_range.call_args
            assert call_args.kwargs["sport_key"] == "basketball_nba"

    @pytest.mark.asyncio
    async def test_converts_events_to_dict_format(self, sample_db_events):
        """Test that Event models are converted to dict format for GameSelector."""
        from odds_cli.commands.backfill import _create_plan_async
        from rich.console import Console

        mock_session = AsyncMock()
        mock_reader = AsyncMock(spec=OddsReader)
        mock_reader.get_events_by_date_range = AsyncMock(return_value=sample_db_events)

        captured_events_by_date = None

        def capture_generate_plan(events_by_date):
            nonlocal captured_events_by_date
            captured_events_by_date = events_by_date
            return {
                "total_games": 3,
                "total_snapshots": 15,
                "estimated_quota_usage": 450,
                "games": [],
                "start_date": "2024-01-15",
                "end_date": "2024-01-17",
            }

        mock_selector = MagicMock()
        mock_selector.generate_backfill_plan = MagicMock(side_effect=capture_generate_plan)

        m_open = mock_open()

        with (
            patch("odds_cli.commands.backfill.async_session_maker") as mock_session_maker,
            patch("odds_lambda.storage.readers.OddsReader", return_value=mock_reader),
            patch("odds_cli.commands.backfill.GameSelector", return_value=mock_selector),
            patch("odds_cli.commands.backfill.console", Console()),
            patch("builtins.open", m_open),
            patch("odds_cli.commands.backfill.Path", MagicMock()),
        ):
            mock_session_maker.return_value.__aenter__.return_value = mock_session

            await _create_plan_async(
                start_date_str="2024-01-15",
                end_date_str="2024-01-17",
                target_games=3,
                output_file="test_plan.json",
            )

            # Verify events were converted to dict format
            assert captured_events_by_date is not None
            assert len(captured_events_by_date) > 0

            # Check structure of converted events
            for _, events in captured_events_by_date.items():
                assert isinstance(events, list)
                for event_dict in events:
                    assert "id" in event_dict
                    assert "home_team" in event_dict
                    assert "away_team" in event_dict
                    assert "commence_time" in event_dict
                    assert "sport_key" in event_dict
                    assert "sport_title" in event_dict

    @pytest.mark.asyncio
    async def test_no_events_shows_error(self):
        """Test that helpful error is shown when no events found in database."""
        import typer
        from odds_cli.commands.backfill import _create_plan_async
        from rich.console import Console

        mock_session = AsyncMock()
        mock_reader = AsyncMock(spec=OddsReader)
        # Return empty list
        mock_reader.get_events_by_date_range = AsyncMock(return_value=[])

        with (
            patch("odds_cli.commands.backfill.async_session_maker") as mock_session_maker,
            patch("odds_lambda.storage.readers.OddsReader", return_value=mock_reader),
            patch("odds_cli.commands.backfill.console", Console()),
        ):
            mock_session_maker.return_value.__aenter__.return_value = mock_session

            # Should raise Exit(1)
            with pytest.raises(typer.Exit) as exc_info:
                await _create_plan_async(
                    start_date_str="2024-01-15",
                    end_date_str="2024-01-17",
                    target_games=3,
                    output_file="test_plan.json",
                )

            assert exc_info.value.exit_code == 1
