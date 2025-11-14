"""Unit tests for backfill plan command with database mode."""

import json
from datetime import datetime, timezone
from io import StringIO
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
            commence_time=datetime(2024, 1, 15, 19, 0, 0, tzinfo=timezone.utc),
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
            commence_time=datetime(2024, 1, 16, 20, 0, 0, tzinfo=timezone.utc),
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
            commence_time=datetime(2024, 1, 17, 19, 30, 0, tzinfo=timezone.utc),
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
    async def test_from_db_queries_database(self, sample_db_events):
        """Test that --from-db mode queries the database instead of API."""
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

            # Execute with from_db=True
            await _create_plan_async(
                start_date_str="2024-01-15",
                end_date_str="2024-01-17",
                target_games=3,
                output_file="test_plan.json",
                sample_interval=1,
                from_db=True,
            )

            # Verify database was queried
            mock_reader.get_events_by_date_range.assert_called_once()
            call_args = mock_reader.get_events_by_date_range.call_args
            assert call_args.kwargs["sport_key"] == "basketball_nba"
            assert call_args.kwargs["status"] == EventStatus.FINAL

    @pytest.mark.asyncio
    async def test_from_db_converts_events_to_dict_format(self, sample_db_events):
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
                sample_interval=1,
                from_db=True,
            )

            # Verify events were converted to dict format
            assert captured_events_by_date is not None
            assert len(captured_events_by_date) > 0

            # Check structure of converted events
            for date_str, events in captured_events_by_date.items():
                assert isinstance(events, list)
                for event_dict in events:
                    assert "id" in event_dict
                    assert "home_team" in event_dict
                    assert "away_team" in event_dict
                    assert "commence_time" in event_dict
                    assert "sport_key" in event_dict
                    assert "sport_title" in event_dict

    @pytest.mark.asyncio
    async def test_from_db_no_events_shows_error(self):
        """Test that helpful error is shown when no events found in database."""
        from odds_cli.commands.backfill import _create_plan_async
        from rich.console import Console

        import typer

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
                    sample_interval=1,
                    from_db=True,
                )

            assert exc_info.value.exit_code == 1

    @pytest.mark.asyncio
    async def test_from_db_adds_data_source_to_plan(self, sample_db_events):
        """Test that data source is added to plan metadata."""
        from odds_cli.commands.backfill import _create_plan_async
        from rich.console import Console

        mock_session = AsyncMock()
        mock_reader = AsyncMock(spec=OddsReader)
        mock_reader.get_events_by_date_range = AsyncMock(return_value=sample_db_events)

        mock_selector = MagicMock()
        mock_selector.generate_backfill_plan = MagicMock(
            return_value={
                "total_games": 3,
                "total_snapshots": 15,
                "estimated_quota_usage": 450,
                "games": [],
            }
        )

        # Capture what was written to the file
        written_content = StringIO()
        m_open = mock_open()
        m_open.return_value.write = written_content.write

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
                sample_interval=1,
                from_db=True,
            )

            # Verify data_source was added to plan
            written_content.seek(0)
            captured_plan = json.loads(written_content.read())
            assert "data_source" in captured_plan
            assert captured_plan["data_source"] == "database"

    @pytest.mark.asyncio
    async def test_api_mode_adds_data_source_as_api(self):
        """Test that API mode sets data source as 'API'."""
        from odds_cli.commands.backfill import _create_plan_async
        from rich.console import Console

        # Mock API client
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get_historical_events = AsyncMock(
            return_value={"data": {"data": []}}  # Empty response
        )

        mock_selector = MagicMock()
        mock_selector.generate_sample_dates = MagicMock(
            return_value=[datetime(2024, 1, 15, tzinfo=timezone.utc)]
        )
        mock_selector.generate_backfill_plan = MagicMock(
            return_value={
                "total_games": 0,
                "total_snapshots": 0,
                "estimated_quota_usage": 0,
                "games": [],
            }
        )

        # Capture what was written to the file
        written_content = StringIO()
        m_open = mock_open()
        m_open.return_value.write = written_content.write

        with (
            patch("odds_cli.commands.backfill.TheOddsAPIClient", return_value=mock_client),
            patch("odds_cli.commands.backfill.GameSelector", return_value=mock_selector),
            patch("odds_cli.commands.backfill.console", Console()),
            patch("builtins.open", m_open),
            patch("odds_cli.commands.backfill.Path", MagicMock()),
            patch("odds_cli.commands.backfill.asyncio.sleep", AsyncMock()),
        ):
            await _create_plan_async(
                start_date_str="2024-01-15",
                end_date_str="2024-01-17",
                target_games=3,
                output_file="test_plan.json",
                sample_interval=1,
                from_db=False,  # API mode
            )

            # Verify data_source was set to API
            written_content.seek(0)
            captured_plan = json.loads(written_content.read())
            assert "data_source" in captured_plan
            assert captured_plan["data_source"] == "API"
