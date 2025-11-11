"""Integration tests for backfill functionality with real database."""

import pytest
from odds_analytics.backfill_executor import BackfillExecutor
from odds_core.models import Event, EventStatus, Odds, OddsSnapshot
from odds_core.time import parse_api_datetime
from odds_lambda.storage.readers import OddsReader
from sqlalchemy import select


def _api_dict_to_event(event_data: dict) -> Event:
    """Convert API response dict to Event instance for testing."""
    # Parse commence_time with timezone awareness
    commence_time = parse_api_datetime(event_data["commence_time"])

    return Event(
        id=event_data["id"],
        sport_key=event_data["sport_key"],
        sport_title=event_data.get("sport_title", event_data["sport_key"]),
        commence_time=commence_time,
        home_team=event_data["home_team"],
        away_team=event_data["away_team"],
        status=EventStatus.SCHEDULED,
    )


@pytest.fixture
def simple_backfill_plan():
    """Minimal backfill plan for integration testing."""
    return {
        "total_games": 1,
        "total_snapshots": 2,
        "estimated_quota_usage": 60,
        "games": [
            {
                "event_id": "integration_test_1",
                "home_team": "Test Lakers",
                "away_team": "Test Celtics",
                "commence_time": "2024-01-15T19:00:00Z",
                "snapshots": [
                    "2024-01-14T19:00:00Z",
                    "2024-01-15T18:30:00Z",
                ],
                "snapshot_count": 2,
            }
        ],
    }


class TestBackfillIntegration:
    """Integration tests with real database operations."""

    @pytest.mark.asyncio
    async def test_full_backfill_flow(
        self, test_session, mock_session_factory, simple_backfill_plan, mock_api_response_factory
    ):
        """Test complete backfill flow writing to database."""
        from unittest.mock import AsyncMock

        # Create mock API client
        mock_client = AsyncMock()
        mock_client.get_historical_odds = AsyncMock(
            return_value=mock_api_response_factory(
                "integration_test_1", "Test Lakers", "Test Celtics"
            )
        )

        # Execute backfill
        async with BackfillExecutor(
            client=mock_client,
            session_factory=mock_session_factory,
            skip_existing=False,
            rate_limit_seconds=0,
        ) as executor:
            result = await executor.execute_plan(simple_backfill_plan)

            assert result.successful_games == 1
            assert result.successful_snapshots == 2
            assert result.failed_snapshots == 0

        # Verify data was written to database
        reader = OddsReader(test_session)

        # Check event was created
        event = await reader.get_event_by_id("integration_test_1")
        assert event is not None
        assert event.home_team == "Test Lakers"
        assert event.away_team == "Test Celtics"

        # Check snapshots were created
        result = await test_session.execute(
            select(OddsSnapshot).where(OddsSnapshot.event_id == "integration_test_1")
        )
        snapshots = list(result.scalars().all())
        assert len(snapshots) == 2

        # Check odds were created
        result = await test_session.execute(
            select(Odds).where(Odds.event_id == "integration_test_1")
        )
        odds_records = list(result.scalars().all())
        # 2 bookmakers × 2 snapshots × 3 outcomes each (fanduel has 4, draftkings has 2)
        # FanDuel: 2 h2h + 2 spreads = 4, DraftKings: 2 h2h = 2
        # Total per snapshot: 6 outcomes, × 2 snapshots = 12
        assert len(odds_records) == 12

    @pytest.mark.asyncio
    async def test_skip_existing_snapshots(
        self, test_session, mock_session_factory, simple_backfill_plan, mock_api_response_factory
    ):
        """Test that existing snapshots are skipped on subsequent runs."""
        from unittest.mock import AsyncMock

        mock_client = AsyncMock()
        mock_client.get_historical_odds = AsyncMock(
            return_value=mock_api_response_factory(
                "integration_test_1", "Test Lakers", "Test Celtics"
            )
        )

        # First execution - insert data
        async with BackfillExecutor(
            client=mock_client,
            session_factory=mock_session_factory,
            skip_existing=False,
            rate_limit_seconds=0,
        ) as executor:
            result1 = await executor.execute_plan(simple_backfill_plan)
            assert result1.successful_snapshots == 2

        # Second execution - should skip existing
        mock_client.get_historical_odds.reset_mock()  # Reset call count

        async with BackfillExecutor(
            client=mock_client,
            session_factory=mock_session_factory,
            skip_existing=True,
            rate_limit_seconds=0,
        ) as executor:
            result2 = await executor.execute_plan(simple_backfill_plan)

            # Should skip both snapshots
            assert result2.skipped_snapshots == 2
            assert result2.successful_snapshots == 0

            # API should not have been called
            mock_client.get_historical_odds.assert_not_called()

    @pytest.mark.asyncio
    async def test_event_upsert(
        self, test_session, mock_session_factory, simple_backfill_plan, mock_api_response_factory
    ):
        """Test that events are properly upserted on multiple runs."""
        from unittest.mock import AsyncMock

        from odds_lambda.storage.writers import OddsWriter

        # Create event with old data
        writer = OddsWriter(test_session)
        event_data = {
            "id": "integration_test_1",
            "sport_key": "basketball_nba",
            "sport_title": "NBA",
            "commence_time": "2024-01-15T19:00:00Z",
            "home_team": "Old Lakers",
            "away_team": "Old Celtics",
        }
        event = _api_dict_to_event(event_data)
        await writer.upsert_event(event)
        await test_session.commit()

        # Run backfill - should update event
        mock_client = AsyncMock()
        mock_client.get_historical_odds = AsyncMock(
            return_value=mock_api_response_factory(
                "integration_test_1", "Test Lakers", "Test Celtics"
            )
        )

        async with BackfillExecutor(
            client=mock_client,
            session_factory=mock_session_factory,
            skip_existing=False,
            rate_limit_seconds=0,
        ) as executor:
            await executor.execute_plan(simple_backfill_plan)

        # Verify event was updated
        await test_session.refresh(await test_session.get(Event, "integration_test_1"))
        event = await test_session.get(Event, "integration_test_1")
        assert event.home_team == "Test Lakers"
        assert event.away_team == "Test Celtics"

    @pytest.mark.asyncio
    async def test_snapshot_exists_check(
        self, test_session, mock_session_factory, simple_backfill_plan, mock_api_response_factory
    ):
        """Test that OddsReader.snapshot_exists works correctly after backfill."""
        from datetime import datetime
        from unittest.mock import AsyncMock

        mock_client = AsyncMock()
        mock_client.get_historical_odds = AsyncMock(
            return_value=mock_api_response_factory(
                "integration_test_1", "Test Lakers", "Test Celtics"
            )
        )

        # Execute backfill
        async with BackfillExecutor(
            client=mock_client,
            session_factory=mock_session_factory,
            skip_existing=False,
            rate_limit_seconds=0,
        ) as executor:
            await executor.execute_plan(simple_backfill_plan)

        # Test snapshot_exists
        reader = OddsReader(test_session)

        # Should find existing snapshots
        exists1 = await reader.snapshot_exists(
            "integration_test_1", datetime(2024, 1, 14, 19, 0, 0)
        )
        assert exists1 is True

        # Should not find non-existent snapshot
        exists2 = await reader.snapshot_exists(
            "integration_test_1", datetime(2024, 1, 10, 19, 0, 0)
        )
        assert exists2 is False
