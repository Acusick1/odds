"""Integration tests for backfill functionality with real database."""

import pytest
from odds_analytics.backfill_executor import BackfillExecutor
from odds_core.models import Event, EventExternalId, EventStatus, OddsSnapshot, Price
from odds_core.time import parse_api_datetime
from odds_core.vocabulary import Source, TimeBasis
from odds_lambda.storage.readers import OddsReader
from sqlalchemy import select

from tests.test_helpers import add_event


def _api_dict_to_event(event_id: int, event_data: dict) -> Event:
    """Convert API response dict to Event instance for testing."""
    # Parse commence_time with timezone awareness
    commence_time = parse_api_datetime(event_data["commence_time"])

    return Event(
        id=event_id,
        sport_key=event_data["sport_key"],
        sport_title=event_data.get("sport_title", event_data["sport_key"]),
        commence_time=commence_time,
        home_team=event_data["home_team"],
        away_team=event_data["away_team"],
        status=EventStatus.SCHEDULED,
    )


async def _store_planned_event(session) -> Event:
    event = _api_dict_to_event(
        1,
        {
            "id": "integration_test_1",
            "sport_key": "basketball_nba",
            "sport_title": "NBA",
            "commence_time": "2024-01-15T19:00:00Z",
            "home_team": "Test Lakers",
            "away_team": "Test Celtics",
        },
    )
    await add_event(session, event)
    session.add(
        EventExternalId(source=Source.ODDS_API, external_id="integration_test_1", event_id=event.id)
    )
    await session.commit()
    return event


@pytest.fixture
def simple_backfill_plan():
    """Minimal backfill plan for integration testing."""
    return {
        "total_games": 1,
        "total_snapshots": 2,
        "estimated_quota_usage": 60,
        "games": [
            {
                "event_id": 1,
                "odds_api_id": "integration_test_1",
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
            sport="basketball_nba",
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
        # Check event was created
        event = (await test_session.execute(select(Event))).scalar_one()
        assert event.home_team == "Test Lakers"
        assert event.away_team == "Test Celtics"

        # Check snapshots were created
        result = await test_session.execute(
            select(OddsSnapshot).where(OddsSnapshot.event_id == event.id)
        )
        snapshots = list(result.scalars().all())
        # One snapshot per market (h2h and spreads) at each of the plan's two times
        assert len(snapshots) == 4

        # Check price rows were created
        result = await test_session.execute(
            select(Price).where(Price.snapshot_id.in_([s.id for s in snapshots]))
        )
        prices = list(result.scalars().all())
        # FanDuel: 2 h2h + 2 spreads = 4, DraftKings: 2 h2h = 2
        # Total per time: 6 prices, × 2 times = 12
        assert len(prices) == 12

    @pytest.mark.asyncio
    async def test_skip_existing_snapshots(
        self, test_session, mock_session_factory, simple_backfill_plan, mock_api_response_factory
    ):
        """Test that existing snapshots are skipped on subsequent runs."""
        from unittest.mock import AsyncMock

        await _store_planned_event(test_session)
        mock_client = AsyncMock()
        mock_client.get_historical_odds = AsyncMock(
            return_value=mock_api_response_factory(
                "integration_test_1", "Test Lakers", "Test Celtics"
            )
        )

        # First execution - insert data
        async with BackfillExecutor(
            sport="basketball_nba",
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
            sport="basketball_nba",
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
    async def test_stored_event_is_found_by_its_api_id(
        self, test_session, mock_session_factory, simple_backfill_plan, mock_api_response_factory
    ):
        """The backfill stores its snapshot on the event holding the API's id, and adds no event."""
        from unittest.mock import AsyncMock

        # Create event with old data
        event_data = {
            "id": "integration_test_1",
            "sport_key": "basketball_nba",
            "sport_title": "NBA",
            "commence_time": "2024-01-15T19:00:00Z",
            "home_team": "Old Lakers",
            "away_team": "Old Celtics",
        }
        event = _api_dict_to_event(1, event_data)
        await add_event(test_session, event)
        test_session.add(
            EventExternalId(
                source=Source.ODDS_API,
                external_id="integration_test_1",
                event_id=event.id,
            )
        )
        await test_session.commit()

        # Run backfill - should attach to the stored event
        mock_client = AsyncMock()
        mock_client.get_historical_odds = AsyncMock(
            return_value=mock_api_response_factory(
                "integration_test_1", "Test Lakers", "Test Celtics"
            )
        )

        async with BackfillExecutor(
            sport="basketball_nba",
            client=mock_client,
            session_factory=mock_session_factory,
            skip_existing=False,
            rate_limit_seconds=0,
        ) as executor:
            await executor.execute_plan(simple_backfill_plan)

        events = (await test_session.execute(select(Event))).scalars().all()
        assert [e.id for e in events] == [1]
        assert events[0].home_team == "Old Lakers"
        snapshots = (await test_session.execute(select(OddsSnapshot))).scalars().all()
        assert {s.event_id for s in snapshots} == {1}
        assert {s.time_basis for s in snapshots} == {TimeBasis.UPPER_BOUND}

    @pytest.mark.asyncio
    async def test_snapshot_exists_check(
        self, test_session, mock_session_factory, simple_backfill_plan, mock_api_response_factory
    ):
        """Test that OddsReader.snapshot_exists works correctly after backfill."""
        from datetime import datetime
        from unittest.mock import AsyncMock

        event = await _store_planned_event(test_session)
        mock_client = AsyncMock()
        mock_client.get_historical_odds = AsyncMock(
            return_value=mock_api_response_factory(
                "integration_test_1", "Test Lakers", "Test Celtics"
            )
        )

        # Execute backfill
        async with BackfillExecutor(
            sport="basketball_nba",
            client=mock_client,
            session_factory=mock_session_factory,
            skip_existing=False,
            rate_limit_seconds=0,
        ) as executor:
            await executor.execute_plan(simple_backfill_plan)

        # Test snapshot_exists
        reader = OddsReader(test_session)

        # Should find existing snapshots
        exists1 = await reader.snapshot_exists(event.id, datetime(2024, 1, 14, 19, 0, 0))
        assert exists1 is True

        # Should not find non-existent snapshot
        exists2 = await reader.snapshot_exists(event.id, datetime(2024, 1, 10, 19, 0, 0))
        assert exists2 is False
