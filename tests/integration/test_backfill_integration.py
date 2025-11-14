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


class TestBackfillPlanFromDatabase:
    """Integration tests for database-based backfill plan generation."""

    @pytest.mark.asyncio
    async def test_plan_from_database_with_real_events(self, test_session):
        """Test generating backfill plan from database with real Event records."""
        from datetime import datetime, timezone

        from odds_analytics.game_selector import GameSelector
        from odds_lambda.storage.readers import OddsReader
        from odds_lambda.storage.writers import OddsWriter

        # Insert test events into database
        writer = OddsWriter(test_session)
        test_events = [
            Event(
                id="db_plan_test_1",
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
                id="db_plan_test_2",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime(2024, 1, 16, 20, 0, 0, tzinfo=timezone.utc),
                home_team="Golden State Warriors",
                away_team="Miami Heat",
                status=EventStatus.FINAL,
                home_score=98,
                away_score=102,
            ),
        ]

        for event in test_events:
            await writer.upsert_event(event)
        await test_session.commit()

        # Query events from database
        reader = OddsReader(test_session)
        events = await reader.get_events_by_date_range(
            start_date=datetime(2024, 1, 15, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 17, tzinfo=timezone.utc),
            sport_key="basketball_nba",
            status=EventStatus.FINAL,
        )

        assert len(events) == 2

        # Convert Event models to dict format (same as CLI does)
        from odds_core.time import utc_isoformat

        events_by_date = {}
        for event in events:
            date_str = utc_isoformat(event.commence_time)
            event_dict = {
                "id": event.id,
                "sport_key": event.sport_key,
                "sport_title": event.sport_title,
                "commence_time": event.commence_time.isoformat(),
                "home_team": event.home_team,
                "away_team": event.away_team,
                "bookmakers": [],
            }
            if date_str not in events_by_date:
                events_by_date[date_str] = []
            events_by_date[date_str].append(event_dict)

        # Generate plan using GameSelector
        selector = GameSelector(
            start_date=datetime(2024, 1, 15, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 17, tzinfo=timezone.utc),
            target_games=2,
            games_per_team=1,
        )

        plan = selector.generate_backfill_plan(events_by_date)

        # Verify plan structure
        assert "total_games" in plan
        assert "total_snapshots" in plan
        assert "estimated_quota_usage" in plan
        assert "games" in plan
        assert plan["total_games"] <= 2

    @pytest.mark.asyncio
    async def test_db_plan_output_matches_api_plan_structure(self, test_session):
        """Test that database-sourced plan has same structure as API-sourced plan."""
        from datetime import datetime, timezone

        from odds_analytics.game_selector import GameSelector
        from odds_core.time import utc_isoformat
        from odds_lambda.storage.readers import OddsReader
        from odds_lambda.storage.writers import OddsWriter

        # Insert test event
        writer = OddsWriter(test_session)
        event = Event(
            id="structure_test_1",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime(2024, 1, 15, 19, 0, 0, tzinfo=timezone.utc),
            home_team="Test Team A",
            away_team="Test Team B",
            status=EventStatus.FINAL,
            home_score=100,
            away_score=95,
        )
        await writer.upsert_event(event)
        await test_session.commit()

        # Query from database and convert to dict (DB path)
        reader = OddsReader(test_session)
        db_events = await reader.get_events_by_date_range(
            start_date=datetime(2024, 1, 15, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 16, tzinfo=timezone.utc),
            sport_key="basketball_nba",
            status=EventStatus.FINAL,
        )

        db_events_by_date = {}
        for event in db_events:
            date_str = utc_isoformat(event.commence_time)
            event_dict = {
                "id": event.id,
                "sport_key": event.sport_key,
                "sport_title": event.sport_title,
                "commence_time": event.commence_time.isoformat(),
                "home_team": event.home_team,
                "away_team": event.away_team,
                "bookmakers": [],
            }
            if date_str not in db_events_by_date:
                db_events_by_date[date_str] = []
            db_events_by_date[date_str].append(event_dict)

        # Simulate API response structure (API path)
        api_events_by_date = {
            utc_isoformat(datetime(2024, 1, 15, 19, 0, 0, tzinfo=timezone.utc)): [
                {
                    "id": "structure_test_1",
                    "sport_key": "basketball_nba",
                    "sport_title": "NBA",
                    "commence_time": "2024-01-15T19:00:00+00:00",
                    "home_team": "Test Team A",
                    "away_team": "Test Team B",
                    "bookmakers": [
                        {"key": "fanduel", "markets": []},
                        {"key": "draftkings", "markets": []},
                    ],
                }
            ]
        }

        # Generate plans from both sources
        selector = GameSelector(
            start_date=datetime(2024, 1, 15, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 16, tzinfo=timezone.utc),
            target_games=1,
            games_per_team=1,
        )

        db_plan = selector.generate_backfill_plan(db_events_by_date)
        api_plan = selector.generate_backfill_plan(api_events_by_date)

        # Verify both plans have identical structure
        assert set(db_plan.keys()) == set(api_plan.keys())
        assert db_plan["total_games"] == api_plan["total_games"]
        assert db_plan["total_snapshots"] == api_plan["total_snapshots"]
        assert db_plan["estimated_quota_usage"] == api_plan["estimated_quota_usage"]

        # Verify game entries have same fields
        if db_plan["games"] and api_plan["games"]:
            db_game = db_plan["games"][0]
            api_game = api_plan["games"][0]
            assert set(db_game.keys()) == set(api_game.keys())

    @pytest.mark.asyncio
    async def test_db_plan_empty_date_range(self, test_session):
        """Test that empty date range returns no games in plan."""
        from datetime import datetime, timezone

        from odds_analytics.game_selector import GameSelector
        from odds_lambda.storage.readers import OddsReader

        # Query empty date range
        reader = OddsReader(test_session)
        events = await reader.get_events_by_date_range(
            start_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2025, 1, 2, tzinfo=timezone.utc),
            sport_key="basketball_nba",
            status=EventStatus.FINAL,
        )

        assert len(events) == 0

        # Generate plan with no events
        selector = GameSelector(
            start_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2025, 1, 2, tzinfo=timezone.utc),
            target_games=10,
            games_per_team=1,
        )

        plan = selector.generate_backfill_plan({})

        # Should return valid plan with no games
        assert plan["total_games"] == 0
        assert plan["total_snapshots"] == 0
        assert plan["estimated_quota_usage"] == 0
        assert plan["games"] == []
