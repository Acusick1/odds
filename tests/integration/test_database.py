"""Integration tests for database operations."""

from datetime import UTC, datetime, timedelta

import pytest
from odds_core.api_models import create_scheduled_event
from odds_core.models import DataQualityLog, Event, EventStatus, FetchLog, Odds
from odds_lambda.storage.readers import OddsReader
from odds_lambda.storage.writers import OddsWriter


class TestDatabaseIntegration:
    """Integration tests for database operations."""

    @pytest.mark.asyncio
    async def test_create_event(self, test_session):
        """Test creating an event in database."""
        writer = OddsWriter(test_session)
        event = Event(
            id="test123",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime.now(UTC),
            home_team="Lakers",
            away_team="Celtics",
        )

        result = await writer.upsert_event(event)
        await test_session.commit()

        assert result.id == "test123"
        assert result.home_team == "Lakers"
        assert result.status == EventStatus.SCHEDULED

    @pytest.mark.asyncio
    async def test_update_event(self, test_session):
        """Test updating an existing event."""
        writer = OddsWriter(test_session)

        # Create event
        event = Event(
            id="test123",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime.now(UTC),
            home_team="Lakers",
            away_team="Celtics",
        )

        await writer.upsert_event(event)
        await test_session.commit()

        # Update event
        updated_event = event.model_copy()
        updated_event.home_team = "Golden State Warriors"
        result = await writer.upsert_event(updated_event)
        await test_session.commit()

        assert result.home_team == "Golden State Warriors"
        assert result.id == "test123"

    @pytest.mark.asyncio
    async def test_store_odds_snapshot(self, test_session, sample_odds_data):
        """Test storing odds snapshot with hybrid storage."""
        writer = OddsWriter(test_session)

        # Create event
        event = Event(
            id=sample_odds_data["id"],
            sport_key=sample_odds_data["sport_key"],
            sport_title=sample_odds_data["sport_title"],
            commence_time=datetime.now(UTC),
            home_team=sample_odds_data["home_team"],
            away_team=sample_odds_data["away_team"],
        )
        await writer.upsert_event(event)

        # Store snapshot
        snapshot, odds_records = await writer.store_odds_snapshot(
            event_id=sample_odds_data["id"],
            raw_data=sample_odds_data,
            validate=False,
        )

        await test_session.commit()

        assert snapshot.event_id == sample_odds_data["id"]
        assert snapshot.bookmaker_count == 2
        assert len(odds_records) > 0  # Should have multiple odds records

        # Verify normalized odds were created
        assert all(isinstance(odds, Odds) for odds in odds_records)
        assert any(odds.market_key == "h2h" for odds in odds_records)
        assert any(odds.market_key == "spreads" for odds in odds_records)
        assert any(odds.market_key == "totals" for odds in odds_records)

    @pytest.mark.asyncio
    async def test_update_event_status(self, test_session):
        """Test updating event status and scores."""
        writer = OddsWriter(test_session)
        reader = OddsReader(test_session)

        # Create event
        event = Event(
            id="test123",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime.now(UTC),
            home_team="Lakers",
            away_team="Celtics",
            status=EventStatus.SCHEDULED,
        )
        await writer.upsert_event(event)
        await test_session.commit()

        # Update to final with scores
        await writer.update_event_status(
            event_id="test123",
            status=EventStatus.FINAL,
            home_score=112,
            away_score=108,
        )
        await test_session.commit()

        # Verify update
        event = await reader.get_event_by_id("test123")
        assert event is not None
        assert event.status == EventStatus.FINAL
        assert event.home_score == 112
        assert event.away_score == 108
        assert event.completed_at is not None

    @pytest.mark.asyncio
    async def test_log_fetch(self, test_session):
        """Test logging fetch operations."""
        writer = OddsWriter(test_session)

        fetch_log = FetchLog(
            sport_key="basketball_nba",
            events_count=10,
            bookmakers_count=8,
            success=True,
            api_quota_remaining=19950,
            response_time_ms=234,
        )
        log = await writer.log_fetch(fetch_log)

        await test_session.commit()

        assert log.sport_key == "basketball_nba"
        assert log.success is True
        assert log.api_quota_remaining == 19950

    @pytest.mark.asyncio
    async def test_log_data_quality_issue(self, test_session):
        """Test logging data quality issues."""
        writer = OddsWriter(test_session)

        # Test with no event_id (event_id is nullable)
        quality_log = DataQualityLog(
            event_id=None,
            severity="warning",
            issue_type="suspicious_odds",
            description="Vig too high",
            raw_data={"vig": 20.5},
        )
        log = await writer.log_data_quality_issue(quality_log)

        await test_session.commit()

        assert log.event_id is None
        assert log.severity == "warning"
        assert log.issue_type == "suspicious_odds"

    @pytest.mark.asyncio
    async def test_get_events_by_date_range(self, test_session):
        """Test querying events by date range."""
        writer = OddsWriter(test_session)
        reader = OddsReader(test_session)

        # Create events with different dates
        now = datetime.now(UTC)

        for i in range(3):
            event_data = {
                "id": f"test{i}",
                "sport_key": "basketball_nba",
                "sport_title": "NBA",
                "commence_time": (now + timedelta(days=i)).isoformat(),
                "home_team": f"Team{i}",
                "away_team": f"Team{i + 1}",
            }
            event = create_scheduled_event(event_data)
            await writer.upsert_event(event)

        await test_session.commit()

        # Query date range (get first 2 out of 3 events)
        start = now - timedelta(hours=1)
        end = now + timedelta(days=1, hours=12)

        events = await reader.get_events_by_date_range(start, end)

        assert len(events) == 2  # Should get 2 out of 3 events (test0 and test1)

    @pytest.mark.asyncio
    async def test_get_events_by_team(self, test_session):
        """Test querying events by team name."""
        writer = OddsWriter(test_session)
        reader = OddsReader(test_session)

        # Create events
        event1 = Event(
            id="test1",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime.now(UTC),
            home_team="Los Angeles Lakers",
            away_team="Boston Celtics",
        )
        event2 = Event(
            id="test2",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime.now(UTC),
            home_team="Golden State Warriors",
            away_team="Los Angeles Lakers",
        )

        await writer.upsert_event(event1)
        await writer.upsert_event(event2)
        await test_session.commit()

        # Query by team
        events = await reader.get_events_by_team("Lakers")

        assert len(events) == 2
        assert all("Lakers" in e.home_team or "Lakers" in e.away_team for e in events)

    @pytest.mark.asyncio
    async def test_get_odds_at_time(self, test_session, sample_odds_data):
        """Test getting odds at specific time (critical for backtesting)."""
        writer = OddsWriter(test_session)
        reader = OddsReader(test_session)

        # Create event
        event = Event(
            id=sample_odds_data["id"],
            sport_key=sample_odds_data["sport_key"],
            sport_title=sample_odds_data["sport_title"],
            commence_time=datetime.now(UTC),
            home_team=sample_odds_data["home_team"],
            away_team=sample_odds_data["away_team"],
        )
        await writer.upsert_event(event)

        # Store snapshot at specific time
        snapshot_time = datetime.now(UTC)
        await writer.store_odds_snapshot(
            event_id=sample_odds_data["id"],
            raw_data=sample_odds_data,
            snapshot_time=snapshot_time,
            validate=False,
        )

        await test_session.commit()

        # Query odds at that time
        odds = await reader.get_odds_at_time(
            event_id=sample_odds_data["id"],
            timestamp=snapshot_time,
            tolerance_minutes=5,
        )

        assert len(odds) > 0
        assert all(o.event_id == sample_odds_data["id"] for o in odds)

    @pytest.mark.asyncio
    async def test_get_database_stats(self, test_session, sample_odds_data):
        """Test getting database statistics."""
        writer = OddsWriter(test_session)
        reader = OddsReader(test_session)

        # Create some data
        event = Event(
            id=sample_odds_data["id"],
            sport_key=sample_odds_data["sport_key"],
            sport_title=sample_odds_data["sport_title"],
            commence_time=datetime.now(UTC),
            home_team=sample_odds_data["home_team"],
            away_team=sample_odds_data["away_team"],
        )
        await writer.upsert_event(event)

        await writer.store_odds_snapshot(
            event_id=sample_odds_data["id"],
            raw_data=sample_odds_data,
            validate=False,
        )

        fetch_log = FetchLog(
            sport_key="basketball_nba",
            events_count=1,
            bookmakers_count=2,
            success=True,
            api_quota_remaining=19950,
        )
        await writer.log_fetch(fetch_log)

        await test_session.commit()

        # Get stats
        stats = await reader.get_database_stats()

        assert "total_events" in stats
        assert "total_odds_records" in stats
        assert "total_snapshots" in stats
        assert stats["total_events"] >= 1
        assert stats["total_odds_records"] > 0
        assert stats["total_snapshots"] >= 1

    @pytest.mark.asyncio
    async def test_bulk_upsert_events_empty_list(self, test_session):
        """Test bulk upsert with empty list returns zero counts."""
        writer = OddsWriter(test_session)

        result = await writer.bulk_upsert_events([])
        await test_session.commit()

        assert result == {"inserted": 0, "updated": 0}

    @pytest.mark.asyncio
    async def test_bulk_upsert_events_all_new(self, test_session):
        """Test bulk upsert with all new events (all inserts)."""
        writer = OddsWriter(test_session)
        reader = OddsReader(test_session)

        now = datetime.now(UTC)
        events = [
            Event(
                id=f"bulk_test_{i}",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=now + timedelta(days=i),
                home_team=f"Team{i}",
                away_team=f"Team{i + 1}",
            )
            for i in range(5)
        ]

        result = await writer.bulk_upsert_events(events)
        await test_session.commit()

        # Verify counts
        assert result == {"inserted": 5, "updated": 0}

        # Verify events exist in database
        for i in range(5):
            event = await reader.get_event_by_id(f"bulk_test_{i}")
            assert event is not None
            assert event.home_team == f"Team{i}"
            assert event.status == EventStatus.SCHEDULED

    @pytest.mark.asyncio
    async def test_bulk_upsert_events_all_existing(self, test_session):
        """Test bulk upsert with all existing events (all updates)."""
        writer = OddsWriter(test_session)
        reader = OddsReader(test_session)

        now = datetime.now(UTC)

        # First, create the events
        initial_events = [
            Event(
                id=f"bulk_test_{i}",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=now + timedelta(days=i),
                home_team=f"Team{i}",
                away_team=f"Team{i + 1}",
            )
            for i in range(5)
        ]

        result = await writer.bulk_upsert_events(initial_events)
        await test_session.commit()
        assert result == {"inserted": 5, "updated": 0}

        # Now, update them with modified data
        updated_events = [
            Event(
                id=f"bulk_test_{i}",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=now + timedelta(days=i + 10),  # Changed
                home_team=f"UpdatedTeam{i}",  # Changed
                away_team=f"UpdatedTeam{i + 1}",  # Changed
            )
            for i in range(5)
        ]

        result = await writer.bulk_upsert_events(updated_events)
        await test_session.commit()

        # Verify counts
        assert result == {"inserted": 0, "updated": 5}

        # Verify events were updated
        for i in range(5):
            event = await reader.get_event_by_id(f"bulk_test_{i}")
            assert event is not None
            assert event.home_team == f"UpdatedTeam{i}"
            assert event.away_team == f"UpdatedTeam{i + 1}"

    @pytest.mark.asyncio
    async def test_bulk_upsert_events_mixed(self, test_session):
        """Test bulk upsert with mixed new and existing events."""
        writer = OddsWriter(test_session)
        reader = OddsReader(test_session)

        now = datetime.now(UTC)

        # First, create some events (will be existing)
        existing_events = [
            Event(
                id=f"bulk_test_{i}",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=now + timedelta(days=i),
                home_team=f"Team{i}",
                away_team=f"Team{i + 1}",
            )
            for i in range(3)
        ]

        result = await writer.bulk_upsert_events(existing_events)
        await test_session.commit()
        assert result == {"inserted": 3, "updated": 0}

        # Now, create a mixed batch: update first 3, insert 2 new ones
        mixed_events = [
            # Update existing (IDs 0-2)
            Event(
                id="bulk_test_0",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=now,
                home_team="UpdatedTeam0",
                away_team="UpdatedTeam1",
            ),
            Event(
                id="bulk_test_1",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=now,
                home_team="UpdatedTeam1",
                away_team="UpdatedTeam2",
            ),
            Event(
                id="bulk_test_2",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=now,
                home_team="UpdatedTeam2",
                away_team="UpdatedTeam3",
            ),
            # New events (IDs 3-4)
            Event(
                id="bulk_test_3",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=now + timedelta(days=3),
                home_team="Team3",
                away_team="Team4",
            ),
            Event(
                id="bulk_test_4",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=now + timedelta(days=4),
                home_team="Team4",
                away_team="Team5",
            ),
        ]

        result = await writer.bulk_upsert_events(mixed_events)
        await test_session.commit()

        # Verify counts
        assert result == {"inserted": 2, "updated": 3}

        # Verify updated events
        event0 = await reader.get_event_by_id("bulk_test_0")
        assert event0.home_team == "UpdatedTeam0"

        # Verify new events
        event3 = await reader.get_event_by_id("bulk_test_3")
        assert event3 is not None
        assert event3.home_team == "Team3"

    @pytest.mark.asyncio
    async def test_bulk_upsert_events_with_helper_functions(self, test_session):
        """Test bulk upsert with events created via helper functions."""
        writer = OddsWriter(test_session)
        reader = OddsReader(test_session)

        # Create events using helper function from api_models
        event_data_list = [
            {
                "id": f"helper_test_{i}",
                "sport_key": "basketball_nba",
                "sport_title": "NBA",
                "commence_time": (datetime.now(UTC) + timedelta(days=i)).isoformat(),
                "home_team": f"Team{i}",
                "away_team": f"Team{i + 1}",
            }
            for i in range(3)
        ]

        events = [create_scheduled_event(data) for data in event_data_list]

        result = await writer.bulk_upsert_events(events)
        await test_session.commit()

        # Verify counts
        assert result == {"inserted": 3, "updated": 0}

        # Verify events exist
        for i in range(3):
            event = await reader.get_event_by_id(f"helper_test_{i}")
            assert event is not None
            assert event.status == EventStatus.SCHEDULED

    @pytest.mark.asyncio
    async def test_bulk_upsert_events_performance(self, test_session):
        """Test bulk upsert performance with 100+ records."""
        import time

        writer = OddsWriter(test_session)
        reader = OddsReader(test_session)

        now = datetime.now(UTC)
        num_events = 150

        # Create 150 events
        events = [
            Event(
                id=f"perf_test_{i}",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=now + timedelta(days=i),
                home_team=f"Team{i}",
                away_team=f"Team{i + 1}",
            )
            for i in range(num_events)
        ]

        # Measure execution time
        start_time = time.time()
        result = await writer.bulk_upsert_events(events)
        await test_session.commit()
        elapsed = time.time() - start_time

        # Verify counts
        assert result == {"inserted": num_events, "updated": 0}

        # Verify performance (should complete in under 2 seconds per requirements)
        assert elapsed < 2.0, f"Bulk upsert took {elapsed:.2f}s, expected < 2.0s"

        # Verify random samples exist
        sample_ids = ["perf_test_0", "perf_test_50", "perf_test_149"]
        for event_id in sample_ids:
            event = await reader.get_event_by_id(event_id)
            assert event is not None

    @pytest.mark.asyncio
    async def test_bulk_upsert_events_preserves_scores(self, test_session):
        """Test that bulk upsert doesn't overwrite completed event scores."""
        writer = OddsWriter(test_session)
        reader = OddsReader(test_session)

        now = datetime.now(UTC)

        # Create completed event with scores
        completed_event = Event(
            id="completed_test",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=now - timedelta(days=1),
            home_team="Lakers",
            away_team="Celtics",
            status=EventStatus.FINAL,
            home_score=108,
            away_score=105,
            completed_at=now,
        )

        await writer.upsert_event(completed_event)
        await test_session.commit()

        # Now try to bulk upsert with updated commence_time but no scores
        update_event = Event(
            id="completed_test",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=now,  # Updated time
            home_team="Lakers",
            away_team="Celtics",
        )

        result = await writer.bulk_upsert_events([update_event])
        await test_session.commit()

        assert result == {"inserted": 0, "updated": 1}

        # Verify scores and status are preserved (not overwritten)
        # Note: This test documents current behavior. The bulk upsert
        # only updates: sport_key, sport_title, home_team, away_team,
        # commence_time, updated_at. It does NOT update status or scores.
        event = await reader.get_event_by_id("completed_test")
        assert event.status == EventStatus.FINAL
        assert event.home_score == 108
        assert event.away_score == 105
