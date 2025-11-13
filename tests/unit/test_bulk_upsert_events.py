"""Unit tests for OddsWriter.bulk_upsert_events() method."""

import time
from datetime import UTC, datetime, timedelta

import pytest
from odds_core.models import Event, EventStatus
from odds_lambda.storage.writers import OddsWriter
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession


@pytest.mark.asyncio
class TestBulkUpsertEvents:
    """Tests for bulk_upsert_events method."""

    async def test_empty_list(self, test_session: AsyncSession):
        """Test handling of empty event list."""
        writer = OddsWriter(test_session)

        result = await writer.bulk_upsert_events([])

        assert result == {"inserted": 0, "updated": 0}

    async def test_insert_new_events(self, test_session: AsyncSession):
        """Test inserting multiple new events."""
        writer = OddsWriter(test_session)

        events = [
            Event(
                id="event1",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime.now(UTC),
                home_team="Lakers",
                away_team="Celtics",
            ),
            Event(
                id="event2",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime.now(UTC) + timedelta(hours=1),
                home_team="Warriors",
                away_team="Heat",
            ),
            Event(
                id="event3",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime.now(UTC) + timedelta(hours=2),
                home_team="Bulls",
                away_team="Nets",
            ),
        ]

        result = await writer.bulk_upsert_events(events)
        await test_session.commit()

        assert result == {"inserted": 3, "updated": 0}

        # Verify events were inserted
        db_result = await test_session.execute(select(Event))
        db_events = db_result.scalars().all()
        assert len(db_events) == 3

        event_ids = {e.id for e in db_events}
        assert event_ids == {"event1", "event2", "event3"}

    async def test_update_existing_events(self, test_session: AsyncSession):
        """Test updating existing events."""
        writer = OddsWriter(test_session)

        # Insert initial events
        initial_events = [
            Event(
                id="event1",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime.now(UTC),
                home_team="Lakers",
                away_team="Celtics",
                status=EventStatus.SCHEDULED,
            ),
            Event(
                id="event2",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime.now(UTC) + timedelta(hours=1),
                home_team="Warriors",
                away_team="Heat",
                status=EventStatus.SCHEDULED,
            ),
        ]

        for event in initial_events:
            test_session.add(event)
        await test_session.commit()

        # Get original created_at timestamps
        test_session.expire_all()
        db_result = await test_session.execute(select(Event).where(Event.id == "event1"))
        original_event1 = db_result.scalar_one()
        original_created_at = original_event1.created_at
        original_updated_at = original_event1.updated_at

        # Wait a moment to ensure updated_at changes
        time.sleep(0.01)

        # Update events with new data
        updated_events = [
            Event(
                id="event1",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime.now(UTC),
                home_team="Lakers",
                away_team="Celtics",
                status=EventStatus.FINAL,
                home_score=110,
                away_score=105,
            ),
            Event(
                id="event2",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime.now(UTC) + timedelta(hours=1),
                home_team="Warriors",
                away_team="Heat",
                status=EventStatus.LIVE,
            ),
        ]

        result = await writer.bulk_upsert_events(updated_events)
        await test_session.commit()

        assert result == {"inserted": 0, "updated": 2}

        # Expire session to get fresh data from DB
        test_session.expire_all()

        # Verify updates
        db_result = await test_session.execute(select(Event).where(Event.id == "event1"))
        updated_event1 = db_result.scalar_one()

        assert updated_event1.status == EventStatus.FINAL
        assert updated_event1.home_score == 110
        assert updated_event1.away_score == 105

        # Verify created_at preserved, updated_at changed
        assert updated_event1.created_at == original_created_at
        assert updated_event1.updated_at > original_updated_at

        # Verify second event
        db_result = await test_session.execute(select(Event).where(Event.id == "event2"))
        updated_event2 = db_result.scalar_one()
        assert updated_event2.status == EventStatus.LIVE

    async def test_mixed_insert_and_update(self, test_session: AsyncSession):
        """Test mixed scenario with both new and existing events."""
        writer = OddsWriter(test_session)

        # Insert one existing event
        existing_event = Event(
            id="event1",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime.now(UTC),
            home_team="Lakers",
            away_team="Celtics",
            status=EventStatus.SCHEDULED,
        )
        test_session.add(existing_event)
        await test_session.commit()

        # Mix of new and existing events
        events = [
            Event(
                id="event1",  # Existing - will update
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime.now(UTC),
                home_team="Lakers",
                away_team="Celtics",
                status=EventStatus.LIVE,
            ),
            Event(
                id="event2",  # New - will insert
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime.now(UTC) + timedelta(hours=1),
                home_team="Warriors",
                away_team="Heat",
            ),
            Event(
                id="event3",  # New - will insert
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime.now(UTC) + timedelta(hours=2),
                home_team="Bulls",
                away_team="Nets",
            ),
        ]

        result = await writer.bulk_upsert_events(events)
        await test_session.commit()

        assert result == {"inserted": 2, "updated": 1}

        # Expire session to get fresh data
        test_session.expire_all()

        # Verify all three events exist
        db_result = await test_session.execute(select(Event))
        db_events = db_result.scalars().all()
        assert len(db_events) == 3

        # Verify event1 was updated
        db_result = await test_session.execute(select(Event).where(Event.id == "event1"))
        event1 = db_result.scalar_one()
        assert event1.status == EventStatus.LIVE

    async def test_performance_100_events(self, test_session: AsyncSession):
        """Test performance with 100+ events (should complete in under 2 seconds)."""
        writer = OddsWriter(test_session)

        # Create 100 events
        events = []
        base_time = datetime.now(UTC)
        for i in range(100):
            events.append(
                Event(
                    id=f"perf_event{i}",
                    sport_key="basketball_nba",
                    sport_title="NBA",
                    commence_time=base_time + timedelta(hours=i),
                    home_team=f"Team{i}",
                    away_team=f"Team{i+1}",
                )
            )

        start_time = time.time()
        result = await writer.bulk_upsert_events(events)
        await test_session.commit()
        elapsed_time = time.time() - start_time

        assert result == {"inserted": 100, "updated": 0}
        assert elapsed_time < 2.0, f"Performance test failed: took {elapsed_time:.2f}s (expected < 2.0s)"

        # Verify all events were inserted
        test_session.expire_all()
        db_result = await test_session.execute(select(Event))
        db_events = db_result.scalars().all()
        assert len(db_events) == 100

    async def test_performance_100_events_mixed(self, test_session: AsyncSession):
        """Test performance with 100+ events including updates (should complete in under 2 seconds)."""
        writer = OddsWriter(test_session)

        # Insert 50 existing events using bulk_upsert to avoid session issues
        existing_events = []
        base_time = datetime.now(UTC)
        for i in range(50):
            existing_events.append(
                Event(
                    id=f"mixed_event{i}",
                    sport_key="basketball_nba",
                    sport_title="NBA",
                    commence_time=base_time + timedelta(hours=i),
                    home_team=f"MixTeam{i}",
                    away_team=f"MixTeam{i+1}",
                    status=EventStatus.SCHEDULED,
                )
            )

        await writer.bulk_upsert_events(existing_events)
        await test_session.commit()

        # Create 100 events (50 updates, 50 inserts)
        events = []
        for i in range(100):
            events.append(
                Event(
                    id=f"mixed_event{i}",
                    sport_key="basketball_nba",
                    sport_title="NBA",
                    commence_time=base_time + timedelta(hours=i),
                    home_team=f"MixTeam{i}",
                    away_team=f"MixTeam{i+1}",
                    status=EventStatus.LIVE if i < 50 else EventStatus.SCHEDULED,
                )
            )

        start_time = time.time()
        result = await writer.bulk_upsert_events(events)
        await test_session.commit()
        elapsed_time = time.time() - start_time

        assert result == {"inserted": 50, "updated": 50}
        assert elapsed_time < 2.0, f"Performance test failed: took {elapsed_time:.2f}s (expected < 2.0s)"

        # Expire session and verify all events exist
        test_session.expire_all()
        db_result = await test_session.execute(
            select(Event).where(Event.id.like("mixed_event%"))
        )
        db_events = db_result.scalars().all()
        assert len(db_events) == 100

    async def test_preserves_created_at_on_update(self, test_session: AsyncSession):
        """Test that created_at is preserved when updating existing events."""
        writer = OddsWriter(test_session)

        # Insert initial event
        initial_event = Event(
            id="event1",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime.now(UTC),
            home_team="Lakers",
            away_team="Celtics",
        )
        test_session.add(initial_event)
        await test_session.commit()

        # Get original created_at
        test_session.expire_all()
        db_result = await test_session.execute(select(Event).where(Event.id == "event1"))
        original_event = db_result.scalar_one()
        original_created_at = original_event.created_at

        # Wait to ensure timestamps would differ
        time.sleep(0.01)

        # Update event
        updated_event = Event(
            id="event1",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime.now(UTC),
            home_team="Lakers Updated",
            away_team="Celtics Updated",
        )

        await writer.bulk_upsert_events([updated_event])
        await test_session.commit()

        # Verify created_at unchanged
        test_session.expire_all()
        db_result = await test_session.execute(select(Event).where(Event.id == "event1"))
        final_event = db_result.scalar_one()

        assert final_event.created_at == original_created_at
        assert final_event.home_team == "Lakers Updated"
        assert final_event.away_team == "Celtics Updated"

    async def test_updates_updated_at_on_conflict(self, test_session: AsyncSession):
        """Test that updated_at is set to current time on conflict."""
        writer = OddsWriter(test_session)

        # Insert initial event
        initial_event = Event(
            id="event1",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime.now(UTC),
            home_team="Lakers",
            away_team="Celtics",
        )
        test_session.add(initial_event)
        await test_session.commit()

        # Get original updated_at
        test_session.expire_all()
        db_result = await test_session.execute(select(Event).where(Event.id == "event1"))
        original_event = db_result.scalar_one()
        original_updated_at = original_event.updated_at

        # Wait to ensure timestamps differ
        time.sleep(0.01)

        # Update event
        updated_event = Event(
            id="event1",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime.now(UTC),
            home_team="Lakers",
            away_team="Celtics",
            status=EventStatus.LIVE,
        )

        await writer.bulk_upsert_events([updated_event])
        await test_session.commit()

        # Verify updated_at changed
        test_session.expire_all()
        db_result = await test_session.execute(select(Event).where(Event.id == "event1"))
        final_event = db_result.scalar_one()

        assert final_event.updated_at > original_updated_at
        assert final_event.status == EventStatus.LIVE
