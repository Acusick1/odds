"""Integration tests for database operations."""

from datetime import UTC, datetime, timedelta

import pytest
from odds_core.api_models import create_scheduled_event
from odds_core.models import DataQualityLog, Event, EventStatus, FetchLog, OddsSnapshot, Price
from odds_core.vocabulary import Market, Source, TimeBasis
from odds_lambda.storage.readers import OddsReader
from odds_lambda.storage.shaped_snapshots import SnapshotShape
from odds_lambda.storage.writers import OddsWriter
from sqlalchemy import select

from tests.test_helpers import add_event


class TestDatabaseIntegration:
    """Integration tests for database operations."""

    @pytest.mark.asyncio
    async def test_create_event(self, test_session):
        """Test creating an event in database."""
        event = Event(
            id="test123",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime.now(UTC),
            home_team="Lakers",
            away_team="Celtics",
        )

        result = await add_event(test_session, event)
        await test_session.commit()

        assert result.id == "test123"
        assert result.home_team == "Lakers"
        assert result.status == EventStatus.SCHEDULED

    @pytest.mark.asyncio
    async def test_store_envelope(self, test_session, sample_odds_data):
        """An envelope is stored as one snapshot per market, each with its price rows."""
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
        await add_event(test_session, event)

        stored = await writer.store_envelope(
            event_id=sample_odds_data["id"],
            envelope=sample_odds_data,
            validate=False,
            shape=SnapshotShape(Source.ODDS_API, TimeBasis.OBSERVED, None),
        )

        await test_session.commit()

        assert stored.inserted
        assert stored.snapshot.event_id == sample_odds_data["id"]
        assert stored.snapshots_written == 3
        snapshots = (
            (
                await test_session.execute(
                    select(OddsSnapshot).where(OddsSnapshot.event_id == sample_odds_data["id"])
                )
            )
            .scalars()
            .all()
        )
        assert {s.market for s in snapshots} == {Market.H2H, Market.SPREADS, Market.TOTALS}
        prices = (
            (
                await test_session.execute(
                    select(Price).where(Price.snapshot_id.in_([s.id for s in snapshots]))
                )
            )
            .scalars()
            .all()
        )
        assert {p.snapshot_id for p in prices} == {s.id for s in snapshots}

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
        await add_event(test_session, event)
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
            await add_event(test_session, event)

        await test_session.commit()

        # Query date range (get first 2 out of 3 events)
        start = now - timedelta(hours=1)
        end = now + timedelta(days=1, hours=12)

        events = await reader.get_events_by_date_range(start, end)

        assert len(events) == 2  # Should get 2 out of 3 events (test0 and test1)

    @pytest.mark.asyncio
    async def test_get_events_by_team(self, test_session):
        """Test querying events by team name."""
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

        await add_event(test_session, event1)
        await add_event(test_session, event2)
        await test_session.commit()

        # Query by team
        events = await reader.get_events_by_team("Lakers")

        assert len(events) == 2
        assert all("Lakers" in e.home_team or "Lakers" in e.away_team for e in events)

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
        await add_event(test_session, event)

        await writer.store_envelope(
            event_id=sample_odds_data["id"],
            envelope=sample_odds_data,
            validate=False,
            shape=SnapshotShape(Source.ODDS_API, TimeBasis.OBSERVED, None),
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
        assert "total_price_rows" in stats
        assert "total_snapshots" in stats
        assert stats["total_events"] >= 1
        assert stats["total_price_rows"] > 0
        assert stats["total_snapshots"] >= 1
