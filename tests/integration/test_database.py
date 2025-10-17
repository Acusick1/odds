"""Integration tests for database operations."""

from datetime import datetime, timedelta

import pytest

from core.models import EventStatus, Odds
from storage.readers import OddsReader
from storage.writers import OddsWriter


class TestDatabaseIntegration:
    """Integration tests for database operations."""

    @pytest.mark.asyncio
    async def test_create_event(self, test_session):
        """Test creating an event in database."""
        writer = OddsWriter(test_session)

        event_data = {
            "id": "test123",
            "sport_key": "basketball_nba",
            "sport_title": "NBA",
            "commence_time": datetime.utcnow().isoformat() + "Z",
            "home_team": "Lakers",
            "away_team": "Celtics",
        }

        event = await writer.upsert_event(event_data)
        await test_session.commit()

        assert event.id == "test123"
        assert event.home_team == "Lakers"
        assert event.status == EventStatus.SCHEDULED

    @pytest.mark.asyncio
    async def test_update_event(self, test_session):
        """Test updating an existing event."""
        writer = OddsWriter(test_session)

        # Create event
        event_data = {
            "id": "test123",
            "sport_key": "basketball_nba",
            "sport_title": "NBA",
            "commence_time": datetime.utcnow().isoformat() + "Z",
            "home_team": "Lakers",
            "away_team": "Celtics",
        }

        await writer.upsert_event(event_data)
        await test_session.commit()

        # Update event
        event_data["home_team"] = "Golden State Warriors"
        updated_event = await writer.upsert_event(event_data)
        await test_session.commit()

        assert updated_event.home_team == "Golden State Warriors"
        assert updated_event.id == "test123"

    @pytest.mark.asyncio
    async def test_store_odds_snapshot(self, test_session, sample_odds_data):
        """Test storing odds snapshot with hybrid storage."""
        writer = OddsWriter(test_session)

        # First create the event
        event_data = {
            "id": sample_odds_data["id"],
            "sport_key": sample_odds_data["sport_key"],
            "sport_title": sample_odds_data["sport_title"],
            "commence_time": sample_odds_data["commence_time"],
            "home_team": sample_odds_data["home_team"],
            "away_team": sample_odds_data["away_team"],
        }
        await writer.upsert_event(event_data)

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
        event_data = {
            "id": "test123",
            "sport_key": "basketball_nba",
            "sport_title": "NBA",
            "commence_time": datetime.utcnow().isoformat() + "Z",
            "home_team": "Lakers",
            "away_team": "Celtics",
        }

        await writer.upsert_event(event_data)
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
        assert event.status == EventStatus.FINAL
        assert event.home_score == 112
        assert event.away_score == 108
        assert event.completed_at is not None

    @pytest.mark.asyncio
    async def test_log_fetch(self, test_session):
        """Test logging fetch operations."""
        writer = OddsWriter(test_session)

        log = await writer.log_fetch(
            sport_key="basketball_nba",
            events_count=10,
            bookmakers_count=8,
            success=True,
            api_quota_remaining=19950,
            response_time_ms=234,
        )

        await test_session.commit()

        assert log.sport_key == "basketball_nba"
        assert log.success is True
        assert log.api_quota_remaining == 19950

    @pytest.mark.asyncio
    async def test_log_data_quality_issue(self, test_session):
        """Test logging data quality issues."""
        writer = OddsWriter(test_session)

        log = await writer.log_data_quality_issue(
            event_id="test123",
            severity="warning",
            issue_type="suspicious_odds",
            description="Vig too high",
            raw_data={"vig": 20.5},
        )

        await test_session.commit()

        assert log.event_id == "test123"
        assert log.severity == "warning"
        assert log.issue_type == "suspicious_odds"

    @pytest.mark.asyncio
    async def test_get_events_by_date_range(self, test_session):
        """Test querying events by date range."""
        writer = OddsWriter(test_session)
        reader = OddsReader(test_session)

        # Create events with different dates
        now = datetime.utcnow()

        for i in range(3):
            event_data = {
                "id": f"test{i}",
                "sport_key": "basketball_nba",
                "sport_title": "NBA",
                "commence_time": (now + timedelta(days=i)).isoformat() + "Z",
                "home_team": f"Team{i}",
                "away_team": f"Team{i+1}",
            }
            await writer.upsert_event(event_data)

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
        event_data_1 = {
            "id": "test1",
            "sport_key": "basketball_nba",
            "sport_title": "NBA",
            "commence_time": datetime.utcnow().isoformat() + "Z",
            "home_team": "Los Angeles Lakers",
            "away_team": "Boston Celtics",
        }

        event_data_2 = {
            "id": "test2",
            "sport_key": "basketball_nba",
            "sport_title": "NBA",
            "commence_time": datetime.utcnow().isoformat() + "Z",
            "home_team": "Golden State Warriors",
            "away_team": "Los Angeles Lakers",
        }

        await writer.upsert_event(event_data_1)
        await writer.upsert_event(event_data_2)
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
        event_data = {
            "id": sample_odds_data["id"],
            "sport_key": sample_odds_data["sport_key"],
            "sport_title": sample_odds_data["sport_title"],
            "commence_time": sample_odds_data["commence_time"],
            "home_team": sample_odds_data["home_team"],
            "away_team": sample_odds_data["away_team"],
        }
        await writer.upsert_event(event_data)

        # Store snapshot at specific time
        snapshot_time = datetime.utcnow()
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
        event_data = {
            "id": sample_odds_data["id"],
            "sport_key": sample_odds_data["sport_key"],
            "sport_title": sample_odds_data["sport_title"],
            "commence_time": sample_odds_data["commence_time"],
            "home_team": sample_odds_data["home_team"],
            "away_team": sample_odds_data["away_team"],
        }
        await writer.upsert_event(event_data)

        await writer.store_odds_snapshot(
            event_id=sample_odds_data["id"],
            raw_data=sample_odds_data,
            validate=False,
        )

        await writer.log_fetch(
            sport_key="basketball_nba",
            events_count=1,
            bookmakers_count=2,
            success=True,
            api_quota_remaining=19950,
        )

        await test_session.commit()

        # Get stats
        stats = await reader.get_database_stats()

        assert "total_events" in stats
        assert "total_odds_records" in stats
        assert "total_snapshots" in stats
        assert stats["total_events"] >= 1
        assert stats["total_odds_records"] > 0
        assert stats["total_snapshots"] >= 1
