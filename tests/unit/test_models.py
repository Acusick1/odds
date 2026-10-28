"""Unit tests for database models."""

from datetime import UTC, datetime

import pytest
from odds_core.models import DataQualityLog, Event, EventStatus, FetchLog, OddsSnapshot
from odds_core.vocabulary import Market, Source, TimeBasis


class TestEventModel:
    """Tests for Event model."""

    def test_event_creation(self):
        """Test creating an Event instance."""
        event = Event(
            id="test123",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime.now(UTC),
            home_team="Lakers",
            away_team="Celtics",
            status=EventStatus.SCHEDULED,
        )

        assert event.id == "test123"
        assert event.sport_key == "basketball_nba"
        assert event.home_team == "Lakers"
        assert event.away_team == "Celtics"
        assert event.status == EventStatus.SCHEDULED
        assert event.home_score is None
        assert event.away_score is None

    def test_event_status_enum(self):
        """Test EventStatus enum values."""
        assert EventStatus.SCHEDULED.value == "scheduled"
        assert EventStatus.LIVE.value == "live"
        assert EventStatus.FINAL.value == "final"
        assert EventStatus.CANCELLED.value == "cancelled"
        assert EventStatus.POSTPONED.value == "postponed"

    def test_event_with_scores(self):
        """Test Event with final scores."""
        event = Event(
            id="test123",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime.now(UTC),
            home_team="Lakers",
            away_team="Celtics",
            status=EventStatus.FINAL,
            home_score=112,
            away_score=108,
            completed_at=datetime.now(UTC),
        )

        assert event.status == EventStatus.FINAL
        assert event.home_score == 112
        assert event.away_score == 108
        assert event.completed_at is not None


class TestEventStatusTransitions:
    @pytest.mark.parametrize(
        ("current", "target"),
        [
            (EventStatus.SCHEDULED, EventStatus.LIVE),
            (EventStatus.SCHEDULED, EventStatus.POSTPONED),
            (EventStatus.LIVE, EventStatus.FINAL),
            (EventStatus.LIVE, EventStatus.SCHEDULED),
            (EventStatus.LIVE, EventStatus.CANCELLED),
            (EventStatus.POSTPONED, EventStatus.SCHEDULED),
            (EventStatus.CANCELLED, EventStatus.FINAL),
            (EventStatus.FINAL, EventStatus.FINAL),
            (EventStatus.SCHEDULED, EventStatus.UNMATCHED),
            (EventStatus.POSTPONED, EventStatus.UNMATCHED),
            (EventStatus.UNMATCHED, EventStatus.SCHEDULED),
        ],
    )
    def test_allowed(self, current: EventStatus, target: EventStatus) -> None:
        assert current.can_become(target)

    @pytest.mark.parametrize("target", [s for s in EventStatus if s is not EventStatus.FINAL])
    def test_final_never_moves_to_another_status(self, target: EventStatus) -> None:
        assert not EventStatus.FINAL.can_become(target)

    @pytest.mark.parametrize(
        ("current", "target"),
        [
            (EventStatus.CANCELLED, EventStatus.SCHEDULED),
            (EventStatus.CANCELLED, EventStatus.POSTPONED),
            (EventStatus.CANCELLED, EventStatus.UNMATCHED),
            (EventStatus.UNMATCHED, EventStatus.FINAL),
            (EventStatus.UNMATCHED, EventStatus.LIVE),
        ],
    )
    def test_refused(self, current: EventStatus, target: EventStatus) -> None:
        assert not current.can_become(target)


class TestOddsSnapshotModel:
    """Tests for OddsSnapshot model."""

    def test_odds_snapshot_creation(self):
        """Test creating an OddsSnapshot instance."""
        snapshot = OddsSnapshot(
            event_id="test123",
            snapshot_time=datetime.now(UTC),
            source=Source.ODDS_API,
            market=Market.H2H,
            time_basis=TimeBasis.OBSERVED,
        )

        assert snapshot.event_id == "test123"
        assert snapshot.source is Source.ODDS_API
        assert snapshot.payload_id is None


class TestDataQualityLogModel:
    """Tests for DataQualityLog model."""

    def test_data_quality_log_creation(self):
        """Test creating a DataQualityLog instance."""
        log = DataQualityLog(
            event_id="test123",
            severity="warning",
            issue_type="suspicious_odds",
            description="Vig too high",
            raw_data={"vig": 20.5},
        )

        assert log.event_id == "test123"
        assert log.severity == "warning"
        assert log.issue_type == "suspicious_odds"
        assert log.description == "Vig too high"
        assert isinstance(log.raw_data, dict)


class TestFetchLogModel:
    """Tests for FetchLog model."""

    def test_fetch_log_creation(self):
        """Test creating a FetchLog instance."""
        log = FetchLog(
            sport_key="basketball_nba",
            events_count=10,
            bookmakers_count=8,
            success=True,
            api_quota_remaining=19950,
            response_time_ms=234,
        )

        assert log.sport_key == "basketball_nba"
        assert log.events_count == 10
        assert log.bookmakers_count == 8
        assert log.success is True
        assert log.api_quota_remaining == 19950
        assert log.response_time_ms == 234
        assert log.error_message is None

    def test_fetch_log_failure(self):
        """Test FetchLog for failed fetch."""
        log = FetchLog(
            sport_key="basketball_nba",
            events_count=0,
            bookmakers_count=0,
            success=False,
            error_message="API timeout",
        )

        assert log.success is False
        assert log.error_message == "API timeout"
