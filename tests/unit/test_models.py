"""Unit tests for database models."""

from datetime import datetime

from core.models import DataQualityLog, Event, EventStatus, FetchLog, Odds, OddsSnapshot


class TestEventModel:
    """Tests for Event model."""

    def test_event_creation(self):
        """Test creating an Event instance."""
        event = Event(
            id="test123",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime.utcnow(),
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
            commence_time=datetime.utcnow(),
            home_team="Lakers",
            away_team="Celtics",
            status=EventStatus.FINAL,
            home_score=112,
            away_score=108,
            completed_at=datetime.utcnow(),
        )

        assert event.status == EventStatus.FINAL
        assert event.home_score == 112
        assert event.away_score == 108
        assert event.completed_at is not None


class TestOddsSnapshotModel:
    """Tests for OddsSnapshot model."""

    def test_odds_snapshot_creation(self, sample_odds_data):
        """Test creating an OddsSnapshot instance."""
        snapshot = OddsSnapshot(
            event_id="test123",
            snapshot_time=datetime.utcnow(),
            raw_data=sample_odds_data,
            bookmaker_count=2,
        )

        assert snapshot.event_id == "test123"
        assert snapshot.bookmaker_count == 2
        assert isinstance(snapshot.raw_data, dict)
        assert "bookmakers" in snapshot.raw_data


class TestOddsModel:
    """Tests for Odds model."""

    def test_odds_creation(self):
        """Test creating an Odds instance."""
        odds = Odds(
            event_id="test123",
            bookmaker_key="fanduel",
            bookmaker_title="FanDuel",
            market_key="h2h",
            outcome_name="Lakers",
            price=-110,
            odds_timestamp=datetime.utcnow(),
            last_update=datetime.utcnow(),
            is_valid=True,
        )

        assert odds.event_id == "test123"
        assert odds.bookmaker_key == "fanduel"
        assert odds.market_key == "h2h"
        assert odds.outcome_name == "Lakers"
        assert odds.price == -110
        assert odds.point is None
        assert odds.is_valid is True

    def test_odds_with_point(self):
        """Test Odds with spread/total line."""
        odds = Odds(
            event_id="test123",
            bookmaker_key="fanduel",
            bookmaker_title="FanDuel",
            market_key="spreads",
            outcome_name="Lakers",
            price=-110,
            point=-2.5,
            odds_timestamp=datetime.utcnow(),
            last_update=datetime.utcnow(),
        )

        assert odds.market_key == "spreads"
        assert odds.point == -2.5

    def test_odds_validation_flag(self):
        """Test Odds validation flag."""
        odds = Odds(
            event_id="test123",
            bookmaker_key="fanduel",
            bookmaker_title="FanDuel",
            market_key="h2h",
            outcome_name="Lakers",
            price=50000,  # Invalid odds
            odds_timestamp=datetime.utcnow(),
            last_update=datetime.utcnow(),
            is_valid=False,
            validation_notes="Price out of valid range",
        )

        assert odds.is_valid is False
        assert odds.validation_notes is not None


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
