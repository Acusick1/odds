"""Unit tests for tier coverage validation."""

from datetime import UTC, datetime, timedelta

import pytest

from core.fetch_tier import FetchTier
from core.models import Event, EventStatus, OddsSnapshot
from storage.tier_validator import TierCoverageReport, TierCoverageValidator


@pytest.fixture
async def sample_event(test_session):
    """Create a sample completed event."""
    event = Event(
        id="test_event_123",
        sport_key="basketball_nba",
        sport_title="NBA",
        commence_time=datetime(2024, 10, 24, 19, 0, 0, tzinfo=UTC),
        home_team="Lakers",
        away_team="Celtics",
        status=EventStatus.FINAL,
        home_score=110,
        away_score=105,
    )
    test_session.add(event)
    await test_session.commit()
    await test_session.refresh(event)
    return event


class TestTierCoverageReport:
    """Test TierCoverageReport dataclass."""

    def test_is_complete_with_all_tiers(self):
        """Test is_complete returns True when no tiers missing."""
        report = TierCoverageReport(
            event_id="test123",
            home_team="Lakers",
            away_team="Celtics",
            commence_time=datetime.now(UTC),
            tiers_present={
                FetchTier.OPENING,
                FetchTier.EARLY,
                FetchTier.SHARP,
                FetchTier.PREGAME,
                FetchTier.CLOSING,
            },
            tiers_missing=set(),
        )

        assert report.is_complete is True

    def test_is_complete_with_missing_tiers(self):
        """Test is_complete returns False when tiers missing."""
        report = TierCoverageReport(
            event_id="test123",
            home_team="Lakers",
            away_team="Celtics",
            commence_time=datetime.now(UTC),
            tiers_present={FetchTier.OPENING, FetchTier.CLOSING},
            tiers_missing={FetchTier.EARLY, FetchTier.SHARP, FetchTier.PREGAME},
        )

        assert report.is_complete is False

    def test_coverage_percentage_full(self):
        """Test coverage percentage with all tiers."""
        report = TierCoverageReport(
            event_id="test123",
            home_team="Lakers",
            away_team="Celtics",
            commence_time=datetime.now(UTC),
            tiers_present={
                FetchTier.OPENING,
                FetchTier.EARLY,
                FetchTier.SHARP,
                FetchTier.PREGAME,
                FetchTier.CLOSING,
            },
            tiers_missing=set(),
        )

        assert report.coverage_percentage == 100.0

    def test_coverage_percentage_partial(self):
        """Test coverage percentage with partial coverage."""
        report = TierCoverageReport(
            event_id="test123",
            home_team="Lakers",
            away_team="Celtics",
            commence_time=datetime.now(UTC),
            tiers_present={FetchTier.OPENING, FetchTier.CLOSING},
            tiers_missing={FetchTier.EARLY, FetchTier.SHARP, FetchTier.PREGAME},
        )

        # 2 out of 5 tiers = 40%
        assert report.coverage_percentage == pytest.approx(40.0)

    def test_coverage_percentage_empty(self):
        """Test coverage percentage with no tiers."""
        report = TierCoverageReport(
            event_id="test123",
            home_team="Lakers",
            away_team="Celtics",
            commence_time=datetime.now(UTC),
            tiers_present=set(),
            tiers_missing={
                FetchTier.OPENING,
                FetchTier.EARLY,
                FetchTier.SHARP,
                FetchTier.PREGAME,
                FetchTier.CLOSING,
            },
        )

        assert report.coverage_percentage == 0.0


class TestTierCoverageValidator:
    """Test TierCoverageValidator class."""

    @pytest.mark.asyncio
    async def test_validate_game_all_tiers_present(self, test_session, sample_event):
        """Test validation passes when all tiers present."""
        # Create snapshots for all 5 tiers
        game_time = sample_event.commence_time
        tiers_and_offsets = [
            (FetchTier.OPENING, -100),  # 100 hours before
            (FetchTier.EARLY, -48),  # 2 days before
            (FetchTier.SHARP, -18),  # 18 hours before
            (FetchTier.PREGAME, -6),  # 6 hours before
            (FetchTier.CLOSING, -1),  # 1 hour before
        ]

        for tier, hours_offset in tiers_and_offsets:
            snapshot = OddsSnapshot(
                event_id=sample_event.id,
                snapshot_time=game_time + timedelta(hours=hours_offset),
                raw_data={"test": "data"},
                bookmaker_count=8,
                fetch_tier=tier.value,
                hours_until_commence=abs(hours_offset),
            )
            test_session.add(snapshot)

        await test_session.commit()

        # Validate
        validator = TierCoverageValidator(test_session)
        report = await validator.validate_game(sample_event.id)

        assert report.is_complete
        assert len(report.tiers_present) == 5
        assert len(report.tiers_missing) == 0
        assert report.total_snapshots == 5

    @pytest.mark.asyncio
    async def test_validate_game_missing_tiers(self, test_session, sample_event):
        """Test validation fails when tiers missing."""
        # Create snapshots for only 3 tiers
        game_time = sample_event.commence_time
        tiers_and_offsets = [
            (FetchTier.OPENING, -100),
            (FetchTier.EARLY, -48),
            (FetchTier.CLOSING, -1),
        ]

        for tier, hours_offset in tiers_and_offsets:
            snapshot = OddsSnapshot(
                event_id=sample_event.id,
                snapshot_time=game_time + timedelta(hours=hours_offset),
                raw_data={"test": "data"},
                bookmaker_count=8,
                fetch_tier=tier.value,
                hours_until_commence=abs(hours_offset),
            )
            test_session.add(snapshot)

        await test_session.commit()

        # Validate
        validator = TierCoverageValidator(test_session)
        report = await validator.validate_game(sample_event.id)

        assert not report.is_complete
        assert len(report.tiers_present) == 3
        assert len(report.tiers_missing) == 2
        assert FetchTier.SHARP in report.tiers_missing
        assert FetchTier.PREGAME in report.tiers_missing

    @pytest.mark.asyncio
    async def test_validate_game_custom_required_tiers(self, test_session, sample_event):
        """Test validation with custom required tiers."""
        # Create snapshots for OPENING and CLOSING only
        game_time = sample_event.commence_time
        tiers_and_offsets = [
            (FetchTier.OPENING, -100),
            (FetchTier.CLOSING, -1),
        ]

        for tier, hours_offset in tiers_and_offsets:
            snapshot = OddsSnapshot(
                event_id=sample_event.id,
                snapshot_time=game_time + timedelta(hours=hours_offset),
                raw_data={"test": "data"},
                bookmaker_count=8,
                fetch_tier=tier.value,
                hours_until_commence=abs(hours_offset),
            )
            test_session.add(snapshot)

        await test_session.commit()

        # Validate with custom required tiers
        validator = TierCoverageValidator(test_session)
        required_tiers = {FetchTier.OPENING, FetchTier.CLOSING}
        report = await validator.validate_game(sample_event.id, required_tiers=required_tiers)

        assert report.is_complete
        assert len(report.tiers_present) == 2
        assert len(report.tiers_missing) == 0

    @pytest.mark.asyncio
    async def test_validate_game_nonexistent_event(self, test_session):
        """Test validation raises error for nonexistent event."""
        validator = TierCoverageValidator(test_session)

        with pytest.raises(ValueError, match="Event .* not found"):
            await validator.validate_game("nonexistent_event_id")

    @pytest.mark.asyncio
    async def test_validate_game_snapshots_by_tier_count(self, test_session, sample_event):
        """Test that snapshots_by_tier counts multiple snapshots per tier."""
        # Create multiple snapshots for same tier
        game_time = sample_event.commence_time

        for i in range(3):
            snapshot = OddsSnapshot(
                event_id=sample_event.id,
                snapshot_time=game_time + timedelta(hours=-1, minutes=-i * 10),
                raw_data={"test": f"data_{i}"},
                bookmaker_count=8,
                fetch_tier=FetchTier.CLOSING.value,
                hours_until_commence=1.0 + (i * 10 / 60),
            )
            test_session.add(snapshot)

        await test_session.commit()

        # Validate
        validator = TierCoverageValidator(test_session)
        report = await validator.validate_game(sample_event.id)

        assert report.total_snapshots == 3
        assert report.snapshots_by_tier[FetchTier.CLOSING] == 3


class TestTierCoverageValidatorDailyValidation:
    """Test daily validation functionality."""

    @pytest.mark.asyncio
    async def test_validate_date_with_complete_games(self, test_session):
        """Test daily validation with all games complete."""
        target_date = datetime(2024, 10, 24, tzinfo=UTC)

        # Create 2 complete games
        for i in range(2):
            event = Event(
                id=f"event_{i}",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=target_date + timedelta(hours=i),
                home_team=f"Home{i}",
                away_team=f"Away{i}",
                status=EventStatus.FINAL,
                home_score=100,
                away_score=95,
            )
            test_session.add(event)

            # Add all 5 tiers for each game
            for tier in FetchTier:
                snapshot = OddsSnapshot(
                    event_id=event.id,
                    snapshot_time=event.commence_time - timedelta(hours=1),
                    raw_data={"test": "data"},
                    bookmaker_count=8,
                    fetch_tier=tier.value,
                    hours_until_commence=1.0,
                )
                test_session.add(snapshot)

        await test_session.commit()

        # Validate
        validator = TierCoverageValidator(test_session)
        report = await validator.validate_date(target_date.date())

        assert report.is_valid
        assert report.total_games == 2
        assert report.complete_games == 2
        assert report.incomplete_games == 0
        assert report.completion_rate == 100.0

    @pytest.mark.asyncio
    async def test_validate_date_with_incomplete_games(self, test_session):
        """Test daily validation with incomplete games."""
        target_date = datetime(2024, 10, 24, tzinfo=UTC)

        # Create 2 games, one complete and one incomplete
        for i in range(2):
            event = Event(
                id=f"event_{i}",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=target_date + timedelta(hours=i),
                home_team=f"Home{i}",
                away_team=f"Away{i}",
                status=EventStatus.FINAL,
                home_score=100,
                away_score=95,
            )
            test_session.add(event)

            # First game gets all tiers, second game only gets 2 tiers
            tiers = FetchTier if i == 0 else [FetchTier.OPENING, FetchTier.CLOSING]

            for tier in tiers:
                snapshot = OddsSnapshot(
                    event_id=event.id,
                    snapshot_time=event.commence_time - timedelta(hours=1),
                    raw_data={"test": "data"},
                    bookmaker_count=8,
                    fetch_tier=tier.value,
                    hours_until_commence=1.0,
                )
                test_session.add(snapshot)

        await test_session.commit()

        # Validate
        validator = TierCoverageValidator(test_session)
        report = await validator.validate_date(target_date.date())

        assert not report.is_valid
        assert report.total_games == 2
        assert report.complete_games == 1
        assert report.incomplete_games == 1
        assert report.completion_rate == 50.0

    @pytest.mark.asyncio
    async def test_validate_date_missing_tier_breakdown(self, test_session):
        """Test that missing tier breakdown is accurate."""
        target_date = datetime(2024, 10, 24, tzinfo=UTC)

        # Create 3 games, each missing CLOSING tier
        for i in range(3):
            event = Event(
                id=f"event_{i}",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=target_date + timedelta(hours=i),
                home_team=f"Home{i}",
                away_team=f"Away{i}",
                status=EventStatus.FINAL,
                home_score=100,
                away_score=95,
            )
            test_session.add(event)

            # Add all tiers except CLOSING
            for tier in [FetchTier.OPENING, FetchTier.EARLY, FetchTier.SHARP, FetchTier.PREGAME]:
                snapshot = OddsSnapshot(
                    event_id=event.id,
                    snapshot_time=event.commence_time - timedelta(hours=1),
                    raw_data={"test": "data"},
                    bookmaker_count=8,
                    fetch_tier=tier.value,
                    hours_until_commence=1.0,
                )
                test_session.add(snapshot)

        await test_session.commit()

        # Validate
        validator = TierCoverageValidator(test_session)
        report = await validator.validate_date(target_date.date())

        # All 3 games should be missing CLOSING tier
        assert report.missing_tier_breakdown[FetchTier.CLOSING] == 3
        # No games should be missing OPENING tier
        assert FetchTier.OPENING not in report.missing_tier_breakdown
