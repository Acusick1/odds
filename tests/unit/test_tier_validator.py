"""Unit tests for tier coverage validation."""

from datetime import UTC, date, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from freezegun import freeze_time
from odds_core.models import Event, EventExternalId, EventStatus, OddsSnapshot
from odds_core.vocabulary import Source, TimeBasis
from odds_lambda.fetch_tier import FetchTier
from odds_lambda.storage.tier_validator import TierCoverageReport, TierCoverageValidator
from sqlalchemy.ext.asyncio import AsyncSession

from tests.test_helpers import TIER_HOURS_BEFORE


@pytest.fixture
async def sample_event(test_session):
    """Create a sample completed event."""
    event = Event(
        id=123,
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
            event_id=123,
            home_team="Lakers",
            away_team="Celtics",
            commence_time=datetime.now(UTC),
            tiers_present=frozenset(
                {
                    FetchTier.OPENING,
                    FetchTier.EARLY,
                    FetchTier.SHARP,
                    FetchTier.PREGAME,
                    FetchTier.CLOSING,
                }
            ),
            tiers_missing=frozenset(),
        )

        assert report.is_complete is True
        assert report.has_tier_warnings is False

    def test_is_complete_with_only_warning_tiers_missing(self):
        """Test is_complete returns True when only warning tiers missing."""
        report = TierCoverageReport(
            event_id=123,
            home_team="Lakers",
            away_team="Celtics",
            commence_time=datetime.now(UTC),
            tiers_present=frozenset({FetchTier.PREGAME, FetchTier.CLOSING}),
            tiers_missing=frozenset({FetchTier.OPENING, FetchTier.EARLY, FetchTier.SHARP}),
        )

        # is_complete should be True since critical tiers are present
        assert report.is_complete is True
        assert report.has_tier_warnings is True
        assert report.critical_tiers_missing == frozenset()
        assert report.warning_tiers_missing == frozenset(
            {FetchTier.OPENING, FetchTier.EARLY, FetchTier.SHARP}
        )

    def test_is_complete_with_critical_tiers_missing(self):
        """Test is_complete returns False when critical tiers missing."""
        report = TierCoverageReport(
            event_id=123,
            home_team="Lakers",
            away_team="Celtics",
            commence_time=datetime.now(UTC),
            tiers_present=frozenset({FetchTier.OPENING, FetchTier.EARLY, FetchTier.SHARP}),
            tiers_missing=frozenset({FetchTier.PREGAME, FetchTier.CLOSING}),
        )

        # is_complete should be False since critical tiers are missing
        assert report.is_complete is False
        assert report.critical_tiers_missing == frozenset({FetchTier.PREGAME, FetchTier.CLOSING})
        assert report.warning_tiers_missing == frozenset()

    def test_is_complete_with_mixed_missing_tiers(self):
        """Test is_complete returns False when any critical tier missing."""
        report = TierCoverageReport(
            event_id=123,
            home_team="Lakers",
            away_team="Celtics",
            commence_time=datetime.now(UTC),
            tiers_present=frozenset({FetchTier.OPENING, FetchTier.CLOSING}),
            tiers_missing=frozenset({FetchTier.EARLY, FetchTier.SHARP, FetchTier.PREGAME}),
        )

        # is_complete should be False since PREGAME (critical) is missing
        assert report.is_complete is False
        assert report.critical_tiers_missing == frozenset({FetchTier.PREGAME})
        assert report.warning_tiers_missing == frozenset({FetchTier.EARLY, FetchTier.SHARP})

    def test_coverage_percentage_full(self):
        """Test coverage percentage with all tiers."""
        report = TierCoverageReport(
            event_id=123,
            home_team="Lakers",
            away_team="Celtics",
            commence_time=datetime.now(UTC),
            tiers_present=frozenset(
                {
                    FetchTier.OPENING,
                    FetchTier.EARLY,
                    FetchTier.SHARP,
                    FetchTier.PREGAME,
                    FetchTier.CLOSING,
                }
            ),
            tiers_missing=frozenset(),
        )

        assert report.coverage_percentage == 100.0

    def test_coverage_percentage_partial(self):
        """Test coverage percentage with partial coverage."""
        report = TierCoverageReport(
            event_id=123,
            home_team="Lakers",
            away_team="Celtics",
            commence_time=datetime.now(UTC),
            tiers_present=frozenset({FetchTier.OPENING, FetchTier.CLOSING}),
            tiers_missing=frozenset({FetchTier.EARLY, FetchTier.SHARP, FetchTier.PREGAME}),
        )

        # 2 out of 5 tiers = 40%
        assert report.coverage_percentage == pytest.approx(40.0)

    def test_coverage_percentage_empty(self):
        """Test coverage percentage with no tiers."""
        report = TierCoverageReport(
            event_id=123,
            home_team="Lakers",
            away_team="Celtics",
            commence_time=datetime.now(UTC),
            tiers_present=frozenset(),
            tiers_missing=frozenset(
                {
                    FetchTier.OPENING,
                    FetchTier.EARLY,
                    FetchTier.SHARP,
                    FetchTier.PREGAME,
                    FetchTier.CLOSING,
                }
            ),
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

        for _tier, hours_offset in tiers_and_offsets:
            snapshot = OddsSnapshot(
                event_id=sample_event.id,
                snapshot_time=game_time + timedelta(hours=hours_offset),
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

        for _tier, hours_offset in tiers_and_offsets:
            snapshot = OddsSnapshot(
                event_id=sample_event.id,
                snapshot_time=game_time + timedelta(hours=hours_offset),
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

        for _tier, hours_offset in tiers_and_offsets:
            snapshot = OddsSnapshot(
                event_id=sample_event.id,
                snapshot_time=game_time + timedelta(hours=hours_offset),
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
    async def test_tiers_come_from_the_time_to_kickoff(self, test_session, sample_event):
        test_session.add(
            OddsSnapshot(
                event_id=sample_event.id,
                snapshot_time=sample_event.commence_time - timedelta(hours=2),
            )
        )
        await test_session.commit()

        report = await TierCoverageValidator(test_session).validate_game(sample_event.id)

        assert report.snapshots_by_tier == {FetchTier.CLOSING: 1}

    @pytest.mark.asyncio
    async def test_coverage_follows_the_time_basis(self, test_session, sample_event):
        kickoff = sample_event.commence_time
        for taken, source, basis in (
            (kickoff - timedelta(hours=40), Source.ODDSPORTAL, TimeBasis.SOURCE_STAMPED),
            (kickoff + timedelta(minutes=20), Source.ODDSPORTAL, TimeBasis.SOURCE_STAMPED),
            (kickoff - timedelta(hours=3), Source.FOOTBALL_DATA_UK, TimeBasis.UPPER_BOUND),
        ):
            test_session.add(
                OddsSnapshot(
                    event_id=sample_event.id,
                    snapshot_time=taken,
                    source=source,
                    time_basis=basis,
                )
            )
        await test_session.commit()

        report = await TierCoverageValidator(test_session).validate_game(sample_event.id)

        assert report.snapshots_by_tier == {
            FetchTier.EARLY: 1,
            FetchTier.CLOSING: 1,
            FetchTier.PREGAME: 1,
        }

    @pytest.mark.asyncio
    async def test_validate_game_nonexistent_event(self, test_session):
        """Test validation raises error for nonexistent event."""
        validator = TierCoverageValidator(test_session)

        with pytest.raises(ValueError, match="Event .* not found"):
            await validator.validate_game(999)

    @pytest.mark.asyncio
    async def test_validate_game_snapshots_by_tier_count(self, test_session, sample_event):
        """Test that snapshots_by_tier counts multiple snapshots per tier."""
        # Create multiple snapshots for same tier
        game_time = sample_event.commence_time

        for i in range(3):
            snapshot = OddsSnapshot(
                event_id=sample_event.id,
                snapshot_time=game_time + timedelta(hours=-1, minutes=-i * 10),
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

        # Create 2 complete games on the day
        for i in range(2):
            event = Event(
                id=i + 1,
                sport_key="basketball_nba",
                sport_title="NBA",
                # Games at noon and 1 PM UTC
                commence_time=target_date.replace(hour=12) + timedelta(hours=i),
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
                    snapshot_time=event.commence_time - timedelta(hours=TIER_HOURS_BEFORE[tier]),
                )
                test_session.add(snapshot)

        await test_session.commit()

        # Validate
        validator = TierCoverageValidator(test_session)
        report = await validator.validate_date(target_date.date(), "basketball_nba")

        assert report.is_valid
        assert report.total_games == 2
        assert report.complete_games == 2
        assert report.incomplete_games == 0
        assert report.completion_rate == 100.0

    @pytest.mark.asyncio
    async def test_validate_date_with_warning_games(self, test_session):
        """Test daily validation with games that have only warning tiers missing."""
        target_date = datetime(2024, 10, 24, tzinfo=UTC)

        # Create 2 games on the day
        # First game gets all tiers, second game only gets critical tiers (PREGAME, CLOSING)
        for i in range(2):
            event = Event(
                id=i + 1,
                sport_key="basketball_nba",
                sport_title="NBA",
                # Games at noon and 1 PM UTC
                commence_time=target_date.replace(hour=12) + timedelta(hours=i),
                home_team=f"Home{i}",
                away_team=f"Away{i}",
                status=EventStatus.FINAL,
                home_score=100,
                away_score=95,
            )
            test_session.add(event)

            # First game gets all tiers, second game only gets critical tiers
            tiers = FetchTier if i == 0 else [FetchTier.PREGAME, FetchTier.CLOSING]

            for tier in tiers:
                snapshot = OddsSnapshot(
                    event_id=event.id,
                    snapshot_time=event.commence_time - timedelta(hours=TIER_HOURS_BEFORE[tier]),
                )
                test_session.add(snapshot)

        await test_session.commit()

        # Validate
        validator = TierCoverageValidator(test_session)
        report = await validator.validate_date(target_date.date(), "basketball_nba")

        # Both games should be "complete" (have critical tiers)
        # but second game should have warnings
        assert report.is_valid  # No critical tier issues
        assert report.has_warnings  # Has warning tier issues
        assert report.total_games == 2
        assert report.complete_games == 2
        assert report.incomplete_games == 0
        assert report.games_with_warnings == 1
        assert report.completion_rate == 100.0

    @pytest.mark.asyncio
    async def test_validate_date_with_critical_tier_missing(self, test_session):
        """Test daily validation fails when critical tiers are missing."""
        target_date = datetime(2024, 10, 24, tzinfo=UTC)

        # Create 2 games on the day
        # Second game missing CLOSING (critical tier)
        for i in range(2):
            event = Event(
                id=i + 1,
                sport_key="basketball_nba",
                sport_title="NBA",
                # Games at noon and 1 PM UTC
                commence_time=target_date.replace(hour=12) + timedelta(hours=i),
                home_team=f"Home{i}",
                away_team=f"Away{i}",
                status=EventStatus.FINAL,
                home_score=100,
                away_score=95,
            )
            test_session.add(event)

            # First game gets all tiers, second game missing CLOSING
            tiers = (
                FetchTier
                if i == 0
                else [FetchTier.OPENING, FetchTier.EARLY, FetchTier.SHARP, FetchTier.PREGAME]
            )

            for tier in tiers:
                snapshot = OddsSnapshot(
                    event_id=event.id,
                    snapshot_time=event.commence_time - timedelta(hours=TIER_HOURS_BEFORE[tier]),
                )
                test_session.add(snapshot)

        await test_session.commit()

        # Validate
        validator = TierCoverageValidator(test_session)
        report = await validator.validate_date(target_date.date(), "basketball_nba")

        # Validation should fail because CLOSING (critical) is missing
        assert not report.is_valid
        assert report.total_games == 2
        assert report.complete_games == 1
        assert report.incomplete_games == 1
        assert report.completion_rate == 50.0

    @pytest.mark.asyncio
    async def test_validate_date_missing_tier_breakdown(self, test_session):
        """Test that missing tier breakdown is accurate."""
        target_date = datetime(2024, 10, 24, tzinfo=UTC)

        # Create 3 games on the day, each missing CLOSING tier
        for i in range(3):
            event = Event(
                id=i + 1,
                sport_key="basketball_nba",
                sport_title="NBA",
                # Games at noon, 1 PM, 2 PM UTC
                commence_time=target_date.replace(hour=12) + timedelta(hours=i),
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
                    snapshot_time=event.commence_time - timedelta(hours=TIER_HOURS_BEFORE[tier]),
                )
                test_session.add(snapshot)

        await test_session.commit()

        # Validate
        validator = TierCoverageValidator(test_session)
        report = await validator.validate_date(target_date.date(), "basketball_nba")

        # All 3 games should be missing CLOSING tier
        assert report.missing_tier_breakdown[FetchTier.CLOSING] == 3
        # No games should be missing OPENING tier
        assert FetchTier.OPENING not in report.missing_tier_breakdown


class TestApiGameChecks:
    """The discovery checks match the API's games to events by their Odds API ids."""

    @pytest.fixture
    async def held(self, test_session, sample_event) -> Event:
        test_session.add(
            EventExternalId(source=Source.ODDS_API, external_id="api-123", event_id=sample_event.id)
        )
        sample_event.status = EventStatus.LIVE
        sample_event.home_score = None
        await test_session.commit()
        return sample_event

    @staticmethod
    def _api_game(game_id: str) -> dict:
        return {
            "id": game_id,
            "home_team": "Lakers",
            "away_team": "Celtics",
            "commence_time": "2024-10-24T19:00:00Z",
            "completed": True,
        }

    async def test_a_game_held_as_an_external_id_is_not_missing(self, test_session, held):
        validator = TierCoverageValidator(test_session)

        missing = await validator._check_missing_games(
            held.commence_time, [held], [self._api_game("api-123"), self._api_game("api-999")]
        )

        assert [game["id"] for game in missing] == ["api-999"]

    async def test_a_held_game_without_scores_is_reported_by_event_id(self, test_session, held):
        validator = TierCoverageValidator(test_session)

        missing = await validator._check_missing_scores(
            held.commence_time, [held], [self._api_game("api-123")]
        )

        assert [game["id"] for game in missing] == [held.id]


def _event(event_id: int, commence_time: datetime, sport_key: str = "basketball_nba") -> Event:
    return Event(
        id=event_id,
        sport_key=sport_key,
        sport_title=sport_key,
        commence_time=commence_time,
        home_team=f"Home{event_id}",
        away_team=f"Away{event_id}",
        status=EventStatus.SCHEDULED,
    )


class TestValidateDateScope:
    """A daily validation covers one sport's games on its league-local day."""

    async def test_only_the_sports_games_on_its_local_day_are_validated(
        self, test_session: AsyncSession
    ) -> None:
        test_session.add_all(
            [
                # 22:00 ET on the 24th
                _event(1, datetime(2024, 10, 25, 2, 0, tzinfo=UTC)),
                # 23:30 ET on the 23rd
                _event(2, datetime(2024, 10, 24, 3, 30, tzinfo=UTC)),
                _event(3, datetime(2024, 10, 24, 19, 0, tzinfo=UTC), "soccer_epl"),
            ]
        )
        await test_session.commit()

        report = await TierCoverageValidator(test_session).validate_date(
            date(2024, 10, 24), "basketball_nba"
        )

        assert [game.event_id for game in report.game_reports] == [1]
        assert report.sport_key == "basketball_nba"


class TestFetchApiGames:
    """Discovery asks /scores for the sport's games and keeps those on the day."""

    @staticmethod
    def _client(games: list[dict]) -> MagicMock:
        client = MagicMock()
        client.get_scores = AsyncMock(return_value=MagicMock(scores_data=games))
        return client

    @freeze_time("2024-10-26 15:00:00")
    async def test_requests_the_sports_scores_and_keeps_the_days_games(
        self, test_session: AsyncSession
    ) -> None:
        games = [
            {"id": "a", "commence_time": "2024-10-24T23:30:00Z"},  # 00:30 London on the 25th
            {"id": "b", "commence_time": "2024-10-24T22:59:00Z"},  # 23:59 London on the 24th
            {"id": "c", "commence_time": "2024-10-25T22:00:00Z"},
        ]
        client = self._client(games)

        with patch("odds_lambda.data_fetcher.TheOddsAPIClient", return_value=client):
            fetched = await TierCoverageValidator(test_session)._fetch_api_games(
                date(2024, 10, 25), "soccer_epl"
            )

        client.get_scores.assert_awaited_once_with(sport="soccer_epl", days_from=2)
        assert [game["id"] for game in fetched] == ["a", "c"]

    @pytest.mark.parametrize("day", [date(2024, 10, 27), date(2024, 10, 22)])
    @freeze_time("2024-10-26 15:00:00")
    async def test_a_day_outside_the_scores_range_is_skipped(
        self, test_session: AsyncSession, day: date
    ) -> None:
        client = self._client([])

        with patch("odds_lambda.data_fetcher.TheOddsAPIClient", return_value=client):
            fetched = await TierCoverageValidator(test_session)._fetch_api_games(
                day, "baseball_mlb"
            )

        assert fetched is None
        client.get_scores.assert_not_awaited()
