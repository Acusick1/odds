"""Unit tests for QualityMetrics module."""

from datetime import UTC, datetime

import pytest
from odds_analytics.quality_metrics import (
    GameCountResult,
    GameMissingScoresResult,
    GameWithOddsResult,
    GameWithScoresResult,
    QualityMetrics,
    TierCoverage,
)
from odds_core.models import Event, EventStatus, OddsSnapshot
from odds_lambda.fetch_tier import FetchTier
from sqlalchemy.ext.asyncio import AsyncSession


@pytest.mark.asyncio
class TestQualityMetricsGameCounts:
    """Tests for get_game_counts() method."""

    async def test_get_game_counts_empty_database(self, test_session: AsyncSession):
        """Test game counts with no data in database."""
        metrics = QualityMetrics(test_session)

        result = await metrics.get_game_counts(
            start_date=datetime(2024, 10, 1, tzinfo=UTC),
            end_date=datetime(2024, 10, 31, tzinfo=UTC),
        )

        assert isinstance(result, GameCountResult)
        assert result.total_games == 0
        assert result.sport_key is None
        assert result.start_date == datetime(2024, 10, 1, tzinfo=UTC)
        assert result.end_date == datetime(2024, 10, 31, tzinfo=UTC)

    async def test_get_game_counts_with_events(self, test_session: AsyncSession):
        """Test game counts with events in database."""
        # Create test events
        events = [
            Event(
                id="event1",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime(2024, 10, 15, 19, 0, tzinfo=UTC),
                home_team="Lakers",
                away_team="Celtics",
            ),
            Event(
                id="event2",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime(2024, 10, 20, 20, 0, tzinfo=UTC),
                home_team="Warriors",
                away_team="Heat",
            ),
            Event(
                id="event3",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime(2024, 10, 25, 21, 0, tzinfo=UTC),
                home_team="Bulls",
                away_team="Nets",
            ),
        ]

        for event in events:
            test_session.add(event)
        await test_session.commit()

        metrics = QualityMetrics(test_session)
        result = await metrics.get_game_counts(
            start_date=datetime(2024, 10, 1, tzinfo=UTC),
            end_date=datetime(2024, 10, 31, tzinfo=UTC),
        )

        assert result.total_games == 3

    async def test_get_game_counts_with_sport_filter(self, test_session: AsyncSession):
        """Test game counts with sport filter."""
        # Create events for different sports
        events = [
            Event(
                id="event1",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime(2024, 10, 15, 19, 0, tzinfo=UTC),
                home_team="Lakers",
                away_team="Celtics",
            ),
            Event(
                id="event2",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime(2024, 10, 20, 20, 0, tzinfo=UTC),
                home_team="Warriors",
                away_team="Heat",
            ),
            Event(
                id="event3",
                sport_key="americanfootball_nfl",
                sport_title="NFL",
                commence_time=datetime(2024, 10, 25, 18, 0, tzinfo=UTC),
                home_team="Patriots",
                away_team="Cowboys",
            ),
        ]

        for event in events:
            test_session.add(event)
        await test_session.commit()

        metrics = QualityMetrics(test_session)

        # Filter for NBA only
        result = await metrics.get_game_counts(
            start_date=datetime(2024, 10, 1, tzinfo=UTC),
            end_date=datetime(2024, 10, 31, tzinfo=UTC),
            sport_key="basketball_nba",
        )

        assert result.total_games == 2
        assert result.sport_key == "basketball_nba"

    async def test_get_game_counts_date_range_boundary(self, test_session: AsyncSession):
        """Test game counts with events on date range boundaries."""
        events = [
            Event(
                id="event1",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime(2024, 10, 1, 0, 0, tzinfo=UTC),  # On start date
                home_team="Lakers",
                away_team="Celtics",
            ),
            Event(
                id="event2",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime(2024, 10, 31, 23, 59, tzinfo=UTC),  # On end date
                home_team="Warriors",
                away_team="Heat",
            ),
            Event(
                id="event3",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime(2024, 11, 1, 0, 0, tzinfo=UTC),  # After end date
                home_team="Bulls",
                away_team="Nets",
            ),
        ]

        for event in events:
            test_session.add(event)
        await test_session.commit()

        metrics = QualityMetrics(test_session)
        result = await metrics.get_game_counts(
            start_date=datetime(2024, 10, 1, tzinfo=UTC),
            end_date=datetime(2024, 10, 31, 23, 59, 59, tzinfo=UTC),
        )

        # Should include events 1 and 2, exclude event 3
        assert result.total_games == 2


@pytest.mark.asyncio
class TestQualityMetricsGamesWithOdds:
    """Tests for get_games_with_odds() method."""

    async def test_get_games_with_odds_empty(self, test_session: AsyncSession):
        """Test games with odds when no data exists."""
        metrics = QualityMetrics(test_session)

        result = await metrics.get_games_with_odds(
            start_date=datetime(2024, 10, 1, tzinfo=UTC),
            end_date=datetime(2024, 10, 31, tzinfo=UTC),
        )

        assert isinstance(result, list)
        assert len(result) == 0

    async def test_get_games_with_odds(self, test_session: AsyncSession):
        """Test retrieving games that have odds snapshots."""
        # Create events
        events = [
            Event(
                id="event1",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime(2024, 10, 15, 19, 0, tzinfo=UTC),
                home_team="Lakers",
                away_team="Celtics",
            ),
            Event(
                id="event2",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime(2024, 10, 20, 20, 0, tzinfo=UTC),
                home_team="Warriors",
                away_team="Heat",
            ),
            Event(
                id="event3",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime(2024, 10, 25, 21, 0, tzinfo=UTC),
                home_team="Bulls",
                away_team="Nets",
            ),
        ]

        for event in events:
            test_session.add(event)
        await test_session.commit()

        # Create odds snapshots for event1 and event2 only
        snapshots = [
            OddsSnapshot(
                event_id="event1",
                snapshot_time=datetime(2024, 10, 15, 18, 0, tzinfo=UTC),
                raw_data={"bookmakers": []},
                bookmaker_count=0,
            ),
            OddsSnapshot(
                event_id="event1",
                snapshot_time=datetime(2024, 10, 15, 18, 30, tzinfo=UTC),
                raw_data={"bookmakers": []},
                bookmaker_count=0,
            ),
            OddsSnapshot(
                event_id="event2",
                snapshot_time=datetime(2024, 10, 20, 19, 0, tzinfo=UTC),
                raw_data={"bookmakers": []},
                bookmaker_count=0,
            ),
        ]

        for snapshot in snapshots:
            test_session.add(snapshot)
        await test_session.commit()

        metrics = QualityMetrics(test_session)
        result = await metrics.get_games_with_odds(
            start_date=datetime(2024, 10, 1, tzinfo=UTC),
            end_date=datetime(2024, 10, 31, tzinfo=UTC),
        )

        # Should return event1 and event2, but not event3
        assert len(result) == 2
        assert all(isinstance(r, GameWithOddsResult) for r in result)

        # Check that event1 has correct snapshot count
        event1_result = next(r for r in result if r.event_id == "event1")
        assert event1_result.snapshot_count == 2
        assert event1_result.home_team == "Lakers"
        assert event1_result.away_team == "Celtics"

        # Check that event2 has correct snapshot count
        event2_result = next(r for r in result if r.event_id == "event2")
        assert event2_result.snapshot_count == 1

    async def test_get_games_with_odds_sport_filter(self, test_session: AsyncSession):
        """Test games with odds filtered by sport."""
        # Create events for different sports
        events = [
            Event(
                id="event1",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime(2024, 10, 15, 19, 0, tzinfo=UTC),
                home_team="Lakers",
                away_team="Celtics",
            ),
            Event(
                id="event2",
                sport_key="americanfootball_nfl",
                sport_title="NFL",
                commence_time=datetime(2024, 10, 20, 18, 0, tzinfo=UTC),
                home_team="Patriots",
                away_team="Cowboys",
            ),
        ]

        for event in events:
            test_session.add(event)
        await test_session.commit()

        # Create snapshots for both events
        snapshots = [
            OddsSnapshot(
                event_id="event1",
                snapshot_time=datetime(2024, 10, 15, 18, 0, tzinfo=UTC),
                raw_data={"bookmakers": []},
                bookmaker_count=0,
            ),
            OddsSnapshot(
                event_id="event2",
                snapshot_time=datetime(2024, 10, 20, 17, 0, tzinfo=UTC),
                raw_data={"bookmakers": []},
                bookmaker_count=0,
            ),
        ]

        for snapshot in snapshots:
            test_session.add(snapshot)
        await test_session.commit()

        metrics = QualityMetrics(test_session)
        result = await metrics.get_games_with_odds(
            start_date=datetime(2024, 10, 1, tzinfo=UTC),
            end_date=datetime(2024, 10, 31, tzinfo=UTC),
            sport_key="basketball_nba",
        )

        # Should only return NBA event
        assert len(result) == 1
        assert result[0].sport_key == "basketball_nba"
        assert result[0].event_id == "event1"


@pytest.mark.asyncio
class TestQualityMetricsGamesWithScores:
    """Tests for get_games_with_scores() method."""

    async def test_get_games_with_scores_empty(self, test_session: AsyncSession):
        """Test games with scores when no data exists."""
        metrics = QualityMetrics(test_session)

        result = await metrics.get_games_with_scores(
            start_date=datetime(2024, 10, 1, tzinfo=UTC),
            end_date=datetime(2024, 10, 31, tzinfo=UTC),
        )

        assert isinstance(result, list)
        assert len(result) == 0

    async def test_get_games_with_scores(self, test_session: AsyncSession):
        """Test retrieving games with final scores."""
        events = [
            Event(
                id="event1",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime(2024, 10, 15, 19, 0, tzinfo=UTC),
                home_team="Lakers",
                away_team="Celtics",
                status=EventStatus.FINAL,
                home_score=112,
                away_score=108,
            ),
            Event(
                id="event2",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime(2024, 10, 20, 20, 0, tzinfo=UTC),
                home_team="Warriors",
                away_team="Heat",
                status=EventStatus.FINAL,
                home_score=105,
                away_score=98,
            ),
            Event(
                id="event3",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime(2024, 10, 25, 21, 0, tzinfo=UTC),
                home_team="Bulls",
                away_team="Nets",
                status=EventStatus.SCHEDULED,
                # No scores yet
            ),
        ]

        for event in events:
            test_session.add(event)
        await test_session.commit()

        metrics = QualityMetrics(test_session)
        result = await metrics.get_games_with_scores(
            start_date=datetime(2024, 10, 1, tzinfo=UTC),
            end_date=datetime(2024, 10, 31, tzinfo=UTC),
        )

        # Should return only events with scores (event1 and event2)
        assert len(result) == 2
        assert all(isinstance(r, GameWithScoresResult) for r in result)

        # Verify scores
        event1_result = next(r for r in result if r.event_id == "event1")
        assert event1_result.home_score == 112
        assert event1_result.away_score == 108
        assert event1_result.home_team == "Lakers"

        event2_result = next(r for r in result if r.event_id == "event2")
        assert event2_result.home_score == 105
        assert event2_result.away_score == 98

    async def test_get_games_with_scores_partial_scores(self, test_session: AsyncSession):
        """Test that games with only one score are excluded."""
        events = [
            Event(
                id="event1",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime(2024, 10, 15, 19, 0, tzinfo=UTC),
                home_team="Lakers",
                away_team="Celtics",
                home_score=112,
                away_score=None,  # Missing away score
            ),
            Event(
                id="event2",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime(2024, 10, 20, 20, 0, tzinfo=UTC),
                home_team="Warriors",
                away_team="Heat",
                home_score=None,  # Missing home score
                away_score=98,
            ),
        ]

        for event in events:
            test_session.add(event)
        await test_session.commit()

        metrics = QualityMetrics(test_session)
        result = await metrics.get_games_with_scores(
            start_date=datetime(2024, 10, 1, tzinfo=UTC),
            end_date=datetime(2024, 10, 31, tzinfo=UTC),
        )

        # Should return no games (both are missing at least one score)
        assert len(result) == 0

    async def test_get_games_with_scores_sport_filter(self, test_session: AsyncSession):
        """Test games with scores filtered by sport."""
        events = [
            Event(
                id="event1",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime(2024, 10, 15, 19, 0, tzinfo=UTC),
                home_team="Lakers",
                away_team="Celtics",
                home_score=112,
                away_score=108,
            ),
            Event(
                id="event2",
                sport_key="americanfootball_nfl",
                sport_title="NFL",
                commence_time=datetime(2024, 10, 20, 18, 0, tzinfo=UTC),
                home_team="Patriots",
                away_team="Cowboys",
                home_score=28,
                away_score=21,
            ),
        ]

        for event in events:
            test_session.add(event)
        await test_session.commit()

        metrics = QualityMetrics(test_session)
        result = await metrics.get_games_with_scores(
            start_date=datetime(2024, 10, 1, tzinfo=UTC),
            end_date=datetime(2024, 10, 31, tzinfo=UTC),
            sport_key="basketball_nba",
        )

        # Should only return NBA event
        assert len(result) == 1
        assert result[0].sport_key == "basketball_nba"
        assert result[0].home_score == 112


@pytest.mark.asyncio
class TestQualityMetricsGamesMissingScores:
    """Tests for get_games_missing_scores() method."""

    async def test_get_games_missing_scores_empty(self, test_session: AsyncSession):
        """Test games missing scores when no data exists."""
        metrics = QualityMetrics(test_session)

        result = await metrics.get_games_missing_scores(
            start_date=datetime(2024, 10, 1, tzinfo=UTC),
            end_date=datetime(2024, 10, 31, tzinfo=UTC),
        )

        assert isinstance(result, list)
        assert len(result) == 0

    async def test_get_games_missing_scores(self, test_session: AsyncSession):
        """Test retrieving games without final scores."""
        events = [
            Event(
                id="event1",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime(2024, 10, 15, 19, 0, tzinfo=UTC),
                home_team="Lakers",
                away_team="Celtics",
                status=EventStatus.FINAL,
                home_score=112,
                away_score=108,
            ),
            Event(
                id="event2",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime(2024, 10, 20, 20, 0, tzinfo=UTC),
                home_team="Warriors",
                away_team="Heat",
                status=EventStatus.SCHEDULED,
                # No scores
            ),
            Event(
                id="event3",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime(2024, 10, 25, 21, 0, tzinfo=UTC),
                home_team="Bulls",
                away_team="Nets",
                status=EventStatus.FINAL,
                # Missing scores but marked as FINAL
            ),
        ]

        for event in events:
            test_session.add(event)
        await test_session.commit()

        metrics = QualityMetrics(test_session)
        result = await metrics.get_games_missing_scores(
            start_date=datetime(2024, 10, 1, tzinfo=UTC),
            end_date=datetime(2024, 10, 31, tzinfo=UTC),
        )

        # Should return event2 and event3 (missing scores)
        assert len(result) == 2
        assert all(isinstance(r, GameMissingScoresResult) for r in result)

        event_ids = {r.event_id for r in result}
        assert event_ids == {"event2", "event3"}

        # Verify status is included
        event2_result = next(r for r in result if r.event_id == "event2")
        assert event2_result.status == EventStatus.SCHEDULED.value
        assert event2_result.home_team == "Warriors"

    async def test_get_games_missing_scores_partial(self, test_session: AsyncSession):
        """Test that games with partial scores are included."""
        events = [
            Event(
                id="event1",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime(2024, 10, 15, 19, 0, tzinfo=UTC),
                home_team="Lakers",
                away_team="Celtics",
                home_score=112,
                away_score=None,  # Missing away score
            ),
            Event(
                id="event2",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime(2024, 10, 20, 20, 0, tzinfo=UTC),
                home_team="Warriors",
                away_team="Heat",
                home_score=None,  # Missing home score
                away_score=98,
            ),
        ]

        for event in events:
            test_session.add(event)
        await test_session.commit()

        metrics = QualityMetrics(test_session)
        result = await metrics.get_games_missing_scores(
            start_date=datetime(2024, 10, 1, tzinfo=UTC),
            end_date=datetime(2024, 10, 31, tzinfo=UTC),
        )

        # Should return both events (both have at least one missing score)
        assert len(result) == 2

    async def test_get_games_missing_scores_sport_filter(self, test_session: AsyncSession):
        """Test games missing scores filtered by sport."""
        events = [
            Event(
                id="event1",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime(2024, 10, 15, 19, 0, tzinfo=UTC),
                home_team="Lakers",
                away_team="Celtics",
                status=EventStatus.SCHEDULED,
            ),
            Event(
                id="event2",
                sport_key="americanfootball_nfl",
                sport_title="NFL",
                commence_time=datetime(2024, 10, 20, 18, 0, tzinfo=UTC),
                home_team="Patriots",
                away_team="Cowboys",
                status=EventStatus.SCHEDULED,
            ),
        ]

        for event in events:
            test_session.add(event)
        await test_session.commit()

        metrics = QualityMetrics(test_session)
        result = await metrics.get_games_missing_scores(
            start_date=datetime(2024, 10, 1, tzinfo=UTC),
            end_date=datetime(2024, 10, 31, tzinfo=UTC),
            sport_key="basketball_nba",
        )

        # Should only return NBA event
        assert len(result) == 1
        assert result[0].sport_key == "basketball_nba"
        assert result[0].event_id == "event1"


@pytest.mark.asyncio
class TestQualityMetricsEdgeCases:
    """Tests for edge cases and error conditions."""

    async def test_empty_date_range(self, test_session: AsyncSession):
        """Test with events outside the date range."""
        events = [
            Event(
                id="event1",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime(2024, 9, 30, 19, 0, tzinfo=UTC),  # Before range
                home_team="Lakers",
                away_team="Celtics",
            ),
            Event(
                id="event2",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime(2024, 11, 1, 20, 0, tzinfo=UTC),  # After range
                home_team="Warriors",
                away_team="Heat",
            ),
        ]

        for event in events:
            test_session.add(event)
        await test_session.commit()

        metrics = QualityMetrics(test_session)

        # Query for October only
        result = await metrics.get_game_counts(
            start_date=datetime(2024, 10, 1, tzinfo=UTC),
            end_date=datetime(2024, 10, 31, tzinfo=UTC),
        )

        assert result.total_games == 0

    async def test_ordered_by_commence_time(self, test_session: AsyncSession):
        """Test that results are ordered by commence_time."""
        events = [
            Event(
                id="event1",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime(2024, 10, 25, 19, 0, tzinfo=UTC),
                home_team="Lakers",
                away_team="Celtics",
                home_score=110,
                away_score=105,
            ),
            Event(
                id="event2",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime(2024, 10, 10, 20, 0, tzinfo=UTC),
                home_team="Warriors",
                away_team="Heat",
                home_score=98,
                away_score=95,
            ),
            Event(
                id="event3",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime(2024, 10, 15, 21, 0, tzinfo=UTC),
                home_team="Bulls",
                away_team="Nets",
                home_score=102,
                away_score=100,
            ),
        ]

        for event in events:
            test_session.add(event)
        await test_session.commit()

        metrics = QualityMetrics(test_session)
        result = await metrics.get_games_with_scores(
            start_date=datetime(2024, 10, 1, tzinfo=UTC),
            end_date=datetime(2024, 10, 31, tzinfo=UTC),
        )

        # Should be ordered by commence_time
        assert len(result) == 3
        assert result[0].event_id == "event2"  # Oct 10
        assert result[1].event_id == "event3"  # Oct 15
        assert result[2].event_id == "event1"  # Oct 25


@pytest.mark.asyncio
class TestQualityMetricsTierCoverage:
    """Tests for get_tier_coverage() method."""

    async def test_get_tier_coverage_empty_database(self, test_session: AsyncSession):
        """Test tier coverage with no data in database."""
        metrics = QualityMetrics(test_session)

        result = await metrics.get_tier_coverage(
            start_date=datetime(2024, 10, 1, tzinfo=UTC),
            end_date=datetime(2024, 10, 31, tzinfo=UTC),
        )

        # Should return coverage for all tiers
        assert isinstance(result, list)
        assert len(result) == 5  # One for each FetchTier

        # All tiers should have zero coverage
        for tier_info in result:
            assert isinstance(tier_info, TierCoverage)
            assert tier_info.games_in_tier_range == 0
            assert tier_info.games_with_tier_snapshots == 0
            assert tier_info.total_snapshots_in_tier == 0
            assert tier_info.coverage_pct == 0.0
            assert tier_info.avg_snapshots_per_game == 0.0

    async def test_get_tier_coverage_with_snapshots(self, test_session: AsyncSession):
        """Test tier coverage with events and snapshots across multiple tiers."""
        # Create test events
        events = [
            Event(
                id="event1",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime(2024, 10, 15, 19, 0, tzinfo=UTC),
                home_team="Lakers",
                away_team="Celtics",
            ),
            Event(
                id="event2",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime(2024, 10, 20, 20, 0, tzinfo=UTC),
                home_team="Warriors",
                away_team="Heat",
            ),
            Event(
                id="event3",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime(2024, 10, 25, 21, 0, tzinfo=UTC),
                home_team="Bulls",
                away_team="Nets",
            ),
        ]

        for event in events:
            test_session.add(event)
        await test_session.commit()

        # Create snapshots with different tiers
        snapshots = [
            # Event 1: Has closing and pregame snapshots
            OddsSnapshot(
                event_id="event1",
                snapshot_time=datetime(2024, 10, 15, 18, 0, tzinfo=UTC),
                raw_data={"bookmakers": []},
                bookmaker_count=0,
                fetch_tier="closing",
                hours_until_commence=1.0,
            ),
            OddsSnapshot(
                event_id="event1",
                snapshot_time=datetime(2024, 10, 15, 18, 30, tzinfo=UTC),
                raw_data={"bookmakers": []},
                bookmaker_count=0,
                fetch_tier="closing",
                hours_until_commence=0.5,
            ),
            OddsSnapshot(
                event_id="event1",
                snapshot_time=datetime(2024, 10, 15, 15, 0, tzinfo=UTC),
                raw_data={"bookmakers": []},
                bookmaker_count=0,
                fetch_tier="pregame",
                hours_until_commence=4.0,
            ),
            # Event 2: Has only sharp tier snapshots
            OddsSnapshot(
                event_id="event2",
                snapshot_time=datetime(2024, 10, 20, 8, 0, tzinfo=UTC),
                raw_data={"bookmakers": []},
                bookmaker_count=0,
                fetch_tier="sharp",
                hours_until_commence=12.0,
            ),
            # Event 3: Has opening and early tier snapshots
            OddsSnapshot(
                event_id="event3",
                snapshot_time=datetime(2024, 10, 22, 21, 0, tzinfo=UTC),
                raw_data={"bookmakers": []},
                bookmaker_count=0,
                fetch_tier="opening",
                hours_until_commence=72.0,
            ),
            OddsSnapshot(
                event_id="event3",
                snapshot_time=datetime(2024, 10, 24, 21, 0, tzinfo=UTC),
                raw_data={"bookmakers": []},
                bookmaker_count=0,
                fetch_tier="early",
                hours_until_commence=24.0,
            ),
        ]

        for snapshot in snapshots:
            test_session.add(snapshot)
        await test_session.commit()

        # Get tier coverage
        metrics = QualityMetrics(test_session)
        result = await metrics.get_tier_coverage(
            start_date=datetime(2024, 10, 1, tzinfo=UTC),
            end_date=datetime(2024, 10, 31, tzinfo=UTC),
        )

        # Should return 5 tiers
        assert len(result) == 5

        # Convert to dict for easier testing
        tier_map = {tier.tier_name: tier for tier in result}

        # Check closing tier (event1 has 2 snapshots)
        closing = tier_map["closing"]
        assert closing.tier == "closing"
        assert closing.hours_range == "0-3 hours before"
        assert closing.expected_interval_hours == 0.5
        assert closing.games_in_tier_range == 3  # All events eligible
        assert closing.games_with_tier_snapshots == 1  # Only event1 has closing snapshots
        assert closing.total_snapshots_in_tier == 2
        assert closing.coverage_pct == pytest.approx(33.33, abs=0.01)
        assert closing.avg_snapshots_per_game == 2.0

        # Check pregame tier (event1 has 1 snapshot)
        pregame = tier_map["pregame"]
        assert pregame.tier == "pregame"
        assert pregame.hours_range == "3-12 hours before"
        assert pregame.expected_interval_hours == 3.0
        assert pregame.games_in_tier_range == 3
        assert pregame.games_with_tier_snapshots == 1
        assert pregame.total_snapshots_in_tier == 1
        assert pregame.coverage_pct == pytest.approx(33.33, abs=0.01)
        assert pregame.avg_snapshots_per_game == 1.0

        # Check sharp tier (event2 has 1 snapshot)
        sharp = tier_map["sharp"]
        assert sharp.tier == "sharp"
        assert sharp.hours_range == "12-24 hours before"
        assert sharp.expected_interval_hours == 12.0
        assert sharp.games_in_tier_range == 3
        assert sharp.games_with_tier_snapshots == 1
        assert sharp.total_snapshots_in_tier == 1
        assert sharp.coverage_pct == pytest.approx(33.33, abs=0.01)

        # Check early tier (event3 has 1 snapshot)
        early = tier_map["early"]
        assert early.tier == "early"
        assert early.hours_range == "1-3 days before"
        assert early.expected_interval_hours == 24.0
        assert early.games_in_tier_range == 3
        assert early.games_with_tier_snapshots == 1
        assert early.total_snapshots_in_tier == 1

        # Check opening tier (event3 has 1 snapshot)
        opening = tier_map["opening"]
        assert opening.tier == "opening"
        assert opening.hours_range == "3+ days before"
        assert opening.expected_interval_hours == 48.0
        assert opening.games_in_tier_range == 3
        assert opening.games_with_tier_snapshots == 1
        assert opening.total_snapshots_in_tier == 1

    async def test_get_tier_coverage_sport_filter(self, test_session: AsyncSession):
        """Test tier coverage with sport filter."""
        # Create events for different sports
        events = [
            Event(
                id="event1",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime(2024, 10, 15, 19, 0, tzinfo=UTC),
                home_team="Lakers",
                away_team="Celtics",
            ),
            Event(
                id="event2",
                sport_key="americanfootball_nfl",
                sport_title="NFL",
                commence_time=datetime(2024, 10, 20, 18, 0, tzinfo=UTC),
                home_team="Patriots",
                away_team="Cowboys",
            ),
        ]

        for event in events:
            test_session.add(event)
        await test_session.commit()

        # Create snapshots for both sports
        snapshots = [
            OddsSnapshot(
                event_id="event1",
                snapshot_time=datetime(2024, 10, 15, 18, 0, tzinfo=UTC),
                raw_data={"bookmakers": []},
                bookmaker_count=0,
                fetch_tier="closing",
                hours_until_commence=1.0,
            ),
            OddsSnapshot(
                event_id="event2",
                snapshot_time=datetime(2024, 10, 20, 17, 0, tzinfo=UTC),
                raw_data={"bookmakers": []},
                bookmaker_count=0,
                fetch_tier="closing",
                hours_until_commence=1.0,
            ),
        ]

        for snapshot in snapshots:
            test_session.add(snapshot)
        await test_session.commit()

        # Get tier coverage filtered by NBA
        metrics = QualityMetrics(test_session)
        result = await metrics.get_tier_coverage(
            start_date=datetime(2024, 10, 1, tzinfo=UTC),
            end_date=datetime(2024, 10, 31, tzinfo=UTC),
            sport_key="basketball_nba",
        )

        # Should only count NBA events
        tier_map = {tier.tier_name: tier for tier in result}
        closing = tier_map["closing"]

        assert closing.games_in_tier_range == 1  # Only NBA event
        assert closing.games_with_tier_snapshots == 1
        assert closing.total_snapshots_in_tier == 1
        assert closing.coverage_pct == 100.0

    async def test_get_tier_coverage_no_snapshots(self, test_session: AsyncSession):
        """Test tier coverage with events but no snapshots."""
        # Create events without snapshots
        events = [
            Event(
                id="event1",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime(2024, 10, 15, 19, 0, tzinfo=UTC),
                home_team="Lakers",
                away_team="Celtics",
            ),
            Event(
                id="event2",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime(2024, 10, 20, 20, 0, tzinfo=UTC),
                home_team="Warriors",
                away_team="Heat",
            ),
        ]

        for event in events:
            test_session.add(event)
        await test_session.commit()

        # Get tier coverage
        metrics = QualityMetrics(test_session)
        result = await metrics.get_tier_coverage(
            start_date=datetime(2024, 10, 1, tzinfo=UTC),
            end_date=datetime(2024, 10, 31, tzinfo=UTC),
        )

        # All tiers should show events but no coverage
        for tier_info in result:
            assert tier_info.games_in_tier_range == 2
            assert tier_info.games_with_tier_snapshots == 0
            assert tier_info.total_snapshots_in_tier == 0
            assert tier_info.coverage_pct == 0.0
            assert tier_info.avg_snapshots_per_game == 0.0

    async def test_get_tier_coverage_dynamic_tier_iteration(self, test_session: AsyncSession):
        """Test that tier coverage dynamically iterates through all FetchTier enum values."""
        # Create one event
        event = Event(
            id="event1",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime(2024, 10, 15, 19, 0, tzinfo=UTC),
            home_team="Lakers",
            away_team="Celtics",
        )
        test_session.add(event)
        await test_session.commit()

        # Get tier coverage
        metrics = QualityMetrics(test_session)
        result = await metrics.get_tier_coverage(
            start_date=datetime(2024, 10, 1, tzinfo=UTC),
            end_date=datetime(2024, 10, 31, tzinfo=UTC),
        )

        # Should have exactly as many results as FetchTier enum values
        assert len(result) == len(FetchTier)

        # Verify all tier names are present
        tier_names = {tier.tier_name for tier in result}
        expected_names = {tier.value for tier in FetchTier}
        assert tier_names == expected_names

        # Verify expected_interval_hours matches tier properties
        tier_map = {tier.tier_name: tier for tier in result}
        for fetch_tier in FetchTier:
            tier_info = tier_map[fetch_tier.value]
            assert tier_info.expected_interval_hours == fetch_tier.interval_hours

    async def test_get_tier_coverage_multiple_snapshots_per_game(self, test_session: AsyncSession):
        """Test avg_snapshots_per_game calculation with multiple snapshots."""
        # Create one event
        event = Event(
            id="event1",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime(2024, 10, 15, 19, 0, tzinfo=UTC),
            home_team="Lakers",
            away_team="Celtics",
        )
        test_session.add(event)
        await test_session.commit()

        # Create 5 closing tier snapshots for the event
        snapshots = [
            OddsSnapshot(
                event_id="event1",
                snapshot_time=datetime(2024, 10, 15, 18, 0, tzinfo=UTC),
                raw_data={"bookmakers": []},
                bookmaker_count=0,
                fetch_tier="closing",
                hours_until_commence=1.0,
            ),
            OddsSnapshot(
                event_id="event1",
                snapshot_time=datetime(2024, 10, 15, 18, 30, tzinfo=UTC),
                raw_data={"bookmakers": []},
                bookmaker_count=0,
                fetch_tier="closing",
                hours_until_commence=0.5,
            ),
            OddsSnapshot(
                event_id="event1",
                snapshot_time=datetime(2024, 10, 15, 17, 30, tzinfo=UTC),
                raw_data={"bookmakers": []},
                bookmaker_count=0,
                fetch_tier="closing",
                hours_until_commence=1.5,
            ),
            OddsSnapshot(
                event_id="event1",
                snapshot_time=datetime(2024, 10, 15, 17, 0, tzinfo=UTC),
                raw_data={"bookmakers": []},
                bookmaker_count=0,
                fetch_tier="closing",
                hours_until_commence=2.0,
            ),
            OddsSnapshot(
                event_id="event1",
                snapshot_time=datetime(2024, 10, 15, 16, 30, tzinfo=UTC),
                raw_data={"bookmakers": []},
                bookmaker_count=0,
                fetch_tier="closing",
                hours_until_commence=2.5,
            ),
        ]

        for snapshot in snapshots:
            test_session.add(snapshot)
        await test_session.commit()

        # Get tier coverage
        metrics = QualityMetrics(test_session)
        result = await metrics.get_tier_coverage(
            start_date=datetime(2024, 10, 1, tzinfo=UTC),
            end_date=datetime(2024, 10, 31, tzinfo=UTC),
        )

        # Find closing tier
        closing = next(tier for tier in result if tier.tier_name == "closing")

        # Should have 1 game with 5 snapshots
        assert closing.games_with_tier_snapshots == 1
        assert closing.total_snapshots_in_tier == 5
        assert closing.avg_snapshots_per_game == 5.0
        assert closing.coverage_pct == 100.0

    async def test_get_tier_coverage_date_range_boundary(self, test_session: AsyncSession):
        """Test tier coverage respects date range boundaries."""
        events = [
            Event(
                id="event1",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime(2024, 9, 30, 19, 0, tzinfo=UTC),  # Before range
                home_team="Lakers",
                away_team="Celtics",
            ),
            Event(
                id="event2",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime(2024, 10, 15, 20, 0, tzinfo=UTC),  # In range
                home_team="Warriors",
                away_team="Heat",
            ),
            Event(
                id="event3",
                sport_key="basketball_nba",
                sport_title="NBA",
                commence_time=datetime(2024, 11, 1, 21, 0, tzinfo=UTC),  # After range
                home_team="Bulls",
                away_team="Nets",
            ),
        ]

        for event in events:
            test_session.add(event)
        await test_session.commit()

        # Create snapshots for all events
        snapshots = [
            OddsSnapshot(
                event_id="event1",
                snapshot_time=datetime(2024, 9, 30, 18, 0, tzinfo=UTC),
                raw_data={"bookmakers": []},
                bookmaker_count=0,
                fetch_tier="closing",
                hours_until_commence=1.0,
            ),
            OddsSnapshot(
                event_id="event2",
                snapshot_time=datetime(2024, 10, 15, 19, 0, tzinfo=UTC),
                raw_data={"bookmakers": []},
                bookmaker_count=0,
                fetch_tier="closing",
                hours_until_commence=1.0,
            ),
            OddsSnapshot(
                event_id="event3",
                snapshot_time=datetime(2024, 11, 1, 20, 0, tzinfo=UTC),
                raw_data={"bookmakers": []},
                bookmaker_count=0,
                fetch_tier="closing",
                hours_until_commence=1.0,
            ),
        ]

        for snapshot in snapshots:
            test_session.add(snapshot)
        await test_session.commit()

        # Get tier coverage for October only
        metrics = QualityMetrics(test_session)
        result = await metrics.get_tier_coverage(
            start_date=datetime(2024, 10, 1, tzinfo=UTC),
            end_date=datetime(2024, 10, 31, 23, 59, 59, tzinfo=UTC),
        )

        # Should only include event2
        closing = next(tier for tier in result if tier.tier_name == "closing")
        assert closing.games_in_tier_range == 1  # Only event2
        assert closing.games_with_tier_snapshots == 1
        assert closing.total_snapshots_in_tier == 1
