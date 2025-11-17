"""Unit tests for QualityMetrics module."""

from datetime import UTC, datetime

import pytest
from odds_analytics.quality_metrics import (
    GameCountResult,
    GameMissingScoresResult,
    GameWithOddsResult,
    GameWithScoresResult,
    QualityMetrics,
)
from odds_core.models import Event, EventStatus, OddsSnapshot
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
