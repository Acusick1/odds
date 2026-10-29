"""Database query module for data quality and coverage statistics."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime

from odds_core.models import Event, OddsSnapshot, Price
from odds_core.vocabulary import Source, TimeBasis
from odds_lambda.fetch_tier import FetchTier
from odds_lambda.tier_utils import snapshot_tiers
from pydantic import BaseModel, Field
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession


class GameCountResult(BaseModel):
    """Result model for game count queries."""

    total_games: int = Field(description="Total number of games in the date range")
    sport_key: str | None = Field(default=None, description="Sport filter applied")
    start_date: datetime = Field(description="Start of date range")
    end_date: datetime = Field(description="End of date range")


class GameWithOddsResult(BaseModel):
    """Result model for games with odds data."""

    event_id: int = Field(description="Event identifier")
    sport_key: str = Field(description="Sport identifier")
    home_team: str = Field(description="Home team name")
    away_team: str = Field(description="Away team name")
    commence_time: datetime = Field(description="Game start time")
    snapshot_count: int = Field(description="Number of odds snapshots")


class GameWithScoresResult(BaseModel):
    """Result model for games with final scores."""

    event_id: int = Field(description="Event identifier")
    sport_key: str = Field(description="Sport identifier")
    home_team: str = Field(description="Home team name")
    away_team: str = Field(description="Away team name")
    commence_time: datetime = Field(description="Game start time")
    home_score: int = Field(description="Final home team score")
    away_score: int = Field(description="Final away team score")


class GameMissingScoresResult(BaseModel):
    """Result model for games missing score data."""

    event_id: int = Field(description="Event identifier")
    sport_key: str = Field(description="Sport identifier")
    home_team: str = Field(description="Home team name")
    away_team: str = Field(description="Away team name")
    commence_time: datetime = Field(description="Game start time")
    status: str = Field(description="Event status")


class TierCoverage(BaseModel):
    """Result model for tier-based coverage analysis."""

    tier: str = Field(description="FetchTier enum value (e.g., 'closing', 'pregame')")
    tier_name: str = Field(description="Lowercase tier identifier")
    hours_range: str = Field(description="Human-readable time window (e.g., '0-3 hours before')")
    expected_interval_hours: float = Field(
        description="Expected sampling interval from tier properties"
    )
    games_in_tier_range: int = Field(description="Games eligible for this tier")
    games_with_tier_snapshots: int = Field(description="Games with at least one snapshot in tier")
    total_snapshots_in_tier: int = Field(description="Total snapshots captured in this tier")
    coverage_pct: float = Field(description="Percentage of eligible games with tier coverage")
    avg_snapshots_per_game: float = Field(description="Average snapshots per game in tier")


class BookmakerCoverage(BaseModel):
    """Result model for bookmaker coverage analysis."""

    bookmaker_key: str = Field(description="Bookmaker identifier (e.g., 'fanduel', 'draftkings')")
    total_games: int = Field(description="Total games in the analyzed date range")
    games_with_odds: int = Field(description="Games where this bookmaker provided odds")
    coverage_pct: float = Field(
        description="Percentage of games with odds (games_with_odds / total_games * 100)"
    )
    total_snapshots: int = Field(description="Snapshots in which this bookmaker has a price")
    avg_snapshots_per_game: float = Field(
        description="Average snapshots per game (total_snapshots / games_with_odds)"
    )


class QualityMetrics:
    """Handles data quality and coverage metric queries."""

    def __init__(self, session: AsyncSession):
        """
        Initialize quality metrics with database session.

        Args:
            session: Async database session
        """
        self.session = session

    async def get_game_counts(
        self,
        start_date: datetime,
        end_date: datetime,
        sport_key: str | None = None,
    ) -> GameCountResult:
        """
        Get total number of games within a date range.

        Args:
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)
            sport_key: Optional sport filter (e.g., "basketball_nba")

        Returns:
            GameCountResult with total count and query parameters

        Example:
            metrics = QualityMetrics(session)
            result = await metrics.get_game_counts(
                start_date=datetime(2024, 10, 1),
                end_date=datetime(2024, 10, 31),
                sport_key="basketball_nba"
            )
            print(f"Total games: {result.total_games}")
        """
        query = select(func.count(Event.id)).where(
            and_(
                Event.commence_time >= start_date,
                Event.commence_time <= end_date,
            )
        )

        if sport_key:
            query = query.where(Event.sport_key == sport_key)

        result = await self.session.execute(query)
        total = result.scalar_one()

        return GameCountResult(
            total_games=total,
            sport_key=sport_key,
            start_date=start_date,
            end_date=end_date,
        )

    async def get_games_with_odds(
        self,
        start_date: datetime,
        end_date: datetime,
        sport_key: str | None = None,
    ) -> list[GameWithOddsResult]:
        """
        Get games that have odds snapshot records.

        Args:
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)
            sport_key: Optional sport filter (e.g., "basketball_nba")

        Returns:
            List of GameWithOddsResult containing games with odds data

        Example:
            metrics = QualityMetrics(session)
            games = await metrics.get_games_with_odds(
                start_date=datetime(2024, 10, 1),
                end_date=datetime(2024, 10, 31),
                sport_key="basketball_nba"
            )
            print(f"Found {len(games)} games with odds data")
        """
        # Join Event with OddsSnapshot to find events with odds
        # Group by event to count snapshots
        query = (
            select(
                Event.id,
                Event.sport_key,
                Event.home_team,
                Event.away_team,
                Event.commence_time,
                func.count(func.distinct(OddsSnapshot.snapshot_time)).label("snapshot_count"),
            )
            .join(OddsSnapshot, Event.id == OddsSnapshot.event_id)
            .where(
                and_(
                    Event.commence_time >= start_date,
                    Event.commence_time <= end_date,
                )
            )
        )

        if sport_key:
            query = query.where(Event.sport_key == sport_key)

        query = query.group_by(
            Event.id,
            Event.sport_key,
            Event.home_team,
            Event.away_team,
            Event.commence_time,
        ).order_by(Event.commence_time)

        result = await self.session.execute(query)
        rows = result.all()

        return [
            GameWithOddsResult(
                event_id=row[0],
                sport_key=row[1],
                home_team=row[2],
                away_team=row[3],
                commence_time=row[4],
                snapshot_count=row[5],
            )
            for row in rows
        ]

    async def get_games_with_scores(
        self,
        start_date: datetime,
        end_date: datetime,
        sport_key: str | None = None,
    ) -> list[GameWithScoresResult]:
        """
        Get games that have final score data.

        Args:
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)
            sport_key: Optional sport filter (e.g., "basketball_nba")

        Returns:
            List of GameWithScoresResult containing games with scores

        Example:
            metrics = QualityMetrics(session)
            games = await metrics.get_games_with_scores(
                start_date=datetime(2024, 10, 1),
                end_date=datetime(2024, 10, 31),
                sport_key="basketball_nba"
            )
            print(f"Found {len(games)} games with final scores")
        """
        query = select(
            Event.id,
            Event.sport_key,
            Event.home_team,
            Event.away_team,
            Event.commence_time,
            Event.home_score,
            Event.away_score,
        ).where(
            and_(
                Event.commence_time >= start_date,
                Event.commence_time <= end_date,
                Event.home_score.is_not(None),
                Event.away_score.is_not(None),
            )
        )

        if sport_key:
            query = query.where(Event.sport_key == sport_key)

        query = query.order_by(Event.commence_time)

        result = await self.session.execute(query)
        rows = result.all()

        return [
            GameWithScoresResult(
                event_id=row[0],
                sport_key=row[1],
                home_team=row[2],
                away_team=row[3],
                commence_time=row[4],
                home_score=row[5],
                away_score=row[6],
            )
            for row in rows
        ]

    async def get_games_missing_scores(
        self,
        start_date: datetime,
        end_date: datetime,
        sport_key: str | None = None,
    ) -> list[GameMissingScoresResult]:
        """
        Get games that are missing final score data.

        Args:
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)
            sport_key: Optional sport filter (e.g., "basketball_nba")

        Returns:
            List of GameMissingScoresResult for games without scores

        Example:
            metrics = QualityMetrics(session)
            games = await metrics.get_games_missing_scores(
                start_date=datetime(2024, 10, 1),
                end_date=datetime(2024, 10, 31),
                sport_key="basketball_nba"
            )
            print(f"Found {len(games)} games missing scores")
        """
        query = select(
            Event.id,
            Event.sport_key,
            Event.home_team,
            Event.away_team,
            Event.commence_time,
            Event.status,
        ).where(
            and_(
                Event.commence_time >= start_date,
                Event.commence_time <= end_date,
                (Event.home_score.is_(None)) | (Event.away_score.is_(None)),
            )
        )

        if sport_key:
            query = query.where(Event.sport_key == sport_key)

        query = query.order_by(Event.commence_time)

        result = await self.session.execute(query)
        rows = result.all()

        return [
            GameMissingScoresResult(
                event_id=row[0],
                sport_key=row[1],
                home_team=row[2],
                away_team=row[3],
                commence_time=row[4],
                status=row[5].value,
            )
            for row in rows
        ]

    async def get_tier_coverage(
        self,
        start_date: datetime,
        end_date: datetime,
        sport_key: str | None = None,
    ) -> list[TierCoverage]:
        """
        Get tier-based coverage analysis showing snapshot distribution across fetch tiers.

        This method dynamically integrates with the FetchTier system to validate that
        intelligent scheduling is working correctly. For each tier (opening, early, sharp,
        pregame, closing), it calculates:
        - How many games should have snapshots in that tier
        - How many games actually have snapshots
        - Coverage percentage and average snapshots per game

        Args:
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)
            sport_key: Optional sport filter (e.g., "basketball_nba")

        Returns:
            List of TierCoverage results, one per tier, ordered from longest to shortest
            time window (opening -> closing)

        Example:
            metrics = QualityMetrics(session)
            coverage = await metrics.get_tier_coverage(
                start_date=datetime(2024, 10, 1, tzinfo=UTC),
                end_date=datetime(2024, 10, 31, tzinfo=UTC),
                sport_key="basketball_nba"
            )
            for tier_info in coverage:
                print(f"{tier_info.tier_name}: {tier_info.coverage_pct}% coverage")
        """
        # Get all events in date range
        events_query = select(func.count(Event.id)).where(
            and_(
                Event.commence_time >= start_date,
                Event.commence_time <= end_date,
            )
        )

        if sport_key:
            events_query = events_query.where(Event.sport_key == sport_key)

        # All events are eligible for all tiers (any game could theoretically have
        # snapshots in any tier depending on when it was discovered)
        games_in_tier_range = (await self.session.execute(events_query)).scalar_one()

        # Each snapshot takes the tier ``snapshot_tiers`` derives over its event's snapshots; an
        # event's snapshot time counts once per tier its snapshots there cover.
        snapshots_query = (
            select(
                OddsSnapshot.event_id,
                OddsSnapshot.snapshot_time,
                OddsSnapshot.source,
                OddsSnapshot.time_basis,
                OddsSnapshot.market,
                Event.commence_time,
            )
            .join(Event, Event.id == OddsSnapshot.event_id)
            .where(
                and_(
                    Event.commence_time >= start_date,
                    Event.commence_time <= end_date,
                )
            )
        )

        if sport_key:
            snapshots_query = snapshots_query.where(Event.sport_key == sport_key)

        by_event: dict[str, list[tuple[datetime, Source | None, TimeBasis | None, str | None]]] = (
            defaultdict(list)
        )
        kickoffs: dict[str, datetime] = {}
        for event_id, snapshot_time, source, time_basis, market, commence_time in (
            await self.session.execute(snapshots_query)
        ).all():
            by_event[event_id].append((snapshot_time, source, time_basis, market))
            kickoffs[event_id] = commence_time

        snapshots_per_event: dict[FetchTier, Counter[str]] = defaultdict(Counter)
        for event_id, snapshots in by_event.items():
            covered = {
                (tier, snapshot[0])
                for snapshot, tier in zip(
                    snapshots, snapshot_tiers(snapshots, kickoffs[event_id]), strict=True
                )
            }
            for tier, _ in covered:
                snapshots_per_event[tier][event_id] += 1

        tier_results = []
        for fetch_tier in FetchTier:
            counts = snapshots_per_event[fetch_tier]
            games_with_tier_snapshots = len(counts)
            total_snapshots_in_tier = sum(counts.values())

            # Calculate coverage percentage
            coverage_pct = (
                (games_with_tier_snapshots / games_in_tier_range * 100)
                if games_in_tier_range > 0
                else 0.0
            )

            # Calculate average snapshots per game
            avg_snapshots_per_game = (
                (total_snapshots_in_tier / games_with_tier_snapshots)
                if games_with_tier_snapshots > 0
                else 0.0
            )

            tier_results.append(
                TierCoverage(
                    tier=fetch_tier.value,
                    tier_name=fetch_tier.value,
                    hours_range=self._get_tier_hours_range(fetch_tier),
                    expected_interval_hours=fetch_tier.interval_hours,
                    games_in_tier_range=games_in_tier_range,
                    games_with_tier_snapshots=games_with_tier_snapshots,
                    total_snapshots_in_tier=total_snapshots_in_tier,
                    coverage_pct=coverage_pct,
                    avg_snapshots_per_game=avg_snapshots_per_game,
                )
            )

        return tier_results

    @staticmethod
    def _get_tier_hours_range(tier: FetchTier) -> str:
        """
        Get human-readable hours range for a fetch tier.

        Args:
            tier: FetchTier enum value

        Returns:
            Human-readable string describing the time window
        """
        ranges = {
            FetchTier.IN_PLAY: "After game start",
            FetchTier.CLOSING: "0-3 hours before",
            FetchTier.PREGAME: "3-12 hours before",
            FetchTier.SHARP: "12-24 hours before",
            FetchTier.EARLY: "1-3 days before",
            FetchTier.OPENING: "3+ days before",
        }
        return ranges[tier]

    async def get_bookmaker_coverage(
        self,
        start_date: datetime,
        end_date: datetime,
        sport_key: str | None = None,
    ) -> list[BookmakerCoverage]:
        """
        Get bookmaker coverage analysis showing availability across games.

        This method identifies systematic gaps in data collection by tracking
        which bookmakers consistently provide odds data and which have missing
        coverage. Useful for identifying reliability issues with specific bookmakers.

        Args:
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)
            sport_key: Optional sport filter (e.g., "basketball_nba")

        Returns:
            List of BookmakerCoverage results, sorted by coverage_pct descending

        Example:
            metrics = QualityMetrics(session)
            coverage = await metrics.get_bookmaker_coverage(
                start_date=datetime(2024, 10, 1, tzinfo=UTC),
                end_date=datetime(2024, 10, 31, tzinfo=UTC),
                sport_key="basketball_nba"
            )
            for book in coverage:
                print(f"{book.bookmaker_key}: {book.coverage_pct}% coverage")
        """
        # First, get total games in date range
        total_games_result = await self.get_game_counts(
            start_date=start_date,
            end_date=end_date,
            sport_key=sport_key,
        )
        total_games = total_games_result.total_games

        # Count distinct events and snapshots per bookmaker
        query = (
            select(
                Price.bookmaker,
                func.count(func.distinct(OddsSnapshot.event_id)).label("games_with_odds"),
                func.count(func.distinct(Price.snapshot_id)).label("total_snapshots"),
            )
            .join(OddsSnapshot, OddsSnapshot.id == Price.snapshot_id)
            .join(Event, Event.id == OddsSnapshot.event_id)
            .where(
                and_(
                    Event.commence_time >= start_date,
                    Event.commence_time <= end_date,
                )
            )
        )

        if sport_key:
            query = query.where(Event.sport_key == sport_key)

        query = query.group_by(Price.bookmaker)

        result = await self.session.execute(query)
        rows = result.all()

        # Build coverage results
        coverage_results = []
        for row in rows:
            bookmaker_key = row[0]
            games_with_odds = row[1]
            total_snapshots = row[2]

            # Calculate coverage percentage
            coverage_pct = (games_with_odds / total_games * 100) if total_games > 0 else 0.0

            # Calculate average snapshots per game
            avg_snapshots_per_game = (
                (total_snapshots / games_with_odds) if games_with_odds > 0 else 0.0
            )

            coverage_results.append(
                BookmakerCoverage(
                    bookmaker_key=bookmaker_key,
                    total_games=total_games,
                    games_with_odds=games_with_odds,
                    coverage_pct=coverage_pct,
                    total_snapshots=total_snapshots,
                    avg_snapshots_per_game=avg_snapshots_per_game,
                )
            )

        # Sort by coverage percentage descending
        coverage_results.sort(key=lambda x: x.coverage_pct, reverse=True)

        return coverage_results
