"""Database query module for data quality and coverage statistics."""

from __future__ import annotations

from datetime import datetime

from odds_core.models import Event, OddsSnapshot
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

    event_id: str = Field(description="Event identifier")
    sport_key: str = Field(description="Sport identifier")
    home_team: str = Field(description="Home team name")
    away_team: str = Field(description="Away team name")
    commence_time: datetime = Field(description="Game start time")
    snapshot_count: int = Field(description="Number of odds snapshots")


class GameWithScoresResult(BaseModel):
    """Result model for games with final scores."""

    event_id: str = Field(description="Event identifier")
    sport_key: str = Field(description="Sport identifier")
    home_team: str = Field(description="Home team name")
    away_team: str = Field(description="Away team name")
    commence_time: datetime = Field(description="Game start time")
    home_score: int = Field(description="Final home team score")
    away_score: int = Field(description="Final away team score")


class GameMissingScoresResult(BaseModel):
    """Result model for games missing score data."""

    event_id: str = Field(description="Event identifier")
    sport_key: str = Field(description="Sport identifier")
    home_team: str = Field(description="Home team name")
    away_team: str = Field(description="Away team name")
    commence_time: datetime = Field(description="Game start time")
    status: str = Field(description="Event status")


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
                func.count(OddsSnapshot.id).label("snapshot_count"),
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
