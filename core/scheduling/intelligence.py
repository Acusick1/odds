"""Game-aware scheduling intelligence for adaptive data collection."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

import structlog

from core.database import async_session_maker
from core.models import Event, EventStatus
from storage.readers import OddsReader

logger = structlog.get_logger()


class FetchTier(Enum):
    """
    Adaptive sampling tiers based on game proximity.

    Each tier represents different data collection frequency needs:
    - CLOSING: Most critical period for line movement
    - PREGAME: Active betting period with frequent updates
    - SHARP: Professional betting period
    - EARLY: Opening line establishment
    - OPENING: Initial line release
    """

    CLOSING = "closing"  # 0-3 hours before: every 30 min
    PREGAME = "pregame"  # 3-12 hours before: every 3 hours
    SHARP = "sharp"  # 12-24 hours before: every 12 hours
    EARLY = "early"  # 1-3 days before: every 24 hours
    OPENING = "opening"  # 3+ days before: every 48 hours

    @property
    def interval_hours(self) -> float:
        """Get interval in hours for this tier."""
        intervals = {
            FetchTier.CLOSING: 0.5,  # 30 minutes
            FetchTier.PREGAME: 3.0,
            FetchTier.SHARP: 12.0,
            FetchTier.EARLY: 24.0,
            FetchTier.OPENING: 48.0,
        }
        return intervals[self]


@dataclass
class ScheduleDecision:
    """
    Decision about whether to execute and when to run next.

    Attributes:
        should_execute: Whether job should run now
        reason: Human-readable explanation for decision
        next_execution: When to schedule next run (None = no more runs needed)
        tier: Which fetch tier applies (None if not applicable)
    """

    should_execute: bool
    reason: str
    next_execution: datetime | None
    tier: FetchTier | None


class SchedulingIntelligence:
    """
    Determines optimal data collection timing based on game schedule.

    Uses adaptive sampling strategy:
    - More frequent fetching as games approach
    - No fetching when no games scheduled (off-season)
    - Self-adjusting based on actual database state
    """

    def __init__(self, lookahead_days: int = 7, session_factory=None):
        """
        Initialize scheduling intelligence.

        Args:
            lookahead_days: How many days ahead to check for games
            session_factory: Optional session factory for testing (defaults to async_session_maker)
        """
        self.lookahead_days = lookahead_days
        self.session_factory = session_factory or async_session_maker

    async def should_execute_fetch(self) -> ScheduleDecision:
        """
        Determine if odds fetch should execute now and when to run next.

        Returns:
            ScheduleDecision with execution recommendation and next schedule

        Logic:
        1. Find closest upcoming game
        2. If no games: don't execute, check again in 24 hours
        3. If game exists: determine tier based on time until game
        4. Calculate next execution based on tier interval
        """
        closest_game = await self.get_closest_game()

        if not closest_game:
            # No upcoming games - check again tomorrow
            return ScheduleDecision(
                should_execute=False,
                reason="No games scheduled in next 7 days",
                next_execution=datetime.utcnow() + timedelta(hours=24),
                tier=None,
            )

        now = datetime.utcnow()
        hours_until = (closest_game.commence_time - now).total_seconds() / 3600

        # Don't fetch for games that already started
        if hours_until < 0:
            return ScheduleDecision(
                should_execute=False,
                reason=f"Closest game already started: {closest_game.home_team} vs {closest_game.away_team}",
                next_execution=now + timedelta(hours=1),  # Check for next game soon
                tier=None,
            )

        # Determine tier and next execution
        tier = self._get_tier_for_hours(hours_until)
        next_execution = now + timedelta(hours=tier.interval_hours)

        return ScheduleDecision(
            should_execute=True,
            reason=f"Game in {hours_until:.1f}h: {closest_game.home_team} vs {closest_game.away_team} ({tier.value} tier)",
            next_execution=next_execution,
            tier=tier,
        )

    async def should_execute_scores(self) -> ScheduleDecision:
        """
        Determine if scores fetch should execute now.

        Scores should be fetched if there are any games in the last 3 days
        that might not have final scores yet.

        Returns:
            ScheduleDecision for scores fetch
        """
        async with self.session_factory() as session:
            reader = OddsReader(session)

            now = datetime.utcnow()
            start_date = now - timedelta(days=3)

            # Check for games that might need score updates
            events = await reader.get_events_by_date_range(
                start_date=start_date,
                end_date=now,
            )

            # Count how many don't have final status
            needs_update = sum(1 for e in events if e.status != EventStatus.FINAL)

            if needs_update > 0:
                return ScheduleDecision(
                    should_execute=True,
                    reason=f"{needs_update} games need score updates",
                    next_execution=now + timedelta(hours=6),  # Check again in 6 hours
                    tier=None,
                )
            else:
                return ScheduleDecision(
                    should_execute=False,
                    reason="No games need score updates",
                    next_execution=now + timedelta(hours=12),  # Check again in 12 hours
                    tier=None,
                )

    async def should_execute_status_update(self) -> ScheduleDecision:
        """
        Determine if status update should execute now.

        Status updates should run if there are scheduled games that might have started.

        Returns:
            ScheduleDecision for status update
        """
        async with self.session_factory() as session:
            reader = OddsReader(session)

            now = datetime.utcnow()

            # Check for games that should be live (started in last 4 hours)
            start_range = now - timedelta(hours=4)
            events = await reader.get_events_by_date_range(
                start_date=start_range,
                end_date=now,
                status=EventStatus.SCHEDULED,
            )

            if len(events) > 0:
                return ScheduleDecision(
                    should_execute=True,
                    reason=f"{len(events)} games may have started",
                    next_execution=now + timedelta(hours=1),  # Check hourly
                    tier=None,
                )
            else:
                return ScheduleDecision(
                    should_execute=False,
                    reason="No games to update",
                    next_execution=now + timedelta(hours=6),  # Check less frequently
                    tier=None,
                )

    async def get_closest_game(self) -> Event | None:
        """
        Find the soonest upcoming scheduled game.

        Returns:
            Event object for closest game, or None if no games scheduled
        """
        async with self.session_factory() as session:
            reader = OddsReader(session)

            now = datetime.utcnow()
            end_date = now + timedelta(days=self.lookahead_days)

            events = await reader.get_events_by_date_range(
                start_date=now,
                end_date=end_date,
                status=EventStatus.SCHEDULED,
            )

            if not events:
                return None

            # Return earliest game
            return min(events, key=lambda e: e.commence_time)

    async def is_nba_season(self) -> bool:
        """
        Heuristic check if we're in NBA season.

        Returns:
            True if games scheduled in next 30 days
        """
        async with self.session_factory() as session:
            reader = OddsReader(session)

            now = datetime.utcnow()
            end_date = now + timedelta(days=30)

            events = await reader.get_events_by_date_range(
                start_date=now,
                end_date=end_date,
                status=EventStatus.SCHEDULED,
            )

            return len(events) > 0

    def _get_tier_for_hours(self, hours_until: float) -> FetchTier:
        """
        Determine fetch tier based on hours until game.

        Args:
            hours_until: Hours until game commence time

        Returns:
            Appropriate FetchTier
        """
        if hours_until <= 3:
            return FetchTier.CLOSING
        elif hours_until <= 12:
            return FetchTier.PREGAME
        elif hours_until <= 24:
            return FetchTier.SHARP
        elif hours_until <= 72:  # 3 days
            return FetchTier.EARLY
        else:
            return FetchTier.OPENING
