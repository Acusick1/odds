"""Game-aware scheduling intelligence for adaptive data collection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import structlog

from core import tier_utils
from odds_core.database import async_session_maker
from odds_lambda.fetch_tier import FetchTier
from odds_core.models import Event, EventStatus
from odds_lambda.storage.readers import OddsReader

logger = structlog.get_logger()


@dataclass(slots=True, frozen=True)
class ScheduleDecision:
    """
    Decision about whether to execute and when to run next (immutable).

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
        1. Find closest upcoming game in database
        2. If no games: EXECUTE to discover new games from API (bootstrap/off-season)
        3. If game already started: don't execute, check again soon
        4. If game exists and upcoming: determine tier based on time until game
        5. Calculate next execution based on tier interval
        """
        closest_game = await self.get_closest_game()

        if not closest_game:
            # No upcoming games in database - fetch from API to discover new games
            # This handles both initial bootstrap and end-of-season scenarios
            return ScheduleDecision(
                should_execute=True,
                reason="No games in database - fetching from API to discover new games",
                next_execution=datetime.now(UTC) + timedelta(hours=24),  # Check daily for new games
                tier=None,
            )

        now = datetime.now(UTC)
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
        tier = tier_utils.calculate_tier(hours_until)
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

            now = datetime.now(UTC)
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

            now = datetime.now(UTC)

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

            now = datetime.now(UTC)
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

            now = datetime.now(UTC)
            end_date = now + timedelta(days=30)

            events = await reader.get_events_by_date_range(
                start_date=now,
                end_date=end_date,
                status=EventStatus.SCHEDULED,
            )

            return len(events) > 0
