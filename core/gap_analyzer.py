"""Gap detection for historical odds data.

Identifies missing or incomplete data in the historical odds collection:
- Missing events: Events in date range with no snapshots
- Incomplete snapshots: Events with fewer snapshots than expected
- Missing tiers: Events missing specific fetch tiers (opening, closing, etc.)
"""

from dataclasses import dataclass
from datetime import UTC, datetime

import structlog
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.fetch_tier import FetchTier
from core.game_selector import GameSelector
from core.models import Event, EventStatus, OddsSnapshot

logger = structlog.get_logger()


@dataclass
class EventGap:
    """Details about a specific event's data gap."""

    event: Event
    current_snapshot_count: int
    expected_snapshot_count: int
    missing_snapshot_times: list[datetime]
    missing_tiers: list[FetchTier]
    tier_coverage: dict[str, int]


@dataclass
class GapReport:
    """Report of data gaps found in historical data."""

    mode: str  # 'events', 'snapshots', 'tiers', 'all'
    start_date: datetime
    end_date: datetime

    # Missing events (events with no snapshots at all)
    missing_events: list[Event]

    # Incomplete events (events with fewer snapshots than expected)
    incomplete_events: list[EventGap]

    # Events missing specific tiers
    events_missing_tiers: list[EventGap]

    # Summary statistics
    total_events_checked: int
    total_gaps_found: int
    estimated_api_calls: int

    def has_gaps(self) -> bool:
        """Check if any gaps were found."""
        return (
            len(self.missing_events) > 0
            or len(self.incomplete_events) > 0
            or len(self.events_missing_tiers) > 0
        )


class GapAnalyzer:
    """Analyzes historical odds data to identify gaps and missing data."""

    # Expected number of snapshots per game (from GameSelector strategy)
    EXPECTED_SNAPSHOTS_PER_GAME = 5

    # Expected tiers for a complete game
    EXPECTED_TIERS = {
        FetchTier.OPENING,
        FetchTier.EARLY,
        FetchTier.SHARP,
        FetchTier.PREGAME,
        FetchTier.CLOSING,
    }

    def __init__(self, session: AsyncSession):
        """
        Initialize gap analyzer.

        Args:
            session: Async database session
        """
        self.session = session

    async def analyze_gaps(
        self,
        start_date: datetime,
        end_date: datetime,
        mode: str = "all",
        sport_key: str = "basketball_nba",
    ) -> GapReport:
        """
        Analyze data gaps in the specified date range.

        Args:
            start_date: Start of date range to analyze
            end_date: End of date range to analyze
            mode: Analysis mode - 'events', 'snapshots', 'tiers', or 'all'
            sport_key: Sport to analyze (default: basketball_nba)

        Returns:
            GapReport with detailed findings

        Note:
            Only analyzes FINAL events (games that have completed) to avoid
            flagging upcoming games as incomplete.
        """
        logger.info(
            "gap_analysis_started",
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            mode=mode,
            sport_key=sport_key,
        )

        missing_events = []
        incomplete_events = []
        events_missing_tiers = []
        total_events_checked = 0

        # Get all FINAL events in date range
        events = await self._get_events_in_range(start_date, end_date, sport_key)
        total_events_checked = len(events)

        for event in events:
            # Get snapshot count and tier coverage for this event
            snapshot_count = await self._get_snapshot_count(event.id)
            tier_coverage = await self._get_tier_coverage(event.id)

            # Check for missing events (no snapshots at all)
            if mode in ("events", "all"):
                if snapshot_count == 0:
                    missing_events.append(event)
                    continue  # Skip further checks for this event

            # Check for incomplete snapshots (fewer than expected)
            if mode in ("snapshots", "all"):
                if 0 < snapshot_count < self.EXPECTED_SNAPSHOTS_PER_GAME:
                    # Calculate which snapshot times are missing
                    missing_times = await self._calculate_missing_snapshot_times(event)

                    gap = EventGap(
                        event=event,
                        current_snapshot_count=snapshot_count,
                        expected_snapshot_count=self.EXPECTED_SNAPSHOTS_PER_GAME,
                        missing_snapshot_times=missing_times,
                        missing_tiers=[],  # Will be filled by tier analysis
                        tier_coverage=tier_coverage,
                    )
                    incomplete_events.append(gap)

            # Check for missing tiers
            if mode in ("tiers", "all"):
                missing_tiers = await self._identify_missing_tiers(event, tier_coverage)

                if missing_tiers:
                    # Check if already in incomplete_events
                    existing_gap = next(
                        (g for g in incomplete_events if g.event.id == event.id), None
                    )

                    if existing_gap:
                        # Update existing gap with tier info
                        existing_gap.missing_tiers = missing_tiers
                    else:
                        # Create new gap entry for tier issues
                        gap = EventGap(
                            event=event,
                            current_snapshot_count=snapshot_count,
                            expected_snapshot_count=self.EXPECTED_SNAPSHOTS_PER_GAME,
                            missing_snapshot_times=[],
                            missing_tiers=missing_tiers,
                            tier_coverage=tier_coverage,
                        )
                        events_missing_tiers.append(gap)

        # Calculate estimated API calls needed to fill gaps
        estimated_calls = self._estimate_api_calls(
            missing_events, incomplete_events, events_missing_tiers
        )

        total_gaps = len(missing_events) + len(incomplete_events) + len(events_missing_tiers)

        logger.info(
            "gap_analysis_completed",
            total_events=total_events_checked,
            missing_events=len(missing_events),
            incomplete_events=len(incomplete_events),
            events_missing_tiers=len(events_missing_tiers),
            estimated_api_calls=estimated_calls,
        )

        return GapReport(
            mode=mode,
            start_date=start_date,
            end_date=end_date,
            missing_events=missing_events,
            incomplete_events=incomplete_events,
            events_missing_tiers=events_missing_tiers,
            total_events_checked=total_events_checked,
            total_gaps_found=total_gaps,
            estimated_api_calls=estimated_calls,
        )

    async def _get_events_in_range(
        self, start_date: datetime, end_date: datetime, sport_key: str
    ) -> list[Event]:
        """
        Get all FINAL events in date range.

        Only returns games that have completed to avoid flagging upcoming games.
        """
        query = (
            select(Event)
            .where(
                and_(
                    Event.commence_time >= start_date,
                    Event.commence_time <= end_date,
                    Event.sport_key == sport_key,
                    Event.status == EventStatus.FINAL,
                )
            )
            .order_by(Event.commence_time)
        )

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def _get_snapshot_count(self, event_id: str) -> int:
        """Get total snapshot count for an event."""
        query = select(func.count(OddsSnapshot.id)).where(OddsSnapshot.event_id == event_id)

        result = await self.session.execute(query)
        return result.scalar() or 0

    async def _get_tier_coverage(self, event_id: str) -> dict[str, int]:
        """Get count of snapshots per tier for an event."""
        query = (
            select(OddsSnapshot.fetch_tier, func.count(OddsSnapshot.id))
            .where(OddsSnapshot.event_id == event_id)
            .where(OddsSnapshot.fetch_tier.isnot(None))
            .group_by(OddsSnapshot.fetch_tier)
        )

        result = await self.session.execute(query)
        rows = result.all()

        return dict(rows)

    async def _calculate_missing_snapshot_times(self, event: Event) -> list[datetime]:
        """
        Calculate which snapshot times are missing for an event.

        Uses the same logic as GameSelector to determine expected times,
        then checks which ones don't exist in the database.
        """
        # Generate expected snapshot times
        selector = GameSelector(
            start_date=event.commence_time,
            end_date=event.commence_time,
            target_games=1,
        )
        expected_times = selector.calculate_snapshot_times(event.commence_time)

        # Get actual snapshot times from database
        query = (
            select(OddsSnapshot.snapshot_time)
            .where(OddsSnapshot.event_id == event.id)
            .order_by(OddsSnapshot.snapshot_time)
        )

        result = await self.session.execute(query)
        actual_times = list(result.scalars().all())

        # Find missing times (with 10-minute tolerance for flexibility)
        missing = []
        for expected_time in expected_times:
            # Check if any actual snapshot is within 10 minutes of expected
            has_snapshot = any(
                abs((actual - expected_time).total_seconds()) <= 600 for actual in actual_times
            )

            if not has_snapshot:
                missing.append(expected_time)

        return missing

    async def _identify_missing_tiers(
        self, event: Event, tier_coverage: dict[str, int]
    ) -> list[FetchTier]:
        """
        Identify which fetch tiers are missing for an event.

        Only checks for tiers that would have been feasible given game timing.
        For example, if the event is in the past, we expect all tiers.
        """
        # Check current time vs game time to determine which tiers should exist
        now = datetime.now(UTC)
        game_is_past = event.commence_time < now

        missing_tiers = []

        # Only expect all tiers for games that have completed
        if game_is_past:
            for tier in self.EXPECTED_TIERS:
                if tier.value not in tier_coverage:
                    missing_tiers.append(tier)

        return missing_tiers

    def _estimate_api_calls(
        self,
        missing_events: list[Event],
        incomplete_events: list[EventGap],
        events_missing_tiers: list[EventGap],
    ) -> int:
        """
        Estimate API calls needed to fill gaps.

        Each snapshot costs: 10 (historical multiplier) × 1 (region) × 3 (markets) = 30
        """
        QUOTA_PER_SNAPSHOT = 30

        # Missing events need all 5 snapshots
        calls_for_missing = len(missing_events) * self.EXPECTED_SNAPSHOTS_PER_GAME * QUOTA_PER_SNAPSHOT

        # Incomplete events need the missing snapshots
        calls_for_incomplete = (
            sum(len(gap.missing_snapshot_times) for gap in incomplete_events) * QUOTA_PER_SNAPSHOT
        )

        # Events missing tiers - estimate based on tier type
        # Closing tier typically has ~6 snapshots (3 hours * 2 per hour)
        # Other tiers typically have 1-2 snapshots
        calls_for_tiers = 0
        for gap in events_missing_tiers:
            for tier in gap.missing_tiers:
                if tier == FetchTier.CLOSING:
                    calls_for_tiers += 6 * QUOTA_PER_SNAPSHOT
                else:
                    calls_for_tiers += 1 * QUOTA_PER_SNAPSHOT

        return calls_for_missing + calls_for_incomplete + calls_for_tiers

    async def generate_backfill_plan_for_gaps(self, gap_report: GapReport) -> dict:
        """
        Generate a backfill plan JSON to fill the detected gaps.

        Returns a plan compatible with the existing backfill execute command.

        Args:
            gap_report: Gap analysis report

        Returns:
            Backfill plan dictionary
        """
        games = []
        total_snapshots = 0

        # Add missing events (need all 5 snapshots)
        for event in gap_report.missing_events:
            selector = GameSelector(
                start_date=event.commence_time,
                end_date=event.commence_time,
                target_games=1,
            )
            snapshot_times = selector.calculate_snapshot_times(event.commence_time)

            from core.time import utc_isoformat

            game_plan = {
                "event_id": event.id,
                "home_team": event.home_team,
                "away_team": event.away_team,
                "commence_time": utc_isoformat(event.commence_time),
                "snapshots": [utc_isoformat(t) for t in snapshot_times],
                "snapshot_count": len(snapshot_times),
            }
            games.append(game_plan)
            total_snapshots += len(snapshot_times)

        # Add incomplete events (only missing snapshots)
        for gap in gap_report.incomplete_events:
            if gap.missing_snapshot_times:
                from core.time import utc_isoformat

                game_plan = {
                    "event_id": gap.event.id,
                    "home_team": gap.event.home_team,
                    "away_team": gap.event.away_team,
                    "commence_time": utc_isoformat(gap.event.commence_time),
                    "snapshots": [utc_isoformat(t) for t in gap.missing_snapshot_times],
                    "snapshot_count": len(gap.missing_snapshot_times),
                }
                games.append(game_plan)
                total_snapshots += len(gap.missing_snapshot_times)

        # Add events with missing tiers
        for gap in gap_report.events_missing_tiers:
            # Generate snapshot times for missing tiers
            missing_times = await self._generate_snapshot_times_for_tiers(
                gap.event, gap.missing_tiers
            )

            if missing_times:
                from core.time import utc_isoformat

                game_plan = {
                    "event_id": gap.event.id,
                    "home_team": gap.event.home_team,
                    "away_team": gap.event.away_team,
                    "commence_time": utc_isoformat(gap.event.commence_time),
                    "snapshots": [utc_isoformat(t) for t in missing_times],
                    "snapshot_count": len(missing_times),
                }
                games.append(game_plan)
                total_snapshots += len(missing_times)

        plan = {
            "total_games": len(games),
            "total_snapshots": total_snapshots,
            "estimated_quota_usage": gap_report.estimated_api_calls,
            "games": games,
            "start_date": gap_report.start_date.isoformat(),
            "end_date": gap_report.end_date.isoformat(),
            "generated_from": f"gap_analysis_{gap_report.mode}",
        }

        return plan

    async def _generate_snapshot_times_for_tiers(
        self, event: Event, missing_tiers: list[FetchTier]
    ) -> list[datetime]:
        """
        Generate snapshot times for specific missing tiers.

        Uses tier intervals to create appropriate snapshot times.
        """
        from datetime import timedelta

        snapshot_times = []
        commence_time = event.commence_time

        for tier in missing_tiers:
            if tier == FetchTier.CLOSING:
                # Generate multiple snapshots for closing (every 30 min for 3 hours)
                for i in range(6):
                    time = commence_time - timedelta(minutes=30 + i * 30)
                    snapshot_times.append(time)
            elif tier == FetchTier.PREGAME:
                # One snapshot at 3 hours before
                snapshot_times.append(commence_time - timedelta(hours=3))
            elif tier == FetchTier.SHARP:
                # One snapshot at 12 hours before
                snapshot_times.append(commence_time - timedelta(hours=12))
            elif tier == FetchTier.EARLY:
                # One snapshot at 24 hours before
                snapshot_times.append(commence_time - timedelta(hours=24))
            elif tier == FetchTier.OPENING:
                # One snapshot at 3 days before
                snapshot_times.append(commence_time - timedelta(days=3))

        return sorted(snapshot_times)
