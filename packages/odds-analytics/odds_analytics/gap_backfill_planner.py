"""Gap detection and backfill planning for missing odds snapshots.

This module identifies missing fetch tier data in historical odds collection and
generates executable backfill plans to fill those gaps. Plans prioritize games by
highest-priority missing tier (closing > pregame > sharp > early > opening) and
ensure all-or-nothing coverage (all missing tiers filled per selected game).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import structlog
from odds_core.time import utc_isoformat
from odds_lambda.fetch_tier import FetchTier
from odds_lambda.storage.readers import OddsReader
from odds_lambda.storage.tier_validator import TierCoverageValidator
from sqlalchemy.ext.asyncio import AsyncSession

from odds_analytics.game_selector import GameSelector

logger = structlog.get_logger()


@dataclass
class GameGapInfo:
    """Information about gaps in a single game."""

    event_id: str
    home_team: str
    away_team: str
    commence_time: datetime
    missing_tiers: frozenset[FetchTier]
    highest_priority_missing: FetchTier  # Tier with highest priority (closest to game)
    missing_snapshot_count: int  # Total missing snapshots across all missing tiers


@dataclass
class GapAnalysis:
    """Analysis of gaps across a date range."""

    start_date: date
    end_date: date
    total_games: int
    games_with_gaps: int
    total_missing_snapshots: int
    games_by_priority: dict[FetchTier, list[GameGapInfo]]  # Tier -> games missing it


class GapBackfillPlanner:
    """Detects gaps in tier coverage and generates backfill plans."""

    # Quota cost per snapshot (historical multiplier � regions � markets)
    # Historical API costs 10x normal rate
    QUOTA_PER_SNAPSHOT = 30  # 10 (historical) � 1 (region) � 3 (markets)

    def __init__(self, session: AsyncSession):
        """
        Initialize gap backfill planner.

        Args:
            session: Async database session
        """
        self.session = session
        self.validator = TierCoverageValidator(session)
        self.reader = OddsReader(session)

    async def analyze_gaps(
        self,
        start_date: date | datetime,
        end_date: date | datetime,
    ) -> GapAnalysis:
        """
        Analyze gaps in tier coverage across a date range.

        Args:
            start_date: Start of date range
            end_date: End of date range

        Returns:
            GapAnalysis with complete gap information

        Example:
            planner = GapBackfillPlanner(session)
            analysis = await planner.analyze_gaps(
                start_date=date(2024, 10, 1),
                end_date=date(2024, 10, 31)
            )
            print(f"Found {analysis.games_with_gaps} games with gaps")
        """
        # Convert to date objects if datetime
        if isinstance(start_date, datetime):
            start_date = start_date.date()
        if isinstance(end_date, datetime):
            end_date = end_date.date()

        logger.info(
            "analyzing_gaps",
            start_date=str(start_date),
            end_date=str(end_date),
        )

        # Get validation reports for date range
        reports = await self.validator.validate_date_range(start_date, end_date)

        # Collect games with gaps
        games_with_gaps: list[GameGapInfo] = []
        games_by_priority: dict[FetchTier, list[GameGapInfo]] = {
            tier: [] for tier in FetchTier.get_priority_order()
        }

        for report in reports:
            # Process each incomplete game in the report
            for game_report in report.game_reports:
                if not game_report.is_complete:
                    # Calculate missing snapshots for this game
                    missing_snapshot_count = await self._calculate_missing_snapshots(
                        game_report.event_id,
                        game_report.commence_time,
                        game_report.tiers_missing,
                    )

                    # Determine highest priority missing tier
                    highest_priority = self._get_highest_priority_missing_tier(
                        game_report.tiers_missing
                    )

                    gap_info = GameGapInfo(
                        event_id=game_report.event_id,
                        home_team=game_report.home_team,
                        away_team=game_report.away_team,
                        commence_time=game_report.commence_time,
                        missing_tiers=game_report.tiers_missing,
                        highest_priority_missing=highest_priority,
                        missing_snapshot_count=missing_snapshot_count,
                    )

                    games_with_gaps.append(gap_info)

                    # Add to priority bucket
                    games_by_priority[highest_priority].append(gap_info)

        total_missing_snapshots = sum(game.missing_snapshot_count for game in games_with_gaps)

        analysis = GapAnalysis(
            start_date=start_date,
            end_date=end_date,
            total_games=sum(r.total_games for r in reports),
            games_with_gaps=len(games_with_gaps),
            total_missing_snapshots=total_missing_snapshots,
            games_by_priority=games_by_priority,
        )

        logger.info(
            "gap_analysis_complete",
            total_games=analysis.total_games,
            games_with_gaps=analysis.games_with_gaps,
            total_missing_snapshots=analysis.total_missing_snapshots,
        )

        return analysis

    async def generate_plan(
        self,
        start_date: date | datetime,
        end_date: date | datetime,
        max_quota: int | None = None,
    ) -> dict:
        """
        Generate a backfill plan to fill gaps in tier coverage.

        Plan Format (compatible with BackfillExecutor):
        {
            "total_games": int,
            "total_snapshots": int,
            "estimated_quota_usage": int,
            "games": [
                {
                    "event_id": str,
                    "home_team": str,
                    "away_team": str,
                    "commence_time": str (ISO format),
                    "snapshots": [str (ISO format), ...],
                    "snapshot_count": int
                },
                ...
            ],
            "start_date": str (ISO format),
            "end_date": str (ISO format)
        }

        Args:
            start_date: Start of date range
            end_date: End of date range
            max_quota: Maximum quota to use (None = unlimited)

        Returns:
            Backfill plan dictionary compatible with BackfillExecutor

        Raises:
            ValueError: If max_quota would be exceeded by a single complete game

        Example:
            planner = GapBackfillPlanner(session)
            plan = await planner.generate_plan(
                start_date=date(2024, 10, 1),
                end_date=date(2024, 10, 31),
                max_quota=3000
            )
            # Execute with: BackfillExecutor.execute_plan(plan)
        """
        # Analyze gaps
        analysis = await self.analyze_gaps(start_date, end_date)

        if analysis.games_with_gaps == 0:
            logger.info("no_gaps_found")
            return {
                "total_games": 0,
                "total_snapshots": 0,
                "estimated_quota_usage": 0,
                "games": [],
                "start_date": str(start_date),
                "end_date": str(end_date),
            }

        # Prioritize games by highest-priority missing tier
        prioritized_games = self._prioritize_games(analysis)

        # Select games within quota constraint (all-or-nothing per game)
        selected_games = await self._select_games_within_quota(prioritized_games, max_quota)

        # Generate plan for selected games
        plan = await self._create_plan(selected_games, start_date, end_date)

        logger.info(
            "backfill_plan_generated",
            total_games=plan["total_games"],
            total_snapshots=plan["total_snapshots"],
            estimated_quota=plan["estimated_quota_usage"],
        )

        return plan

    async def _calculate_missing_snapshots(
        self,
        event_id: str,
        commence_time: datetime,
        missing_tiers: frozenset[FetchTier],
    ) -> int:
        """
        Calculate number of missing snapshots for a game.

        Args:
            event_id: Event identifier
            commence_time: Game start time
            missing_tiers: Tiers with missing data

        Returns:
            Total count of missing snapshots
        """
        # Use GameSelector to calculate expected snapshot times
        selector = GameSelector(
            start_date=commence_time,
            end_date=commence_time,
        )

        expected_snapshots = selector.calculate_snapshot_times(commence_time)

        # Check which snapshots are actually missing
        missing_count = 0
        for snapshot_time in expected_snapshots:
            # Only count if snapshot doesn't exist
            exists = await self.reader.snapshot_exists(
                event_id=event_id,
                snapshot_time=snapshot_time,
                tolerance_minutes=5,
            )

            if not exists:
                missing_count += 1

        return missing_count

    def _get_highest_priority_missing_tier(self, missing_tiers: frozenset[FetchTier]) -> FetchTier:
        """
        Get the highest priority tier from a set of missing tiers.

        Priority order: CLOSING > PREGAME > SHARP > EARLY > OPENING

        Args:
            missing_tiers: Set of missing tiers

        Returns:
            Highest priority missing tier

        Example:
            >>> missing = frozenset([FetchTier.OPENING, FetchTier.CLOSING])
            >>> planner._get_highest_priority_missing_tier(missing)
            <FetchTier.CLOSING: 'closing'>
        """
        priority_order = FetchTier.get_priority_order()

        for tier in priority_order:
            if tier in missing_tiers:
                return tier

        # Should never reach here if missing_tiers is non-empty
        raise ValueError("No missing tiers found")

    def _prioritize_games(self, analysis: GapAnalysis) -> list[GameGapInfo]:
        """
        Prioritize games by highest-priority missing tier.

        Games are sorted by:
        1. Highest priority missing tier (CLOSING first, OPENING last)
        2. Commence time (earliest first)

        Args:
            analysis: Gap analysis with games grouped by priority

        Returns:
            List of GameGapInfo sorted by priority
        """
        prioritized: list[GameGapInfo] = []

        # Process tiers in priority order (CLOSING to OPENING)
        for tier in FetchTier.get_priority_order():
            games_for_tier = analysis.games_by_priority.get(tier, [])

            # Sort games by commence time within each tier
            games_sorted = sorted(games_for_tier, key=lambda g: g.commence_time)

            prioritized.extend(games_sorted)

        logger.info(
            "games_prioritized",
            total_games=len(prioritized),
            by_priority={
                tier.value: len(analysis.games_by_priority.get(tier, []))
                for tier in FetchTier.get_priority_order()
            },
        )

        return prioritized

    async def _select_games_within_quota(
        self,
        prioritized_games: list[GameGapInfo],
        max_quota: int | None,
    ) -> list[GameGapInfo]:
        """
        Select games to fill within quota constraint.

        Uses all-or-nothing approach: only include complete games within quota limit.
        Partial games are never included.

        Args:
            prioritized_games: Games sorted by priority
            max_quota: Maximum quota to use (None = unlimited)

        Returns:
            List of games to include in backfill plan

        Raises:
            ValueError: If max_quota is set but first game exceeds it
        """
        if max_quota is None:
            logger.info("no_quota_limit_using_all_games", total_games=len(prioritized_games))
            return prioritized_games

        selected: list[GameGapInfo] = []
        quota_used = 0

        for game in prioritized_games:
            game_quota_cost = game.missing_snapshot_count * self.QUOTA_PER_SNAPSHOT

            # Check if adding this complete game would exceed quota
            if quota_used + game_quota_cost <= max_quota:
                selected.append(game)
                quota_used += game_quota_cost
            else:
                # Can't fit this game - stop (all-or-nothing)
                logger.info(
                    "quota_limit_reached",
                    selected_games=len(selected),
                    quota_used=quota_used,
                    max_quota=max_quota,
                    next_game_cost=game_quota_cost,
                )
                break

        # Validate that at least one game fits within quota
        if max_quota is not None and len(selected) == 0 and len(prioritized_games) > 0:
            first_game_cost = prioritized_games[0].missing_snapshot_count * self.QUOTA_PER_SNAPSHOT
            raise ValueError(
                f"Max quota {max_quota} is too low. First game requires {first_game_cost} quota units. "
                f"Increase max_quota or run without quota limit."
            )

        logger.info(
            "games_selected_within_quota",
            selected_games=len(selected),
            quota_used=quota_used,
            max_quota=max_quota,
        )

        return selected

    async def _create_plan(
        self,
        selected_games: list[GameGapInfo],
        start_date: date | datetime,
        end_date: date | datetime,
    ) -> dict:
        """
        Create backfill plan from selected games.

        Args:
            selected_games: Games to include in plan
            start_date: Start of date range
            end_date: End of date range

        Returns:
            Backfill plan dictionary compatible with BackfillExecutor
        """
        games_list = []
        total_snapshots = 0

        selector = GameSelector(
            start_date=start_date
            if isinstance(start_date, datetime)
            else datetime.combine(start_date, datetime.min.time()),
            end_date=end_date
            if isinstance(end_date, datetime)
            else datetime.combine(end_date, datetime.min.time()),
        )

        for game in selected_games:
            # Calculate all expected snapshot times
            expected_snapshots = selector.calculate_snapshot_times(game.commence_time)

            # Filter to only missing snapshots
            missing_snapshot_times = []
            for snapshot_time in expected_snapshots:
                exists = await self.reader.snapshot_exists(
                    event_id=game.event_id,
                    snapshot_time=snapshot_time,
                    tolerance_minutes=5,
                )

                if not exists:
                    missing_snapshot_times.append(snapshot_time)

            game_plan = {
                "event_id": game.event_id,
                "home_team": game.home_team,
                "away_team": game.away_team,
                "commence_time": utc_isoformat(game.commence_time),
                "snapshots": [utc_isoformat(s) for s in missing_snapshot_times],
                "snapshot_count": len(missing_snapshot_times),
            }

            games_list.append(game_plan)
            total_snapshots += len(missing_snapshot_times)

        estimated_quota = total_snapshots * self.QUOTA_PER_SNAPSHOT

        return {
            "total_games": len(games_list),
            "total_snapshots": total_snapshots,
            "estimated_quota_usage": estimated_quota,
            "games": games_list,
            "start_date": str(start_date),
            "end_date": str(end_date),
        }
