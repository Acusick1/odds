"""Tier coverage validation for data quality assurance.

This module validates that all required fetch tiers have been collected for games,
ensuring ML-ready data completeness across the adaptive sampling spectrum.
"""

from dataclasses import dataclass, field
from datetime import UTC, date, datetime

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from core.fetch_tier import FetchTier
from storage.readers import OddsReader

logger = structlog.get_logger()


@dataclass
class TierCoverageReport:
    """Coverage report for a single game."""

    event_id: str
    home_team: str
    away_team: str
    commence_time: datetime
    tiers_present: set[FetchTier] = field(default_factory=set)
    tiers_missing: set[FetchTier] = field(default_factory=set)
    total_snapshots: int = 0
    snapshots_by_tier: dict[FetchTier, int] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        """Check if all required tiers are present."""
        return len(self.tiers_missing) == 0

    @property
    def coverage_percentage(self) -> float:
        """Calculate percentage of required tiers present."""
        total_required = len(self.tiers_present) + len(self.tiers_missing)
        if total_required == 0:
            return 0.0
        return (len(self.tiers_present) / total_required) * 100


@dataclass
class DailyValidationReport:
    """Aggregate validation report for a date."""

    target_date: date
    validation_time: datetime
    total_games: int = 0
    complete_games: int = 0
    incomplete_games: int = 0
    games_by_tier_coverage: dict[int, int] = field(
        default_factory=dict
    )  # num_tiers -> count of games
    missing_tier_breakdown: dict[FetchTier, int] = field(
        default_factory=dict
    )  # tier -> count of games missing it
    game_reports: list[TierCoverageReport] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Check if all games have complete coverage."""
        return self.incomplete_games == 0

    @property
    def completion_rate(self) -> float:
        """Calculate percentage of games with complete coverage."""
        if self.total_games == 0:
            return 0.0
        return (self.complete_games / self.total_games) * 100


class TierCoverageValidator:
    """Validates fetch tier coverage for data completeness."""

    # All 5 tiers by default
    DEFAULT_REQUIRED_TIERS = {
        FetchTier.OPENING,
        FetchTier.EARLY,
        FetchTier.SHARP,
        FetchTier.PREGAME,
        FetchTier.CLOSING,
    }

    def __init__(self, session: AsyncSession):
        """
        Initialize validator with database session.

        Args:
            session: Async database session
        """
        self.session = session
        self.reader = OddsReader(session)

    async def validate_game(
        self, event_id: str, required_tiers: set[FetchTier] | None = None
    ) -> TierCoverageReport:
        """
        Validate tier coverage for a single game.

        Args:
            event_id: Event identifier
            required_tiers: Set of required tiers (defaults to all 5)

        Returns:
            TierCoverageReport with coverage details

        Example:
            validator = TierCoverageValidator(session)
            report = await validator.validate_game(
                event_id="abc123",
                required_tiers={FetchTier.OPENING, FetchTier.CLOSING}
            )
            if not report.is_complete:
                print(f"Missing tiers: {report.tiers_missing}")
        """
        required_tiers = required_tiers or self.DEFAULT_REQUIRED_TIERS

        # Get event details
        event = await self.reader.get_event_by_id(event_id)
        if not event:
            raise ValueError(f"Event {event_id} not found")

        # Get tier coverage
        tier_coverage = await self.reader.get_tier_coverage_for_event(event_id)

        # Get all snapshots to count total
        all_snapshots = await self.reader.get_snapshots_by_tier(event_id)

        # Convert string tier names to FetchTier enum
        tiers_present = set()
        snapshots_by_tier = {}
        for tier_str, count in tier_coverage.items():
            try:
                tier = FetchTier(tier_str)
                tiers_present.add(tier)
                snapshots_by_tier[tier] = count
            except ValueError:
                logger.warning("invalid_tier_value", tier=tier_str, event_id=event_id)

        # Calculate missing tiers
        tiers_missing = required_tiers - tiers_present

        return TierCoverageReport(
            event_id=event.id,
            home_team=event.home_team,
            away_team=event.away_team,
            commence_time=event.commence_time,
            tiers_present=tiers_present,
            tiers_missing=tiers_missing,
            total_snapshots=len(all_snapshots),
            snapshots_by_tier=snapshots_by_tier,
        )

    async def validate_date(
        self,
        target_date: date | datetime,
        required_tiers: set[FetchTier] | None = None,
    ) -> DailyValidationReport:
        """
        Validate tier coverage for all games on a specific date.

        Args:
            target_date: Date to validate (by game commence_time)
            required_tiers: Set of required tiers (defaults to all 5)

        Returns:
            DailyValidationReport with aggregate statistics

        Example:
            from datetime import date
            validator = TierCoverageValidator(session)
            report = await validator.validate_date(
                target_date=date(2024, 10, 24)
            )
            print(f"Complete: {report.complete_games}/{report.total_games}")
            if not report.is_valid:
                print(f"Missing tier breakdown: {report.missing_tier_breakdown}")
        """
        required_tiers = required_tiers or self.DEFAULT_REQUIRED_TIERS

        # Get all final games for the date
        games = await self.reader.get_games_by_date(target_date)

        # Initialize report
        report = DailyValidationReport(
            target_date=target_date if isinstance(target_date, date) else target_date.date(),
            validation_time=datetime.now(UTC),
            total_games=len(games),
        )

        # Validate each game
        for game in games:
            try:
                game_report = await self.validate_game(game.id, required_tiers)
                report.game_reports.append(game_report)

                # Update aggregate statistics
                if game_report.is_complete:
                    report.complete_games += 1
                else:
                    report.incomplete_games += 1

                # Track games by tier coverage count
                num_tiers = len(game_report.tiers_present)
                report.games_by_tier_coverage[num_tiers] = (
                    report.games_by_tier_coverage.get(num_tiers, 0) + 1
                )

                # Track which tiers are missing across all games
                for missing_tier in game_report.tiers_missing:
                    report.missing_tier_breakdown[missing_tier] = (
                        report.missing_tier_breakdown.get(missing_tier, 0) + 1
                    )

            except Exception as e:
                logger.error(
                    "game_validation_failed",
                    event_id=game.id,
                    error=str(e),
                )
                continue

        logger.info(
            "daily_validation_completed",
            target_date=str(target_date),
            total_games=report.total_games,
            complete_games=report.complete_games,
            incomplete_games=report.incomplete_games,
        )

        return report

    async def validate_date_range(
        self,
        start_date: date | datetime,
        end_date: date | datetime,
        required_tiers: set[FetchTier] | None = None,
    ) -> list[DailyValidationReport]:
        """
        Validate tier coverage for a range of dates.

        Args:
            start_date: Start of date range
            end_date: End of date range (inclusive)
            required_tiers: Set of required tiers (defaults to all 5)

        Returns:
            List of DailyValidationReport, one per date

        Example:
            from datetime import date
            validator = TierCoverageValidator(session)
            reports = await validator.validate_date_range(
                start_date=date(2024, 10, 20),
                end_date=date(2024, 10, 25)
            )
            for report in reports:
                print(f"{report.target_date}: {report.completion_rate:.1f}% complete")
        """
        reports = []

        # Convert to date objects if datetime
        if isinstance(start_date, datetime):
            start_date = start_date.date()
        if isinstance(end_date, datetime):
            end_date = end_date.date()

        # Iterate through each date
        current_date = start_date
        while current_date <= end_date:
            report = await self.validate_date(current_date, required_tiers)
            reports.append(report)

            # Move to next day
            current_date = date.fromordinal(current_date.toordinal() + 1)

        return reports
