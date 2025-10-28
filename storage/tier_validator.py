"""Tier coverage validation for data quality assurance.

This module validates that all required fetch tiers have been collected for games,
ensuring ML-ready data completeness across the adaptive sampling spectrum.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from core.fetch_tier import FetchTier
from storage.readers import OddsReader

logger = structlog.get_logger()


@dataclass(slots=True, frozen=True)
class TierCoverageReport:
    """Coverage report for a single game (immutable)."""

    event_id: str
    home_team: str
    away_team: str
    commence_time: datetime
    tiers_present: frozenset[FetchTier] = field(default_factory=frozenset)
    tiers_missing: frozenset[FetchTier] = field(default_factory=frozenset)
    total_snapshots: int = 0
    snapshots_by_tier: dict[FetchTier, int] = field(default_factory=dict)
    # Score validation
    has_final_scores: bool = False
    home_score: int | None = None
    away_score: int | None = None
    # Validation issues
    validation_issues: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_complete(self) -> bool:
        """Check if all required tiers are present."""
        return len(self.tiers_missing) == 0

    @property
    def is_fully_valid(self) -> bool:
        """Check if game has complete tier coverage AND no validation issues."""
        return self.is_complete and len(self.validation_issues) == 0

    @property
    def coverage_percentage(self) -> float:
        """Calculate percentage of required tiers present."""
        total_required = len(self.tiers_present) + len(self.tiers_missing)
        if total_required == 0:
            return 0.0
        return (len(self.tiers_present) / total_required) * 100


@dataclass(slots=True, frozen=True)
class DailyValidationReport:
    """Aggregate validation report for a date (immutable)."""

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
    game_reports: tuple[TierCoverageReport, ...] = field(default_factory=tuple)
    # Score validation
    games_missing_scores: int = 0
    # Game discovery validation
    missing_games: tuple[dict, ...] = field(default_factory=tuple)  # Games from API not in DB

    @property
    def is_valid(self) -> bool:
        """Check if all games have complete coverage."""
        return self.incomplete_games == 0

    @property
    def is_fully_valid(self) -> bool:
        """Check if validation passed with no tier, score, or discovery issues."""
        return (
            self.incomplete_games == 0
            and self.games_missing_scores == 0
            and len(self.missing_games) == 0
        )

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
        self,
        event_id: str,
        required_tiers: set[FetchTier] | None = None,
        check_scores: bool = True,
    ) -> TierCoverageReport:
        """
        Validate tier coverage for a single game.

        Args:
            event_id: Event identifier
            required_tiers: Set of required tiers (defaults to all 5)
            check_scores: Whether to validate final scores exist for FINAL games

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
        from core.models import EventStatus

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
        snapshots_by_tier_dict = {}
        for tier_str, count in tier_coverage.items():
            try:
                tier = FetchTier(tier_str)
                tiers_present.add(tier)
                snapshots_by_tier_dict[tier] = count
            except ValueError:
                logger.warning("invalid_tier_value", tier=tier_str, event_id=event_id)

        # Calculate missing tiers
        tiers_missing = required_tiers - tiers_present

        # Validate scores for FINAL games
        has_final_scores = False
        validation_issues = []

        if check_scores and event.status == EventStatus.FINAL:
            if event.home_score is not None and event.away_score is not None:
                has_final_scores = True
            else:
                validation_issues.append("Missing final scores for completed game")
                logger.warning(
                    "missing_final_scores",
                    event_id=event_id,
                    status=event.status,
                    home_score=event.home_score,
                    away_score=event.away_score,
                )

        return TierCoverageReport(
            event_id=event.id,
            home_team=event.home_team,
            away_team=event.away_team,
            commence_time=event.commence_time,
            tiers_present=frozenset(tiers_present),
            tiers_missing=frozenset(tiers_missing),
            total_snapshots=len(all_snapshots),
            snapshots_by_tier=snapshots_by_tier_dict,
            has_final_scores=has_final_scores,
            home_score=event.home_score,
            away_score=event.away_score,
            validation_issues=tuple(validation_issues),
        )

    async def validate_date(
        self,
        target_date: date | datetime,
        required_tiers: set[FetchTier] | None = None,
        check_scores: bool = True,
        check_discovery: bool = False,
    ) -> DailyValidationReport:
        """
        Validate tier coverage for all games on a specific date.

        Args:
            target_date: Date to validate (by game commence_time)
            required_tiers: Set of required tiers (defaults to all 5)
            check_scores: Whether to validate final scores exist for FINAL games
            check_discovery: Whether to check for missing games via API

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

        # Temporary collections for building the report
        game_reports_list = []
        complete_count = 0
        incomplete_count = 0
        games_by_tier_dict = {}
        missing_tier_dict = {}
        games_missing_scores = 0

        # Validate each game
        for game in games:
            try:
                game_report = await self.validate_game(game.id, required_tiers, check_scores)
                game_reports_list.append(game_report)

                # Update aggregate statistics
                if game_report.is_complete:
                    complete_count += 1
                else:
                    incomplete_count += 1

                # Track score issues
                if check_scores and "Missing final scores" in game_report.validation_issues:
                    games_missing_scores += 1

                # Track games by tier coverage count
                num_tiers = len(game_report.tiers_present)
                games_by_tier_dict[num_tiers] = games_by_tier_dict.get(num_tiers, 0) + 1

                # Track which tiers are missing across all games
                for missing_tier in game_report.tiers_missing:
                    missing_tier_dict[missing_tier] = missing_tier_dict.get(missing_tier, 0) + 1

            except Exception as e:
                logger.error(
                    "game_validation_failed",
                    event_id=game.id,
                    error=str(e),
                )
                continue

        # Check for missing games via API if requested
        missing_games_list = []
        if check_discovery:
            try:
                missing_games_list = await self._check_missing_games(target_date, games)
            except Exception as e:
                logger.error(
                    "game_discovery_check_failed",
                    target_date=str(target_date),
                    error=str(e),
                )

        # Build immutable report
        report = DailyValidationReport(
            target_date=target_date if isinstance(target_date, date) else target_date.date(),
            validation_time=datetime.now(UTC),
            total_games=len(games),
            complete_games=complete_count,
            incomplete_games=incomplete_count,
            games_by_tier_coverage=games_by_tier_dict,
            missing_tier_breakdown=missing_tier_dict,
            game_reports=tuple(game_reports_list),
            games_missing_scores=games_missing_scores,
            missing_games=tuple(missing_games_list),
        )

        logger.info(
            "daily_validation_completed",
            target_date=str(target_date),
            total_games=report.total_games,
            complete_games=report.complete_games,
            incomplete_games=report.incomplete_games,
        )

        return report

    async def _check_missing_games(
        self, target_date: date | datetime, db_games: list
    ) -> list[dict]:
        """
        Check for games from The Odds API that aren't in our database.

        Args:
            target_date: Date to check
            db_games: List of Event objects from database

        Returns:
            List of missing game dictionaries with id, home_team, away_team
        """
        from core.data_fetcher import TheOddsAPIClient

        # Get games from API for this date
        api_client = TheOddsAPIClient()

        # Calculate days_from parameter for API (0 = today, 1 = yesterday, etc.)
        from datetime import date as date_type

        if isinstance(target_date, datetime):
            target_date = target_date.date()

        today = date_type.today()
        days_from = (today - target_date).days

        # Only check for recent dates (API supports up to 3 days)
        if days_from < 0 or days_from > 3:
            logger.warning(
                "game_discovery_check_skipped",
                target_date=str(target_date),
                days_from=days_from,
                reason="Date outside API /scores endpoint range (0-3 days)",
            )
            return []

        try:
            # Fetch scores from API
            response = await api_client.get_scores(
                sport="basketball_nba",
                days_from=days_from,
            )

            api_games = response.scores_data

            # Build set of game IDs in our database
            db_game_ids = {game.id for game in db_games}

            # Find games in API response that aren't in our DB
            missing = []
            for api_game in api_games:
                if api_game.get("id") not in db_game_ids:
                    # Check if game was completed (has scores)
                    if api_game.get("completed", False):
                        missing.append(
                            {
                                "id": api_game.get("id"),
                                "home_team": api_game.get("home_team"),
                                "away_team": api_game.get("away_team"),
                                "commence_time": api_game.get("commence_time"),
                                "home_score": api_game.get("scores", [{}])[0].get("score")
                                if api_game.get("scores")
                                else None,
                                "away_score": api_game.get("scores", [{}])[1].get("score")
                                if len(api_game.get("scores", [])) > 1
                                else None,
                            }
                        )

            if missing:
                logger.warning(
                    "missing_games_detected",
                    target_date=str(target_date),
                    missing_count=len(missing),
                    missing_game_ids=[g["id"] for g in missing],
                )

            return missing

        except Exception as e:
            logger.error(
                "api_scores_fetch_failed",
                target_date=str(target_date),
                error=str(e),
            )
            raise

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
