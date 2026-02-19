"""Database write operations for NBA injury report data."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import structlog
from odds_core.injury_models import InjuryReport
from odds_core.models import Event
from sqlalchemy import and_, or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.injury_fetcher import InjuryRecord
from odds_lambda.polymarket_matching import normalize_team

logger = structlog.get_logger(__name__)


class InjuryWriter:
    """Handles all write operations for injury report data."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def upsert_injury_reports(self, records: list[InjuryRecord]) -> int:
        """Insert or update injury report records with automatic event matching.

        Uses PostgreSQL ON CONFLICT DO UPDATE on the composite unique constraint
        for idempotent upserts. Event matching is performed per unique
        (team, game_date) pair and cached within the batch.

        Returns:
            Number of rows upserted.
        """
        if not records:
            return 0

        # Cache event_id lookups per (canonical_team, game_date) to avoid
        # redundant queries within a single batch
        event_cache: dict[tuple[str, date], str | None] = {}

        report_dicts: list[dict] = []
        for record in records:
            canonical_team = normalize_team(record.team)
            cache_key = (canonical_team or record.team, record.game_date)

            if cache_key not in event_cache:
                event_cache[cache_key] = await self._match_event(cache_key[0], record.game_date)

            report_dicts.append(
                {
                    "report_time": record.report_time,
                    "game_date": record.game_date,
                    "game_time_et": record.game_time_et,
                    "matchup": record.matchup,
                    "team": record.team,
                    "player_name": record.player_name,
                    "status": record.status,
                    "reason": record.reason,
                    "event_id": event_cache[cache_key],
                }
            )

        stmt = insert(InjuryReport).values(report_dicts)
        set_ = {
            col.name: stmt.excluded[col.name]
            for col in InjuryReport.__table__.columns
            if col.name not in ("id", "created_at")
        }
        stmt = stmt.on_conflict_do_update(
            constraint="uq_injury_report_time_team_player_date",
            set_=set_,
        )
        await self.session.execute(stmt)
        await self.session.flush()

        matched = sum(1 for v in event_cache.values() if v is not None)
        logger.info(
            "injury_reports_upserted",
            count=len(report_dicts),
            events_matched=matched,
            events_unmatched=len(event_cache) - matched,
        )
        return len(report_dicts)

    async def _match_event(self, team_name: str, game_date: date) -> str | None:
        """Match a team + game_date to a sportsbook Event record.

        Converts the ET calendar date to a UTC window covering that single
        day. NBA games start between ~noon ET and ~11 PM ET; the window
        spans game_date 10:00 ET to game_date+1 06:00 ET to handle early
        tipoffs and late-night UTC rollovers.

        Returns:
            event_id if exactly one match found, None otherwise.
        """
        from odds_core.time import EASTERN

        day_start_et = datetime(game_date.year, game_date.month, game_date.day, 10, tzinfo=EASTERN)
        day_end_et = day_start_et + timedelta(hours=20)  # game_date+1 06:00 ET
        window_start = day_start_et.astimezone(UTC)
        window_end = day_end_et.astimezone(UTC)

        query = select(Event.id).where(
            and_(
                Event.commence_time >= window_start,
                Event.commence_time <= window_end,
                or_(Event.home_team == team_name, Event.away_team == team_name),
            )
        )
        result = await self.session.execute(query)
        candidates = list(result.scalars().all())

        if len(candidates) == 1:
            return candidates[0]

        if len(candidates) > 1:
            logger.warning(
                "injury_event_ambiguous_match",
                team=team_name,
                game_date=str(game_date),
                candidates=len(candidates),
            )
        return None
