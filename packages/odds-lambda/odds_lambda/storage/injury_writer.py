"""Database write operations for NBA injury report data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import structlog
from odds_core.injury_models import InjuryReport
from odds_core.team import normalize_team_name
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.injury_fetcher import InjuryRecord
from odds_lambda.storage.nba_event_match import match_nba_event

logger = structlog.get_logger(__name__)


@dataclass(slots=True, frozen=True)
class UpsertResult:
    """Rows written and the ``(team, game_date)`` pairs no event matched."""

    rows: int
    unmatched: tuple[tuple[str, date], ...]


class InjuryWriter:
    """Handles all write operations for injury report data."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def upsert_injury_reports(self, records: list[InjuryRecord]) -> UpsertResult:
        """Insert or update injury report records with automatic event matching.

        Uses PostgreSQL ON CONFLICT DO UPDATE on the composite unique constraint
        for idempotent upserts. Event matching is performed per unique
        (team, game_date) pair and cached within the batch.

        Returns:
            Rows upserted and the canonical ``(team, game_date)`` pairs that matched no event.
        """
        if not records:
            return UpsertResult(0, ())

        # Cache event_id lookups per (canonical_team, game_date) to avoid
        # redundant queries within a single batch
        event_cache: dict[tuple[str, date], int | None] = {}

        report_dicts: list[dict] = []
        for record in records:
            cache_key = (normalize_team_name(record.team), record.game_date)

            if cache_key not in event_cache:
                event_cache[cache_key] = await match_nba_event(
                    self.session, cache_key[0], record.game_date, source="injury_report"
                )

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
        unmatched = tuple(pair for pair, event_id in event_cache.items() if event_id is None)
        return UpsertResult(len(report_dicts), unmatched)
