"""Database write operations for ESPN game data."""

from __future__ import annotations

from dataclasses import asdict

import structlog
from odds_core.epl_data_models import EspnGame, EspnGameRecord
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger(__name__)


class EspnGameWriter:
    """Handles all write operations for ESPN game data."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def upsert_games(self, records: list[EspnGameRecord]) -> int:
        """Insert or update ESPN game records.

        Uses PostgreSQL ON CONFLICT DO UPDATE on the composite unique constraint
        (date, team, competition) for idempotent upserts.

        Returns:
            Number of rows upserted.
        """
        if not records:
            return 0

        dicts = [asdict(r) for r in records]
        batch_size = 1000
        for i in range(0, len(dicts), batch_size):
            batch = dicts[i : i + batch_size]
            stmt = insert(EspnGame).values(batch)
            set_ = {
                col.name: stmt.excluded[col.name]
                for col in EspnGame.__table__.columns
                if col.name not in ("id", "created_at")
            }
            stmt = stmt.on_conflict_do_update(
                constraint="uq_espn_game_date_team_comp",
                set_=set_,
            )
            await self.session.execute(stmt)

        await self.session.flush()

        logger.info("espn_games_upserted", count=len(records))
        return len(records)
