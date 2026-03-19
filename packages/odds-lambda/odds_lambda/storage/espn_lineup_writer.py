"""Database write operations for ESPN lineup data."""

from __future__ import annotations

from dataclasses import asdict

import structlog
from odds_core.epl_data_models import EspnLineup, EspnLineupRecord
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger(__name__)


class EspnLineupWriter:
    """Handles all write operations for ESPN lineup data."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def upsert_lineups(self, records: list[EspnLineupRecord]) -> int:
        """Insert or update ESPN lineup records.

        Uses PostgreSQL ON CONFLICT DO UPDATE on the composite unique constraint
        (date, team, player_id) for idempotent upserts.

        Returns:
            Number of rows upserted.
        """
        if not records:
            return 0

        dicts = [asdict(r) for r in records]
        batch_size = 1000
        for i in range(0, len(dicts), batch_size):
            batch = dicts[i : i + batch_size]
            stmt = insert(EspnLineup).values(batch)
            set_ = {
                col.name: stmt.excluded[col.name]
                for col in EspnLineup.__table__.columns
                if col.name not in ("id", "created_at")
            }
            stmt = stmt.on_conflict_do_update(
                constraint="uq_espn_lineup_date_team_player",
                set_=set_,
            )
            await self.session.execute(stmt)

        await self.session.flush()

        logger.info("espn_lineups_upserted", count=len(records))
        return len(records)
