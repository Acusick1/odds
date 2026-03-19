"""Database write operations for ESPN lineup data."""

from __future__ import annotations

from typing import Any

import structlog
from odds_core.epl_data_models import EspnLineup
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger(__name__)


class EspnLineupWriter:
    """Handles all write operations for ESPN lineup data."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def upsert_lineups(self, lineup_dicts: list[dict[str, Any]]) -> int:
        """Insert or update ESPN lineup records.

        Uses PostgreSQL ON CONFLICT DO UPDATE on the composite unique constraint
        (date, team, player_id) for idempotent upserts.

        Args:
            lineup_dicts: List of dicts with keys matching EspnLineup columns.

        Returns:
            Number of rows upserted.
        """
        if not lineup_dicts:
            return 0

        batch_size = 1000
        for i in range(0, len(lineup_dicts), batch_size):
            batch = lineup_dicts[i : i + batch_size]
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

        logger.info("espn_lineups_upserted", count=len(lineup_dicts))
        return len(lineup_dicts)
