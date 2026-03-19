"""Database write operations for FPL availability data."""

from __future__ import annotations

from dataclasses import asdict

import structlog
from odds_core.epl_data_models import FplAvailability, FplAvailabilityRecord
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger(__name__)


class FplAvailabilityWriter:
    """Handles all write operations for FPL availability data."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def upsert_availability(self, records: list[FplAvailabilityRecord]) -> int:
        """Insert or update FPL availability records.

        Uses PostgreSQL ON CONFLICT DO UPDATE on the composite unique constraint
        (snapshot_time, gameweek, player_code, season) for idempotent upserts.

        Returns:
            Number of rows upserted.
        """
        if not records:
            return 0

        dicts = [asdict(r) for r in records]
        batch_size = 1000
        for i in range(0, len(dicts), batch_size):
            batch = dicts[i : i + batch_size]
            stmt = insert(FplAvailability).values(batch)
            set_ = {
                col.name: stmt.excluded[col.name]
                for col in FplAvailability.__table__.columns
                if col.name not in ("id", "created_at")
            }
            stmt = stmt.on_conflict_do_update(
                constraint="uq_fpl_avail_snapshot_gw_player_season",
                set_=set_,
            )
            await self.session.execute(stmt)

        await self.session.flush()

        logger.info("fpl_availability_upserted", count=len(records))
        return len(records)
