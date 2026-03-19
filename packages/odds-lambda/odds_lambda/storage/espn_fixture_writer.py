"""Database write operations for ESPN fixture data."""

from __future__ import annotations

from typing import Any

import structlog
from odds_core.epl_data_models import EspnFixture
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger(__name__)


class EspnFixtureWriter:
    """Handles all write operations for ESPN fixture data."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def upsert_fixtures(self, fixture_dicts: list[dict[str, Any]]) -> int:
        """Insert or update ESPN fixture records.

        Uses PostgreSQL ON CONFLICT DO UPDATE on the composite unique constraint
        (date, team, competition) for idempotent upserts.

        Args:
            fixture_dicts: List of dicts with keys matching EspnFixture columns.

        Returns:
            Number of rows upserted.
        """
        if not fixture_dicts:
            return 0

        batch_size = 1000
        for i in range(0, len(fixture_dicts), batch_size):
            batch = fixture_dicts[i : i + batch_size]
            stmt = insert(EspnFixture).values(batch)
            set_ = {
                col.name: stmt.excluded[col.name]
                for col in EspnFixture.__table__.columns
                if col.name not in ("id", "created_at")
            }
            stmt = stmt.on_conflict_do_update(
                constraint="uq_espn_fixture_date_team_comp",
                set_=set_,
            )
            await self.session.execute(stmt)

        await self.session.flush()

        logger.info("espn_fixtures_upserted", count=len(fixture_dicts))
        return len(fixture_dicts)
