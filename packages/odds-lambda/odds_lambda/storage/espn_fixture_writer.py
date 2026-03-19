"""Database write operations for ESPN fixture data."""

from __future__ import annotations

from dataclasses import asdict

import structlog
from odds_core.epl_data_models import EspnFixture, EspnFixtureRecord
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger(__name__)


class EspnFixtureWriter:
    """Handles all write operations for ESPN fixture data."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def upsert_fixtures(self, records: list[EspnFixtureRecord]) -> int:
        """Insert or update ESPN fixture records.

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

        logger.info("espn_fixtures_upserted", count=len(records))
        return len(records)
