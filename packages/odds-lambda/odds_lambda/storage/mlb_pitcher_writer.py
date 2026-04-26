"""Database write operations for MLB probable-pitcher snapshots."""

from __future__ import annotations

from dataclasses import asdict

import structlog
from odds_core.mlb_data_models import MlbProbablePitchers, MlbProbablePitchersRecord
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger(__name__)


class MlbPitcherWriter:
    """Append-only writer for ``mlb_probable_pitchers`` snapshots.

    Idempotent on the ``(game_pk, fetched_at)`` unique constraint via
    ``ON CONFLICT DO NOTHING`` — re-running a fetch with the same
    ``fetched_at`` is a no-op, while a fresh ``fetched_at`` always inserts a
    new snapshot row.
    """

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def insert_snapshots(self, records: list[MlbProbablePitchersRecord]) -> int:
        """Insert probable-pitcher snapshot records.

        Returns the number of input records (not the number of newly inserted
        rows; ``ON CONFLICT DO NOTHING`` swallows duplicates silently).
        """
        if not records:
            return 0

        dicts = [asdict(r) for r in records]
        batch_size = 1000
        for i in range(0, len(dicts), batch_size):
            batch = dicts[i : i + batch_size]
            stmt = insert(MlbProbablePitchers).values(batch)
            stmt = stmt.on_conflict_do_nothing(
                constraint="uq_mlb_probable_pitchers_game_fetched",
            )
            await self.session.execute(stmt)

        await self.session.flush()

        logger.info("mlb_probable_pitchers_inserted", count=len(records))
        return len(records)
