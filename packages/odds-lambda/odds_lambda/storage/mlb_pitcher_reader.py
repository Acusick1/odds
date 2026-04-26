"""Database read operations for MLB probable-pitcher snapshots."""

from __future__ import annotations

from datetime import datetime

import structlog
from odds_core.mlb_data_models import MlbProbablePitchers
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger(__name__)


class MlbPitcherReader:
    """Reader for ``mlb_probable_pitchers`` snapshots."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_latest_in_window(
        self,
        start: datetime,
        end: datetime,
    ) -> list[MlbProbablePitchers]:
        """Return the latest snapshot per ``game_pk`` whose ``commence_time`` falls in ``[start, end]``.

        Uses ``DISTINCT ON (game_pk)`` against an ordering that picks the
        newest ``fetched_at`` per ``game_pk``, then re-sorts by
        ``commence_time`` ascending so the caller sees games in kickoff order.
        """
        if start > end:
            raise ValueError("start must be <= end")

        # DISTINCT ON requires the leading ORDER BY column to match.
        inner = (
            select(MlbProbablePitchers)
            .where(MlbProbablePitchers.commence_time >= start)
            .where(MlbProbablePitchers.commence_time <= end)
            .distinct(MlbProbablePitchers.game_pk)
            .order_by(
                MlbProbablePitchers.game_pk,
                MlbProbablePitchers.fetched_at.desc(),
            )
        )
        result = await self.session.execute(inner)
        rows = list(result.scalars().all())
        rows.sort(key=lambda r: r.commence_time)
        logger.info(
            "mlb_probable_pitchers_loaded",
            count=len(rows),
            start=start.isoformat(),
            end=end.isoformat(),
        )
        return rows
