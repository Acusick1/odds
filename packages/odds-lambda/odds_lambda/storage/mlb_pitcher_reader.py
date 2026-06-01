"""Database read operations for MLB probable-pitcher snapshots."""

from __future__ import annotations

from datetime import datetime

import structlog
from odds_core.mlb_data_models import (
    MlbProbablePitchers,
    MlbProbablePitchersRecord,
    select_latest_in_window,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger(__name__)


def _to_record(row: MlbProbablePitchers) -> MlbProbablePitchersRecord:
    """Map a persisted snapshot row to the in-memory domain record.

    Keeps the SQLModel persistence type from escaping the storage layer — the
    reader speaks in domain records, the same type the fetcher produces.
    """
    return MlbProbablePitchersRecord(
        game_pk=row.game_pk,
        commence_time=row.commence_time,
        fetched_at=row.fetched_at,
        home_team=row.home_team,
        away_team=row.away_team,
        game_type=row.game_type,
        home_pitcher_name=row.home_pitcher_name,
        home_pitcher_id=row.home_pitcher_id,
        away_pitcher_name=row.away_pitcher_name,
        away_pitcher_id=row.away_pitcher_id,
    )


class MlbPitcherReader:
    """Reader for ``mlb_probable_pitchers`` snapshots."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_latest_in_window(
        self,
        start: datetime,
        end: datetime,
    ) -> list[MlbProbablePitchersRecord]:
        """Return the latest snapshot per ``game_pk`` whose ``commence_time`` is in ``[start, end]``.

        SQL bounds the rows transferred to the commence-time window; the
        latest-per-``game_pk`` selection and ordering are delegated to
        :func:`select_latest_in_window` so the cached read path and the live
        read-through path apply one identical rule. Returns domain records,
        ordered by ``commence_time`` ascending.
        """
        if start > end:
            raise ValueError("start must be <= end")

        stmt = (
            select(MlbProbablePitchers)
            .where(MlbProbablePitchers.commence_time >= start)
            .where(MlbProbablePitchers.commence_time <= end)
        )
        result = await self.session.execute(stmt)
        records = [_to_record(row) for row in result.scalars().all()]
        selected = select_latest_in_window(records, start, end)
        logger.info(
            "mlb_probable_pitchers_loaded",
            count=len(selected),
            start=start.isoformat(),
            end=end.isoformat(),
        )
        return selected
