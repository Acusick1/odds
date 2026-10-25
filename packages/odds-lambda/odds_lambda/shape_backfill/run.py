"""The backfill's steps in the order they run."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.shape_backfill.events import (
    ExternalIdReport,
    OddsportalHistoryReport,
    TwinMerge,
    UnmatchedReport,
    attach_oddsportal_history,
    merge_twins,
    migrate_unmatched,
    oddsportal_history_games,
    record_stored_ids,
    stored_external_ids,
)
from odds_lambda.shape_backfill.legacy_polymarket import (
    LegacyPolymarketReport,
    migrate_legacy_polymarket,
)
from odds_lambda.shape_backfill.snapshots import (
    SnapshotBackfillReport,
    backfill_snapshots,
    capture_cutoff,
    preflight_snapshots,
)


@dataclass(frozen=True)
class BackfillRun:
    cutoff: datetime
    twins: list[TwinMerge]
    unmatched: UnmatchedReport
    external_ids: ExternalIdReport
    oddsportal_history: OddsportalHistoryReport | None
    snapshots: SnapshotBackfillReport
    legacy_polymarket: LegacyPolymarketReport


async def run_backfill(
    session: AsyncSession,
    *,
    apply: bool,
    batch_size: int,
    limit: int | None = None,
    oddsportal_dir: Path | None = None,
    map_out: Path | None = None,
) -> BackfillRun:
    """Merge twins, mark UNMATCHED events, record stored external ids (and, given
    ``oddsportal_dir``, the historical OddsPortal ids), fill the snapshots stored before the
    cutoff taken at the start, then migrate the legacy Polymarket prices. Without ``apply``
    every step counts what it would write and writes nothing.

    Twins merge first so the snapshot step sees one event per match; events take their ids
    after merging, so a merge never has to drop one.
    """
    cutoff = await capture_cutoff(session)
    twins = await merge_twins(session, apply=apply)
    unmatched = await migrate_unmatched(session, apply=apply)
    external_ids = await record_stored_ids(session, await stored_external_ids(session), apply=apply)
    history = None
    if oddsportal_dir is not None:
        history = await attach_oddsportal_history(
            session, oddsportal_history_games(oddsportal_dir), apply=apply
        )
    if apply:
        snapshots = await backfill_snapshots(
            session, cutoff=cutoff, batch_size=batch_size, limit=limit, map_out=map_out
        )
    else:
        snapshots = await preflight_snapshots(
            session, cutoff=cutoff, batch_size=batch_size, limit=limit
        )
    legacy = await migrate_legacy_polymarket(session, apply=apply)
    return BackfillRun(cutoff, twins, unmatched, external_ids, history, snapshots, legacy)
