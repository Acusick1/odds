"""The backfill's steps in the order they run."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.shape_backfill.events import (
    ExternalIdReport,
    MergeDropsRefused,
    OddsportalHistoryReport,
    TwinMerge,
    UnmatchedReport,
    attach_oddsportal_history,
    merge_twins,
    migrate_unmatched,
    oddsportal_history_games,
    plan_twin_merges,
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
    check_references,
    scan_snapshots,
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
    allow_merge_drops: bool = False,
) -> BackfillRun:
    """Merge twins, mark UNMATCHED events, record stored external ids (and, given
    ``oddsportal_dir``, the historical OddsPortal ids), fill the snapshots stored before the
    cutoff taken at the start, then migrate the legacy Polymarket prices. Without ``apply``
    every step counts what it would write and writes nothing.

    Before anything is written, the twin merges and the snapshot step are planned read-only,
    and the run stops when a candidate decision recorded a snapshot the backfill would split or
    delete, or a merge would delete candidate decisions or paper trades (unless
    ``allow_merge_drops``). Twins merge first so the snapshot step sees one event per match; a
    merge moves snapshots between events, so the snapshot step is planned again after one.

    Raises:
        SplitSnapshotReferenced, MergeDropsRefused: before any write.
    """
    cutoff = await capture_cutoff(session)
    twins = await plan_twin_merges(session)
    scan = await scan_snapshots(session, cutoff=cutoff, batch_size=batch_size)
    if apply:
        await check_references(session, scan)
        if not allow_merge_drops and any(t.drops_decisions for t in twins):
            raise MergeDropsRefused(
                "twin merges would delete candidate decisions or paper trades; rerun with "
                "--allow-merge-drops to accept: "
                + "; ".join(f"{t.kept_id}: {t.rows_deleted}" for t in twins if t.drops_decisions)
            )
        twins = await merge_twins(session, twins, allow_drops=allow_merge_drops)
        if any(t.error is None for t in twins):
            scan = await scan_snapshots(session, cutoff=cutoff, batch_size=batch_size)
    unmatched = await migrate_unmatched(session, apply=apply)
    external_ids = await record_stored_ids(session, await stored_external_ids(session), apply=apply)
    history = None
    if oddsportal_dir is not None:
        history = await attach_oddsportal_history(
            session, oddsportal_history_games(oddsportal_dir), apply=apply
        )
    if apply:
        snapshots = await backfill_snapshots(
            session, scan, batch_size=batch_size, limit=limit, map_out=map_out
        )
    else:
        snapshots = scan.report
    legacy = await migrate_legacy_polymarket(session, apply=apply)
    return BackfillRun(cutoff, twins, unmatched, external_ids, history, snapshots, legacy)
