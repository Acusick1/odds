"""cleanup oddsportal frankenstein ghost snapshots

Revision ID: 8a76c9607193
Revises: cc937c8cd362
Create Date: 2026-04-20 12:20:43.856131

Data migration: delete odds snapshots (and their child odds + predictions
rows) that were written to an ``op_live_*`` event more than 24h after the
event's ``commence_time``. That signature is unambiguous for the OddsPortal
H2H-hydration frankenstein pattern — live-market odds landing against an
event whose stale metadata points at a past meeting between the same teams.

See PR #338 for the containment fix that stops new writes of this shape.
This migration removes the pollution that accumulated before that fix
deployed. Snapshot-level rather than event-level deletion so mixed events
(legitimate pre-match snapshots plus a later frankenstein write) are
partially cleaned without losing real data; the event row is only
dropped when every snapshot on it was frankenstein.

Idempotent: re-running after the writer fix deploys is a no-op because
the predicate matches nothing once the adapter's stale-match filter is
live.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "8a76c9607193"
down_revision = "cc937c8cd362"
branch_labels = None
depends_on = None


# Snapshots counted as frankenstein: written more than 24h after the event's
# stated commence_time, on an ``op_live_*`` event. 24h is wide enough to leave
# any legitimate in-play scrape (extra-innings MLB tops out ~5h post-first-
# pitch) unaffected, while still catching pollution by days-to-months.
_POLLUTED_PREDICATE = """
    os.event_id LIKE 'op_live_%%'
    AND os.snapshot_time > e.commence_time + INTERVAL '24 hours'
"""


_BLAST_RADIUS_QUERY = sa.text(
    f"""
    SELECT e.sport_key, COUNT(*) AS n
    FROM odds_snapshots os
    JOIN events e ON e.id = os.event_id
    WHERE {_POLLUTED_PREDICATE}
    GROUP BY e.sport_key
    ORDER BY n DESC
    """
)


# Snapshot ids to delete, with their parent event and timestamp so child
# odds rows can be removed in the same pass.
_POLLUTED_SNAPSHOTS_QUERY = sa.text(
    f"""
    SELECT os.id AS snapshot_id, os.event_id, os.snapshot_time
    FROM odds_snapshots os
    JOIN events e ON e.id = os.event_id
    WHERE {_POLLUTED_PREDICATE}
    """
)


# After deleting polluted snapshots, drop any ``op_live_*`` event that has
# no remaining snapshots AND no paper_trades / match_briefs referencing it.
# Non-empty events are retained — they hold legitimate pre-match snapshots
# that survive the snapshot-level delete.
_EMPTY_GHOST_EVENTS_QUERY = sa.text(
    """
    SELECT e.id, e.sport_key
    FROM events e
    WHERE e.id LIKE 'op_live_%'
      AND NOT EXISTS (SELECT 1 FROM odds_snapshots os WHERE os.event_id = e.id)
      AND NOT EXISTS (SELECT 1 FROM paper_trades pt WHERE pt.event_id = e.id)
      AND NOT EXISTS (SELECT 1 FROM match_briefs mb WHERE mb.event_id = e.id)
    """
)


# Events emptied by the snapshot delete but retained because paper_trades /
# match_briefs still reference them. Logged for manual review — a trade or
# brief on an event whose snapshots were all frankenstein signals a real
# data-quality incident we want a human to triage.
_RETAINED_FK_EVENTS_QUERY = sa.text(
    """
    SELECT e.id, e.sport_key,
           (SELECT COUNT(*) FROM paper_trades pt WHERE pt.event_id = e.id) AS trades,
           (SELECT COUNT(*) FROM match_briefs mb WHERE mb.event_id = e.id) AS briefs
    FROM events e
    WHERE e.id LIKE 'op_live_%'
      AND NOT EXISTS (SELECT 1 FROM odds_snapshots os WHERE os.event_id = e.id)
      AND (
          EXISTS (SELECT 1 FROM paper_trades pt WHERE pt.event_id = e.id)
          OR EXISTS (SELECT 1 FROM match_briefs mb WHERE mb.event_id = e.id)
      )
    """
)


_DELETE_PREDICTIONS = sa.text("DELETE FROM predictions WHERE snapshot_id = :snapshot_id")
_DELETE_ODDS = sa.text(
    "DELETE FROM odds WHERE event_id = :event_id AND odds_timestamp = :snapshot_time"
)
_DELETE_SNAPSHOT = sa.text("DELETE FROM odds_snapshots WHERE id = :snapshot_id")
_DELETE_EVENT = sa.text("DELETE FROM events WHERE id = :event_id")


def upgrade() -> None:
    conn = op.get_bind()

    breakdown = conn.execute(_BLAST_RADIUS_QUERY).fetchall()
    total_polluted = sum(r.n for r in breakdown)
    if breakdown:
        summary = ", ".join(f"{r.sport_key}={r.n}" for r in breakdown)
        print(f"Deleting {total_polluted} polluted snapshots: {summary}")
    else:
        print("No polluted snapshots found — nothing to delete")
        return

    rows = conn.execute(_POLLUTED_SNAPSHOTS_QUERY).fetchall()

    snapshots_deleted = 0
    odds_rows_deleted = 0
    predictions_deleted = 0

    for row in rows:
        pred_result = conn.execute(_DELETE_PREDICTIONS, {"snapshot_id": row.snapshot_id})
        predictions_deleted += pred_result.rowcount or 0

        odds_result = conn.execute(
            _DELETE_ODDS,
            {"event_id": row.event_id, "snapshot_time": row.snapshot_time},
        )
        odds_rows_deleted += odds_result.rowcount or 0

        conn.execute(_DELETE_SNAPSHOT, {"snapshot_id": row.snapshot_id})
        snapshots_deleted += 1

    retained = conn.execute(_RETAINED_FK_EVENTS_QUERY).fetchall()
    if retained:
        sample = [(r.id, r.trades, r.briefs) for r in retained[:5]]
        print(
            f"Retaining {len(retained)} now-empty event rows with "
            f"paper_trades/match_briefs references — manual review needed. "
            f"Sample (id, trades, briefs): {sample}"
        )

    empty_events = conn.execute(_EMPTY_GHOST_EVENTS_QUERY).fetchall()
    events_deleted = 0
    events_by_sport: dict[str, int] = {}
    for row in empty_events:
        conn.execute(_DELETE_EVENT, {"event_id": row.id})
        events_deleted += 1
        events_by_sport[row.sport_key] = events_by_sport.get(row.sport_key, 0) + 1

    if events_by_sport:
        summary = ", ".join(f"{k}={v}" for k, v in sorted(events_by_sport.items()))
        print(f"Deleted {events_deleted} empty ghost event rows: {summary}")
    elif empty_events:
        # Defensive — should never hit this path because events_by_sport is
        # populated from the same loop that does the delete.
        print(f"Deleted {events_deleted} empty ghost event rows")
    else:
        print("No empty ghost event rows to delete")

    print(
        f"Totals: {snapshots_deleted} snapshots, {odds_rows_deleted} odds rows, "
        f"{predictions_deleted} predictions, {events_deleted} events deleted"
    )


def downgrade() -> None:
    raise NotImplementedError(
        "Data-only migration cannot be reversed — the deleted snapshots were "
        "pollution with no legitimate source data to restore."
    )
