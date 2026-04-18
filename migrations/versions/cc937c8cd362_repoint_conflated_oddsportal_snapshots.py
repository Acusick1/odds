"""repoint conflated oddsportal snapshots

Revision ID: cc937c8cd362
Revises: 917adbe0576d
Create Date: 2026-04-18 14:32:05.233779

Data migration: back-to-back same-matchup games (e.g. MLB series) were silently
merged by the old ±24h team+date match window in find_or_create_event. The
second day's scrape wrote its odds onto the first day's event row. This
migration detects conflated snapshots (where event.commence_time diverges from
snapshot_time + hours_until_commence by more than 2h) and re-points them to
the correct event row, creating new rows where none exist.

Idempotent: re-running after the writer fix deploys is a no-op because new
snapshots will already sit on the correct event.
"""

from __future__ import annotations

from datetime import datetime

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "cc937c8cd362"
down_revision = "917adbe0576d"
branch_labels = None
depends_on = None


def _team_abbrev(name: str) -> str:
    """Python port of odds_core.team.team_abbrev.

    Inlined to keep the migration self-contained and stable against future
    changes to the team module.
    """
    words = name.split()
    if len(words) == 1:
        return words[0][:3].upper()
    return (words[0][:3] + words[-1][:3]).upper()


def _build_event_id(home_team: str, away_team: str, commence_time: datetime) -> str:
    """Python port of OddsWriter._build_event_id.

    Includes the match start time down to the minute so same-day
    doubleheaders produce distinct ids.
    """
    home_abbrev = _team_abbrev(home_team)
    away_abbrev = _team_abbrev(away_team)
    date_str = commence_time.strftime("%Y-%m-%dT%H%M")
    return f"op_live_{home_abbrev}_{away_abbrev}_{date_str}"


# Divergence predicate shared by every query below: a snapshot whose parent
# event's commence_time is >2h away from the true match time implied by
# (snapshot_time + hours_until_commence). Kept as a string so it can be
# composed into each query without duplicating the condition.
_DIVERGENCE_PREDICATE = """
    os.hours_until_commence IS NOT NULL
    AND os.hours_until_commence > 0
    AND ABS(
        EXTRACT(
            EPOCH FROM (
                e.commence_time
                - (os.snapshot_time + (os.hours_until_commence || ' hours')::interval)
            )
        ) / 3600
    ) > 2
"""


# (event_id, snapshot_time) pairs inside the divergent set that are backed by
# more than one odds_snapshots row. These cannot be repointed safely because
# _REPOINT_ODDS keys on (old_event_id, odds_timestamp) and cannot tell which
# odds rows belong to which snapshot when multiple snapshots share the same
# timestamp. The upgrade() function logs these for manual review and excludes
# them from the divergent set it actually processes.
_OFFENDING_PAIRS_QUERY = sa.text(
    f"""
    SELECT os.event_id, os.snapshot_time, COUNT(*) AS n
    FROM odds_snapshots os
    JOIN events e ON e.id = os.event_id
    WHERE {_DIVERGENCE_PREDICATE}
    GROUP BY os.event_id, os.snapshot_time
    HAVING COUNT(*) > 1
    """
)


# Snapshots where the event row's commence_time is more than 2h away from
# the true match time implied by (snapshot_time + hours_until_commence).
# Excludes rows whose (event_id, snapshot_time) appears more than once in the
# divergent set — see _OFFENDING_PAIRS_QUERY for why.
#
# Intentionally sport-agnostic: the >2h divergence signature is valid across
# sports, and the correctness of the repoint does not depend on sport_key.
# The upgrade() function logs a per-sport breakdown so the blast radius is
# explicit when the migration runs.
_DIVERGENCE_QUERY = sa.text(
    f"""
    WITH offending_pairs AS (
        SELECT os.event_id, os.snapshot_time
        FROM odds_snapshots os
        JOIN events e ON e.id = os.event_id
        WHERE {_DIVERGENCE_PREDICATE}
        GROUP BY os.event_id, os.snapshot_time
        HAVING COUNT(*) > 1
    )
    SELECT
        os.id AS snapshot_id,
        os.event_id AS old_event_id,
        os.snapshot_time,
        os.hours_until_commence,
        os.snapshot_time + (os.hours_until_commence || ' hours')::interval
            AS true_commence,
        e.home_team,
        e.away_team,
        e.sport_key,
        e.sport_title
    FROM odds_snapshots os
    JOIN events e ON e.id = os.event_id
    WHERE {_DIVERGENCE_PREDICATE}
      AND NOT EXISTS (
          SELECT 1 FROM offending_pairs op
          WHERE op.event_id = os.event_id
            AND op.snapshot_time = os.snapshot_time
      )
    ORDER BY os.snapshot_time
    """
)


# Per-sport breakdown of the divergence set (excluding offending pairs),
# logged before repointing runs so the blast radius is explicit in migration
# output and matches the set actually processed.
_DIVERGENCE_BREAKDOWN_QUERY = sa.text(
    f"""
    WITH offending_pairs AS (
        SELECT os.event_id, os.snapshot_time
        FROM odds_snapshots os
        JOIN events e ON e.id = os.event_id
        WHERE {_DIVERGENCE_PREDICATE}
        GROUP BY os.event_id, os.snapshot_time
        HAVING COUNT(*) > 1
    )
    SELECT e.sport_key, COUNT(*) AS n
    FROM odds_snapshots os
    JOIN events e ON e.id = os.event_id
    WHERE {_DIVERGENCE_PREDICATE}
      AND NOT EXISTS (
          SELECT 1 FROM offending_pairs op
          WHERE op.event_id = os.event_id
            AND op.snapshot_time = os.snapshot_time
      )
    GROUP BY e.sport_key
    ORDER BY n DESC
    """
)


# Find an existing event within ±2h of true_commence for the same matchup.
_FIND_EVENT_QUERY = sa.text(
    """
    SELECT id
    FROM events
    WHERE home_team = :home_team
      AND away_team = :away_team
      AND sport_key = :sport_key
      AND ABS(EXTRACT(EPOCH FROM (commence_time - :true_commence))) <= 7200
    ORDER BY ABS(EXTRACT(EPOCH FROM (commence_time - :true_commence)))
    LIMIT 1
    """
)


_EVENT_EXISTS_QUERY = sa.text(
    """
    SELECT home_team, away_team, sport_key, commence_time
    FROM events
    WHERE id = :id
    """
)


# Pre-flight counts of paper_trades / match_briefs rows that will be
# repointed, so the migration logs the blast radius before issuing UPDATEs.
_PAPER_TRADES_COUNT_QUERY = sa.text(
    "SELECT COUNT(*) FROM paper_trades WHERE event_id = ANY(:old_ids)"
)
_PAPER_TRADES_SAMPLE_QUERY = sa.text(
    "SELECT id, event_id FROM paper_trades WHERE event_id = ANY(:old_ids) LIMIT 5"
)
_MATCH_BRIEFS_COUNT_QUERY = sa.text(
    "SELECT COUNT(*) FROM match_briefs WHERE event_id = ANY(:old_ids)"
)
_MATCH_BRIEFS_SAMPLE_QUERY = sa.text(
    "SELECT id, event_id FROM match_briefs WHERE event_id = ANY(:old_ids) LIMIT 5"
)


_INSERT_EVENT = sa.text(
    """
    INSERT INTO events (
        id, sport_key, sport_title, commence_time, home_team, away_team,
        status, created_at, updated_at
    )
    VALUES (
        :id, :sport_key, :sport_title, :commence_time, :home_team, :away_team,
        'SCHEDULED'::eventstatus, NOW(), NOW()
    )
    """
)


_REPOINT_SNAPSHOT = sa.text("UPDATE odds_snapshots SET event_id = :new_id WHERE id = :snapshot_id")


_REPOINT_ODDS = sa.text(
    """
    UPDATE odds
    SET event_id = :new_id
    WHERE event_id = :old_id
      AND odds_timestamp = :snapshot_time
    """
)


# Predictions are keyed by (event_id, snapshot_id) and inherit the snapshot's
# event. Keep them consistent with the snapshot's new event_id.
_REPOINT_PREDICTIONS = sa.text(
    "UPDATE predictions SET event_id = :new_id WHERE snapshot_id = :snapshot_id"
)


# paper_trades and match_briefs both have plain (non-unique) event_id FK
# columns with no unique constraint involving event_id — simple UPDATE is
# safe. Re-pointed per (old_event_id -> new_event_id) pair so trades and
# briefs follow their event's snapshots to the corrected row.
_REPOINT_PAPER_TRADES = sa.text(
    "UPDATE paper_trades SET event_id = :new_id WHERE event_id = :old_id"
)
_REPOINT_MATCH_BRIEFS = sa.text(
    "UPDATE match_briefs SET event_id = :new_id WHERE event_id = :old_id"
)


def upgrade() -> None:
    conn = op.get_bind()

    # (event_id, snapshot_time) pairs with multiple divergent odds_snapshots
    # rows cannot be repointed safely — _REPOINT_ODDS keys on
    # (old_event_id, odds_timestamp) and has no way to attribute odds to the
    # correct snapshot when two share a timestamp. Log these for later manual
    # review and continue with the safe remainder.
    offending = conn.execute(_OFFENDING_PAIRS_QUERY).fetchall()
    # Set of old event ids that contain at least one offending pair. Their
    # safe sibling snapshots still get repointed at the snapshot level, but
    # paper_trades / match_briefs on those events must NOT be moved: the
    # event's game-level attribution is ambiguous while unhealed pairs remain.
    offending_event_ids: set[str] = {r.event_id for r in offending}
    if offending:
        sample = [(r.event_id, r.snapshot_time.isoformat()) for r in offending[:5]]
        print(
            f"Skipping {len(offending)} (event_id, snapshot_time) pairs with "
            "multiple divergent snapshots — these need manual review. "
            f"Sample: {sample}."
        )

    # Log the per-sport breakdown of the divergent set that WILL be processed
    # (offending pairs excluded), so the migration output makes the blast
    # radius explicit and matches the rows actually moved.
    breakdown = conn.execute(_DIVERGENCE_BREAKDOWN_QUERY).fetchall()
    if breakdown:
        summary = ", ".join(f"{r.sport_key}={r.n}" for r in breakdown)
        total = sum(r.n for r in breakdown)
        print(f"Repointing {total} snapshots: {summary}")
    elif offending:
        # The filtered set is empty ONLY because every divergent row was in
        # an offending pair. Make that explicit rather than implying the DB
        # is clean.
        print(
            f"Repointing 0 snapshots: all {len(offending)} divergent pairs "
            "were skipped as offending — manual review needed"
        )
    else:
        print("Repointing 0 snapshots: no safely-repointable divergent rows found")

    rows = conn.execute(_DIVERGENCE_QUERY).fetchall()

    repointed = 0
    events_created = 0
    odds_rows_repointed = 0
    # Map of old_event_id -> new_event_id; populated as we re-point snapshots
    # so we can propagate the same re-point to paper_trades and match_briefs
    # exactly once per (old, new) pair after the loop.
    event_remap: dict[str, str] = {}

    for row in rows:
        true_commence = row.true_commence

        # Look for an existing event row with matching teams within ±2h.
        existing = conn.execute(
            _FIND_EVENT_QUERY,
            {
                "home_team": row.home_team,
                "away_team": row.away_team,
                "sport_key": row.sport_key,
                "true_commence": true_commence,
            },
        ).fetchone()

        if existing is not None:
            new_event_id = existing[0]
        else:
            # No row for the true match — mint one using the op_live_ scheme.
            new_event_id = _build_event_id(row.home_team, row.away_team, true_commence)
            already_there = conn.execute(_EVENT_EXISTS_QUERY, {"id": new_event_id}).fetchone()
            if already_there is None:
                conn.execute(
                    _INSERT_EVENT,
                    {
                        "id": new_event_id,
                        "sport_key": row.sport_key,
                        "sport_title": row.sport_title,
                        "commence_time": true_commence,
                        "home_team": row.home_team,
                        "away_team": row.away_team,
                    },
                )
                events_created += 1
            else:
                # The minted id already exists but wasn't picked up by
                # _FIND_EVENT_QUERY's ±2h window. Verify it actually
                # describes the same match before reusing — a collision on
                # a different matchup would silently corrupt data.
                existing_commence = already_there.commence_time
                delta_seconds = abs((existing_commence - true_commence).total_seconds())
                if (
                    already_there.home_team != row.home_team
                    or already_there.away_team != row.away_team
                    or already_there.sport_key != row.sport_key
                    or delta_seconds > 7200
                ):
                    raise RuntimeError(
                        f"Minted event id {new_event_id!r} already exists but "
                        f"does not match expected values. Expected "
                        f"(home={row.home_team!r}, away={row.away_team!r}, "
                        f"sport={row.sport_key!r}, commence={true_commence.isoformat()}); "
                        f"found (home={already_there.home_team!r}, "
                        f"away={already_there.away_team!r}, "
                        f"sport={already_there.sport_key!r}, "
                        f"commence={existing_commence.isoformat()})."
                    )

        if new_event_id == row.old_event_id:
            # Defensive: should not happen — divergence query guarantees the
            # old event's commence_time differs by >2h from true_commence.
            continue

        # Re-point the snapshot and its child odds rows (matched by timestamp).
        conn.execute(
            _REPOINT_SNAPSHOT,
            {"new_id": new_event_id, "snapshot_id": row.snapshot_id},
        )
        odds_result = conn.execute(
            _REPOINT_ODDS,
            {
                "new_id": new_event_id,
                "old_id": row.old_event_id,
                "snapshot_time": row.snapshot_time,
            },
        )
        odds_rows_repointed += odds_result.rowcount or 0
        conn.execute(
            _REPOINT_PREDICTIONS,
            {"new_id": new_event_id, "snapshot_id": row.snapshot_id},
        )
        # Only queue paper_trades/match_briefs repoint if the old event has
        # NO offending pairs. When an event hosts both safe and unsafe
        # divergent snapshots, the safe snapshot still moves but aggregate
        # (event-FK) data stays put — we cannot tell which trades/briefs
        # belong to the unhealed pair's game vs the safe one's.
        if row.old_event_id not in offending_event_ids:
            event_remap[row.old_event_id] = new_event_id
        repointed += 1

    # Log how many events were held back from the paper_trades/match_briefs
    # repoint because they still host offending pairs.
    skipped_events = offending_event_ids & {r.old_event_id for r in rows}
    if skipped_events:
        sample = list(skipped_events)[:5]
        print(
            f"Skipping paper_trades/match_briefs repoint for {len(skipped_events)} "
            f"events that also contain offending (event, snapshot_time) pairs: {sample}"
        )

    # Re-point paper_trades and match_briefs once per (old, new) pair. These
    # tables are not keyed by snapshot, so the move follows the event as a
    # whole. Skip self-remaps defensively (already filtered above, but cheap).
    paper_trades_repointed = 0
    match_briefs_repointed = 0
    old_ids = [old_id for old_id, new_id in event_remap.items() if old_id != new_id]

    # Pre-flight: log blast radius per table before issuing UPDATEs so
    # Alembic output makes the change auditable.
    pt_total = 0
    if old_ids:
        pt_total = conn.execute(_PAPER_TRADES_COUNT_QUERY, {"old_ids": old_ids}).scalar() or 0
    if pt_total == 0:
        print("paper_trades: 0 rows to repoint")
    else:
        pt_sample = conn.execute(_PAPER_TRADES_SAMPLE_QUERY, {"old_ids": old_ids}).fetchall()
        sample_pairs = [(r.id, r.event_id) for r in pt_sample]
        print(f"paper_trades: {pt_total} rows to repoint. Sample: {sample_pairs}")

    mb_total = 0
    if old_ids:
        mb_total = conn.execute(_MATCH_BRIEFS_COUNT_QUERY, {"old_ids": old_ids}).scalar() or 0
    if mb_total == 0:
        print("match_briefs: 0 rows to repoint")
    else:
        mb_sample = conn.execute(_MATCH_BRIEFS_SAMPLE_QUERY, {"old_ids": old_ids}).fetchall()
        sample_pairs = [(r.id, r.event_id) for r in mb_sample]
        print(f"match_briefs: {mb_total} rows to repoint. Sample: {sample_pairs}")

    for old_id, new_id in event_remap.items():
        if old_id == new_id:
            continue
        if pt_total > 0:
            pt_result = conn.execute(_REPOINT_PAPER_TRADES, {"new_id": new_id, "old_id": old_id})
            paper_trades_repointed += pt_result.rowcount or 0
        if mb_total > 0:
            mb_result = conn.execute(_REPOINT_MATCH_BRIEFS, {"new_id": new_id, "old_id": old_id})
            match_briefs_repointed += mb_result.rowcount or 0

    print(
        f"Repointed {repointed} conflated snapshots "
        f"({odds_rows_repointed} odds rows), created {events_created} event rows, "
        f"repointed {paper_trades_repointed} paper_trades and "
        f"{match_briefs_repointed} match_briefs rows"
    )


def downgrade() -> None:
    raise NotImplementedError(
        "Data-only migration cannot be reversed — the prior assignment of "
        "snapshots to the wrong event row was the bug being fixed."
    )
