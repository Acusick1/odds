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


# Snapshots where the event row's commence_time is more than 2h away from
# the true match time implied by (snapshot_time + hours_until_commence).
#
# Intentionally sport-agnostic: the >2h divergence signature is valid across
# sports, and the correctness of the repoint does not depend on sport_key.
# The upgrade() function logs a per-sport breakdown so the blast radius is
# explicit when the migration runs.
_DIVERGENCE_QUERY = sa.text(
    """
    SELECT
        os.id AS snapshot_id,
        os.event_id AS old_event_id,
        os.snapshot_time,
        os.hours_until_commence,
        os.snapshot_time + make_interval(hours => os.hours_until_commence)
            AS true_commence,
        e.home_team,
        e.away_team,
        e.sport_key,
        e.sport_title
    FROM odds_snapshots os
    JOIN events e ON e.id = os.event_id
    WHERE os.hours_until_commence IS NOT NULL
      AND os.hours_until_commence > 0
      AND ABS(
          EXTRACT(
              EPOCH FROM (
                  e.commence_time
                  - (os.snapshot_time + make_interval(hours => os.hours_until_commence))
              )
          ) / 3600
      ) > 2
    ORDER BY os.snapshot_time
    """
)


# Per-sport breakdown of the divergence set, logged before repointing runs so
# the blast radius is explicit in migration output.
_DIVERGENCE_BREAKDOWN_QUERY = sa.text(
    """
    SELECT e.sport_key, COUNT(*) AS n
    FROM odds_snapshots os
    JOIN events e ON e.id = os.event_id
    WHERE os.hours_until_commence IS NOT NULL
      AND os.hours_until_commence > 0
      AND ABS(
          EXTRACT(
              EPOCH FROM (
                  e.commence_time
                  - (os.snapshot_time + make_interval(hours => os.hours_until_commence))
              )
          ) / 3600
      ) > 2
    GROUP BY e.sport_key
    ORDER BY n DESC
    """
)


# Sanity check for the uniqueness assumption embedded in _REPOINT_ODDS: that
# (event_id, odds_timestamp) uniquely identifies the odds rows belonging to
# one snapshot, so UPDATE needs no snapshot_id tiebreaker. This holds because
# each scrape run writes exactly one snapshot per event with a single
# snapshot_time value shared by all its odds rows. The query below flags any
# event where two or more snapshots share the same snapshot_time within the
# divergent set — that would indicate overlapping scrapes and mean the UPDATE
# could drag unrelated odds rows along with the one we intend to move.
_ODDS_TIMESTAMP_SANITY_QUERY = sa.text(
    """
    SELECT os.event_id, os.snapshot_time, COUNT(*) AS n
    FROM odds_snapshots os
    JOIN events e ON e.id = os.event_id
    WHERE os.hours_until_commence IS NOT NULL
      AND os.hours_until_commence > 0
      AND ABS(
          EXTRACT(
              EPOCH FROM (
                  e.commence_time
                  - (os.snapshot_time + make_interval(hours => os.hours_until_commence))
              )
          ) / 3600
      ) > 2
    GROUP BY os.event_id, os.snapshot_time
    HAVING COUNT(*) > 1
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


_EVENT_EXISTS_QUERY = sa.text("SELECT 1 FROM events WHERE id = :id")


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

    # Log the per-sport breakdown of the divergent set before touching
    # anything, so the migration output makes the blast radius explicit.
    breakdown = conn.execute(_DIVERGENCE_BREAKDOWN_QUERY).fetchall()
    if breakdown:
        summary = ", ".join(f"{r.sport_key}={r.n}" for r in breakdown)
        total = sum(r.n for r in breakdown)
        print(f"Repointing {total} snapshots: {summary}")
    else:
        print("Repointing 0 snapshots: no divergent rows found")

    # Sanity-check the uniqueness assumption behind _REPOINT_ODDS. If this
    # fires, review manually before applying — the UPDATE could drag odds
    # rows from an unrelated overlapping snapshot.
    dup_snapshots = conn.execute(_ODDS_TIMESTAMP_SANITY_QUERY).fetchall()
    if dup_snapshots:
        print(
            f"WARNING: {len(dup_snapshots)} (event_id, snapshot_time) pairs in the "
            "divergent set have multiple odds_snapshots rows — _REPOINT_ODDS may "
            "move unintended rows. Sample: "
            f"{[(r.event_id, r.snapshot_time.isoformat(), r.n) for r in dup_snapshots[:5]]}"
        )

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
        event_remap[row.old_event_id] = new_event_id
        repointed += 1

    # Re-point paper_trades and match_briefs once per (old, new) pair. These
    # tables are not keyed by snapshot, so the move follows the event as a
    # whole. Skip self-remaps defensively (already filtered above, but cheap).
    paper_trades_repointed = 0
    match_briefs_repointed = 0
    for old_id, new_id in event_remap.items():
        if old_id == new_id:
            continue
        pt_result = conn.execute(_REPOINT_PAPER_TRADES, {"new_id": new_id, "old_id": old_id})
        paper_trades_repointed += pt_result.rowcount or 0
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
