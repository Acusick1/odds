"""merge duplicate OddsPortal events and settle stale events

Revision ID: d8858af1ee85
Revises: e7f2a1b3c4d5
Create Date: 2026-04-11 12:00:00.000000

One-time data migration:
1. For each op_live_* event that has a matching Odds API event (same teams,
   commence_time within +/-24h), reassign all child rows (odds_snapshots, odds,
   predictions, paper_trades) to the API event and delete the duplicate.
2. Bulk-settle past SCHEDULED/LIVE EPL events to FINAL.
3. Bulk-settle stale NBA API events to FINAL.
"""

import sqlalchemy as sa
from alembic import op

revision = "d8858af1ee85"
down_revision = "e7f2a1b3c4d5"
branch_labels = None
depends_on = None

# Child tables WITHOUT unique constraints involving event_id — simple UPDATE is safe.
_CHILD_TABLES_SIMPLE = [
    "odds_snapshots",
    "odds",
    "paper_trades",
    "data_quality_logs",
    "polymarket_events",
    "nba_team_game_logs",
    "nba_injury_reports",
]


def _reassign_predictions(conn: sa.Connection, old_id: str, new_id: str) -> None:
    """Reassign predictions from old event to new, handling unique constraint conflicts.

    predictions has UniqueConstraint(event_id, snapshot_id, model_name). If the API event
    already has a prediction for the same snapshot+model, the OP duplicate row is deleted
    (API data is authoritative). Non-conflicting rows are moved via UPDATE.
    """
    conn.execute(
        sa.text(
            """
            DELETE FROM predictions
            WHERE event_id = :old
              AND (snapshot_id, model_name) IN (
                  SELECT snapshot_id, model_name
                  FROM predictions
                  WHERE event_id = :new
              )
            """
        ).bindparams(old=old_id, new=new_id)
    )
    conn.execute(
        sa.text("UPDATE predictions SET event_id = :new WHERE event_id = :old").bindparams(
            new=new_id, old=old_id
        )
    )


def upgrade() -> None:
    conn = op.get_bind()

    # --- Step 1: merge op_live_* duplicates into their Odds API counterparts ---

    op_live_rows = conn.execute(
        sa.text(
            """
            SELECT id, home_team, away_team, commence_time, sport_key
            FROM events
            WHERE id LIKE 'op_live_%'
            """
        )
    ).fetchall()

    merged = 0
    for row in op_live_rows:
        op_id, home, away, commence, sport = row

        # Find matching Odds API event: same teams, within 24h, NOT an op_live_* id
        api_match = conn.execute(
            sa.text(
                """
                SELECT id FROM events
                WHERE id NOT LIKE 'op_live_%'
                  AND home_team = :home
                  AND away_team = :away
                  AND sport_key = :sport
                  AND ABS(EXTRACT(EPOCH FROM (commence_time - :commence))) <= 86400
                LIMIT 1
                """
            ).bindparams(home=home, away=away, sport=sport, commence=commence),
        ).fetchone()

        if api_match is None:
            continue

        api_id = api_match[0]

        # Reassign child rows from op_live event to the API event.
        # Tables without unique constraints on event_id: simple UPDATE.
        for table in _CHILD_TABLES_SIMPLE:
            conn.execute(
                sa.text(f"UPDATE {table} SET event_id = :new WHERE event_id = :old").bindparams(
                    new=api_id, old=op_id
                )
            )

        # Tables with unique constraints involving event_id: delete conflicts first.
        _reassign_predictions(conn, old_id=op_id, new_id=api_id)

        # Delete the duplicate op_live event
        conn.execute(sa.text("DELETE FROM events WHERE id = :id").bindparams(id=op_id))
        merged += 1

    print(f"Merged {merged} op_live_* duplicate events into Odds API events")

    # --- Step 2: settle stale past EPL events (SCHEDULED/LIVE -> FINAL) ---

    epl_settled = conn.execute(
        sa.text(
            """
            UPDATE events
            SET status = 'FINAL'::eventstatus, updated_at = NOW()
            WHERE sport_key = 'soccer_epl'
              AND status IN ('SCHEDULED'::eventstatus, 'LIVE'::eventstatus)
              AND commence_time < NOW()
            """
        )
    ).rowcount
    print(f"Settled {epl_settled} stale EPL events to FINAL")

    # --- Step 3: settle stale NBA API events ---

    nba_settled = conn.execute(
        sa.text(
            """
            UPDATE events
            SET status = 'FINAL'::eventstatus, updated_at = NOW()
            WHERE sport_key LIKE 'basketball_nba%'
              AND status IN ('SCHEDULED'::eventstatus, 'LIVE'::eventstatus)
              AND commence_time < NOW()
            """
        )
    ).rowcount
    print(f"Settled {nba_settled} stale NBA events to FINAL")


def downgrade() -> None:
    raise NotImplementedError(
        "Data-only migration cannot be reversed — merged snapshots and "
        "status changes require re-scraping to restore."
    )
