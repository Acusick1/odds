"""link each odds row to its snapshot

Revision ID: b6e3d9f2a4c7
Revises: d3f8a6b1c5e9
Create Date: 2026-10-25 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

revision = "b6e3d9f2a4c7"
down_revision = "d3f8a6b1c5e9"
branch_labels = None
depends_on = None

# A row links to the one snapshot at its event and time whose raw_data quotes its bookmaker,
# market, outcome and point. A row no snapshot quotes, or several do, stays NULL (#535). The
# bookmaker key is trimmed of the ASCII whitespace Python's str.strip removes, as the writer
# stored it. Only snapshots at a time that has odds rows are expanded.
LINK_BY_KEY = r"""
UPDATE odds o SET snapshot_id = k.snapshot_id
FROM (
    SELECT s.event_id, s.snapshot_time,
           btrim(b->>'key', E' \t\n\r\x0b\x0c\x1c\x1d\x1e\x1f') AS bookmaker_key,
           m->>'key' AS market_key, x->>'name' AS outcome_name,
           (x->>'point')::float AS point, min(s.id) AS snapshot_id
    FROM odds_snapshots s
    CROSS JOIN LATERAL json_array_elements(s.raw_data->'bookmakers') b
    CROSS JOIN LATERAL json_array_elements(b->'markets') m
    CROSS JOIN LATERAL json_array_elements(m->'outcomes') x
    WHERE EXISTS (
        SELECT 1 FROM odds r
        WHERE r.event_id = s.event_id AND r.odds_timestamp = s.snapshot_time
    )
    GROUP BY 1, 2, 3, 4, 5, 6
    HAVING count(DISTINCT s.id) = 1
) k
WHERE o.event_id = k.event_id AND o.odds_timestamp = k.snapshot_time
  AND o.bookmaker_key = k.bookmaker_key AND o.market_key = k.market_key
  AND o.outcome_name = k.outcome_name AND o.point IS NOT DISTINCT FROM k.point
"""


def upgrade() -> None:
    op.execute(sa.text("SET lock_timeout = '10s'"))
    op.add_column("odds", sa.Column("snapshot_id", sa.Integer(), nullable=True))
    op.execute(sa.text(LINK_BY_KEY))
    # Index and key after the backfill, so the UPDATE does not maintain them row by row.
    op.create_index("ix_odds_snapshot_id", "odds", ["snapshot_id"])
    op.create_foreign_key(
        "odds_snapshot_id_fkey", "odds", "odds_snapshots", ["snapshot_id"], ["id"]
    )


def downgrade() -> None:
    op.drop_constraint("odds_snapshot_id_fkey", "odds", type_="foreignkey")
    op.drop_index("ix_odds_snapshot_id", table_name="odds")
    op.drop_column("odds", "snapshot_id")
