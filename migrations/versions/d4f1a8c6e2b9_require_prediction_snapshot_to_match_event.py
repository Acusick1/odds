"""require a prediction's snapshot to belong to its event

Revision ID: d4f1a8c6e2b9
Revises: c3a9e51d7b20
Create Date: 2026-10-24 10:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

revision = "d4f1a8c6e2b9"
down_revision = "c3a9e51d7b20"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(sa.text("SET lock_timeout = '10s'"))
    op.execute(
        sa.text(
            "DELETE FROM predictions p USING odds_snapshots s"
            " WHERE s.id = p.snapshot_id AND p.event_id <> s.event_id"
        )
    )
    op.create_unique_constraint("uq_odds_snapshot_id_event", "odds_snapshots", ["id", "event_id"])
    op.drop_constraint("predictions_snapshot_id_fkey", "predictions", type_="foreignkey")
    op.create_foreign_key(
        "fk_prediction_snapshot_event",
        "predictions",
        "odds_snapshots",
        ["snapshot_id", "event_id"],
        ["id", "event_id"],
        deferrable=True,
        initially="DEFERRED",
    )


def downgrade() -> None:
    op.drop_constraint("fk_prediction_snapshot_event", "predictions", type_="foreignkey")
    op.create_foreign_key(
        "predictions_snapshot_id_fkey", "predictions", "odds_snapshots", ["snapshot_id"], ["id"]
    )
    op.drop_constraint("uq_odds_snapshot_id_event", "odds_snapshots", type_="unique")
