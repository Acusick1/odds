"""drop the predictions table and the snapshot key only it referenced

Revision ID: e7b2c4d9a1f8
Revises: e8b1c4d7a2f6
Create Date: 2026-10-24 14:00:00.000000

"""

import sqlalchemy as sa
import sqlmodel
from alembic import op

revision = "e7b2c4d9a1f8"
down_revision = "e8b1c4d7a2f6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(sa.text("SET lock_timeout = '10s'"))
    op.drop_table("predictions")
    op.drop_constraint("uq_odds_snapshot_id_event", "odds_snapshots", type_="unique")


def downgrade() -> None:
    op.create_unique_constraint("uq_odds_snapshot_id_event", "odds_snapshots", ["id", "event_id"])
    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("event_id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("snapshot_id", sa.Integer(), nullable=False),
        sa.Column("model_name", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("model_version", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("predicted_clv", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["event_id"], ["events.id"]),
        sa.ForeignKeyConstraint(
            ["snapshot_id", "event_id"],
            ["odds_snapshots.id", "odds_snapshots.event_id"],
            name="fk_prediction_snapshot_event",
            deferrable=True,
            initially="DEFERRED",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "event_id", "snapshot_id", "model_name", name="uq_prediction_event_snap_model"
        ),
    )
    op.create_index(
        "ix_prediction_model_created", "predictions", ["model_name", "created_at"], unique=False
    )
    op.create_index(op.f("ix_predictions_event_id"), "predictions", ["event_id"], unique=False)
    op.create_index(op.f("ix_predictions_model_name"), "predictions", ["model_name"], unique=False)
    op.create_index(
        op.f("ix_predictions_snapshot_id"), "predictions", ["snapshot_id"], unique=False
    )
