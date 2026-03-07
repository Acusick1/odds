"""add prediction table

Revision ID: 019ba3d7f14b
Revises: 49b9789986fa
Create Date: 2026-03-07 08:03:34.866911

"""

import sqlalchemy as sa
import sqlmodel
from alembic import op

# revision identifiers, used by Alembic.
revision = "019ba3d7f14b"
down_revision = "49b9789986fa"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("event_id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("snapshot_id", sa.Integer(), nullable=False),
        sa.Column("model_name", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("model_version", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("predicted_clv", sa.Float(), nullable=False),
        sa.Column("scored_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["event_id"],
            ["events.id"],
        ),
        sa.ForeignKeyConstraint(
            ["snapshot_id"],
            ["odds_snapshots.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "event_id", "snapshot_id", "model_name", name="uq_prediction_event_snap_model"
        ),
    )
    op.create_index(
        "ix_prediction_model_scored", "predictions", ["model_name", "scored_at"], unique=False
    )
    op.create_index(op.f("ix_predictions_event_id"), "predictions", ["event_id"], unique=False)
    op.create_index(op.f("ix_predictions_model_name"), "predictions", ["model_name"], unique=False)
    op.create_index(
        op.f("ix_predictions_snapshot_id"), "predictions", ["snapshot_id"], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_predictions_snapshot_id"), table_name="predictions")
    op.drop_index(op.f("ix_predictions_model_name"), table_name="predictions")
    op.drop_index(op.f("ix_predictions_event_id"), table_name="predictions")
    op.drop_index("ix_prediction_model_scored", table_name="predictions")
    op.drop_table("predictions")
