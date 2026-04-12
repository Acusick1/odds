"""add match_briefs table

Revision ID: f1a2b3c4d5e6
Revises: d8858af1ee85
Create Date: 2026-04-12 12:20:52.679712

"""

import sqlalchemy as sa
import sqlmodel
from alembic import op

# revision identifiers, used by Alembic.
revision = "f1a2b3c4d5e6"
down_revision = "d8858af1ee85"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "match_briefs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("event_id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column(
            "checkpoint",
            sa.Enum("CONTEXT", "DECISION", name="briefcheckpoint"),
            nullable=False,
        ),
        sa.Column("brief_text", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("sharp_price_at_brief", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["event_id"],
            ["events.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_match_briefs_event_id"), "match_briefs", ["event_id"], unique=False)
    op.create_index(
        "ix_match_briefs_event_checkpoint",
        "match_briefs",
        ["event_id", "checkpoint"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_match_briefs_event_checkpoint", table_name="match_briefs")
    op.drop_index(op.f("ix_match_briefs_event_id"), table_name="match_briefs")
    op.drop_table("match_briefs")
    op.execute("DROP TYPE IF EXISTS briefcheckpoint")
