"""add agent_wakeups table

Revision ID: a7b3c9d2e1f0
Revises: 920aee156d9c
Create Date: 2026-04-15 18:00:00.000000

"""

import sqlalchemy as sa
import sqlmodel
from alembic import op

# revision identifiers, used by Alembic.
revision = "a7b3c9d2e1f0"
down_revision = "920aee156d9c"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_wakeups",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("sport_key", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("requested_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("reason", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("sport_key", name="uq_agent_wakeup_sport_key"),
    )
    op.create_index(
        op.f("ix_agent_wakeups_sport_key"),
        "agent_wakeups",
        ["sport_key"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_agent_wakeups_sport_key"), table_name="agent_wakeups")
    op.drop_table("agent_wakeups")
