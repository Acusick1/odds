"""add api_football_lineups table

Revision ID: b44debb4b677
Revises: 771d5b1b451c
Create Date: 2026-04-12 23:00:00.000000

"""

import sqlalchemy as sa
import sqlmodel
from alembic import op

# revision identifiers, used by Alembic.
revision = "b44debb4b677"
down_revision = "771d5b1b451c"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "api_football_lineups",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("event_id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("fixture_id", sa.Integer(), nullable=False),
        sa.Column("team_name", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("team_id", sa.Integer(), nullable=False),
        sa.Column("formation", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("coach", sa.JSON(), nullable=True),
        sa.Column("start_xi", sa.JSON(), nullable=False),
        sa.Column("substitutes", sa.JSON(), nullable=False),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("fixture_id", "team_id", name="uq_api_football_lineup_fixture_team"),
    )
    op.create_index("ix_api_football_lineup_event", "api_football_lineups", ["event_id"])
    op.create_index(
        op.f("ix_api_football_lineups_fixture_id"), "api_football_lineups", ["fixture_id"]
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_api_football_lineups_fixture_id"), table_name="api_football_lineups")
    op.drop_index("ix_api_football_lineup_event", table_name="api_football_lineups")
    op.drop_table("api_football_lineups")
