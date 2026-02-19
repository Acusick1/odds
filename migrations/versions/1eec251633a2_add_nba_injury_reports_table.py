"""add nba_injury_reports table

Revision ID: 1eec251633a2
Revises: b02bc04a440e
Create Date: 2026-02-19 14:30:04.987653

"""

import sqlalchemy as sa
import sqlmodel
from alembic import op

# revision identifiers, used by Alembic.
revision = "1eec251633a2"
down_revision = "b02bc04a440e"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "nba_injury_reports",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("report_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("game_date", sa.Date(), nullable=False),
        sa.Column("game_time_et", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("matchup", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("team", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("player_name", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "OUT", "QUESTIONABLE", "DOUBTFUL", "PROBABLE", "AVAILABLE", name="injurystatus"
            ),
            nullable=False,
        ),
        sa.Column("reason", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("event_id", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["event_id"],
            ["events.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "report_time",
            "team",
            "player_name",
            "game_date",
            name="uq_injury_report_time_team_player_date",
        ),
    )
    op.create_index(
        "ix_injury_event_report_time",
        "nba_injury_reports",
        ["event_id", "report_time"],
        unique=False,
    )
    op.create_index(
        op.f("ix_nba_injury_reports_event_id"), "nba_injury_reports", ["event_id"], unique=False
    )
    op.create_index(
        op.f("ix_nba_injury_reports_report_time"),
        "nba_injury_reports",
        ["report_time"],
        unique=False,
    )
    op.create_index(
        op.f("ix_nba_injury_reports_team"), "nba_injury_reports", ["team"], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_nba_injury_reports_team"), table_name="nba_injury_reports")
    op.drop_index(op.f("ix_nba_injury_reports_report_time"), table_name="nba_injury_reports")
    op.drop_index(op.f("ix_nba_injury_reports_event_id"), table_name="nba_injury_reports")
    op.drop_index("ix_injury_event_report_time", table_name="nba_injury_reports")
    op.drop_table("nba_injury_reports")
