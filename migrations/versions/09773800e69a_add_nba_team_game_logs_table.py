"""add nba_team_game_logs table

Revision ID: 09773800e69a
Revises: 1eec251633a2
Create Date: 2026-02-19 22:51:08.650137

"""

import sqlalchemy as sa
import sqlmodel
from alembic import op

# revision identifiers, used by Alembic.
revision = "09773800e69a"
down_revision = "1eec251633a2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "nba_team_game_logs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("nba_game_id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("team_id", sa.Integer(), nullable=False),
        sa.Column("team_abbreviation", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("game_date", sa.Date(), nullable=True),
        sa.Column("matchup", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("wl", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("season", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("pts", sa.Integer(), nullable=True),
        sa.Column("fgm", sa.Integer(), nullable=True),
        sa.Column("fga", sa.Integer(), nullable=True),
        sa.Column("fg3m", sa.Integer(), nullable=True),
        sa.Column("fg3a", sa.Integer(), nullable=True),
        sa.Column("ftm", sa.Integer(), nullable=True),
        sa.Column("fta", sa.Integer(), nullable=True),
        sa.Column("oreb", sa.Integer(), nullable=True),
        sa.Column("dreb", sa.Integer(), nullable=True),
        sa.Column("reb", sa.Integer(), nullable=True),
        sa.Column("ast", sa.Integer(), nullable=True),
        sa.Column("stl", sa.Integer(), nullable=True),
        sa.Column("blk", sa.Integer(), nullable=True),
        sa.Column("tov", sa.Integer(), nullable=True),
        sa.Column("pf", sa.Integer(), nullable=True),
        sa.Column("plus_minus", sa.Integer(), nullable=True),
        sa.Column("event_id", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["event_id"],
            ["events.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("nba_game_id", "team_id", name="uq_game_log_game_team"),
    )
    op.create_index("ix_game_log_season", "nba_team_game_logs", ["season"], unique=False)
    op.create_index(
        "ix_game_log_team_date",
        "nba_team_game_logs",
        ["team_abbreviation", "game_date"],
        unique=False,
    )
    op.create_index(
        op.f("ix_nba_team_game_logs_event_id"),
        "nba_team_game_logs",
        ["event_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_nba_team_game_logs_event_id"), table_name="nba_team_game_logs")
    op.drop_index("ix_game_log_team_date", table_name="nba_team_game_logs")
    op.drop_index("ix_game_log_season", table_name="nba_team_game_logs")
    op.drop_table("nba_team_game_logs")
