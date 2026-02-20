"""add nba_player_season_stats table

Revision ID: 49b9789986fa
Revises: 09773800e69a
Create Date: 2026-02-20 16:15:02.132921

"""

import sqlalchemy as sa
import sqlmodel
from alembic import op

# revision identifiers, used by Alembic.
revision = "49b9789986fa"
down_revision = "09773800e69a"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "nba_player_season_stats",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("player_id", sa.Integer(), nullable=False),
        sa.Column("player_name", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("team_id", sa.Integer(), nullable=False),
        sa.Column("team_abbreviation", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("season", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("minutes", sa.Float(), nullable=False),
        sa.Column("games_played", sa.Integer(), nullable=False),
        sa.Column("on_off_rtg", sa.Float(), nullable=True),
        sa.Column("on_def_rtg", sa.Float(), nullable=True),
        sa.Column("usage", sa.Float(), nullable=True),
        sa.Column("ts_pct", sa.Float(), nullable=True),
        sa.Column("efg_pct", sa.Float(), nullable=True),
        sa.Column("assists", sa.Integer(), nullable=False),
        sa.Column("turnovers", sa.Integer(), nullable=False),
        sa.Column("rebounds", sa.Integer(), nullable=False),
        sa.Column("steals", sa.Integer(), nullable=False),
        sa.Column("blocks", sa.Integer(), nullable=False),
        sa.Column("points", sa.Integer(), nullable=False),
        sa.Column("plus_minus", sa.Float(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("player_id", "season", name="uq_player_season_stats_player_season"),
    )
    op.create_index(
        "ix_player_season_stats_season", "nba_player_season_stats", ["season"], unique=False
    )
    op.create_index(
        "ix_player_season_stats_team",
        "nba_player_season_stats",
        ["team_abbreviation", "season"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_player_season_stats_team", table_name="nba_player_season_stats")
    op.drop_index("ix_player_season_stats_season", table_name="nba_player_season_stats")
    op.drop_table("nba_player_season_stats")
