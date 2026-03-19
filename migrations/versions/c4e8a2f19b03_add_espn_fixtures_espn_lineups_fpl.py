"""add espn_fixtures, espn_lineups, fpl_availability tables

Revision ID: c4e8a2f19b03
Revises: 019ba3d7f14b
Create Date: 2026-03-19 12:00:00.000000

"""

import sqlalchemy as sa
import sqlmodel
from alembic import op

# revision identifiers, used by Alembic.
revision = "c4e8a2f19b03"
down_revision = "019ba3d7f14b"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ESPN fixtures
    op.create_table(
        "espn_fixtures",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("team", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("opponent", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("competition", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("match_round", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("home_away", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("score_team", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("score_opponent", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("status", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("season", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("date", "team", "competition", name="uq_espn_fixture_date_team_comp"),
    )
    op.create_index("ix_espn_fixture_team_season", "espn_fixtures", ["team", "season"])
    op.create_index(op.f("ix_espn_fixtures_date"), "espn_fixtures", ["date"])
    op.create_index(op.f("ix_espn_fixtures_team"), "espn_fixtures", ["team"])
    op.create_index(op.f("ix_espn_fixtures_season"), "espn_fixtures", ["season"])

    # ESPN lineups
    op.create_table(
        "espn_lineups",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("date", sa.DateTime(timezone=True), nullable=False),
        sa.Column("home_team", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("away_team", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("team", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("player_id", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("player_name", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("position", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("starter", sa.Boolean(), nullable=False),
        sa.Column("formation_place", sa.Integer(), nullable=False),
        sa.Column("season", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("date", "team", "player_id", name="uq_espn_lineup_date_team_player"),
    )
    op.create_index("ix_espn_lineup_team_season", "espn_lineups", ["team", "season"])
    op.create_index(op.f("ix_espn_lineups_date"), "espn_lineups", ["date"])
    op.create_index(op.f("ix_espn_lineups_team"), "espn_lineups", ["team"])
    op.create_index(op.f("ix_espn_lineups_season"), "espn_lineups", ["season"])

    # FPL availability
    op.create_table(
        "fpl_availability",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("snapshot_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("gameweek", sa.Integer(), nullable=False),
        sa.Column("season", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("player_code", sa.Integer(), nullable=False),
        sa.Column("player_name", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("team", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("position", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("chance_of_playing", sa.Float(), nullable=False),
        sa.Column("status", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("news", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "snapshot_time",
            "gameweek",
            "player_code",
            "season",
            name="uq_fpl_avail_snapshot_gw_player_season",
        ),
    )
    op.create_index("ix_fpl_avail_team_season", "fpl_availability", ["team", "season"])
    op.create_index(
        op.f("ix_fpl_availability_snapshot_time"), "fpl_availability", ["snapshot_time"]
    )
    op.create_index(op.f("ix_fpl_availability_team"), "fpl_availability", ["team"])
    op.create_index(op.f("ix_fpl_availability_season"), "fpl_availability", ["season"])


def downgrade() -> None:
    op.drop_table("fpl_availability")
    op.drop_table("espn_lineups")
    op.drop_table("espn_fixtures")
