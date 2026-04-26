"""add mlb_probable_pitchers table

Revision ID: e6f24ef60549
Revises: beaf16386050
Create Date: 2026-04-26 14:42:04.516881

"""

import sqlalchemy as sa
import sqlmodel
from alembic import op

# revision identifiers, used by Alembic.
revision = "e6f24ef60549"
down_revision = "beaf16386050"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "mlb_probable_pitchers",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("game_pk", sa.Integer(), nullable=False),
        sa.Column("commence_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("fetched_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("home_team", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("away_team", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("game_type", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("home_pitcher_name", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("home_pitcher_id", sa.Integer(), nullable=True),
        sa.Column("away_pitcher_name", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.Column("away_pitcher_id", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("game_pk", "fetched_at", name="uq_mlb_probable_pitchers_game_fetched"),
    )
    op.create_index(
        op.f("ix_mlb_probable_pitchers_commence_time"),
        "mlb_probable_pitchers",
        ["commence_time"],
        unique=False,
    )
    op.create_index(
        op.f("ix_mlb_probable_pitchers_fetched_at"),
        "mlb_probable_pitchers",
        ["fetched_at"],
        unique=False,
    )
    # No standalone index on game_pk: the unique constraint on
    # (game_pk, fetched_at) provides one via leftmost-prefix.


def downgrade() -> None:
    op.drop_index(op.f("ix_mlb_probable_pitchers_fetched_at"), table_name="mlb_probable_pitchers")
    op.drop_index(
        op.f("ix_mlb_probable_pitchers_commence_time"),
        table_name="mlb_probable_pitchers",
    )
    op.drop_table("mlb_probable_pitchers")
