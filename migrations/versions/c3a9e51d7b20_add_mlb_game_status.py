"""add game status and start-time flag to mlb_probable_pitchers

Revision ID: c3a9e51d7b20
Revises: b529d3c1a7e4
Create Date: 2026-10-19 09:00:00.000000

"""

import sqlalchemy as sa
import sqlmodel
from alembic import op

revision = "c3a9e51d7b20"
down_revision = "b529d3c1a7e4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "mlb_probable_pitchers",
        sa.Column("game_status", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    )
    op.add_column(
        "mlb_probable_pitchers",
        sa.Column("start_time_tbd", sa.Boolean(), server_default=sa.false(), nullable=False),
    )


def downgrade() -> None:
    op.drop_column("mlb_probable_pitchers", "start_time_tbd")
    op.drop_column("mlb_probable_pitchers", "game_status")
