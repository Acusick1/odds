"""add server default for created_at on EPL data tables

Revision ID: 3387940c2f83
Revises: c4e8a2f19b03
Create Date: 2026-03-19 21:38:35.421577

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "3387940c2f83"
down_revision = "c4e8a2f19b03"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for table in ("espn_fixtures", "espn_lineups", "fpl_availability"):
        op.alter_column(
            table,
            "created_at",
            server_default=sa.func.now(),
        )


def downgrade() -> None:
    for table in ("espn_fixtures", "espn_lineups", "fpl_availability"):
        op.alter_column(
            table,
            "created_at",
            server_default=None,
        )
