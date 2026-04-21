"""add state column to espn_fixtures

Revision ID: beaf16386050
Revises: 8a76c9607193
Create Date: 2026-04-21 13:36:34.478454

"""

import sqlalchemy as sa
import sqlmodel
from alembic import op

# revision identifiers, used by Alembic.
revision = "beaf16386050"
down_revision = "8a76c9607193"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "espn_fixtures",
        sa.Column("state", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("espn_fixtures", "state")
