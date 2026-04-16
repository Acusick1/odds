"""drop checkpoint column from match_briefs

Revision ID: 9e505a326804
Revises: a7b3c9d2e1f0
Create Date: 2026-04-15 23:47:29.438680

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "9e505a326804"
down_revision = "a7b3c9d2e1f0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_index("ix_match_briefs_event_checkpoint", table_name="match_briefs")
    op.drop_column("match_briefs", "checkpoint")
    op.execute("DROP TYPE IF EXISTS briefcheckpoint")


def downgrade() -> None:
    op.execute("CREATE TYPE briefcheckpoint AS ENUM ('CONTEXT', 'DECISION')")
    op.add_column(
        "match_briefs",
        sa.Column(
            "checkpoint",
            postgresql.ENUM("CONTEXT", "DECISION", name="briefcheckpoint", create_type=False),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_match_briefs_event_checkpoint",
        "match_briefs",
        ["event_id", "checkpoint"],
        unique=False,
    )
