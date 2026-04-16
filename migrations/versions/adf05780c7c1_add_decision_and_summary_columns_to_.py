"""add decision and summary columns to match_briefs

Revision ID: adf05780c7c1
Revises: 9e505a326804
Create Date: 2026-04-16 09:55:04.153523

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "adf05780c7c1"
down_revision = "9e505a326804"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE TYPE briefdecision AS ENUM ('WATCHING', 'BET', 'SKIP')")
    op.add_column(
        "match_briefs",
        sa.Column(
            "decision",
            sa.Enum("WATCHING", "BET", "SKIP", name="briefdecision", create_type=False),
            nullable=True,
        ),
    )
    op.add_column(
        "match_briefs",
        sa.Column("summary", sa.String(), nullable=True),
    )
    # Backfill existing rows before adding NOT NULL constraints
    op.execute("UPDATE match_briefs SET decision = 'WATCHING' WHERE decision IS NULL")
    op.execute("UPDATE match_briefs SET summary = '' WHERE summary IS NULL")
    op.alter_column("match_briefs", "decision", nullable=False)
    op.alter_column("match_briefs", "summary", nullable=False)


def downgrade() -> None:
    op.drop_column("match_briefs", "summary")
    op.drop_column("match_briefs", "decision")
    op.execute("DROP TYPE IF EXISTS briefdecision")
