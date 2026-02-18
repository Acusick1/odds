"""retag in-play snapshots

Revision ID: b02bc04a440e
Revises: 7a9cf0e00ca5
Create Date: 2026-02-14 12:09:17.519839

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "b02bc04a440e"
down_revision = "7a9cf0e00ca5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        UPDATE odds_snapshots
        SET fetch_tier = 'in_play'
        WHERE hours_until_commence IS NOT NULL
          AND hours_until_commence < 0
          AND fetch_tier = 'closing'
        """
    )


def downgrade() -> None:
    op.execute(
        """
        UPDATE odds_snapshots
        SET fetch_tier = 'closing'
        WHERE hours_until_commence IS NOT NULL
          AND hours_until_commence < 0
          AND fetch_tier = 'in_play'
        """
    )
