"""recalculate_historical_fetch_tiers

Revision ID: 9ff81f073e6b
Revises: d5abd419df31
Create Date: 2025-10-30 11:48:41.008544

This migration recalculates fetch_tier and hours_until_commence for all existing
odds_snapshots based on their snapshot_time and the associated event's commence_time.

This fixes a bug where tiers were incorrectly assigned based on the closest game
instead of each event's individual timing.
"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "9ff81f073e6b"
down_revision = "d5abd419df31"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    Recalculate fetch_tier and hours_until_commence for all snapshots.

    The calculation logic matches core/tier_utils.py:
    - CLOSING: 0-3 hours before game
    - PREGAME: 3-12 hours before game
    - SHARP: 12-24 hours before game
    - EARLY: 1-3 days (24-72 hours) before game
    - OPENING: 3+ days (72+ hours) before game
    """
    connection = op.get_bind()

    # Update all snapshots with recalculated values
    # This uses a SQL CASE statement to replicate the tier calculation logic
    connection.execute(
        sa.text(
            """
        UPDATE odds_snapshots
        SET
            hours_until_commence = EXTRACT(EPOCH FROM (e.commence_time - odds_snapshots.snapshot_time)) / 3600,
            fetch_tier = CASE
                WHEN EXTRACT(EPOCH FROM (e.commence_time - odds_snapshots.snapshot_time)) / 3600 <= 3 THEN 'closing'
                WHEN EXTRACT(EPOCH FROM (e.commence_time - odds_snapshots.snapshot_time)) / 3600 <= 12 THEN 'pregame'
                WHEN EXTRACT(EPOCH FROM (e.commence_time - odds_snapshots.snapshot_time)) / 3600 <= 24 THEN 'sharp'
                WHEN EXTRACT(EPOCH FROM (e.commence_time - odds_snapshots.snapshot_time)) / 3600 <= 72 THEN 'early'
                ELSE 'opening'
            END
        FROM events e
        WHERE odds_snapshots.event_id = e.id
    """
        )
    )

    # Log the migration
    print("✓ Recalculated fetch_tier and hours_until_commence for all historical snapshots")


def downgrade() -> None:
    """
    Downgrade is not implemented as we cannot restore the incorrect historical values.

    The original values were incorrect due to a bug, so there's no meaningful way to
    revert this migration. If needed, you could set all values to NULL:

    UPDATE odds_snapshots SET fetch_tier = NULL, hours_until_commence = NULL;
    """
    # We cannot meaningfully downgrade a data fix migration
    # The old values were incorrect, so there's nothing to restore to
    print("⚠ Downgrade not implemented - original values were incorrect")
