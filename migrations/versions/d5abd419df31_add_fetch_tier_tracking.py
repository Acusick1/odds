"""add_fetch_tier_tracking

Revision ID: d5abd419df31
Revises: db489260d72b
Create Date: 2025-10-25 18:50:56.054751

Adds fetch_tier and hours_until_commence columns to odds_snapshots table
for tier coverage validation and ML feature engineering.
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "d5abd419df31"
down_revision = "db489260d72b"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add fetch_tier tracking columns to odds_snapshots."""
    # Add fetch_tier column (nullable to support existing data)
    op.add_column("odds_snapshots", sa.Column("fetch_tier", sa.String(), nullable=True))

    # Add hours_until_commence column for ML features
    op.add_column("odds_snapshots", sa.Column("hours_until_commence", sa.Float(), nullable=True))

    # Create index on fetch_tier for efficient querying
    op.create_index("ix_odds_snapshots_fetch_tier", "odds_snapshots", ["fetch_tier"])

    # Create composite index on event_id + fetch_tier
    op.create_index("ix_event_tier", "odds_snapshots", ["event_id", "fetch_tier"])

    # Backfill tiers for existing snapshots
    # This SQL calculates the tier based on snapshot_time and commence_time
    op.execute(
        """
        UPDATE odds_snapshots os
        SET
            hours_until_commence = EXTRACT(EPOCH FROM (e.commence_time - os.snapshot_time)) / 3600.0,
            fetch_tier = CASE
                WHEN EXTRACT(EPOCH FROM (e.commence_time - os.snapshot_time)) / 3600.0 <= 3 THEN 'closing'
                WHEN EXTRACT(EPOCH FROM (e.commence_time - os.snapshot_time)) / 3600.0 <= 12 THEN 'pregame'
                WHEN EXTRACT(EPOCH FROM (e.commence_time - os.snapshot_time)) / 3600.0 <= 24 THEN 'sharp'
                WHEN EXTRACT(EPOCH FROM (e.commence_time - os.snapshot_time)) / 3600.0 <= 72 THEN 'early'
                ELSE 'opening'
            END
        FROM events e
        WHERE os.event_id = e.id
    """
    )


def downgrade() -> None:
    """Remove fetch_tier tracking columns from odds_snapshots."""
    # Drop indexes first
    op.drop_index("ix_event_tier", table_name="odds_snapshots")
    op.drop_index("ix_odds_snapshots_fetch_tier", table_name="odds_snapshots")

    # Drop columns
    op.drop_column("odds_snapshots", "hours_until_commence")
    op.drop_column("odds_snapshots", "fetch_tier")
