"""migrate_to_timezone_aware_datetimes

Revision ID: db489260d72b
Revises: 1b46df42e39c
Create Date: 2025-10-22 21:57:25.237483

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "db489260d72b"
down_revision = "1b46df42e39c"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Convert all TIMESTAMP columns to TIMESTAMPTZ (timezone-aware)."""
    # Events table
    op.alter_column(
        "events",
        "commence_time",
        type_=sa.DateTime(timezone=True),
        existing_type=sa.DateTime(),
        existing_nullable=False,
        postgresql_using="commence_time AT TIME ZONE 'UTC'",
    )
    op.alter_column(
        "events",
        "completed_at",
        type_=sa.DateTime(timezone=True),
        existing_type=sa.DateTime(),
        existing_nullable=True,
        postgresql_using="completed_at AT TIME ZONE 'UTC'",
    )
    op.alter_column(
        "events",
        "created_at",
        type_=sa.DateTime(timezone=True),
        existing_type=sa.DateTime(),
        existing_nullable=False,
        postgresql_using="created_at AT TIME ZONE 'UTC'",
    )
    op.alter_column(
        "events",
        "updated_at",
        type_=sa.DateTime(timezone=True),
        existing_type=sa.DateTime(),
        existing_nullable=False,
        postgresql_using="updated_at AT TIME ZONE 'UTC'",
    )

    # Odds table
    op.alter_column(
        "odds",
        "odds_timestamp",
        type_=sa.DateTime(timezone=True),
        existing_type=sa.DateTime(),
        existing_nullable=False,
        postgresql_using="odds_timestamp AT TIME ZONE 'UTC'",
    )
    op.alter_column(
        "odds",
        "last_update",
        type_=sa.DateTime(timezone=True),
        existing_type=sa.DateTime(),
        existing_nullable=False,
        postgresql_using="last_update AT TIME ZONE 'UTC'",
    )
    op.alter_column(
        "odds",
        "created_at",
        type_=sa.DateTime(timezone=True),
        existing_type=sa.DateTime(),
        existing_nullable=False,
        postgresql_using="created_at AT TIME ZONE 'UTC'",
    )

    # Odds snapshots table
    op.alter_column(
        "odds_snapshots",
        "snapshot_time",
        type_=sa.DateTime(timezone=True),
        existing_type=sa.DateTime(),
        existing_nullable=False,
        postgresql_using="snapshot_time AT TIME ZONE 'UTC'",
    )
    op.alter_column(
        "odds_snapshots",
        "created_at",
        type_=sa.DateTime(timezone=True),
        existing_type=sa.DateTime(),
        existing_nullable=False,
        postgresql_using="created_at AT TIME ZONE 'UTC'",
    )

    # Fetch logs table
    op.alter_column(
        "fetch_logs",
        "fetch_time",
        type_=sa.DateTime(timezone=True),
        existing_type=sa.DateTime(),
        existing_nullable=False,
        postgresql_using="fetch_time AT TIME ZONE 'UTC'",
    )

    # Data quality logs table
    op.alter_column(
        "data_quality_logs",
        "created_at",
        type_=sa.DateTime(timezone=True),
        existing_type=sa.DateTime(),
        existing_nullable=False,
        postgresql_using="created_at AT TIME ZONE 'UTC'",
    )


def downgrade() -> None:
    """Revert TIMESTAMPTZ columns back to TIMESTAMP (timezone-naive)."""
    # Events table
    op.alter_column(
        "events",
        "commence_time",
        type_=sa.DateTime(),
        existing_type=sa.DateTime(timezone=True),
        existing_nullable=False,
        postgresql_using="commence_time AT TIME ZONE 'UTC'",
    )
    op.alter_column(
        "events",
        "completed_at",
        type_=sa.DateTime(),
        existing_type=sa.DateTime(timezone=True),
        existing_nullable=True,
        postgresql_using="completed_at AT TIME ZONE 'UTC'",
    )
    op.alter_column(
        "events",
        "created_at",
        type_=sa.DateTime(),
        existing_type=sa.DateTime(timezone=True),
        existing_nullable=False,
        postgresql_using="created_at AT TIME ZONE 'UTC'",
    )
    op.alter_column(
        "events",
        "updated_at",
        type_=sa.DateTime(),
        existing_type=sa.DateTime(timezone=True),
        existing_nullable=False,
        postgresql_using="updated_at AT TIME ZONE 'UTC'",
    )

    # Odds table
    op.alter_column(
        "odds",
        "odds_timestamp",
        type_=sa.DateTime(),
        existing_type=sa.DateTime(timezone=True),
        existing_nullable=False,
        postgresql_using="odds_timestamp AT TIME ZONE 'UTC'",
    )
    op.alter_column(
        "odds",
        "last_update",
        type_=sa.DateTime(),
        existing_type=sa.DateTime(timezone=True),
        existing_nullable=False,
        postgresql_using="last_update AT TIME ZONE 'UTC'",
    )
    op.alter_column(
        "odds",
        "created_at",
        type_=sa.DateTime(),
        existing_type=sa.DateTime(timezone=True),
        existing_nullable=False,
        postgresql_using="created_at AT TIME ZONE 'UTC'",
    )

    # Odds snapshots table
    op.alter_column(
        "odds_snapshots",
        "snapshot_time",
        type_=sa.DateTime(),
        existing_type=sa.DateTime(timezone=True),
        existing_nullable=False,
        postgresql_using="snapshot_time AT TIME ZONE 'UTC'",
    )
    op.alter_column(
        "odds_snapshots",
        "created_at",
        type_=sa.DateTime(),
        existing_type=sa.DateTime(timezone=True),
        existing_nullable=False,
        postgresql_using="created_at AT TIME ZONE 'UTC'",
    )

    # Fetch logs table
    op.alter_column(
        "fetch_logs",
        "fetch_time",
        type_=sa.DateTime(),
        existing_type=sa.DateTime(timezone=True),
        existing_nullable=False,
        postgresql_using="fetch_time AT TIME ZONE 'UTC'",
    )

    # Data quality logs table
    op.alter_column(
        "data_quality_logs",
        "created_at",
        type_=sa.DateTime(),
        existing_type=sa.DateTime(timezone=True),
        existing_nullable=False,
        postgresql_using="created_at AT TIME ZONE 'UTC'",
    )
