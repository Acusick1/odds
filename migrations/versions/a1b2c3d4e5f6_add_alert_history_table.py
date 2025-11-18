"""add_alert_history_table

Revision ID: a1b2c3d4e5f6
Revises: 9ff81f073e6b
Create Date: 2025-11-18 12:20:00.000000

Adds alert_history table for alert deduplication tracking in health monitoring system.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy import JSON, Column, DateTime, Index, Integer, String

# revision identifiers, used by Alembic.
revision = "a1b2c3d4e5f6"
down_revision = "9ff81f073e6b"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create alert_history table."""
    op.create_table(
        "alert_history",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("alert_type", sa.String(), nullable=False),
        sa.Column("severity", sa.String(), nullable=False),
        sa.Column("message", sa.String(), nullable=False),
        sa.Column("sent_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("context", JSON, nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indexes for efficient querying
    op.create_index("ix_alert_history_alert_type", "alert_history", ["alert_type"])
    op.create_index("ix_alert_history_sent_at", "alert_history", ["sent_at"])
    op.create_index("ix_alert_type_sent_at", "alert_history", ["alert_type", "sent_at"])


def downgrade() -> None:
    """Drop alert_history table."""
    # Drop indexes first
    op.drop_index("ix_alert_type_sent_at", table_name="alert_history")
    op.drop_index("ix_alert_history_sent_at", table_name="alert_history")
    op.drop_index("ix_alert_history_alert_type", table_name="alert_history")

    # Drop table
    op.drop_table("alert_history")
