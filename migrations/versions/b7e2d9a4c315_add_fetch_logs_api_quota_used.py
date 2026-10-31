"""add fetch_logs.api_quota_used, the credits used since the key's last quota reset

Revision ID: b7e2d9a4c315
Revises: 75c92251aa25
Create Date: 2026-10-31 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

revision = "b7e2d9a4c315"
down_revision = "75c92251aa25"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(sa.text("SET lock_timeout = '10s'"))
    op.add_column("fetch_logs", sa.Column("api_quota_used", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("fetch_logs", "api_quota_used")
