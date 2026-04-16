"""drop scrape_jobs table

Revision ID: 917adbe0576d
Revises: adf05780c7c1
Create Date: 2026-04-16 14:54:49.924126

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "917adbe0576d"
down_revision = "adf05780c7c1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_index("ix_scrape_jobs_league_status", table_name="scrape_jobs")
    op.drop_index(op.f("ix_scrape_jobs_league"), table_name="scrape_jobs")
    op.drop_table("scrape_jobs")
    sa.Enum("PENDING", "RUNNING", "COMPLETED", "FAILED", name="scrapejobstatus").drop(op.get_bind())


def downgrade() -> None:
    sa.Enum("PENDING", "RUNNING", "COMPLETED", "FAILED", name="scrapejobstatus").create(
        op.get_bind()
    )
    op.create_table(
        "scrape_jobs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("league", sa.String(), nullable=False),
        sa.Column("market", sa.String(), nullable=False),
        sa.Column(
            "status",
            sa.Enum("PENDING", "RUNNING", "COMPLETED", "FAILED", name="scrapejobstatus"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("matches_scraped", sa.Integer(), nullable=True),
        sa.Column("matches_converted", sa.Integer(), nullable=True),
        sa.Column("events_matched", sa.Integer(), nullable=True),
        sa.Column("events_created", sa.Integer(), nullable=True),
        sa.Column("snapshots_stored", sa.Integer(), nullable=True),
        sa.Column("error_message", sa.String(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_scrape_jobs_league"), "scrape_jobs", ["league"], unique=False)
    op.create_index(
        "ix_scrape_jobs_league_status", "scrape_jobs", ["league", "market", "status"], unique=False
    )
