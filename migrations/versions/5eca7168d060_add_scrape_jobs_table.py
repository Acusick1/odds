"""add scrape_jobs table

Revision ID: 5eca7168d060
Revises: 771d5b1b451c
Create Date: 2026-04-14 22:29:02.211432

"""

import sqlalchemy as sa
import sqlmodel
from alembic import op

# revision identifiers, used by Alembic.
revision = "5eca7168d060"
down_revision = "771d5b1b451c"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "scrape_jobs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("league", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column("market", sqlmodel.sql.sqltypes.AutoString(), nullable=False),
        sa.Column(
            "status",
            sa.Enum("pending", "running", "completed", "failed", name="scrapejobstatus"),
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
        sa.Column("error_message", sqlmodel.sql.sqltypes.AutoString(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_scrape_jobs_league"), "scrape_jobs", ["league"], unique=False)
    op.create_index(
        "ix_scrape_jobs_league_status", "scrape_jobs", ["league", "market", "status"], unique=False
    )
    op.create_index(
        "ix_scrape_jobs_pending",
        "scrape_jobs",
        ["status", "created_at"],
        unique=False,
        postgresql_where=sa.text("status = 'pending'"),
    )


def downgrade() -> None:
    op.drop_index(
        "ix_scrape_jobs_pending",
        table_name="scrape_jobs",
        postgresql_where=sa.text("status = 'pending'"),
    )
    op.drop_index("ix_scrape_jobs_league_status", table_name="scrape_jobs")
    op.drop_index(op.f("ix_scrape_jobs_league"), table_name="scrape_jobs")
    op.drop_table("scrape_jobs")
    sa.Enum("pending", "running", "completed", "failed", name="scrapejobstatus").drop(op.get_bind())
