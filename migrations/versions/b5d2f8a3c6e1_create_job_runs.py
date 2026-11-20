"""create job_runs

Revision ID: b5d2f8a3c6e1
Revises: a7c3e9f1b2d4
Create Date: 2026-11-20 14:00:00.000000

One row per scheduled job run, with its start, finish and outcome, which the odds-web
performance page charts.
"""

import sqlalchemy as sa
from alembic import op

revision = "b5d2f8a3c6e1"
down_revision = "a7c3e9f1b2d4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "job_runs",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("job_name", sa.Text(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("outcome", sa.Text(), nullable=False),
        sa.CheckConstraint("finished_at >= started_at", name="ck_job_runs_finished_after_start"),
        sa.CheckConstraint(
            "outcome IN ('failed', 'succeeded', 'timed_out')", name="ck_job_runs_outcome"
        ),
    )
    op.create_index("ix_job_runs_job_name_started_at", "job_runs", ["job_name", "started_at"])
    op.create_index("ix_job_runs_started_at", "job_runs", ["started_at"])


def downgrade() -> None:
    op.drop_table("job_runs")
