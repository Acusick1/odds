"""drop legacy_event_ids

Revision ID: f8fdc2b8e985
Revises: c8e4a1f6d2b9
Create Date: 2026-11-28 12:00:00.000000

``legacy_event_ids`` held each event's string id from before 75c92251aa25 re-keyed events to
the integer id, and only that migration's downgrade read it. Without it the string ids are
gone, so this downgrade refuses and 75c92251aa25's downgrade cannot be reached past it.
"""

import sqlalchemy as sa
from alembic import op

revision = "f8fdc2b8e985"
down_revision = "c8e4a1f6d2b9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(sa.text("SET lock_timeout = '10s'"))
    op.drop_table("legacy_event_ids")


def downgrade() -> None:
    raise NotImplementedError(
        "legacy_event_ids is dropped and the pre-75c92251aa25 string event ids cannot be"
        " restored; restore a database dump taken before this migration instead"
    )
