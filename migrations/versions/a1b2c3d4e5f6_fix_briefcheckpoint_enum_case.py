"""fix briefcheckpoint enum labels to uppercase

SQLAlchemy sends the Python enum .name (CONTEXT, DECISION) not .value
(context, decision). The initial migration created lowercase labels,
causing InvalidTextRepresentationError on insert.

Revision ID: a1b2c3d4e5f6
Revises: fc55da171cdb
Create Date: 2026-04-12 21:00:00.000000

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "a1b2c3d4e5f6"
down_revision = "fc55da171cdb"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TYPE briefcheckpoint RENAME VALUE 'context' TO 'CONTEXT'")
    op.execute("ALTER TYPE briefcheckpoint RENAME VALUE 'decision' TO 'DECISION'")


def downgrade() -> None:
    op.execute("ALTER TYPE briefcheckpoint RENAME VALUE 'CONTEXT' TO 'context'")
    op.execute("ALTER TYPE briefcheckpoint RENAME VALUE 'DECISION' TO 'decision'")
