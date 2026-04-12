"""fix briefcheckpoint enum labels to uppercase

Revision ID: 771d5b1b451c
Revises: fc55da171cdb
Create Date: 2026-04-12 20:48:32.183640

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "771d5b1b451c"
down_revision = "fc55da171cdb"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TYPE briefcheckpoint RENAME VALUE 'context' TO 'CONTEXT'")
    op.execute("ALTER TYPE briefcheckpoint RENAME VALUE 'decision' TO 'DECISION'")


def downgrade() -> None:
    op.execute("ALTER TYPE briefcheckpoint RENAME VALUE 'CONTEXT' TO 'context'")
    op.execute("ALTER TYPE briefcheckpoint RENAME VALUE 'DECISION' TO 'decision'")
