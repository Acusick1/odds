"""create the pg_stat_statements extension

Revision ID: a7c3e9f1b2d4
Revises: e4b8c2d6a1f3
Create Date: 2026-11-20 12:00:00.000000

Creates the extension whose statistics the odds-web performance page reads. The server must also
preload it (``shared_preload_libraries``, set in ``docker-compose.yml``) before its views answer;
creating it needs no preload. The downgrade drops the extension, its statistics and the web role's
functions over it, which ``odds_web.grants`` defines and which would otherwise be left broken.
"""

import sqlalchemy as sa
from alembic import op

revision = "a7c3e9f1b2d4"
down_revision = "e4b8c2d6a1f3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(sa.text("CREATE EXTENSION IF NOT EXISTS pg_stat_statements SCHEMA public"))


def downgrade() -> None:
    op.execute(sa.text("DROP FUNCTION IF EXISTS public.odds_web_query_stats()"))
    op.execute(sa.text("DROP FUNCTION IF EXISTS public.odds_web_query_stats_reset()"))
    op.execute(sa.text("DROP EXTENSION IF EXISTS pg_stat_statements"))
