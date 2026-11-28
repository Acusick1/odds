"""rename match_briefs to briefs and espn_fixtures to espn_games

Revision ID: d3a7f1c9e2b4
Revises: f8fdc2b8e985
Create Date: 2026-11-28 15:00:00.000000

Renames the two tables with their primary keys, id sequences, indexes and constraints, the
``espn_games.match_round`` column to ``game_round``, and the ``fetch-espn-fixtures`` job's
recorded runs and alerts to its new name ``fetch-espn-games``, so its heartbeat history carries
over. The ``briefdecision`` enum keeps its name. Privileges follow the tables. Nothing is dropped
or copied; the downgrade renames every object back.
"""

import sqlalchemy as sa
from alembic import op

revision = "d3a7f1c9e2b4"
down_revision = "f8fdc2b8e985"
branch_labels = None
depends_on = None

# (table, new table, id sequence, new id sequence)
_TABLES = (
    ("match_briefs", "briefs", "match_briefs_id_seq", "briefs_id_seq"),
    ("espn_fixtures", "espn_games", "espn_fixtures_id_seq", "espn_games_id_seq"),
)
# (new table, old name, new name); renaming a primary key or unique constraint renames its index.
_CONSTRAINTS = (
    ("briefs", "match_briefs_pkey", "briefs_pkey"),
    ("briefs", "match_briefs_event_id_fkey", "briefs_event_id_fkey"),
    ("espn_games", "espn_fixtures_pkey", "espn_games_pkey"),
    ("espn_games", "uq_espn_fixture_date_team_comp", "uq_espn_game_date_team_comp"),
)
# (old name, new name) of each plain index.
_INDEXES = (
    ("ix_match_briefs_event_id", "ix_briefs_event_id"),
    ("ix_espn_fixtures_date", "ix_espn_games_date"),
    ("ix_espn_fixtures_team", "ix_espn_games_team"),
    ("ix_espn_fixtures_season", "ix_espn_games_season"),
    ("ix_espn_fixture_team_season", "ix_espn_game_team_season"),
)
# (new table, old column, new column)
_COLUMNS = (("espn_games", "match_round", "game_round"),)
# (table, column) holding a job name, alone or after a ``<kind>:`` prefix.
_JOB_NAME_COLUMNS = (("job_runs", "job_name"), ("alert_history", "alert_type"))
_OLD_JOB = "fetch-espn-fixtures"
_NEW_JOB = "fetch-espn-games"


def _rename_job(old: str, new: str) -> None:
    for table, column in _JOB_NAME_COLUMNS:
        op.execute(
            sa.text(
                f"UPDATE {table} SET {column} = replace({column}, :old, :new) "
                f"WHERE {column} LIKE :pattern"
            ).bindparams(old=old, new=new, pattern=f"%{old}%")
        )


def upgrade() -> None:
    op.execute(sa.text("SET lock_timeout = '10s'"))
    for table, new_table, seq, new_seq in _TABLES:
        op.rename_table(table, new_table)
        op.execute(sa.text(f"ALTER SEQUENCE {seq} RENAME TO {new_seq}"))
    for table, old, new in _CONSTRAINTS:
        op.execute(sa.text(f"ALTER TABLE {table} RENAME CONSTRAINT {old} TO {new}"))
    for old, new in _INDEXES:
        op.execute(sa.text(f"ALTER INDEX {old} RENAME TO {new}"))
    for table, old, new in _COLUMNS:
        op.alter_column(table, old, new_column_name=new)
    _rename_job(_OLD_JOB, _NEW_JOB)


def downgrade() -> None:
    op.execute(sa.text("SET lock_timeout = '10s'"))
    _rename_job(_NEW_JOB, _OLD_JOB)
    for table, old, new in _COLUMNS:
        op.alter_column(table, new, new_column_name=old)
    for old, new in _INDEXES:
        op.execute(sa.text(f"ALTER INDEX {new} RENAME TO {old}"))
    for table, old, new in _CONSTRAINTS:
        op.execute(sa.text(f"ALTER TABLE {table} RENAME CONSTRAINT {new} TO {old}"))
    for table, new_table, seq, new_seq in _TABLES:
        op.execute(sa.text(f"ALTER SEQUENCE {new_seq} RENAME TO {seq}"))
        op.rename_table(new_table, table)
