"""rename match_briefs to briefs and espn_fixtures to espn_games

Revision ID: b9d4e1a7c3f5
Revises: e4b8c2d6a1f3
Create Date: 2026-11-19 14:00:00.000000

Renames the two tables, their primary keys, id sequences, indexes and constraints to name the
brief and the game, and the two columns that named a fixture or a match for a game
(``espn_games.game_round``, ``candidate_decisions.input_game``). The ``briefdecision`` enum
keeps its name. Privileges follow the tables, so no grant changes. Nothing is dropped or copied;
the downgrade renames every object back.
"""

import sqlalchemy as sa
from alembic import op

revision = "b9d4e1a7c3f5"
down_revision = "e4b8c2d6a1f3"
branch_labels = None
depends_on = None

# (table, new table, id sequence, new id sequence)
_TABLES = (
    ("match_briefs", "briefs", "match_briefs_id_seq", "briefs_id_seq"),
    ("espn_fixtures", "espn_games", "espn_fixtures_id_seq", "espn_games_id_seq"),
)
# (new table, old index name, new index name) of each plain index.
_INDEXES = (
    ("briefs", "ix_match_briefs_event_id", "ix_briefs_event_id"),
    ("espn_games", "ix_espn_fixtures_date", "ix_espn_games_date"),
    ("espn_games", "ix_espn_fixtures_team", "ix_espn_games_team"),
    ("espn_games", "ix_espn_fixtures_season", "ix_espn_games_season"),
    ("espn_games", "ix_espn_fixture_team_season", "ix_espn_game_team_season"),
)
# (new table, old name, new name) of each constraint; renaming one renames the index behind a
# primary key or unique constraint.
_CONSTRAINTS = (
    ("briefs", "match_briefs_pkey", "briefs_pkey"),
    ("briefs", "match_briefs_event_id_fkey", "briefs_event_id_fkey"),
    ("espn_games", "espn_fixtures_pkey", "espn_games_pkey"),
    ("espn_games", "uq_espn_fixture_date_team_comp", "uq_espn_game_date_team_comp"),
)
# (new table, old column, new column)
_COLUMNS = (
    ("espn_games", "match_round", "game_round"),
    ("candidate_decisions", "input_fixture", "input_game"),
)


def upgrade() -> None:
    op.execute(sa.text("SET lock_timeout = '10s'"))
    for table, new_table, seq, new_seq in _TABLES:
        op.rename_table(table, new_table)
        op.execute(sa.text(f"ALTER SEQUENCE {seq} RENAME TO {new_seq}"))
    for table, old, new in _CONSTRAINTS:
        op.execute(sa.text(f"ALTER TABLE {table} RENAME CONSTRAINT {old} TO {new}"))
    for _, old, new in _INDEXES:
        op.execute(sa.text(f"ALTER INDEX {old} RENAME TO {new}"))
    for table, old, new in _COLUMNS:
        op.alter_column(table, old, new_column_name=new)


def downgrade() -> None:
    op.execute(sa.text("SET lock_timeout = '10s'"))
    for table, old, new in _COLUMNS:
        op.alter_column(table, new, new_column_name=old)
    for _, old, new in _INDEXES:
        op.execute(sa.text(f"ALTER INDEX {new} RENAME TO {old}"))
    for table, old, new in _CONSTRAINTS:
        op.execute(sa.text(f"ALTER TABLE {table} RENAME CONSTRAINT {new} TO {old}"))
    for table, new_table, seq, new_seq in _TABLES:
        op.execute(sa.text(f"ALTER SEQUENCE {new_seq} RENAME TO {seq}"))
        op.rename_table(new_table, table)
