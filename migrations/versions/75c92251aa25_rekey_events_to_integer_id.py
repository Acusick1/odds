"""re-key events to the integer id and drop the string event ids

Revision ID: 75c92251aa25
Revises: b8e2d5f7a1c3
Create Date: 2026-10-29 12:00:00.000000

``events`` keeps one key, the integer ``surrogate_id``, renamed to ``id``; the string ``id`` is
dropped and every foreign key to ``events`` references the integer (#712). The string id was
the first source's id: the Odds API's own id, or one built from teams and kick-off (``op_``,
``op_live_``, ``fduk_``). An Odds API id is recorded as the event's ``odds_api`` external id
before it goes, so the resolver still finds the event by it; the built ids were never a
source's, and nothing matches on them.

Every string id is kept in ``legacy_event_ids``, which only the downgrade reads. The upgrade
refuses, changing nothing, when a foreign key to ``events`` is not one it re-keys, a row names
no event, or an Odds API id could not be recorded; and it fails, rolling back, when any table's
rows, or the events they reference, differ after the re-key (``_guards``, ``_check_rekeyed``).
Alert contexts that list event ids are re-keyed with them. The downgrade restores the string
ids from ``legacy_event_ids``; an event created since takes its Odds API id, else
``event_<id>``.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import sqlalchemy as sa
from alembic import op

revision = "75c92251aa25"
down_revision = "b8e2d5f7a1c3"
branch_labels = None
depends_on = None

logger = logging.getLogger("alembic.runtime.migration")

# Tables whose event_id references the string id; nullable ones may hold no event.
_STRING_KEYED = (
    "candidate_decisions",
    "data_quality_logs",
    "match_briefs",
    "nba_injury_reports",
    "nba_team_game_logs",
    "odds_snapshots",
    "paper_trades",
)
# Tables whose event_id already references the integer key.
_INTEGER_KEYED = ("event_external_ids", "health_check_acknowledgements")

# A string id built from teams and kick-off, or by this downgrade, rather than issued by a source.
_BUILT_ID = "e.id LIKE 'op\\_%' OR e.id LIKE 'fduk\\_%' OR e.id LIKE 'event\\_%'"

LEGACY_TABLE_DDL = (
    "CREATE TABLE legacy_event_ids ("
    " event_id BIGINT PRIMARY KEY,"
    " legacy_id VARCHAR NOT NULL CONSTRAINT uq_legacy_event_ids_legacy_id UNIQUE)"
)

_ODDS_API_ID_GUARD = (
    "events whose Odds API id is not their odds_api external id",
    "SELECT count(*) FROM events e WHERE NOT (" + _BUILT_ID + ") AND NOT EXISTS ("
    " SELECT 1 FROM event_external_ids x WHERE x.event_id = e.surrogate_id"
    " AND x.source = 'odds_api' AND x.external_id = e.id)",
)


def _guards(string_keyed: Sequence[str]) -> list[tuple[str, str]]:
    """(what the rows are, a query counting them); any count refuses the upgrade."""
    return [
        *(
            (
                f"{table} rows whose event_id names no event",
                f"SELECT count(*) FROM {table} t WHERE t.event_id IS NOT NULL"
                " AND NOT EXISTS (SELECT 1 FROM events e WHERE e.id = t.event_id)",
            )
            for table in string_keyed
        ),
        _ODDS_API_ID_GUARD,
    ]


# The odds_api id of each event keyed by an Odds API id that holds none yet.
_RECORD_ODDS_API_IDS = (
    "INSERT INTO event_external_ids (source, external_id, event_id, created_at)"
    " SELECT 'odds_api', e.id, e.surrogate_id, now() FROM events e"
    " WHERE NOT (" + _BUILT_ID + ") AND NOT EXISTS ("
    " SELECT 1 FROM event_external_ids x WHERE x.event_id = e.surrogate_id"
    " AND x.source = 'odds_api') AND NOT EXISTS ("
    " SELECT 1 FROM event_external_ids x WHERE x.source = 'odds_api' AND x.external_id = e.id)"
)


@dataclass(frozen=True)
class _Referencing:
    """The tables whose event_id has a foreign key to ``events``, by the key they reference
    before the upgrade. A database may lack the NBA tables."""

    string_keyed: tuple[str, ...]
    integer_keyed: tuple[str, ...]

    @property
    def tables(self) -> tuple[str, ...]:
        return (*self.string_keyed, *self.integer_keyed)


def _referencing(bind: sa.engine.Connection) -> _Referencing:
    """The tables referencing ``events``; refuses a foreign key this migration does not re-key."""
    found = {
        (row[0], row[1])
        for row in bind.execute(
            sa.text(
                "SELECT r.relname, c.conname FROM pg_constraint c"
                " JOIN pg_class r ON r.oid = c.conrelid"
                " WHERE c.contype = 'f' AND c.confrelid = 'events'::regclass"
            )
        )
    }
    known = {(table, _fk(table)) for table in (*_STRING_KEYED, *_INTEGER_KEYED)}
    if unknown := found - known:
        raise RuntimeError(
            f"refusing to re-key events: foreign keys to events this migration does not re-key: "
            f"{sorted(unknown)}"
        )
    tables = {table for table, _ in found}
    return _Referencing(
        tuple(t for t in _STRING_KEYED if t in tables),
        tuple(t for t in _INTEGER_KEYED if t in tables),
    )


def _fk(table: str) -> str:
    return f"{table}_event_id_fkey"


def _references(
    bind: sa.engine.Connection, referencing: _Referencing, string_keyed: bool
) -> dict[str, tuple[int, int, int]]:
    """Per referencing table: its rows, its rows naming an event, and the sum of the integer
    keys of the events they name. ``string_keyed`` is the schema before the upgrade, where the
    integer key is ``surrogate_id`` and the string-keyed tables reference the string ``id``."""
    key = "surrogate_id" if string_keyed else "id"
    totals: dict[str, tuple[int, int, int]] = {}
    for table in referencing.tables:
        joined = "id" if table in referencing.string_keyed or not string_keyed else "surrogate_id"
        row = bind.execute(
            sa.text(
                f"SELECT count(*), count(t.event_id), coalesce(sum(e.{key}), 0)"
                f" FROM {table} t LEFT JOIN events e ON e.{joined} = t.event_id"
            )
        ).one()
        totals[table] = (row[0], row[1], int(row[2]))
    return totals


def _check_rekeyed(
    before: dict[str, tuple[int, int, int]], after: dict[str, tuple[int, int, int]]
) -> None:
    changed = [
        f"{table} {before[table]} -> {after[table]}"
        for table in before
        if before[table] != after[table]
    ]
    if changed:
        raise RuntimeError(
            "the re-key changed the rows referencing events or the events they name "
            "(rows, rows naming an event, sum of event keys): " + "; ".join(changed)
        )


def _drop_foreign_keys(referencing: _Referencing) -> None:
    for table in referencing.tables:
        op.drop_constraint(_fk(table), table, type_="foreignkey")


def _create_foreign_keys(
    referencing: _Referencing, integer_column: str, string_column: str
) -> None:
    for table in referencing.string_keyed:
        op.create_foreign_key(_fk(table), table, "events", ["event_id"], [string_column])
    for table in referencing.integer_keyed:
        op.create_foreign_key(_fk(table), table, "events", ["event_id"], [integer_column])


def _rename_identity_sequence(column: str, name: str) -> None:
    op.execute(
        sa.text(
            "DO $$ BEGIN EXECUTE format('ALTER SEQUENCE %s RENAME TO "
            f"{name}', pg_get_serial_sequence('events', '{column}')); END $$"
        )
    )


def _rekey_alert_contexts(match: str, value: str) -> None:
    """Replace each id an alert context's ``event_ids`` lists by the other key of its event;
    an id naming no event stays as it is."""
    op.execute(
        sa.text(
            "UPDATE alert_history a SET context = jsonb_set("
            "a.context::jsonb, '{event_ids}', m.ids)::json FROM ("
            f" SELECT h.id, jsonb_agg(coalesce(to_jsonb({value}), x.v) ORDER BY x.n) AS ids"
            " FROM alert_history h"
            " CROSS JOIN LATERAL jsonb_array_elements(h.context::jsonb -> 'event_ids')"
            " WITH ORDINALITY x(v, n)"
            f" LEFT JOIN events e ON {match}"
            " WHERE jsonb_typeof(h.context::jsonb -> 'event_ids') = 'array'"
            " GROUP BY h.id) m WHERE a.id = m.id"
        )
    )


def upgrade() -> None:
    op.execute(sa.text("SET lock_timeout = '10s'"))
    bind = op.get_bind()
    referencing = _referencing(bind)

    recorded = bind.execute(sa.text(_RECORD_ODDS_API_IDS)).rowcount
    logger.info("recorded %s Odds API ids as odds_api external ids", recorded)
    held = [
        f"{count} {what}"
        for what, query in _guards(referencing.string_keyed)
        if (count := bind.execute(sa.text(query)).scalar_one())
    ]
    if held:
        raise RuntimeError("refusing to re-key events: " + "; ".join(held))

    before = _references(bind, referencing, string_keyed=True)
    op.execute(sa.text(LEGACY_TABLE_DDL))
    op.execute(
        sa.text(
            "INSERT INTO legacy_event_ids (event_id, legacy_id) SELECT surrogate_id, id FROM events"
        )
    )
    _rekey_alert_contexts("e.id = x.v #>> '{}'", "e.surrogate_id")

    _drop_foreign_keys(referencing)
    op.execute(
        sa.text(
            "CREATE FUNCTION pg_temp.event_key(legacy VARCHAR) RETURNS BIGINT LANGUAGE sql STABLE"
            " AS $$ SELECT surrogate_id FROM events WHERE id = legacy $$"
        )
    )
    for table in referencing.string_keyed:
        op.execute(
            sa.text(
                f"ALTER TABLE {table} ALTER COLUMN event_id TYPE BIGINT"
                " USING pg_temp.event_key(event_id)"
            )
        )
    op.execute(sa.text("DROP FUNCTION pg_temp.event_key(VARCHAR)"))

    op.drop_constraint("events_pkey", "events", type_="primary")
    op.drop_column("events", "id")
    op.drop_constraint("uq_events_surrogate_id", "events", type_="unique")
    op.alter_column("events", "surrogate_id", new_column_name="id")
    _rename_identity_sequence("id", "events_id_seq")
    op.create_primary_key("events_pkey", "events", ["id"])
    _create_foreign_keys(referencing, integer_column="id", string_column="id")

    _check_rekeyed(before, _references(bind, referencing, string_keyed=False))


def downgrade() -> None:
    op.execute(sa.text("SET lock_timeout = '10s'"))
    bind = op.get_bind()
    referencing = _referencing(bind)
    before = _references(bind, referencing, string_keyed=False)

    # Events created since the upgrade take their Odds API id, as the Odds API writers keyed
    # them, else a built one.
    op.execute(
        sa.text(
            "INSERT INTO legacy_event_ids (event_id, legacy_id)"
            " SELECT e.id, CASE WHEN x.external_id IS NOT NULL AND NOT EXISTS ("
            " SELECT 1 FROM legacy_event_ids l WHERE l.legacy_id = x.external_id)"
            " THEN x.external_id ELSE 'event_' || e.id END"
            " FROM events e LEFT JOIN event_external_ids x"
            " ON x.event_id = e.id AND x.source = 'odds_api'"
            " WHERE NOT EXISTS (SELECT 1 FROM legacy_event_ids l WHERE l.event_id = e.id)"
        )
    )

    _drop_foreign_keys(referencing)
    op.execute(
        sa.text(
            "CREATE FUNCTION pg_temp.legacy_key(key BIGINT) RETURNS VARCHAR LANGUAGE sql STABLE"
            " AS $$ SELECT legacy_id FROM legacy_event_ids WHERE event_id = key $$"
        )
    )
    for table in referencing.string_keyed:
        op.execute(
            sa.text(
                f"ALTER TABLE {table} ALTER COLUMN event_id TYPE VARCHAR"
                " USING pg_temp.legacy_key(event_id)"
            )
        )
    op.execute(sa.text("DROP FUNCTION pg_temp.legacy_key(BIGINT)"))

    op.add_column("events", sa.Column("legacy_id", sa.VARCHAR(), nullable=True))
    op.execute(
        sa.text(
            "UPDATE events e SET legacy_id = l.legacy_id FROM legacy_event_ids l"
            " WHERE l.event_id = e.id"
        )
    )
    op.alter_column("events", "legacy_id", nullable=False)
    op.drop_constraint("events_pkey", "events", type_="primary")
    op.alter_column("events", "id", new_column_name="surrogate_id")
    _rename_identity_sequence("surrogate_id", "events_surrogate_id_seq")
    op.create_unique_constraint("uq_events_surrogate_id", "events", ["surrogate_id"])
    op.alter_column("events", "legacy_id", new_column_name="id")
    op.create_primary_key("events_pkey", "events", ["id"])
    _create_foreign_keys(referencing, integer_column="surrogate_id", string_column="id")
    _rekey_alert_contexts(
        "e.surrogate_id = CASE WHEN jsonb_typeof(x.v) = 'number' THEN (x.v #>> '{}')::bigint END",
        "e.id",
    )
    op.drop_table("legacy_event_ids")

    _check_rekeyed(before, _references(bind, referencing, string_keyed=True))
