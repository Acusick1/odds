"""shape the odds rows no snapshot holds into snapshots and price rows

Revision ID: a6f3c1e9d2b4
Revises: 4d7e2a9c1b36
Create Date: 2026-10-28 16:00:00.000000

The OddsPortal live scraper stored one snapshot per market at one scrape time. For EPL scrapes
from March to August 2026 only one market's snapshot of a scrape survived, so the other
market's ``odds`` rows hold the only copy of its prices and carry no ``snapshot_id`` (#677).

Each (event, market, time) group of those rows becomes an ``oddsportal`` snapshot with time
basis ``observed``, as its surviving sibling is, holding an envelope rebuilt from the rows as
its payload and the price rows ``odds_lambda.price_rows.expand_envelope`` reads from it, the
reader every writer uses. The rows are then linked to it. Only groups beside an OddsPortal
``observed`` snapshot of the same event and time are shaped; any other row stays without a
snapshot, and the next migration refuses to drop ``odds`` while one remains.

The upgrade raises when an envelope does not read back as exactly its rows, one price row each,
or when the snapshot key is already held. The reader is imported only when a group is found, so
a database without such rows migrates whatever the reader's later shape.

The downgrade removes the snapshots whose payload carries ``REBUILT_MARKER`` and whose ``odds``
rows still exist, with their prices and payloads, and unlinks those rows. Once the next
migration has dropped ``odds``, the rebuilt snapshots are the only copy of those prices, so a
downgrade past it keeps them, and a later upgrade finds no unlinked row to shape again.
"""

import json
from collections import defaultdict
from datetime import datetime
from typing import Any

import sqlalchemy as sa
from alembic import op

revision = "a6f3c1e9d2b4"
down_revision = "4d7e2a9c1b36"
branch_labels = None
depends_on = None

REBUILT_MARKER = "rebuilt_from"
_ORPHAN_ROWS = sa.text(
    """
    SELECT o.id, o.event_id, o.odds_timestamp, o.market_key, o.bookmaker_key,
           o.bookmaker_title, o.outcome_name, o.point, o.price_decimal, o.last_update,
           o.created_at, e.home_team, e.away_team
    FROM odds o
    JOIN events e ON e.id = o.event_id
    WHERE o.snapshot_id IS NULL
      AND EXISTS (
          SELECT 1 FROM odds_snapshots s
          WHERE s.event_id = o.event_id
            AND s.snapshot_time = o.odds_timestamp
            AND s.source = 'oddsportal'
            AND s.time_basis = 'observed'
      )
    ORDER BY o.event_id, o.odds_timestamp, o.market_key, o.id
    """
)


def _envelope(rows: list[Any]) -> dict[str, Any]:
    books: dict[str, dict[str, Any]] = {}
    for row in rows:
        book = books.setdefault(
            row.bookmaker_key,
            {
                "key": row.bookmaker_key,
                "title": row.bookmaker_title,
                "last_update": row.last_update.isoformat(),
                "markets": [{"key": row.market_key, "outcomes": []}],
            },
        )
        outcome: dict[str, Any] = {"name": row.outcome_name, "price_decimal": row.price_decimal}
        if row.point is not None:
            outcome["point"] = row.point
        book["markets"][0]["outcomes"].append(outcome)
    return {"source": "oddsportal_live", REBUILT_MARKER: "odds", "bookmakers": list(books.values())}


def upgrade() -> None:
    op.execute(sa.text("SET lock_timeout = '10s'"))
    bind = op.get_bind()
    groups: dict[tuple[str, datetime, str], list[Any]] = defaultdict(list)
    for row in bind.execute(_ORPHAN_ROWS):
        groups[(row.event_id, row.odds_timestamp, row.market_key)].append(row)
    if not groups:
        return

    from odds_core.vocabulary import Source
    from odds_lambda.price_rows import expand_envelope

    prices: list[dict[str, Any]] = []
    for (event_id, snapshot_time, market_key), rows in groups.items():
        envelope = _envelope(rows)
        expanded = expand_envelope(
            envelope, Source.ODDSPORTAL, rows[0].home_team, rows[0].away_team
        )
        if (
            expanded.issues
            or [m.market.value for m in expanded.markets] != [market_key]
            or len(expanded.markets[0].prices) != len(rows)
        ):
            raise RuntimeError(
                f"odds rows of {event_id} {market_key} at {snapshot_time.isoformat()} do not read "
                f"back as one price row each: {len(rows)} rows, markets "
                f"{[m.market.value for m in expanded.markets]}, issues "
                f"{[i.description for i in expanded.issues]}"
            )
        created_at = min(row.created_at for row in rows)
        payload_id = bind.execute(
            sa.text(
                "INSERT INTO snapshot_payloads (payload, received_at)"
                " VALUES (CAST(:payload AS jsonb), :received_at) RETURNING id"
            ),
            {"payload": json.dumps(envelope), "received_at": created_at},
        ).scalar_one()
        snapshot_id = bind.execute(
            sa.text(
                "INSERT INTO odds_snapshots (event_id, snapshot_time, bookmaker_count, created_at,"
                " source, market, time_basis, payload_id)"
                " VALUES (:event_id, :snapshot_time, :bookmaker_count, :created_at,"
                " 'oddsportal', :market, 'observed', :payload_id)"
                " ON CONFLICT ON CONSTRAINT uq_odds_snapshots_event_source_market_time DO NOTHING"
                " RETURNING id"
            ),
            {
                "event_id": event_id,
                "snapshot_time": snapshot_time,
                "bookmaker_count": len(envelope["bookmakers"]),
                "created_at": created_at,
                "market": market_key,
                "payload_id": payload_id,
            },
        ).scalar_one_or_none()
        if snapshot_id is None:
            raise RuntimeError(
                f"an oddsportal {market_key} snapshot of {event_id} at "
                f"{snapshot_time.isoformat()} is already stored, yet its odds rows have none"
            )
        prices += [
            {
                "snapshot_id": snapshot_id,
                "bookmaker": spec.bookmaker,
                "outcome": spec.outcome.value,
                "line": spec.line,
                "side": spec.side.value,
                "price": spec.price,
                "changed_at": spec.changed_at,
            }
            for spec in expanded.markets[0].prices
        ]
        bind.execute(
            sa.text("UPDATE odds SET snapshot_id = :snapshot_id WHERE id = ANY(:ids)"),
            {"snapshot_id": snapshot_id, "ids": [row.id for row in rows]},
        )
    bind.execute(
        sa.text(
            "INSERT INTO prices (snapshot_id, bookmaker, outcome, line, side, price, changed_at)"
            " VALUES (:snapshot_id, :bookmaker, :outcome, :line, :side, :price, :changed_at)"
        ),
        prices,
    )


def downgrade() -> None:
    op.execute(sa.text("SET lock_timeout = '10s'"))
    op.execute(
        sa.text(
            "CREATE TEMP TABLE rebuilt_with_odds ON COMMIT DROP AS"
            " SELECT s.id, s.payload_id FROM odds_snapshots s"
            " JOIN snapshot_payloads p ON p.id = s.payload_id"
            f" WHERE p.payload ? '{REBUILT_MARKER}'"
            " AND EXISTS (SELECT 1 FROM odds o WHERE o.snapshot_id = s.id)"
        )
    )
    op.execute(
        sa.text(
            "UPDATE odds SET snapshot_id = NULL"
            " WHERE snapshot_id IN (SELECT id FROM rebuilt_with_odds)"
        )
    )
    op.execute(sa.text("DELETE FROM odds_snapshots WHERE id IN (SELECT id FROM rebuilt_with_odds)"))
    op.execute(
        sa.text(
            "DELETE FROM snapshot_payloads WHERE id IN (SELECT payload_id FROM rebuilt_with_odds)"
            " AND NOT EXISTS (SELECT 1 FROM odds_snapshots s WHERE s.payload_id = snapshot_payloads.id)"
        )
    )
    op.execute(sa.text("DROP TABLE rebuilt_with_odds"))
