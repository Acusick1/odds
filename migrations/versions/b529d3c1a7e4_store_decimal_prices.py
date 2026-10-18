"""store decimal prices

Revision ID: b529d3c1a7e4
Revises: a513c0de3way
Create Date: 2026-10-18 20:00:00.000000

Replaces ``odds.price`` (integer American odds, truncated from decimal sources)
with ``odds.price_decimal``, the source's own decimal price.

Rows are first converted from the stored American price, which is exact for The
Odds API (it quotes integer American odds). Snapshots whose original decimal
price survives are then re-derived, in ``raw_data`` and in the matching ``odds``
rows:

- ``betfair_api``: the runner's ``best_back`` in ``betfair_meta``.
- ``oddsportal`` and ``oddsportal_live``: OddsPortal quotes two-decimal prices and
  ``round()`` conversion to American is injective on them, so the stored price has
  exactly one two-decimal preimage.

The old converter wrote -10000 for decimal prices at or below 1.0; those stay as
stored. ``football_data_uk`` snapshots are not decodable (three-decimal prices); re-run
``scripts/ingest_football_data_uk.py`` to rewrite them.

Dropping ``price`` is destructive: stop the scheduler before migrating.
"""

from __future__ import annotations

import json
from typing import Any

import sqlalchemy as sa
from alembic import op

revision = "b529d3c1a7e4"
down_revision = "a513c0de3way"
branch_labels = None
depends_on = None

BETFAIR_SOURCE = "betfair_api"
ODDSPORTAL_SOURCES = ("oddsportal", "oddsportal_live")
BATCH_SIZE = 500
MAX_ODDSPORTAL_DECIMAL = 101.0
INVALID_PRICE_SENTINEL = -10000


def _american(decimal_odds: float) -> int:
    if decimal_odds >= 2.0:
        return round((decimal_odds - 1) * 100)
    return round(-100 / (decimal_odds - 1))


def _oddsportal_preimages() -> dict[int, float]:
    """Map each American price to its two-decimal preimage, dropping ambiguous ones."""
    preimages: dict[int, list[float]] = {}
    for cents in range(1, round((MAX_ODDSPORTAL_DECIMAL - 1) * 100) + 1):
        decimal_odds = round(1 + cents / 100, 2)
        preimages.setdefault(_american(decimal_odds), []).append(decimal_odds)
    return {
        american: found[0]
        for american, found in preimages.items()
        if len(found) == 1 and american != INVALID_PRICE_SENTINEL
    }


def _rewrite_betfair(raw_data: dict[str, Any]) -> bool:
    changed = False
    for bookmaker in raw_data.get("bookmakers", []):
        runners = [
            runner
            for runner in bookmaker.get("betfair_meta", {}).get("runners", [])
            if runner.get("best_back") is not None
        ]
        for market in bookmaker.get("markets", []):
            outcomes = market.get("outcomes", [])
            if len(outcomes) != len(runners):
                continue
            for outcome, runner in zip(outcomes, runners, strict=True):
                if "price" in outcome:
                    outcome["price_decimal"] = float(runner["best_back"])
                    del outcome["price"]
                    changed = True
    return changed


def _rewrite_oddsportal(raw_data: dict[str, Any], preimages: dict[int, float]) -> bool:
    changed = False
    for bookmaker in raw_data.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            for outcome in market.get("outcomes", []):
                decimal_odds = preimages.get(outcome.get("price"))  # type: ignore[arg-type]
                if decimal_odds is not None:
                    outcome["price_decimal"] = decimal_odds
                    del outcome["price"]
                    changed = True
    return changed


def _rederive_raw_data(connection: sa.Connection) -> None:
    preimages = _oddsportal_preimages()
    sources = (BETFAIR_SOURCE, *ODDSPORTAL_SOURCES)
    select = sa.text(
        "SELECT id, raw_data FROM odds_snapshots "
        "WHERE id > :last_id AND raw_data->>'source' IN :sources ORDER BY id LIMIT :batch"
    ).bindparams(sa.bindparam("sources", expanding=True))
    update = sa.text("UPDATE odds_snapshots SET raw_data = CAST(:raw_data AS json) WHERE id = :id")

    last_id = 0
    while True:
        rows = connection.execute(
            select, {"last_id": last_id, "sources": sources, "batch": BATCH_SIZE}
        ).all()
        if not rows:
            return
        updates = []
        for snapshot_id, raw_data in rows:
            payload = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
            if payload.get("source") == BETFAIR_SOURCE:
                changed = _rewrite_betfair(payload)
            else:
                changed = _rewrite_oddsportal(payload, preimages)
            if changed:
                updates.append({"id": snapshot_id, "raw_data": json.dumps(payload)})
        if updates:
            connection.execute(update, updates)
        last_id = rows[-1][0]


ODDS_FROM_RAW_DATA_SQL = """
UPDATE odds
SET price_decimal = derived.price_decimal
FROM (
    SELECT s.event_id,
           s.snapshot_time,
           bookmaker ->> 'key' AS bookmaker_key,
           market ->> 'key' AS market_key,
           outcome ->> 'name' AS outcome_name,
           CAST(outcome ->> 'point' AS double precision) AS point,
           CAST(outcome ->> 'price_decimal' AS double precision) AS price_decimal
    FROM odds_snapshots s,
         jsonb_array_elements(CAST(s.raw_data AS jsonb) -> 'bookmakers') AS bookmaker,
         jsonb_array_elements(bookmaker -> 'markets') AS market,
         jsonb_array_elements(market -> 'outcomes') AS outcome
    WHERE s.raw_data ->> 'source' IN ('betfair_api', 'oddsportal', 'oddsportal_live')
      AND outcome ? 'price_decimal'
) AS derived
WHERE odds.event_id = derived.event_id
  AND odds.odds_timestamp = derived.snapshot_time
  AND odds.bookmaker_key = derived.bookmaker_key
  AND odds.market_key = derived.market_key
  AND odds.outcome_name = derived.outcome_name
  AND odds.point IS NOT DISTINCT FROM derived.point
  AND odds.price_decimal IS DISTINCT FROM derived.price_decimal
"""


def upgrade() -> None:
    op.add_column("odds", sa.Column("price_decimal", sa.Float(), nullable=True))
    op.execute(
        "UPDATE odds SET price_decimal = CASE WHEN price > 0 "
        "THEN price / 100.0 + 1 ELSE 100.0 / abs(price) + 1 END"
    )
    op.alter_column("odds", "price_decimal", nullable=False)
    op.drop_column("odds", "price")

    connection = op.get_bind()
    _rederive_raw_data(connection)
    connection.execute(sa.text(ODDS_FROM_RAW_DATA_SQL))


def _restore_american(raw_data: dict[str, Any]) -> bool:
    changed = False
    for bookmaker in raw_data.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            for outcome in market.get("outcomes", []):
                if "price_decimal" in outcome:
                    decimal_odds = float(outcome.pop("price_decimal"))
                    outcome["price"] = _american(decimal_odds)
                    changed = True
    return changed


def _restore_raw_data(connection: sa.Connection) -> None:
    select = sa.text(
        "SELECT id, raw_data FROM odds_snapshots "
        "WHERE id > :last_id AND CAST(raw_data AS text) LIKE '%price_decimal%' "
        "ORDER BY id LIMIT :batch"
    )
    update = sa.text("UPDATE odds_snapshots SET raw_data = CAST(:raw_data AS json) WHERE id = :id")
    last_id = 0
    while True:
        rows = connection.execute(select, {"last_id": last_id, "batch": BATCH_SIZE}).all()
        if not rows:
            return
        updates = []
        for snapshot_id, raw_data in rows:
            payload = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
            if _restore_american(payload):
                updates.append({"id": snapshot_id, "raw_data": json.dumps(payload)})
        if updates:
            connection.execute(update, updates)
        last_id = rows[-1][0]


def downgrade() -> None:
    op.add_column("odds", sa.Column("price", sa.Integer(), nullable=True))
    op.execute(
        "UPDATE odds SET price = CASE WHEN price_decimal >= 2.0 "
        "THEN round((price_decimal - 1) * 100) "
        "ELSE round(-100 / (price_decimal - 1)) END"
    )
    op.alter_column("odds", "price", nullable=False)
    op.drop_column("odds", "price_decimal")
    _restore_raw_data(op.get_bind())
