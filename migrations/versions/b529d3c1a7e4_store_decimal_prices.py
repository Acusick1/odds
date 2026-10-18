"""store decimal prices

Revision ID: b529d3c1a7e4
Revises: a513c0de3way
Create Date: 2026-10-18 20:00:00.000000

Replaces ``odds.price`` (integer American odds, truncated from decimal sources)
with ``odds.price_decimal``, the source's own decimal price.

Every row is filled in one UPDATE: from the snapshot's re-derived decimal price
when that price maps back to the row's own stored American price, otherwise from
the stored American price, which is exact for The Odds API (it quotes integer
American odds). Snapshots whose original decimal price survives are re-derived in
``raw_data``:

- ``betfair_api``: the runner's ``best_back`` in ``betfair_meta``.
- ``oddsportal`` and ``oddsportal_live``: OddsPortal quotes two-decimal prices and
  ``round()`` conversion to American is injective on them, so the stored price has
  exactly one two-decimal preimage.

The old converter wrote -10000 for decimal prices at or below 1.0. Such a price
decodes like any other (to 1.01), except in OddsPortal markets whose implied
probabilities sum above ``GARBAGE_OVERROUND``: those are source garbage, so the
market is removed from the snapshot's ``raw_data`` and its ``odds`` rows are
deleted. ``football_data_uk`` snapshots are not decodable (three-decimal prices); re-run
``scripts/ingest_football_data_uk.py`` to rewrite them.

``odds`` rows have no snapshot id, so a row is matched to a snapshot outcome by
(event, snapshot time, bookmaker, market, outcome, point), which is not unique
when one event-time has several snapshots. A derived price is accepted for a row
only if it converts back to that row's own stored price (rounding for OddsPortal,
truncation for Betfair).

Dropping ``price`` is destructive and deleted rows are not restored by the
downgrade: stop the scheduler before migrating, and run VACUUM ANALYZE odds after.
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
GARBAGE_OVERROUND = 1.3
PRICE_CHECK = "ck_odds_price_decimal_gt_one"


def _american(decimal_odds: float) -> int:
    if decimal_odds >= 2.0:
        return round((decimal_odds - 1) * 100)
    return round(-100 / (decimal_odds - 1))


def _truncated_american(decimal_odds: float) -> int:
    if decimal_odds >= 2.0:
        return int((decimal_odds - 1) * 100)
    return int(-100 / (decimal_odds - 1))


def _implied(american: int) -> float:
    return 100 / (american + 100) if american > 0 else -american / (-american + 100)


def _oddsportal_preimages() -> dict[int, float]:
    """Map each American price to its two-decimal preimage, dropping ambiguous ones."""
    preimages: dict[int, list[float]] = {}
    for cents in range(1, round((MAX_ODDSPORTAL_DECIMAL - 1) * 100) + 1):
        decimal_odds = round(1 + cents / 100, 2)
        preimages.setdefault(_american(decimal_odds), []).append(decimal_odds)
    return {american: found[0] for american, found in preimages.items() if len(found) == 1}


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
                best_back = float(runner["best_back"])
                if outcome.get("price") == _truncated_american(best_back):
                    outcome["price_decimal"] = best_back
                    del outcome["price"]
                    changed = True
    return changed


def _is_garbage_market(market: dict[str, Any]) -> bool:
    outcomes = market.get("outcomes", [])
    if not any(o.get("price") == INVALID_PRICE_SENTINEL for o in outcomes):
        return False
    totals: dict[Any, float] = {}
    for outcome in outcomes:
        if isinstance(outcome.get("price"), int):
            point = outcome.get("point")
            totals[point] = totals.get(point, 0.0) + _implied(outcome["price"])
    return any(total > GARBAGE_OVERROUND for total in totals.values())


def _rewrite_oddsportal(
    raw_data: dict[str, Any], preimages: dict[int, float]
) -> tuple[bool, list[dict[str, Any]]]:
    changed = False
    quarantined: list[dict[str, Any]] = []
    bookmakers = raw_data.get("bookmakers", [])
    for bookmaker in bookmakers:
        kept = []
        for market in bookmaker.get("markets", []):
            if _is_garbage_market(market):
                quarantined.extend(
                    {
                        "bookmaker_key": bookmaker.get("key"),
                        "market_key": market.get("key"),
                        "outcome_name": outcome.get("name"),
                        "point": outcome.get("point"),
                        "price": outcome.get("price"),
                    }
                    for outcome in market.get("outcomes", [])
                )
                changed = True
                continue
            kept.append(market)
            for outcome in market.get("outcomes", []):
                decimal_odds = preimages.get(outcome.get("price"))  # type: ignore[arg-type]
                if decimal_odds is not None:
                    outcome["price_decimal"] = decimal_odds
                    del outcome["price"]
                    changed = True
        bookmaker["markets"] = kept
    raw_data["bookmakers"] = [b for b in bookmakers if b.get("markets")]
    return changed, quarantined


def _rederive_raw_data(connection: sa.Connection) -> None:
    preimages = _oddsportal_preimages()
    sources = (BETFAIR_SOURCE, *ODDSPORTAL_SOURCES)
    select = sa.text(
        "SELECT id, event_id, snapshot_time, raw_data FROM odds_snapshots "
        "WHERE id > :last_id AND raw_data->>'source' IN :sources ORDER BY id LIMIT :batch"
    ).bindparams(sa.bindparam("sources", expanding=True))
    update = sa.text(
        "UPDATE odds_snapshots SET raw_data = CAST(:raw_data AS json), "
        "bookmaker_count = :bookmaker_count WHERE id = :id"
    )
    quarantine = sa.text(
        "INSERT INTO odds_quarantine (event_id, odds_timestamp, bookmaker_key, market_key, "
        "outcome_name, point, price) VALUES (:event_id, :odds_timestamp, :bookmaker_key, "
        ":market_key, :outcome_name, :point, :price)"
    )

    last_id = 0
    while True:
        rows = connection.execute(
            select, {"last_id": last_id, "sources": sources, "batch": BATCH_SIZE}
        ).all()
        if not rows:
            return
        updates = []
        quarantined = []
        for snapshot_id, event_id, snapshot_time, raw_data in rows:
            payload = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
            if payload.get("source") == BETFAIR_SOURCE:
                changed = _rewrite_betfair(payload)
            else:
                changed, dropped = _rewrite_oddsportal(payload, preimages)
                quarantined.extend(
                    {**row, "event_id": event_id, "odds_timestamp": snapshot_time}
                    for row in dropped
                )
            if changed:
                updates.append(
                    {
                        "id": snapshot_id,
                        "raw_data": json.dumps(payload),
                        "bookmaker_count": len(payload["bookmakers"]),
                    }
                )
        if updates:
            connection.execute(update, updates)
        if quarantined:
            connection.execute(quarantine, quarantined)
        last_id = rows[-1][0]


ROUND_AMERICAN_SQL = """
CASE WHEN {d} >= 2.0 THEN round(CAST(({d} - 1) * 100 AS double precision))
     ELSE round(CAST(-100 / ({d} - 1) AS double precision)) END
"""

TRUNCATED_AMERICAN_SQL = """
CASE WHEN {d} >= 2.0 THEN trunc(CAST(({d} - 1) * 100 AS double precision))
     ELSE trunc(CAST(-100 / ({d} - 1) AS double precision)) END
"""

CREATE_QUARANTINE_SQL = """
CREATE TEMPORARY TABLE odds_quarantine (
    event_id varchar, odds_timestamp timestamptz, bookmaker_key varchar,
    market_key varchar, outcome_name varchar, point double precision, price integer
)
"""

DELETE_QUARANTINED_SQL = """
DELETE FROM odds
USING odds_quarantine q
WHERE odds.event_id = q.event_id
  AND odds.odds_timestamp = q.odds_timestamp
  AND odds.bookmaker_key = q.bookmaker_key
  AND odds.market_key = q.market_key
  AND odds.outcome_name = q.outcome_name
  AND odds.point IS NOT DISTINCT FROM q.point
  AND odds.price = q.price
"""

DERIVED_PRICES_SQL = f"""
CREATE TEMPORARY TABLE odds_derived_price AS
SELECT DISTINCT ON (odds.id) odds.id, derived.price_decimal
FROM (
    SELECT s.event_id,
           s.snapshot_time,
           s.raw_data ->> 'source' AS source,
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
JOIN odds
  ON odds.event_id = derived.event_id
 AND odds.odds_timestamp = derived.snapshot_time
 AND odds.bookmaker_key = derived.bookmaker_key
 AND odds.market_key = derived.market_key
 AND odds.outcome_name = derived.outcome_name
 AND odds.point IS NOT DISTINCT FROM derived.point
WHERE odds.price = CASE WHEN derived.source = 'betfair_api'
    THEN {TRUNCATED_AMERICAN_SQL.format(d="derived.price_decimal")}
    ELSE {ROUND_AMERICAN_SQL.format(d="derived.price_decimal")} END
ORDER BY odds.id
"""

FILL_PRICE_DECIMAL_SQL = """
UPDATE odds
SET price_decimal = COALESCE(
    (SELECT t.price_decimal FROM odds_derived_price t WHERE t.id = odds.id),
    CASE WHEN odds.price > 0 THEN odds.price / 100.0 + 1 ELSE 100.0 / abs(odds.price) + 1 END
)
"""


def upgrade() -> None:
    connection = op.get_bind()
    op.add_column("odds", sa.Column("price_decimal", sa.Float(), nullable=True))

    connection.execute(sa.text(CREATE_QUARANTINE_SQL))
    _rederive_raw_data(connection)
    connection.execute(sa.text(DELETE_QUARANTINED_SQL))
    connection.execute(sa.text(DERIVED_PRICES_SQL))
    connection.execute(sa.text("CREATE UNIQUE INDEX ON odds_derived_price (id)"))
    connection.execute(sa.text(FILL_PRICE_DECIMAL_SQL))
    connection.execute(sa.text("DROP TABLE odds_derived_price"))
    connection.execute(sa.text("DROP TABLE odds_quarantine"))

    op.alter_column("odds", "price_decimal", nullable=False)
    op.create_check_constraint(PRICE_CHECK, "odds", "price_decimal > 1.0")
    op.drop_column("odds", "price")


def _restore_american(raw_data: dict[str, Any]) -> bool:
    changed = False
    for bookmaker in raw_data.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            for outcome in market.get("outcomes", []):
                if "price_decimal" in outcome:
                    outcome["price"] = _american(float(outcome.pop("price_decimal")))
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
        "THEN round(CAST((price_decimal - 1) * 100 AS double precision)) "
        "ELSE round(CAST(-100 / (price_decimal - 1) AS double precision)) END"
    )
    op.alter_column("odds", "price", nullable=False)
    op.drop_constraint(PRICE_CHECK, "odds", type_="check")
    op.drop_column("odds", "price_decimal")
    _restore_raw_data(op.get_bind())
