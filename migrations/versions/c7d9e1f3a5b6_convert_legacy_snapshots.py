"""Convert legacy snapshot envelopes before removing the old representations."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import timedelta
from typing import Any

import sqlalchemy as sa
from alembic import op
from odds_core.vocabulary import Source
from odds_lambda.price_rows import ExpandedEnvelope, expand_envelope, stored_market

revision = "c7d9e1f3a5b6"
down_revision = "a6f3c1e9d2b4"
branch_labels = None
depends_on = None


def _classify(raw: dict[str, Any], row: sa.RowMapping) -> tuple[Source, str]:
    books = raw.get("bookmakers")
    if not isinstance(books, list) or not books:
        raise ValueError("missing bookmakers")
    tag = raw.get("source")
    if tag == "betfair_api":
        if any(
            not isinstance(b, dict) or not isinstance(b.get("betfair_meta"), dict) for b in books
        ):
            raise ValueError("malformed Betfair envelope")
        return Source.BETFAIR, "observed"
    if tag is not None or "source" in raw:
        raise ValueError(f"unsupported source {tag!r}")
    if not all(
        isinstance(raw.get(k), str) and raw[k]
        for k in ("id", "sport_key", "home_team", "away_team", "commence_time")
    ):
        raise ValueError("unrecognizable untagged Odds API envelope")
    for book in books:
        if (
            not isinstance(book, dict)
            or not book.get("key")
            or not book.get("last_update")
            or not book.get("markets")
        ):
            raise ValueError("malformed Odds API bookmaker")
        for market in book["markets"]:
            if not market.get("outcomes") or any("price" not in o for o in market["outcomes"]):
                raise ValueError("malformed Odds API outcomes")
    gap = row["created_at"] - row["snapshot_time"]
    return Source.ODDS_API, "upper_bound" if gap > timedelta(hours=1) else "observed"


def upgrade() -> None:
    bind = op.get_bind()
    rows = (
        bind.execute(
            sa.text(
                "SELECT s.*, e.home_team, e.away_team,"
                " EXISTS (SELECT 1 FROM prices p WHERE p.snapshot_id=s.id) AS has_prices"
                " FROM odds_snapshots s JOIN events e ON e.id=s.event_id ORDER BY s.id"
            )
        )
        .mappings()
        .all()
    )
    plans: list[tuple[sa.RowMapping, Source, str, ExpandedEnvelope, dict[str, str]]] = []
    keys: set[tuple[Any, ...]] = {
        (r["event_id"], r["source"], r["market"], r["snapshot_time"])
        for r in rows
        if r["source"] is not None
    }
    for row in rows:
        sid = row["id"]
        if row["source"] is not None:
            if (
                row["time_basis"] is None
                or row["payload_id"] is None
                or (row["market"] is not None and not row["has_prices"])
            ):
                raise RuntimeError(f"snapshot {sid}: partial shaped state")
            continue
        if (
            any(
                row[k] is not None
                for k in (
                    "market",
                    "time_basis",
                    "payload_id",
                    "in_play",
                    "market_status",
                    "total_matched",
                    "market_start_time",
                )
            )
            or row["has_prices"]
        ):
            raise RuntimeError(f"snapshot {sid}: partial shaped state")
        try:
            raw = row["raw_data"]
            if not isinstance(raw, dict):
                raise ValueError("raw_data is not an object")
            source, basis = _classify(raw, row)
            expanded = expand_envelope(raw, source, row["home_team"], row["away_team"])
            if (
                expanded.issues
                or not expanded.markets
                or any(not m.prices for m in expanded.markets)
            ):
                raise ValueError(f"parser issues or empty markets: {expanded.issues}")
            relinks: dict[str, str] = {}
            for book in raw["bookmakers"]:
                for market in book["markets"]:
                    read = stored_market(market)
                    if read is None:
                        raise ValueError("unsupported market")
                    key = market["key"]
                    if key in relinks and relinks[key] != read[0].value:
                        raise ValueError("ambiguous market relinking")
                    relinks[key] = read[0].value
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            raise RuntimeError(f"snapshot {sid}: {exc}") from exc
        old_markets = (
            bind.execute(
                sa.text("SELECT DISTINCT market_key FROM odds WHERE snapshot_id=:id"), {"id": sid}
            )
            .scalars()
            .all()
        )
        if any(m not in relinks for m in old_markets):
            raise RuntimeError(f"snapshot {sid}: odds market absent from envelope")
        if (
            len(expanded.markets) > 1
            and bind.execute(
                sa.text(
                    "SELECT EXISTS (SELECT 1 FROM candidate_decisions WHERE :id = ANY(input_snapshot_ids))"
                ),
                {"id": sid},
            ).scalar_one()
        ):
            raise RuntimeError(f"snapshot {sid}: referenced split snapshot")
        for market in expanded.markets:
            key = (row["event_id"], source.value, market.market.value, row["snapshot_time"])
            if key in keys:
                raise RuntimeError(f"snapshot {sid}: snapshot-key collision {key}")
            keys.add(key)
        plans.append((row, source, basis, expanded, relinks))

    for row, source, basis, expanded, relinks in plans:
        payload_id = bind.execute(
            sa.text(
                "INSERT INTO snapshot_payloads (payload,received_at) VALUES (CAST(:raw AS jsonb),:created_at) RETURNING id"
            ),
            {"raw": json.dumps(row["raw_data"]), "created_at": row["created_at"]},
        ).scalar_one()
        ids: dict[str, int] = {}
        for index, market in enumerate(expanded.markets):
            values: dict[str, Any] = {
                "event_id": row["event_id"],
                "snapshot_time": row["snapshot_time"],
                "created_at": row["created_at"],
                "source": source.value,
                "time_basis": basis,
                "payload_id": payload_id,
                "market": market.market.value,
                **asdict(market.meta),
            }
            if index == 0:
                sid = row["id"]
                bind.execute(
                    sa.text(
                        "UPDATE odds_snapshots SET "
                        + ",".join(f"{k}=:{k}" for k in values)
                        + " WHERE id=:id"
                    ),
                    {**values, "id": sid},
                )
            else:
                values.update(
                    {
                        k: row[k]
                        for k in (
                            "bookmaker_count",
                            "api_request_id",
                            "fetch_tier",
                            "hours_until_commence",
                        )
                    }
                )
                values["raw_data"] = json.dumps(row["raw_data"])
                columns = ",".join(values)
                params = ",".join(
                    "CAST(:raw_data AS json)" if k == "raw_data" else f":{k}" for k in values
                )
                sid = bind.execute(
                    sa.text(
                        f"INSERT INTO odds_snapshots ({columns}) VALUES ({params}) RETURNING id"
                    ),
                    values,
                ).scalar_one()
            ids[market.market.value] = sid
            prices: list[dict[str, Any]] = [
                {"snapshot_id": sid, **asdict(p)} for p in market.prices
            ]
            columns = ",".join(prices[0])
            params = ",".join(f":{k}" for k in prices[0])
            bind.execute(sa.text(f"INSERT INTO prices ({columns}) VALUES ({params})"), prices)
        for old_market, market_key in relinks.items():
            bind.execute(
                sa.text(
                    "UPDATE odds SET snapshot_id=:new_id WHERE snapshot_id=:old_id AND market_key=:market"
                ),
                {"new_id": ids[market_key], "old_id": row["id"], "market": old_market},
            )


def downgrade() -> None:
    # The following drop migration cannot restore legacy data; keep its surviving archive.
    pass
