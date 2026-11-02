from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from odds_core.vocabulary import Source
from odds_lambda.price_rows import expand_envelope
from sqlalchemy import text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import AsyncConnection

from tests.unit.conftest import seed

CONVERSION = "c7d9e1f3a5b6_convert_legacy_snapshots"
DROP = "b8e2d5f7a1c3_drop_old_odds_representations"


def load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, Path("migrations/versions") / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def step(connection: AsyncConnection, name: str, direction: str = "upgrade") -> None:
    module = load(name)

    def run(sync_connection: Connection) -> None:
        with Operations.context(MigrationContext.configure(sync_connection)):
            getattr(module, direction)()

    await connection.run_sync(run)


@pytest.mark.asyncio(loop_scope="session")
class TestConvertLegacySnapshots:
    async def test_conversion_unblocks_drop_and_preserves_quotes(
        self, converted_input: AsyncConnection
    ) -> None:
        c = converted_input
        with pytest.raises(RuntimeError, match="refusing to drop"):
            await step(c, DROP)
        await step(c, CONVERSION)
        assert (await c.execute(text("SELECT count(*) FROM prices"))).scalar_one() == 21
        assert (await c.execute(text("SELECT count(*) FROM odds_snapshots"))).scalar_one() == 4
        assert (
            await c.execute(
                text(
                    "SELECT count(*) FROM odds o JOIN odds_snapshots s ON s.id=o.snapshot_id WHERE o.market_key<>s.market"
                )
            )
        ).scalar_one() == 0
        assert (
            await c.execute(
                text(
                    "SELECT count(*) FROM odds_snapshots s JOIN snapshot_payloads p ON p.id=s.payload_id WHERE s.raw_data::jsonb=p.payload AND s.created_at=p.received_at"
                )
            )
        ).scalar_one() == 4
        await step(c, CONVERSION)
        await step(c, DROP)
        await step(c, DROP, "downgrade")
        await step(c, CONVERSION, "downgrade")
        assert (await c.execute(text("SELECT count(*) FROM prices"))).scalar_one() == 21

    async def test_exact_archives_timestamps_prices_and_metadata(
        self, converted_input: AsyncConnection
    ) -> None:
        c = converted_input
        data: dict[str, Any] = json.loads(
            Path("tests/fixtures/legacy_snapshots/representative.json").read_text()
        )
        betfair = data["snapshots"][1]
        betfair["raw_data"]["bookmakers"][0]["betfair_meta"]["market_start_time"] = betfair[
            "snapshot_time"
        ]
        await c.execute(
            text("UPDATE odds_snapshots SET raw_data=CAST(:raw AS json) WHERE id=:id"),
            {"raw": json.dumps(betfair["raw_data"]), "id": betfair["id"]},
        )
        await step(c, CONVERSION)
        for original in data["snapshots"]:
            source = Source.BETFAIR if original["raw_data"].get("source") else Source.ODDS_API
            expanded = expand_envelope(
                original["raw_data"], source, original["home_team"], original["away_team"]
            )
            rows = (
                (
                    await c.execute(
                        text(
                            "SELECT s.*,p.payload,p.received_at FROM odds_snapshots s JOIN snapshot_payloads p ON p.id=s.payload_id WHERE s.event_id=:event AND s.snapshot_time=:time ORDER BY s.id"
                        ),
                        {
                            "event": original["event_id"],
                            "time": datetime.fromisoformat(original["snapshot_time"]),
                        },
                    )
                )
                .mappings()
                .all()
            )
            assert rows[0]["id"] == original["id"]
            assert len(rows) == len(expanded.markets)
            assert len({r["payload_id"] for r in rows}) == 1
            for row, market in zip(rows, expanded.markets, strict=True):
                assert row["payload"] == original["raw_data"]
                assert (
                    row["created_at"]
                    == row["received_at"]
                    == datetime.fromisoformat(original["created_at"])
                )
                assert row["snapshot_time"] == datetime.fromisoformat(original["snapshot_time"])
                assert row["source"] == source.value
                assert row["time_basis"] == "observed"
                for key, value in asdict(market.meta).items():
                    assert row[key] == value
                prices = (
                    (
                        await c.execute(
                            text(
                                "SELECT bookmaker,outcome,line,side,price,size,matched,changed_at FROM prices WHERE snapshot_id=:id ORDER BY id"
                            ),
                            {"id": row["id"]},
                        )
                    )
                    .mappings()
                    .all()
                )
                assert [dict(p) for p in prices] == [asdict(p) for p in market.prices]

    @pytest.mark.parametrize(
        "gap,basis",
        [(timedelta(hours=1), "observed"), (timedelta(hours=1, microseconds=1), "upper_bound")],
    )
    async def test_historical_gap(
        self, converted_input: AsyncConnection, gap: timedelta, basis: str
    ) -> None:
        await converted_input.execute(
            text("UPDATE odds_snapshots SET created_at=snapshot_time+:gap WHERE id=1"), {"gap": gap}
        )
        await step(converted_input, CONVERSION)
        assert (
            await converted_input.execute(text("SELECT time_basis FROM odds_snapshots WHERE id=1"))
        ).scalar_one() == basis

    @pytest.mark.parametrize(
        "mutation,match",
        [
            ("UPDATE odds_snapshots SET raw_data='null'::json WHERE id=17", "not an object"),
            (
                "UPDATE odds_snapshots SET raw_data=jsonb_set(raw_data::jsonb,'{source}','\"unknown\"') WHERE id=17",
                "unsupported source",
            ),
            (
                "UPDATE odds_snapshots SET raw_data=jsonb_set(raw_data::jsonb,'{source}','null') WHERE id=1",
                "unsupported source",
            ),
            (
                "UPDATE odds_snapshots SET raw_data=jsonb_set(raw_data::jsonb,'{bookmakers}','[]') WHERE id=17",
                "missing bookmakers",
            ),
            (
                "UPDATE odds_snapshots SET raw_data=jsonb_set(raw_data::jsonb,'{bookmakers,0,markets,0,outcomes,0,name}','\"Mystery\"') WHERE id=1",
                "parser issues",
            ),
            ("UPDATE odds_snapshots SET time_basis='observed' WHERE id=17", "partial shaped state"),
            ("UPDATE odds_snapshots SET source='betfair' WHERE id=17", "partial shaped state"),
            (
                "INSERT INTO prices(snapshot_id,bookmaker,outcome,side,price) VALUES(17,'betfair','home','back',2)",
                "partial shaped state",
            ),
            (
                "INSERT INTO snapshot_payloads(id,payload) VALUES(99,'{}'); UPDATE odds_snapshots SET payload_id=99 WHERE id=17",
                "partial shaped state",
            ),
            ("INSERT INTO candidate_decisions VALUES(ARRAY[1])", "referenced split"),
            ("UPDATE odds SET market_key='unknown' WHERE snapshot_id=17", "odds market absent"),
        ],
    )
    async def test_refuses_without_writing(
        self, converted_input: AsyncConnection, mutation: str, match: str
    ) -> None:
        c = converted_input
        for sql in mutation.split("; "):
            await c.execute(text(sql))
        before = (await c.execute(text("SELECT count(*) FROM snapshot_payloads"))).scalar_one()
        with pytest.raises(RuntimeError, match=match):
            await step(c, CONVERSION)
        assert (
            await c.execute(text("SELECT count(*) FROM snapshot_payloads"))
        ).scalar_one() == before
        assert (await c.execute(text("SELECT count(*) FROM odds_snapshots"))).scalar_one() == 2
        assert (
            await c.execute(text("SELECT source FROM odds_snapshots WHERE id=1"))
        ).scalar_one() is None

    @pytest.mark.parametrize("shaped", [False, True])
    async def test_refuses_key_collision(
        self, converted_input: AsyncConnection, shaped: bool
    ) -> None:
        c = converted_input
        if shaped:
            await c.execute(text("INSERT INTO snapshot_payloads(id,payload) VALUES(99,'{}')"))
        await c.execute(
            text(
                "INSERT INTO odds_snapshots(event_id,snapshot_time,created_at,raw_data,bookmaker_count,source,market,time_basis,payload_id) SELECT event_id,snapshot_time,created_at,raw_data,bookmaker_count,:source,:market,:basis,:payload FROM odds_snapshots WHERE id=1"
            ),
            {
                "source": "odds_api" if shaped else None,
                "market": "h2h" if shaped else None,
                "basis": "observed" if shaped else None,
                "payload": 99 if shaped else None,
            },
        )
        if shaped:
            await c.execute(
                text(
                    "INSERT INTO prices(snapshot_id,bookmaker,outcome,side,price) SELECT max(id),'fanduel','home','back',2 FROM odds_snapshots"
                )
            )
        with pytest.raises(RuntimeError, match="snapshot-key collision"):
            await step(c, CONVERSION)
        assert (
            await c.execute(text("SELECT count(*) FROM snapshot_payloads"))
        ).scalar_one() == int(shaped)

    async def test_lay_relinking_and_matched_metadata(
        self, legacy_connection: AsyncConnection
    ) -> None:
        data: dict[str, Any] = json.loads(
            Path("tests/fixtures/legacy_snapshots/representative.json").read_text()
        )
        original = data["snapshots"][0]
        book = original["raw_data"]["bookmakers"][0]
        book["betfair_matched"] = {"home": 24.5, "away": 31.25}
        book["markets"][0]["key"] = "h2h_lay"
        for row in data["odds"]:
            if (
                row["snapshot_id"] == 1
                and row["bookmaker_key"] == book["key"]
                and row["market_key"] == "h2h"
            ):
                row["market_key"] = "h2h_lay"
        await seed(legacy_connection, data)
        await step(legacy_connection, CONVERSION)
        assert (
            await legacy_connection.execute(
                text(
                    "SELECT count(*) FROM odds o JOIN odds_snapshots s ON s.id=o.snapshot_id WHERE o.market_key='h2h_lay' AND s.market='h2h'"
                )
            )
        ).scalar_one() == 2
        assert (
            await legacy_connection.execute(
                text(
                    "SELECT matched FROM prices WHERE snapshot_id=1 AND bookmaker='fanduel' AND outcome='home' AND side='lay'"
                )
            )
        ).scalar_one() == Decimal("24.5")
        await step(legacy_connection, DROP)

    async def test_already_converted_is_unchanged(self, converted_input: AsyncConnection) -> None:
        c = converted_input
        await step(c, CONVERSION)
        before = (
            await c.execute(text("SELECT jsonb_agg(to_jsonb(s) ORDER BY id) FROM odds_snapshots s"))
        ).scalar_one()
        await step(c, CONVERSION)
        await step(c, CONVERSION, "downgrade")
        assert (
            await c.execute(text("SELECT jsonb_agg(to_jsonb(s) ORDER BY id) FROM odds_snapshots s"))
        ).scalar_one() == before
        assert (await c.execute(text("SELECT count(*) FROM prices"))).scalar_one() == 21

    async def test_lay_with_draw_relinks_to_three_way(
        self, legacy_connection: AsyncConnection
    ) -> None:
        data: dict[str, Any] = json.loads(
            Path("tests/fixtures/legacy_snapshots/representative.json").read_text()
        )
        original = data["snapshots"][0]
        book = original["raw_data"]["bookmakers"][0]
        market = book["markets"][0]
        market["key"] = "h2h_lay"
        market["outcomes"].append({"name": "Draw", "price": 200})
        book["markets"] = [market]
        original["raw_data"]["bookmakers"] = [book]
        data["snapshots"] = [original]
        data["odds"] = [
            o
            for o in data["odds"]
            if o["snapshot_id"] == 1
            and o["bookmaker_key"] == book["key"]
            and o["market_key"] == "h2h"
        ]
        for row in data["odds"]:
            row["market_key"] = "h2h_lay"
        await seed(legacy_connection, data)
        await legacy_connection.execute(text("INSERT INTO candidate_decisions VALUES(ARRAY[1])"))
        await step(legacy_connection, CONVERSION)
        assert (
            await legacy_connection.execute(text("SELECT market FROM odds_snapshots WHERE id=1"))
        ).scalar_one() == "1x2"
        assert (
            await legacy_connection.execute(text("SELECT count(*) FROM prices WHERE side='lay'"))
        ).scalar_one() == 3
        await step(legacy_connection, DROP)


class TestLegacySnapshotRevisionChain:
    def test_revision_chain(self) -> None:
        assert load(CONVERSION).down_revision == "a6f3c1e9d2b4"
        assert load(DROP).down_revision == load(CONVERSION).revision
