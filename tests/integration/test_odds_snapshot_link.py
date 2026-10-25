"""Tests for odds.snapshot_id: the b6e3d9f2a4c7 backfill and the readers that set it (#535)."""

from __future__ import annotations

import importlib.util
import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from odds_core.models import Event, EventStatus, Odds, OddsSnapshot
from odds_core.snapshot_utils import extract_odds_from_snapshot
from odds_lambda.storage.writers import OddsWriter
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

MOMENT = datetime(2026, 10, 18, 12, 0, tzinfo=UTC)


def _load_migration() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[2]
        / "migrations"
        / "versions"
        / "b6e3d9f2a4c7_add_odds_snapshot_id.py"
    )
    spec = importlib.util.spec_from_file_location("b6e3d9f2a4c7_migration", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _raw(bookmaker: str, market: str, prices: dict[str, float], point: float | None = None) -> dict:
    outcomes: list[dict[str, Any]] = [
        {"name": name, "price_decimal": price} | ({"point": point} if point is not None else {})
        for name, price in prices.items()
    ]
    return {
        "source": "oddsportal_live",
        "bookmakers": [
            {
                "key": bookmaker,
                "title": bookmaker,
                "markets": [{"key": market, "outcomes": outcomes}],
            }
        ],
    }


class TestBackfill:
    @staticmethod
    def _upgrade(connection: Any) -> None:
        with Operations.context(MigrationContext.configure(connection)):
            _load_migration().upgrade()

    @pytest.fixture
    async def legacy_session(self, pglite_async_session: AsyncSession) -> AsyncSession:
        session = pglite_async_session
        await session.execute(text("ALTER TABLE odds DROP COLUMN snapshot_id"))
        session.add(
            Event(
                id="e1",
                sport_key="baseball_mlb",
                sport_title="MLB",
                commence_time=MOMENT + timedelta(hours=2),
                home_team="A",
                away_team="B",
                status=EventStatus.SCHEDULED,
            )
        )
        await session.flush()
        return session

    async def _snapshot(self, session: AsyncSession, at: datetime, raw: dict) -> int:
        return (
            await session.execute(
                text(
                    "INSERT INTO odds_snapshots (event_id, snapshot_time, raw_data,"
                    " bookmaker_count, created_at) VALUES ('e1', :t, :r, 1, now()) RETURNING id"
                ),
                {"t": at, "r": json.dumps(raw)},
            )
        ).scalar_one()

    async def _row(
        self,
        session: AsyncSession,
        at: datetime,
        bookmaker: str,
        market: str,
        outcome: str,
        price: float,
        point: float | None = None,
    ) -> int:
        return (
            await session.execute(
                text(
                    "INSERT INTO odds (event_id, bookmaker_key, bookmaker_title, market_key,"
                    " outcome_name, price_decimal, point, odds_timestamp, last_update,"
                    " created_at, is_valid) VALUES ('e1', :b, :b, :m, :n, :p, :pt, :t, :t,"
                    " now(), true) RETURNING id"
                ),
                {"b": bookmaker, "m": market, "n": outcome, "p": price, "pt": point, "t": at},
            )
        ).scalar_one()

    async def _links(self, session: AsyncSession) -> dict[int, int | None]:
        rows = await session.execute(text("SELECT id, snapshot_id FROM odds"))
        return dict(rows.tuples().all())

    async def test_sole_snapshot_and_unique_key_link_and_shared_key_stays_null(
        self, legacy_session: AsyncSession
    ) -> None:
        session = legacy_session
        sole_at = MOMENT
        split_at = MOMENT + timedelta(minutes=5)
        clash_at = MOMENT + timedelta(minutes=10)
        sole = await self._snapshot(session, sole_at, _raw("bet365", "h2h", {"A": 1.8}))
        sole_row = await self._row(session, sole_at, "bet365", "h2h", "A", 1.8)
        # One scrape stored per market at one time: each key is quoted by one snapshot.
        h2h = await self._snapshot(session, split_at, _raw("bet365", "h2h", {"A": 1.9}))
        totals = await self._snapshot(
            session, split_at, _raw("bet365", "totals", {"Over": 1.95}, point=8.5)
        )
        h2h_row = await self._row(session, split_at, "bet365", "h2h", "A", 1.9)
        totals_row = await self._row(session, split_at, "bet365", "totals", "Over", 1.95, 8.5)
        # Two games on one event: both snapshots quote 7bet A.
        await self._snapshot(session, clash_at, _raw("7bet", "h2h", {"A": 1.6}))
        await self._snapshot(session, clash_at, _raw("7bet", "h2h", {"A": 1.71}))
        clash_rows = [
            await self._row(session, clash_at, "7bet", "h2h", "A", price) for price in (1.6, 1.71)
        ]

        connection = await session.connection()
        await connection.run_sync(self._upgrade)

        links = await self._links(session)
        assert links[sole_row] == sole
        assert links[h2h_row] == h2h
        assert links[totals_row] == totals
        assert [links[r] for r in clash_rows] == [None, None]

    async def test_sole_snapshot_that_does_not_quote_the_row_leaves_it_null(
        self, legacy_session: AsyncSession
    ) -> None:
        session = legacy_session
        # Only the totals snapshot of this scrape survived; the 1x2 row is an orphan.
        await self._snapshot(session, MOMENT, _raw("bet365", "totals", {"Over": 1.9}, point=2.5))
        orphan = await self._row(session, MOMENT, "bet365", "1x2", "A", 2.0)

        connection = await session.connection()
        await connection.run_sync(self._upgrade)

        assert (await self._links(session))[orphan] is None

    async def test_bookmaker_key_is_trimmed_as_the_writer_stored_it(
        self, legacy_session: AsyncSession
    ) -> None:
        session = legacy_session
        snapshot = await self._snapshot(session, MOMENT, _raw(" bet365\t", "h2h", {"A": 1.8}))
        row = await self._row(session, MOMENT, "bet365", "h2h", "A", 1.8)

        connection = await session.connection()
        await connection.run_sync(self._upgrade)

        assert (await self._links(session))[row] == snapshot

    async def test_key_matches_on_point(self, legacy_session: AsyncSession) -> None:
        session = legacy_session
        low = await self._snapshot(
            session, MOMENT, _raw("bet365", "totals", {"Over": 1.9}, point=8.5)
        )
        high = await self._snapshot(
            session, MOMENT, _raw("bet365", "totals", {"Over": 2.1}, point=9.5)
        )
        low_row = await self._row(session, MOMENT, "bet365", "totals", "Over", 1.9, 8.5)
        high_row = await self._row(session, MOMENT, "bet365", "totals", "Over", 2.1, 9.5)

        connection = await session.connection()
        await connection.run_sync(self._upgrade)

        links = await self._links(session)
        assert (links[low_row], links[high_row]) == (low, high)


class TestExtractedOddsCarryTheirSnapshot:
    def test_extract_sets_the_snapshot_id(self) -> None:
        snapshot = OddsSnapshot(
            id=42,
            event_id="e1",
            snapshot_time=MOMENT,
            raw_data=_raw("bet365", "h2h", {"A": 1.8, "B": 2.0}),
            bookmaker_count=1,
        )

        odds = extract_odds_from_snapshot(snapshot, "e1")

        assert [o.snapshot_id for o in odds] == [42, 42]


class TestEveryWriterLinksItsRows:
    """New odds rows always name their snapshot; only legacy orphans are NULL."""

    KNOWN_WRITERS = {
        Path("packages/odds-core/odds_core/snapshot_utils.py"),
        Path("packages/odds-lambda/odds_lambda/storage/writers.py"),
        Path("packages/odds-lambda/odds_lambda/copy_from_production.py"),
    }

    def test_no_other_code_builds_odds_rows(self) -> None:
        root = Path(__file__).resolve().parents[2]
        builds_odds = re.compile(r"(?<![\w.])(?<!class )Odds\(|insert\(Odds\)")
        found = {
            path.relative_to(root)
            for top in ("packages", "scripts")
            for path in (root / top).rglob("*.py")
            if "/tests/" not in str(path) and builds_odds.search(path.read_text())
        }
        assert found == self.KNOWN_WRITERS

    @pytest.mark.parametrize(
        "raw",
        [
            _raw("bet365", "h2h", {"A": 1.8, "B": 2.0}),
            {**_raw("betfair_exchange", "h2h", {"A": 1.8}), "source": "betfair_api"},
            {**_raw("polymarket", "h2h", {"A": 1.8}), "source": "polymarket"},
            {
                "bookmakers": [
                    {
                        "key": "fanduel",
                        "title": "FanDuel",
                        "markets": [{"key": "h2h", "outcomes": [{"name": "A", "price": -120}]}],
                    }
                ]
            },
        ],
    )
    async def test_store_odds_snapshot_links_every_row(
        self, pglite_async_session: AsyncSession, raw: dict
    ) -> None:
        session = pglite_async_session
        session.add(
            Event(
                id="e1",
                sport_key="baseball_mlb",
                sport_title="MLB",
                commence_time=MOMENT + timedelta(hours=2),
                home_team="A",
                away_team="B",
                status=EventStatus.SCHEDULED,
            )
        )
        await session.flush()

        snapshot, _ = await OddsWriter(session).store_odds_snapshot(
            "e1", raw, snapshot_time=MOMENT, validate=False
        )
        await session.commit()

        links = (await session.execute(select(Odds.snapshot_id))).scalars().all()
        assert links and set(links) == {snapshot.id}
