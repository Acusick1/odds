"""Tests for the b529d3c1a7e4_store_decimal_prices migration's re-derivation logic."""

from __future__ import annotations

import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from odds_core.models import Event, EventStatus
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


def _load_migration() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[2]
        / "migrations"
        / "versions"
        / "b529d3c1a7e4_store_decimal_prices.py"
    )
    spec = importlib.util.spec_from_file_location("b529d3c1a7e4_migration", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def migration() -> ModuleType:
    return _load_migration()


def _legacy_oddsportal_american(decimal_odds: float) -> int:
    if decimal_odds >= 2.0:
        return round((decimal_odds - 1) * 100)
    return round(-100 / (decimal_odds - 1))


class TestOddsPortalPreimages:
    def test_every_two_decimal_price_round_trips(self, migration: ModuleType) -> None:
        preimages = migration._oddsportal_preimages()
        for cents in range(102, 10001):
            decimal_odds = round(1 + cents / 100, 2)
            assert preimages[_legacy_oddsportal_american(decimal_odds)] == decimal_odds

    def test_sentinel_decodes_like_any_price(self, migration: ModuleType) -> None:
        assert migration._oddsportal_preimages()[migration.INVALID_PRICE_SENTINEL] == 1.01

    def test_rewrite_replaces_decodable_prices_only(self, migration: ModuleType) -> None:
        raw_data = {
            "source": "oddsportal_live",
            "bookmakers": [
                {
                    "markets": [
                        {
                            "outcomes": [
                                {"name": "A", "price": -110},
                                {"name": "B", "price": 2000},
                            ]
                        }
                    ]
                }
            ],
        }
        changed, quarantined = migration._rewrite_oddsportal(
            raw_data, migration._oddsportal_preimages()
        )
        assert changed
        assert quarantined == []
        outcomes = raw_data["bookmakers"][0]["markets"][0]["outcomes"]
        assert outcomes[0] == {"name": "A", "price_decimal": 1.91}
        assert outcomes[1] == {"name": "B", "price_decimal": 21.0}

    def test_garbage_market_is_removed(self, migration: ModuleType) -> None:
        raw_data = {
            "bookmakers": [
                {
                    "key": "b",
                    "markets": [
                        {"key": "1x2", "outcomes": [{"name": n, "price": -10000} for n in "ABC"]}
                    ],
                }
            ]
        }
        changed, quarantined = migration._rewrite_oddsportal(
            raw_data, migration._oddsportal_preimages()
        )
        assert changed
        assert raw_data["bookmakers"] == []
        assert len(quarantined) == 3


def migration_truncated(decimal_odds: float) -> int:
    return _load_migration()._truncated_american(decimal_odds)


class TestBetfairRederivation:
    @staticmethod
    def _bookmaker(best_backs: list[float | None]) -> dict[str, Any]:
        return {
            "markets": [
                {
                    "outcomes": [
                        {"name": f"o{i}", "price": migration_truncated(back)}
                        for i, back in enumerate(best_backs)
                        if back is not None
                    ]
                }
            ],
            "betfair_meta": {"runners": [{"best_back": back} for back in best_backs]},
        }

    def test_outcomes_take_runner_best_back(self, migration: ModuleType) -> None:
        raw_data = {"bookmakers": [self._bookmaker([1.64, None, 5.7])]}
        assert migration._rewrite_betfair(raw_data)
        outcomes = raw_data["bookmakers"][0]["markets"][0]["outcomes"]
        assert [o["price_decimal"] for o in outcomes] == [1.64, 5.7]
        assert all("price" not in o for o in outcomes)

    def test_runner_that_does_not_truncate_to_stored_price_is_left_alone(
        self, migration: ModuleType
    ) -> None:
        raw_data = {"bookmakers": [self._bookmaker([1.64])]}
        raw_data["bookmakers"][0]["markets"][0]["outcomes"][0]["price"] = -150
        assert not migration._rewrite_betfair(raw_data)

    def test_misaligned_runners_are_left_alone(self, migration: ModuleType) -> None:
        bookmaker = self._bookmaker([1.64, 5.7])
        bookmaker["betfair_meta"]["runners"].append({"best_back": 3.0})
        raw_data = {"bookmakers": [bookmaker]}
        assert not migration._rewrite_betfair(raw_data)

    def test_restore_american_inverts_to_nearest_quote(self, migration: ModuleType) -> None:
        raw_data = {
            "bookmakers": [{"markets": [{"outcomes": [{"name": "a", "price_decimal": 1.64}]}]}]
        }
        assert migration._restore_american(raw_data)
        assert raw_data["bookmakers"][0]["markets"][0]["outcomes"] == [{"name": "a", "price": -156}]


def _snapshot(source: str | None, bookmaker: str, market: str, outcomes: list[dict]) -> dict:
    raw: dict = {
        "bookmakers": [{"key": bookmaker, "markets": [{"key": market, "outcomes": outcomes}]}]
    }
    if source is not None:
        raw["source"] = source
    return raw


class TestUpgradeAndDowngrade:
    MOMENT = datetime(2026, 10, 18, 12, 0, tzinfo=UTC)

    @staticmethod
    def _run(connection: Any, migration: ModuleType, step: str) -> None:
        with Operations.context(MigrationContext.configure(connection)):
            getattr(migration, step)()

    @pytest.fixture
    async def legacy_session(self, pglite_async_session: AsyncSession) -> AsyncSession:
        session = pglite_async_session
        for statement in (
            "ALTER TABLE odds DROP CONSTRAINT ck_odds_price_decimal_gt_one",
            "ALTER TABLE odds DROP COLUMN price_decimal",
            "ALTER TABLE odds ADD COLUMN price integer NOT NULL DEFAULT 0",
            "ALTER TABLE odds ALTER COLUMN price DROP DEFAULT",
            # The migration's SQL aliases a JSON element as ``market``, which a later
            # odds_snapshots.market column would shadow.
            "ALTER TABLE odds_snapshots DROP COLUMN market",
        ):
            await session.execute(text(statement))
        session.add(
            Event(
                id="e1",
                sport_key="soccer_epl",
                sport_title="EPL",
                commence_time=self.MOMENT,
                home_team="A",
                away_team="B",
                status=EventStatus.SCHEDULED,
            )
        )
        await session.flush()
        return session

    async def _add(
        self, session: AsyncSession, raw: dict, rows: list[tuple[str, str, str, int]]
    ) -> None:
        await session.execute(
            text(
                "INSERT INTO odds_snapshots (event_id, snapshot_time, raw_data, bookmaker_count, "
                "hours_until_commence, created_at) "
                "VALUES ('e1', :t, CAST(:raw AS json), :n, 1.0, now())"
            ),
            {"t": self.MOMENT, "raw": json.dumps(raw), "n": len(raw["bookmakers"])},
        )
        for bookmaker, market, name, price in rows:
            await session.execute(
                text(
                    "INSERT INTO odds (event_id, bookmaker_key, bookmaker_title, market_key, "
                    "outcome_name, price, odds_timestamp, last_update, created_at, is_valid) "
                    "VALUES ('e1', :b, :b, :m, :n, :p, :t, :t, now(), true)"
                ),
                {"b": bookmaker, "m": market, "n": name, "p": price, "t": self.MOMENT},
            )

    async def test_rows_take_their_own_snapshot_price_and_garbage_is_removed(
        self, migration: ModuleType, legacy_session: AsyncSession
    ) -> None:
        session = legacy_session
        # Two scrapes of the same event-time: the same keys carry different prices.
        await self._add(
            session,
            _snapshot("oddsportal_live", "7bet", "h2h", [{"name": "A", "price": -167}]),
            [("7bet", "h2h", "A", -167)],
        )
        await self._add(
            session,
            _snapshot("oddsportal_live", "7bet", "h2h", [{"name": "A", "price": -141}]),
            [("7bet", "h2h", "A", -141)],
        )
        betfair = _snapshot(
            "betfair_api", "betfair_exchange", "1x2", [{"name": "A", "price": -156}]
        )
        betfair["bookmakers"][0]["betfair_meta"] = {"runners": [{"best_back": 1.64}]}
        await self._add(session, betfair, [("betfair_exchange", "1x2", "A", -156)])
        await self._add(
            session,
            _snapshot(
                "oddsportal_live",
                "bet365",
                "h2h",
                [{"name": "A", "price": -10000}, {"name": "B", "price": 2000}],
            ),
            [("bet365", "h2h", "A", -10000), ("bet365", "h2h", "B", 2000)],
        )
        await self._add(
            session,
            _snapshot(
                "oddsportal_live",
                "junk",
                "1x2",
                [{"name": n, "price": -10000} for n in ("A", "Draw", "B")],
            ),
            [("junk", "1x2", n, -10000) for n in ("A", "Draw", "B")],
        )
        await self._add(
            session,
            _snapshot(None, "fanduel", "1x2", [{"name": "A", "price": -110}]),
            [("fanduel", "1x2", "A", -110)],
        )
        await session.flush()

        connection = await session.connection()
        await connection.run_sync(lambda c: self._run(c, migration, "upgrade"))

        rows = (
            await session.execute(
                text("SELECT bookmaker_key, outcome_name, price_decimal FROM odds ORDER BY id")
            )
        ).all()
        assert [(b, n, pytest.approx(d, abs=1e-6)) for b, n, d in rows] == [
            ("7bet", "A", 1.6),
            ("7bet", "A", 1.71),
            ("betfair_exchange", "A", 1.64),
            ("bet365", "A", 1.01),
            ("bet365", "B", 21.0),
            ("fanduel", "A", pytest.approx(1.90909, abs=1e-5)),
        ]
        junk = (
            await session.execute(
                text(
                    "SELECT count(*) FROM odds_snapshots WHERE CAST(raw_data AS text) LIKE '%junk%'"
                )
            )
        ).scalar_one()
        assert junk == 0
        remaining = (
            (await session.execute(text("SELECT bookmaker_count FROM odds_snapshots ORDER BY id")))
            .scalars()
            .all()
        )
        assert remaining[4] == 0

        await connection.run_sync(lambda c: self._run(c, migration, "downgrade"))
        prices = (await session.execute(text("SELECT price FROM odds ORDER BY id"))).scalars().all()
        assert prices == [-167, -141, -156, -10000, 2000, -110]
