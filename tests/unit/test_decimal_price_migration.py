"""Tests for the b529d3c1a7e4_store_decimal_prices migration's re-derivation logic."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


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
