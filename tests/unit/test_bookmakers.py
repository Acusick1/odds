"""Unit tests for bookmaker role definitions."""

from __future__ import annotations

import pytest
from odds_core.bookmakers import (
    NON_BETTABLE_BOOKMAKERS,
    ZERO_MARGIN_BOOKMAKERS,
    is_reference_only_snapshot,
)


class TestBookmakerRoles:
    @pytest.mark.parametrize("key", ["polymarket", "pinnacle"])
    def test_unbettable_venues(self, key: str) -> None:
        assert key in NON_BETTABLE_BOOKMAKERS

    def test_exchanges_are_bettable(self) -> None:
        assert "betfair_exchange" not in NON_BETTABLE_BOOKMAKERS

    def test_bet365_is_bettable(self) -> None:
        assert "bet365" not in NON_BETTABLE_BOOKMAKERS

    def test_pinnacle_keeps_a_margin(self) -> None:
        assert "pinnacle" not in ZERO_MARGIN_BOOKMAKERS
        assert {"polymarket", "betfair_exchange"} <= ZERO_MARGIN_BOOKMAKERS


class TestIsReferenceOnlySnapshot:
    def test_polymarket_only(self) -> None:
        assert is_reference_only_snapshot({"bookmakers": [{"key": "polymarket"}]})

    def test_polymarket_with_retail(self) -> None:
        raw = {"bookmakers": [{"key": "polymarket"}, {"key": "bet365"}]}
        assert not is_reference_only_snapshot(raw)

    def test_exchange_only_is_not_reference_only(self) -> None:
        assert not is_reference_only_snapshot({"bookmakers": [{"key": "betfair_exchange"}]})

    @pytest.mark.parametrize("raw", [None, {}, {"bookmakers": []}])
    def test_empty(self, raw: dict | None) -> None:
        assert not is_reference_only_snapshot(raw)
