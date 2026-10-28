"""Unit tests for bookmaker role definitions."""

from __future__ import annotations

import pytest
from odds_core.bookmakers import (
    NON_BETTABLE_BOOKMAKERS,
    ZERO_MARGIN_BOOKMAKERS,
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
