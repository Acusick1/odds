"""Unit tests for bookmaker role definitions."""

from __future__ import annotations

import pytest
from odds_core.bookmakers import (
    NON_BETTABLE_BOOKMAKERS,
    UK_RETAIL_BOOKMAKERS,
    ZERO_MARGIN_BOOKMAKERS,
    canonical_bookmaker,
)
from odds_core.vocabulary import PRICE_SOURCES, Source


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


class TestUkRetailRoster:
    @pytest.mark.parametrize("key", ["casumo", "leovegas", "livescorebet"])
    def test_uk_licensed_odds_api_books_are_retail(self, key: str) -> None:
        assert canonical_bookmaker(Source.ODDS_API, key) == key
        assert key in UK_RETAIL_BOOKMAKERS

    @pytest.mark.parametrize("key", ["virginbet", "grosvenor", "mrgreen"])
    def test_duplicate_and_unlicensed_books_are_not_retail(self, key: str) -> None:
        assert canonical_bookmaker(Source.ODDS_API, key) not in UK_RETAIL_BOOKMAKERS


class TestLegacyBetfairKey:
    def test_odds_api_betfair_is_the_exchange(self) -> None:
        assert canonical_bookmaker(Source.ODDS_API, "betfair") == "betfair_exchange"

    @pytest.mark.parametrize("source", [s for s in PRICE_SOURCES if s is not Source.ODDS_API])
    def test_other_sources_keep_betfair_as_the_sportsbook(self, source: Source) -> None:
        assert canonical_bookmaker(source, "betfair") == "betfair_sportsbook"
