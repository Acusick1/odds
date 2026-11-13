"""Unit tests for bookmaker role definitions."""

from __future__ import annotations

import pytest
from odds_core.bookmakers import (
    BOOK_SETS,
    NON_BETTABLE_BOOKMAKERS,
    UK_RETAIL_BOOKMAKERS,
    ZERO_MARGIN_BOOKMAKERS,
    BookSet,
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


class TestBookSets:
    def test_every_named_set_is_defined(self) -> None:
        assert set(BOOK_SETS) == set(BookSet)

    def test_the_full_set_is_the_roster(self) -> None:
        assert BOOK_SETS[BookSet.UK_RETAIL] == UK_RETAIL_BOOKMAKERS

    def test_the_odds_api_set_is_the_twelve_roster_books_its_uk_region_carries(self) -> None:
        books = BOOK_SETS[BookSet.ODDS_API_UK]
        assert len(books) == 12 and books < UK_RETAIL_BOOKMAKERS
        assert not books & {"bet365", "midnite", "betmgm", "betano", "spreadex", "bwin"}

    def test_the_sets_are_read_only(self) -> None:
        with pytest.raises(TypeError):
            BOOK_SETS[BookSet.UK_RETAIL] = frozenset()  # type: ignore[index]


class TestLegacyBetfairKey:
    def test_odds_api_betfair_is_the_exchange(self) -> None:
        assert canonical_bookmaker(Source.ODDS_API, "betfair") == "betfair_exchange"

    @pytest.mark.parametrize("source", [s for s in PRICE_SOURCES if s is not Source.ODDS_API])
    def test_other_sources_keep_betfair_as_the_sportsbook(self, source: Source) -> None:
        assert canonical_bookmaker(source, "betfair") == "betfair_sportsbook"
