"""Unit tests for the Betfair Exchange client's catalogue and book requests."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from odds_lambda.betfair.client import BetfairEvent, BetfairExchangeClient
from odds_lambda.betfair.constants import SPORT_CONFIG

START = datetime(2026, 11, 1, 15, 0, tzinfo=UTC)
EVENT = BetfairEvent("35001", "Arsenal v Chelsea", START, 40)


def _catalogue(market_id: str, market_type: str, runners: list[tuple[int, str]]) -> Any:
    return SimpleNamespace(
        market_id=market_id,
        description=SimpleNamespace(market_type=market_type),
        event=SimpleNamespace(id="35001", name="Arsenal v Chelsea"),
        market_start_time=START,
        runners=[SimpleNamespace(selection_id=i, runner_name=n) for i, n in runners],
    )


def _market_book(market_id: str, selection_ids: list[int]) -> Any:
    def runner(selection_id: int) -> Any:
        offer = SimpleNamespace(price=2.0, size=100.0)
        return SimpleNamespace(
            selection_id=selection_id,
            ex=SimpleNamespace(available_to_back=[offer], available_to_lay=[offer]),
            last_price_traded=2.0,
        )

    return SimpleNamespace(
        market_id=market_id,
        status="OPEN",
        inplay=False,
        total_matched=1000.0,
        runners=[runner(i) for i in selection_ids],
    )


def _client(catalogues: list[Any], books: list[Any]) -> tuple[BetfairExchangeClient, MagicMock]:
    client = BetfairExchangeClient(username="u", password="p", app_key="k")
    betting = MagicMock()
    betting.list_market_catalogue.return_value = catalogues
    betting.list_market_book.return_value = books
    client._trading = MagicMock(betting=betting)
    return client, betting


class TestListBooks:
    def test_epl_requests_match_odds_and_over_under_2_5_in_one_catalogue_call(self) -> None:
        client, betting = _client([], [])

        client.list_books(SPORT_CONFIG["soccer_epl"], [EVENT])

        betting.list_market_catalogue.assert_called_once()
        kwargs = betting.list_market_catalogue.call_args.kwargs
        assert kwargs["filter"]["marketTypeCodes"] == ["MATCH_ODDS", "OVER_UNDER_25"]
        assert "MARKET_DESCRIPTION" in kwargs["market_projection"]

    def test_mlb_requests_match_odds_only(self) -> None:
        client, betting = _client([], [])

        client.list_books(SPORT_CONFIG["baseball_mlb"], [EVENT])

        kwargs = betting.list_market_catalogue.call_args.kwargs
        assert kwargs["filter"]["marketTypeCodes"] == ["MATCH_ODDS"]

    def test_each_book_carries_its_market_type_and_runner_names(self) -> None:
        catalogues = [
            _catalogue("1.1", "MATCH_ODDS", [(1, "Arsenal"), (2, "Chelsea"), (3, "The Draw")]),
            _catalogue(
                "1.2", "OVER_UNDER_25", [(47972, "Under 2.5 Goals"), (47973, "Over 2.5 Goals")]
            ),
        ]
        books = [_market_book("1.1", [1, 2, 3]), _market_book("1.2", [47972, 47973])]
        client, betting = _client(catalogues, books)

        result = client.list_books(SPORT_CONFIG["soccer_epl"], [EVENT])

        assert [(b.market_id, b.market_type) for b in result] == [
            ("1.1", "MATCH_ODDS"),
            ("1.2", "OVER_UNDER_25"),
        ]
        assert [r.runner_name for r in result[1].runners] == ["Under 2.5 Goals", "Over 2.5 Goals"]
        betting.list_market_book.assert_called_once()
        assert betting.list_market_book.call_args.kwargs["market_ids"] == ["1.1", "1.2"]

    def test_a_catalogue_without_a_market_type_is_skipped(self) -> None:
        catalogues = [
            _catalogue("1.1", "MATCH_ODDS", [(1, "Arsenal"), (2, "Chelsea"), (3, "The Draw")]),
            _catalogue(
                "1.2", "OVER_UNDER_25", [(47972, "Under 2.5 Goals"), (47973, "Over 2.5 Goals")]
            ),
        ]
        catalogues[1].description = None
        books = [_market_book("1.1", [1, 2, 3]), _market_book("1.2", [47972, 47973])]
        client, _ = _client(catalogues, books)

        result = client.list_books(SPORT_CONFIG["soccer_epl"], [EVENT])

        assert [b.market_id for b in result] == ["1.1"]
