"""Unit tests for the devigged price of a book's side, and Polymarket's price."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from odds_analytics.retail_ev import _polymarket_price
from odds_core.book_price import polymarket_price
from odds_core.close import event_close
from odds_core.market_prices import Outcome
from odds_core.quotes import Quote
from odds_core.vocabulary import Market, Source
from odds_lambda.candidate_rules.market_state import _polymarket

from tests.test_helpers import make_nba_event, make_quotes

KICKOFF = datetime(2026, 11, 7, 0, 0, tzinfo=UTC)
EVENT = make_nba_event(1, KICKOFF)
HOME, AWAY = Outcome.HOME, Outcome.AWAY


def polymarket_snapshot(yes: dict[str, float], ask: dict[str, float] | None = None) -> list[Quote]:
    meta: dict[str, dict[str, float]] = {"yes_probabilities": yes}
    if ask is not None:
        meta |= {"best_ask": ask, "best_bid": {o: p - 0.02 for o, p in ask.items()}}
    entry = {
        "key": "polymarket",
        "markets": [{"key": "h2h", "outcomes": []}],
        "polymarket_meta": meta,
    }
    return make_quotes(EVENT, 1, KICKOFF - timedelta(minutes=5), [entry], "polymarket")


class TestPolymarketPrice:
    def test_is_the_devigged_yes_midpoints(self) -> None:
        price = polymarket_price(polymarket_snapshot({"home": 0.6, "away": 0.44}), "h2h")
        assert price is not None and price.source is Source.POLYMARKET
        assert price.fair == pytest.approx({HOME: 0.6 / 1.04, AWAY: 0.44 / 1.04})

    def test_its_overround_is_that_of_the_yes_asks(self) -> None:
        snapshot = polymarket_snapshot(
            {"home": 0.6, "away": 0.44}, ask={"home": 0.62, "away": 0.46}
        )
        price = polymarket_price(snapshot, "h2h")
        assert price is not None and price.overround == pytest.approx(0.08)

    def test_a_missing_midpoint_is_no_price(self) -> None:
        assert polymarket_price(polymarket_snapshot({"home": 0.6}), "h2h") is None

    def test_every_polymarket_reader_takes_its_price(self) -> None:
        snapshot = polymarket_snapshot(
            {"home": 0.58, "away": 0.44}, ask={"home": 0.59, "away": 0.45}
        )
        price = polymarket_price(snapshot, "h2h")
        assert price is not None
        state_price = _polymarket(snapshot, Market.H2H)[None]
        feature_price = _polymarket_price(snapshot, "h2h")
        close = event_close(snapshot, KICKOFF, EVENT.sport_key, "h2h")
        assert feature_price is not None and close is not None
        assert state_price.fair == feature_price.fair == close.fair == price.fair
