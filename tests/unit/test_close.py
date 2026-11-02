"""Unit tests for the Betfair Exchange close of an event."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from itertools import count

import pytest
from odds_core.close import CLOSES, close_grace, close_spec, event_close, pick_close
from odds_core.close import FDUK_CLOSE as FDUK_CLOSE_SPEC
from odds_core.exchange_price import MAX_EXCHANGE_OVERROUND, ExchangePrice, exchange_price
from odds_core.market_prices import market_outcomes
from odds_core.models import Event
from odds_core.odds_math import devig_probabilities
from odds_core.quotes import Quote
from odds_core.vocabulary import Market, PriceOutcome, Side, Source, TimeBasis

from tests.test_helpers import (
    betfair_exchange_bookmaker,
    make_epl_event,
    make_nba_event,
    make_quotes,
    three_way_bookmaker,
)

KICKOFF = datetime(2025, 10, 4, 14, 0, tzinfo=UTC)
EVENT = make_epl_event(1, KICKOFF)
_ids = count(1)
EARLY = (2.2, 3.5, 3.8)
LATE = (2.3, 3.4, 3.6)
FDUK_CLOSE = (2.25, 3.45, 3.7)
WIDE = (1.8, 3.2, 3.2)


def _betfair(
    minutes_before: float,
    prices: tuple[float, float, float],
    inplay: bool = False,
    betfair_start: datetime = KICKOFF,
) -> list[Quote]:
    book = betfair_exchange_bookmaker(
        prices, inplay=inplay, market_start_time=betfair_start.isoformat()
    )
    return make_quotes(
        EVENT, next(_ids), KICKOFF - timedelta(minutes=minutes_before), [book], "betfair_api"
    )


def _fduk_close(prices: tuple[float, float, float]) -> list[Quote]:
    return make_quotes(
        EVENT,
        next(_ids),
        KICKOFF,
        [three_way_bookmaker("betfair_exchange", prices)],
        "football_data_uk",
        "kickoff",
    )


def _close(*snapshots: list[Quote], **kwargs: float) -> ExchangePrice | None:
    return event_close([q for s in snapshots for q in s], KICKOFF, "soccer_epl", "1x2", **kwargs)


def _fair(prices: tuple[float, float, float]) -> dict:
    probs = devig_probabilities(*(1 / p for p in prices))
    return dict(zip(market_outcomes("1x2"), probs, strict=True))


class TestEventClose:
    def test_last_live_snapshot_in_window_wins(self) -> None:
        close = _close(_betfair(20, EARLY), _betfair(10, LATE), _fduk_close(FDUK_CLOSE))
        assert close is not None
        assert (close.source, close.time_basis) == (Source.BETFAIR, TimeBasis.OBSERVED)
        assert close.time == KICKOFF - timedelta(minutes=10)
        assert close.fair == pytest.approx(_fair(LATE))

    def test_order_of_snapshots_does_not_matter(self) -> None:
        close = _close(_betfair(10, LATE), _betfair(20, EARLY))
        assert close is not None and close.fair == pytest.approx(_fair(LATE))

    def test_in_play_snapshot_is_skipped(self) -> None:
        close = _close(_betfair(10, LATE), _betfair(1, (5.0, 5.0, 1.5), inplay=True))
        assert close is not None and close.fair == pytest.approx(_fair(LATE))

    def test_latest_pre_play_snapshot_wins_whatever_the_betfair_start(self) -> None:
        betfair_start = KICKOFF - timedelta(minutes=25)
        close = _close(
            _betfair(20, EARLY, betfair_start=betfair_start),
            _betfair(5, LATE, betfair_start=betfair_start),
            _betfair(2, (5.0, 5.0, 1.5), inplay=True, betfair_start=betfair_start),
            _betfair(-5, (2.0, 3.6, 4.4), betfair_start=betfair_start),
        )
        assert close is not None
        assert close.time == KICKOFF - timedelta(minutes=5)
        assert close.fair == pytest.approx(_fair(LATE))

    def test_snapshot_outside_window_falls_back_to_fduk(self) -> None:
        close = _close(_betfair(45, EARLY), _fduk_close(FDUK_CLOSE))
        assert close is not None
        assert (close.source, close.time_basis) == (Source.FOOTBALL_DATA_UK, TimeBasis.CLOSE)
        assert close.fair == pytest.approx(_fair(FDUK_CLOSE))

    def test_wide_overround_snapshot_is_skipped(self) -> None:
        close = _close(_betfair(20, EARLY), _betfair(10, WIDE))
        assert close is not None and close.fair == pytest.approx(_fair(EARLY))

    def test_overround_cap_is_a_parameter(self) -> None:
        close = _close(_betfair(10, WIDE), max_exchange_overround=1.0)
        assert close is not None and close.fair == pytest.approx(_fair(WIDE))

    def test_wide_live_and_no_fduk_has_no_close(self) -> None:
        assert _close(_betfair(10, WIDE)) is None

    def test_no_exchange_snapshots_has_no_close(self) -> None:
        retail = make_quotes(
            EVENT,
            next(_ids),
            KICKOFF - timedelta(minutes=10),
            [three_way_bookmaker("bet365", (2.0, 3.4, 4.0))],
            "oddsportal_live",
        )
        assert _close(retail) is None
        assert _close() is None

    def test_live_exchange_in_a_retail_snapshot_is_not_a_close(self) -> None:
        oddsportal = make_quotes(
            EVENT,
            next(_ids),
            KICKOFF - timedelta(minutes=10),
            [betfair_exchange_bookmaker(LATE)],
            "oddsportal_live",
        )
        assert _close(oddsportal) is None

    def test_default_cap_is_the_retail_ev_cap(self) -> None:
        assert MAX_EXCHANGE_OVERROUND == 0.05

    def test_result_is_an_exchange_price(self) -> None:
        close = _close(_betfair(10, LATE))
        assert isinstance(close, ExchangePrice)
        assert sum(close.fair.values()) == pytest.approx(1.0)


class TestExchangePrice:
    def test_needs_a_back_price_for_every_outcome(self) -> None:
        assert exchange_price(_betfair(10, LATE)[:2], "1x2") is None
        assert exchange_price([], "1x2") is None

    def test_reads_only_exchange_backs(self) -> None:
        quotes = _betfair(10, LATE)
        price = exchange_price(quotes, "1x2")
        assert price is not None and price.fair == pytest.approx(_fair(LATE))
        assert price.overround == pytest.approx(sum(1 / p for p in LATE) - 1)


def _totals(minutes_before: float, lines: dict[float, tuple[float, float]]) -> list[Quote]:
    """A live Betfair snapshot's totals backs, (over, under) per line."""
    [template, *_] = _betfair(minutes_before, LATE)
    return [
        replace(template, market=Market.TOTALS, outcome=outcome, line=line, price=price)
        for line, (over, under) in lines.items()
        for outcome, price in ((PriceOutcome.OVER, over), (PriceOutcome.UNDER, under))
    ]


class TestLineClose:
    def test_each_line_has_its_own_close(self) -> None:
        quoted = _totals(10, {2.5: (2.0, 2.0), 3.5: (4.0, 4.0 / 3)})
        on_2_5 = event_close(quoted, KICKOFF, "soccer_epl", "totals", line=2.5)
        on_3_5 = event_close(quoted, KICKOFF, "soccer_epl", "totals", line=3.5)
        assert on_2_5 is not None and on_2_5.fair[PriceOutcome.OVER] == pytest.approx(0.5)
        assert on_3_5 is not None and on_3_5.fair[PriceOutcome.OVER] == pytest.approx(0.25)

    def test_the_last_snapshot_pricing_the_line_wins(self) -> None:
        quoted = _totals(20, {2.5: (2.0, 2.0)}) + _totals(10, {3.5: (4.0, 4.0 / 3)})
        close = event_close(quoted, KICKOFF, "soccer_epl", "totals", line=2.5)
        assert close is not None and close.fair[PriceOutcome.OVER] == pytest.approx(0.5)

    def test_a_line_without_prices_has_no_close(self) -> None:
        assert (
            event_close(_totals(10, {2.5: (2.0, 2.0)}), KICKOFF, "soccer_epl", "totals", line=1.5)
            is None
        )

    def test_a_one_sided_line_has_no_price(self) -> None:
        quoted = [q for q in _totals(10, {2.5: (2.0, 2.0)}) if q.outcome is PriceOutcome.OVER]
        assert exchange_price(quoted, "totals", 2.5) is None


def _betfair_totals_envelope(minutes_before: float, over: float, under: float) -> list[Quote]:
    """The quotes of a Betfair job ``OVER_UNDER_25`` entry, read from its runners."""
    runners = [
        {"runner_name": "Under 2.5 Goals", "best_back": under},
        {"runner_name": "Over 2.5 Goals", "best_back": over},
    ]
    entry = {
        "key": "betfair_exchange",
        "markets": [
            {
                "key": "totals",
                "outcomes": [
                    {"name": "Over", "point": 2.5, "price_decimal": over},
                    {"name": "Under", "point": 2.5, "price_decimal": under},
                ],
            }
        ],
        "betfair_meta": {"inplay": False, "market_status": "OPEN", "runners": runners},
    }
    return make_quotes(
        EVENT, next(_ids), KICKOFF - timedelta(minutes=minutes_before), [entry], "betfair_api"
    )


class TestBetfairTotalsClose:
    def test_the_betfair_job_totals_entry_closes_on_line_2_5(self) -> None:
        quoted = _betfair_totals_envelope(20, 2.1, 1.9) + _betfair_totals_envelope(5, 2.0, 2.0)

        close = event_close(quoted, KICKOFF, "soccer_epl", "totals", line=2.5)

        assert close is not None
        assert close.time == KICKOFF - timedelta(minutes=5)
        assert close.fair[PriceOutcome.OVER] == pytest.approx(0.5)

    def test_a_two_way_back_book_wider_than_the_cap_is_not_a_close(self) -> None:
        # 1/1.9 + 1/1.9 - 1 = 5.3%, over the 5% cap; 1/1.95 + 1/1.95 - 1 = 2.6% is kept.
        quoted = _betfair_totals_envelope(20, 1.95, 1.95) + _betfair_totals_envelope(5, 1.9, 1.9)

        close = event_close(quoted, KICKOFF, "soccer_epl", "totals", line=2.5)

        assert close is not None and close.time == KICKOFF - timedelta(minutes=20)


class TestPickClose:
    def test_in_play_price_is_not_the_close(self) -> None:
        price = exchange_price(_betfair(10, LATE, inplay=True), "1x2")
        assert price is not None and price.in_play
        assert pick_close([price], KICKOFF, FDUK_CLOSE_SPEC) is None


NBA_EVENT = make_nba_event(2, KICKOFF)
BETFAIR_NBA = (1.8, 2.1)
POLYMARKET_NBA = (0.55, 0.46)


def _nba_betfair(
    minutes_before: float, prices: tuple[float, float], inplay: bool = False
) -> list[Quote]:
    runners = [
        {"runner_name": name, "best_back": price}
        for name, price in zip((NBA_EVENT.home_team, NBA_EVENT.away_team), prices, strict=True)
    ]
    entry = {
        "key": "betfair_exchange",
        "markets": [{"key": "h2h", "outcomes": []}],
        "betfair_meta": {"inplay": inplay, "market_status": "OPEN", "runners": runners},
    }
    return make_quotes(
        NBA_EVENT, next(_ids), KICKOFF - timedelta(minutes=minutes_before), [entry], "betfair_api"
    )


def _polymarket(
    minutes_before: float, probabilities: tuple[float, float], spread: float = 0.02
) -> list[Quote]:
    home, away = probabilities
    half = spread / 2
    entry = {
        "key": "polymarket",
        "markets": [{"key": "h2h", "outcomes": []}],
        "polymarket_meta": {
            "yes_probabilities": {"home": home, "away": away},
            "best_ask": {"home": home + half, "away": away + half},
            "best_bid": {"home": home - half, "away": away - half},
        },
    }
    return make_quotes(
        NBA_EVENT, next(_ids), KICKOFF - timedelta(minutes=minutes_before), [entry], "polymarket"
    )


def _nba_close(*snapshots: list[Quote], event: Event = NBA_EVENT) -> ExchangePrice | None:
    quoted = [q for s in snapshots for q in s]
    return event_close(quoted, KICKOFF, event.sport_key, "h2h")


def _two_way_fair(prices: tuple[float, float]) -> dict:
    probs = devig_probabilities(*(1 / p for p in prices))
    return dict(zip(market_outcomes("h2h"), probs, strict=True))


class TestNbaClose:
    def test_a_live_betfair_price_in_the_window_is_the_close(self) -> None:
        close = _nba_close(_nba_betfair(5, BETFAIR_NBA), _polymarket(1, POLYMARKET_NBA))
        assert close is not None
        assert close.source is Source.BETFAIR
        assert close.fair == pytest.approx(_two_way_fair(BETFAIR_NBA))

    def test_polymarket_mid_is_the_close_without_a_live_betfair_price(self) -> None:
        close = _nba_close(
            _nba_betfair(45, BETFAIR_NBA),
            _polymarket(10, (0.4, 0.62)),
            _polymarket(2, POLYMARKET_NBA),
        )
        assert close is not None
        assert (close.source, close.time_basis) == (Source.POLYMARKET, TimeBasis.OBSERVED)
        assert close.time == KICKOFF - timedelta(minutes=2)
        assert close.fair == pytest.approx(_two_way_fair((1 / 0.55, 1 / 0.46)))
        assert sum(close.fair.values()) == pytest.approx(1.0)

    def test_an_in_play_betfair_price_falls_back_to_polymarket(self) -> None:
        close = _nba_close(
            _nba_betfair(5, BETFAIR_NBA, inplay=True), _polymarket(3, POLYMARKET_NBA)
        )
        assert close is not None and close.source is Source.POLYMARKET

    def test_a_polymarket_price_outside_the_window_is_not_a_close(self) -> None:
        assert _nba_close(_polymarket(45, POLYMARKET_NBA)) is None

    def test_a_polymarket_price_after_tip_off_is_not_a_close(self) -> None:
        assert _nba_close(_polymarket(-5, POLYMARKET_NBA)) is None

    def test_neither_source_has_no_close(self) -> None:
        assert _nba_close() is None
        assert _nba_close(_nba_betfair(45, BETFAIR_NBA)) is None

    def test_a_polymarket_snapshot_with_a_wide_back_book_is_not_a_close(self) -> None:
        assert _nba_close(_polymarket(2, POLYMARKET_NBA, spread=0.2)) is None

    def test_a_wide_back_book_falls_back_to_an_earlier_narrow_snapshot(self) -> None:
        close = _nba_close(_polymarket(10, (0.4, 0.62)), _polymarket(2, POLYMARKET_NBA, spread=0.2))
        assert close is not None and close.time == KICKOFF - timedelta(minutes=10)

    def test_a_mid_only_snapshot_is_a_close_with_no_liquidity_check(self) -> None:
        mids = [q for q in _polymarket(2, POLYMARKET_NBA, spread=0.2) if q.side is Side.MID]
        close = _nba_close(mids)
        assert close is not None and close.source is Source.POLYMARKET

    def test_polymarket_is_not_the_close_of_epl(self) -> None:
        quoted = _polymarket(2, POLYMARKET_NBA)
        assert event_close(quoted, KICKOFF, "soccer_epl", "h2h") is None


class TestCloseDefinedPerSport:
    def test_epl_and_nba_have_closes(self) -> None:
        assert close_spec("soccer_epl", "1x2") is not None
        assert close_spec("soccer_epl", "totals") is not None
        assert close_spec("basketball_nba", "h2h") is not None

    def test_mlb_h2h_has_no_close(self) -> None:
        mlb = make_nba_event(3, KICKOFF, sport_key="baseball_mlb")
        quoted = _nba_betfair(5, BETFAIR_NBA)
        assert close_spec("baseball_mlb", "h2h") is None
        assert event_close(quoted, KICKOFF, mlb.sport_key, "h2h") is None

    def test_nba_has_no_spreads_or_totals_close(self) -> None:
        assert close_spec("basketball_nba", "totals") is None
        assert close_spec("basketball_nba", "spreads") is None

    def test_only_the_fduk_close_arrives_after_kick_off(self) -> None:
        assert close_grace("soccer_epl", "1x2") == timedelta(days=7)
        assert close_grace("basketball_nba", "h2h") == timedelta(0)
        assert close_grace("baseball_mlb", "h2h") == timedelta(0)

    def test_every_fallback_is_read_from_the_sides_it_names(self) -> None:
        for spec in CLOSES.values():
            assert spec.fallback is None or spec.fallback.side in spec.fallback.sides
