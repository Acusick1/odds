"""Unit tests for the Betfair Exchange close of an event."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from odds_core.betfair_close import MAX_EXCHANGE_OVERROUND, ExchangePrice, event_close
from odds_core.market_prices import market_outcomes
from odds_core.models import Event, OddsSnapshot
from odds_core.odds_math import devig_probabilities
from odds_core.snapshot_sources import FdukTimeBasis, SnapshotKind

from tests.test_helpers import make_epl_event, make_tagged_snapshot, three_way_bookmaker

KICKOFF = datetime(2025, 10, 4, 14, 0, tzinfo=UTC)
EVENT_ID = "fduk_ARS_CHE_2025-10-04"
EARLY = (2.2, 3.5, 3.8)
LATE = (2.3, 3.4, 3.6)
FDUK_CLOSE = (2.25, 3.45, 3.7)
WIDE = (1.8, 3.2, 3.2)


def _betfair(
    minutes_before: float,
    prices: tuple[float, float, float],
    inplay: bool = False,
    betfair_start: datetime = KICKOFF,
):
    meta = {
        "inplay": inplay,
        "market_status": "OPEN",
        "market_start_time": betfair_start.isoformat(),
    }
    book = three_way_bookmaker("betfair_exchange", prices, betfair_meta=meta)
    return make_tagged_snapshot(
        EVENT_ID, KICKOFF - timedelta(minutes=minutes_before), [book], "betfair_api"
    )


def _fduk_close(prices: tuple[float, float, float]) -> OddsSnapshot:
    return make_tagged_snapshot(
        EVENT_ID,
        KICKOFF,
        [three_way_bookmaker("betfair_exchange", prices)],
        "football_data_uk",
        FdukTimeBasis.KICKOFF,
    )


def _fair(prices: tuple[float, float, float]) -> dict:
    probs = devig_probabilities(*(1 / p for p in prices))
    return dict(zip(market_outcomes("1x2"), probs, strict=True))


@pytest.fixture
def event() -> Event:
    return make_epl_event(EVENT_ID, KICKOFF)


class TestEventClose:
    def test_last_live_snapshot_in_window_wins(self, event: Event) -> None:
        snapshots = [_betfair(20, EARLY), _betfair(10, LATE), _fduk_close(FDUK_CLOSE)]
        close = event_close(event, snapshots, "1x2")
        assert close is not None
        assert close.kind is SnapshotKind.BETFAIR_LIVE
        assert close.time == KICKOFF - timedelta(minutes=10)
        assert close.fair == pytest.approx(_fair(LATE))

    def test_order_of_snapshots_does_not_matter(self, event: Event) -> None:
        close = event_close(event, [_betfair(10, LATE), _betfair(20, EARLY)], "1x2")
        assert close is not None and close.fair == pytest.approx(_fair(LATE))

    def test_in_play_snapshot_is_skipped(self, event: Event) -> None:
        snapshots = [_betfair(10, LATE), _betfair(1, (5.0, 5.0, 1.5), inplay=True)]
        close = event_close(event, snapshots, "1x2")
        assert close is not None and close.fair == pytest.approx(_fair(LATE))

    def test_latest_pre_play_snapshot_wins_whatever_the_betfair_start(self, event: Event) -> None:
        betfair_start = KICKOFF - timedelta(minutes=25)
        snapshots = [
            _betfair(20, EARLY, betfair_start=betfair_start),
            _betfair(5, LATE, betfair_start=betfair_start),
            _betfair(2, (5.0, 5.0, 1.5), inplay=True, betfair_start=betfair_start),
            _betfair(-5, (2.0, 3.6, 4.4), betfair_start=betfair_start),
        ]
        close = event_close(event, snapshots, "1x2")
        assert close is not None
        assert close.time == KICKOFF - timedelta(minutes=5)
        assert close.fair == pytest.approx(_fair(LATE))

    def test_snapshot_outside_window_falls_back_to_fduk(self, event: Event) -> None:
        snapshots = [_betfair(45, EARLY), _fduk_close(FDUK_CLOSE)]
        close = event_close(event, snapshots, "1x2")
        assert close is not None
        assert close.kind is SnapshotKind.FDUK_CLOSING
        assert close.fair == pytest.approx(_fair(FDUK_CLOSE))

    def test_wide_overround_snapshot_is_skipped(self, event: Event) -> None:
        snapshots = [_betfair(20, EARLY), _betfair(10, WIDE)]
        close = event_close(event, snapshots, "1x2")
        assert close is not None and close.fair == pytest.approx(_fair(EARLY))

    def test_overround_cap_is_a_parameter(self, event: Event) -> None:
        close = event_close(event, [_betfair(10, WIDE)], "1x2", max_exchange_overround=1.0)
        assert close is not None and close.fair == pytest.approx(_fair(WIDE))

    def test_wide_live_and_no_fduk_has_no_close(self, event: Event) -> None:
        assert event_close(event, [_betfair(10, WIDE)], "1x2") is None

    def test_no_exchange_snapshots_has_no_close(self, event: Event) -> None:
        retail = make_tagged_snapshot(
            EVENT_ID,
            KICKOFF - timedelta(minutes=10),
            [three_way_bookmaker("bet365", (2.0, 3.4, 4.0))],
            "oddsportal_live",
        )
        assert event_close(event, [retail], "1x2") is None
        assert event_close(event, [], "1x2") is None

    def test_default_cap_is_the_retail_ev_cap(self) -> None:
        assert MAX_EXCHANGE_OVERROUND == 0.05

    def test_result_is_an_exchange_price(self, event: Event) -> None:
        close = event_close(event, [_betfair(10, LATE)], "1x2")
        assert isinstance(close, ExchangePrice)
        assert sum(close.fair.values()) == pytest.approx(1.0)
