"""Tests for fetch_betfair_exchange job helpers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from odds_lambda.betfair.client import BetfairBook
from odds_lambda.betfair.constants import SPORT_CONFIG
from odds_lambda.jobs.fetch_betfair_exchange import (
    CLOSING_INTERVAL_HOURS,
    FAR_INTERVAL_HOURS,
    PREGAME_INTERVAL_HOURS,
    _interval_for_kickoff,
    _should_skip_book,
)


def _book(
    *,
    inplay: bool = False,
    status: str = "OPEN",
    runners_count: int = 3,
) -> BetfairBook:
    from odds_lambda.betfair.client import BetfairRunner

    runners = [
        BetfairRunner(
            selection_id=i,
            runner_name=f"R{i}",
            best_back=2.0,
            best_lay=2.05,
            back_size=10.0,
            lay_size=10.0,
            last_price_traded=2.0,
        )
        for i in range(runners_count)
    ]
    return BetfairBook(
        market_id="1.x",
        betfair_event_id="x",
        betfair_event_name="A v B",
        market_start_time=datetime(2026, 4, 25, 14, 0, tzinfo=UTC),
        market_status=status,
        inplay=inplay,
        total_matched=1000.0,
        runners=runners,
    )


class TestIntervalForKickoff:
    def test_no_kickoff_returns_far(self) -> None:
        assert _interval_for_kickoff(None) == FAR_INTERVAL_HOURS

    def test_imminent_kickoff_returns_closing(self) -> None:
        now = datetime(2026, 4, 25, 12, 0, tzinfo=UTC)
        ko = now + timedelta(hours=2)  # < 3h => CLOSING
        assert _interval_for_kickoff(ko, now=now) == CLOSING_INTERVAL_HOURS

    def test_pregame_window(self) -> None:
        now = datetime(2026, 4, 25, 12, 0, tzinfo=UTC)
        ko = now + timedelta(hours=6)
        assert _interval_for_kickoff(ko, now=now) == PREGAME_INTERVAL_HOURS

    def test_far_out_kickoff(self) -> None:
        now = datetime(2026, 4, 25, 12, 0, tzinfo=UTC)
        ko = now + timedelta(hours=24)
        assert _interval_for_kickoff(ko, now=now) == FAR_INTERVAL_HOURS


class TestShouldSkipBook:
    def test_inplay_skipped(self) -> None:
        assert _should_skip_book(_book(inplay=True), SPORT_CONFIG["soccer_epl"])

    @pytest.mark.parametrize("status", ["CLOSED", "INACTIVE", "closed"])
    def test_closed_skipped(self, status: str) -> None:
        assert _should_skip_book(_book(status=status), SPORT_CONFIG["soccer_epl"])

    def test_open_3way_kept(self) -> None:
        assert not _should_skip_book(_book(runners_count=3), SPORT_CONFIG["soccer_epl"])

    def test_2way_for_3way_market_skipped(self) -> None:
        # EPL is 3-way h2h, but only 2 runners → skip (corrupt)
        assert _should_skip_book(_book(runners_count=2), SPORT_CONFIG["soccer_epl"])

    def test_2way_for_2way_market_kept(self) -> None:
        assert not _should_skip_book(_book(runners_count=2), SPORT_CONFIG["baseball_mlb"])
