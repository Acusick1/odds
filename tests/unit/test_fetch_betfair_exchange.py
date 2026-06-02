"""Tests for fetch_betfair_exchange job helpers."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest
from odds_lambda.betfair.client import BetfairBook
from odds_lambda.betfair.constants import SPORT_CONFIG
from odds_lambda.fetch_tier import FetchTier
from odds_lambda.jobs.fetch_betfair_exchange import CADENCE, _should_skip_book, main
from odds_lambda.scheduling.jobs import JobContext


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


class TestBetfairCadence:
    """Validate the BFE-specific cadence values (aggressive cheap-API polling)."""

    def test_closing_band(self) -> None:
        assert CADENCE.interval_for(FetchTier.CLOSING) == 0.25

    def test_pregame_band(self) -> None:
        assert CADENCE.interval_for(FetchTier.PREGAME) == 0.5

    def test_far_bands_all_2h(self) -> None:
        assert CADENCE.interval_for(FetchTier.SHARP) == 2.0
        assert CADENCE.interval_for(FetchTier.EARLY) == 2.0
        assert CADENCE.interval_for(FetchTier.OPENING) == 2.0

    def test_no_game_band(self) -> None:
        assert CADENCE.no_game == 2.0


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


class TestBetfairSeasonGate:
    """Verify the season gate skips the Betfair login/fetch when no fixture is near."""

    @staticmethod
    @asynccontextmanager
    async def _noop_alert_context(name: str) -> AsyncIterator[None]:
        yield

    @pytest.mark.asyncio
    async def test_season_gate_skips_fetch_when_no_fixture(self) -> None:
        """Deep offseason (no fixture): pre-schedule fires, login/fetch is skipped."""
        mock_backend = AsyncMock()
        mock_backend.get_backend_name = Mock(return_value="test")

        mock_client_cls = Mock()

        with (
            patch(
                "odds_lambda.jobs.fetch_betfair_exchange.BetfairExchangeClient",
                mock_client_cls,
            ),
            patch(
                "odds_lambda.scheduling.helpers.get_scheduler_backend",
                return_value=mock_backend,
            ),
            patch("odds_lambda.jobs.fetch_betfair_exchange.get_settings") as mock_settings,
            patch(
                "odds_lambda.jobs.fetch_betfair_exchange.job_alert_context",
                side_effect=self._noop_alert_context,
            ),
            patch(
                "odds_lambda.scheduling.decision.get_next_kickoff",
                new_callable=AsyncMock,
                # No upcoming fixture → the season gate stays shut.
                return_value=None,
            ),
        ):
            mock_settings.return_value.betfair.enabled = True
            mock_settings.return_value.betfair.sports = ["soccer_epl"]
            mock_settings.return_value.scheduler.dry_run = False
            mock_settings.return_value.scheduler.lead_days_for = lambda _sport: 7

            await main(JobContext(sport="soccer_epl"))

        # Pre-schedule still fires so the chain survives the offseason.
        assert mock_backend.schedule_next_execution.call_count == 1
        # The Betfair client is never constructed and login never runs.
        mock_client_cls.assert_not_called()
