"""The Betfair and Polymarket collectors check in without doing any work while their sport is dormant."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from odds_lambda.jobs import fetch_betfair_exchange, fetch_polymarket_odds
from odds_lambda.scheduling.decision import DORMANCY_HORIZON_DAYS
from odds_lambda.scheduling.jobs import JobContext

EPL = "soccer_epl"


def _settings() -> MagicMock:
    settings = MagicMock()
    settings.scheduler.dry_run = False
    settings.betfair.enabled = True
    settings.polymarket.enabled = True
    return settings


@pytest.fixture
def heartbeats() -> list[str]:
    return []


@pytest.fixture
def alert_context(heartbeats: list[str]) -> Callable[[str], AbstractAsyncContextManager[None]]:
    @asynccontextmanager
    async def context(name: str) -> AsyncIterator[None]:
        heartbeats.append(name)
        yield

    return context


@pytest.fixture
def booked() -> Iterator[SimpleNamespace]:
    with (
        patch.object(fetch_betfair_exchange, "self_schedule", new_callable=AsyncMock) as betfair,
        patch.object(fetch_polymarket_odds, "self_schedule", new_callable=AsyncMock) as polymarket,
    ):
        yield SimpleNamespace(betfair=betfair, polymarket=polymarket)


class TestBetfairDormancy:
    @pytest.mark.asyncio
    async def test_a_dormant_run_never_logs_in_and_writes_a_heartbeat(
        self,
        booked: SimpleNamespace,
        heartbeats: list[str],
        alert_context: Callable[[str], AbstractAsyncContextManager[None]],
    ) -> None:
        with (
            patch.object(fetch_betfair_exchange, "get_settings", return_value=_settings()),
            patch.object(fetch_betfair_exchange, "job_alert_context", alert_context),
            patch.object(fetch_betfair_exchange, "BetfairExchangeClient") as client,
            patch(
                "odds_lambda.collection_calendar.next_fixture_kickoff",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch.object(
                fetch_betfair_exchange,
                "next_betfair_kickoff",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            await fetch_betfair_exchange.main(JobContext(sport=EPL))

        client.assert_not_called()
        assert heartbeats == ["fetch-betfair-exchange-epl"]
        booked.betfair.assert_awaited_once()
        next_time = booked.betfair.call_args.kwargs["next_time"]
        assert next_time - datetime.now(UTC) < timedelta(
            hours=fetch_betfair_exchange.CADENCE.no_game, minutes=1
        )

    @pytest.mark.asyncio
    async def test_a_fixture_inside_the_horizon_logs_in(
        self,
        booked: SimpleNamespace,
        heartbeats: list[str],
        alert_context: Callable[[str], AbstractAsyncContextManager[None]],
    ) -> None:
        settings = _settings()
        with (
            patch.object(fetch_betfair_exchange, "get_settings", return_value=settings),
            patch.object(fetch_betfair_exchange, "job_alert_context", alert_context),
            patch.object(fetch_betfair_exchange, "resolve_cert_paths", return_value=(None, None)),
            patch.object(fetch_betfair_exchange, "BetfairExchangeClient") as client,
            patch.object(
                fetch_betfair_exchange, "require_rows_for_fixtures", new_callable=AsyncMock
            ),
            patch.object(
                fetch_betfair_exchange,
                "ingest_sport",
                new_callable=AsyncMock,
                return_value=fetch_betfair_exchange.IngestionStats(sport=EPL),
            ),
            patch(
                "odds_lambda.collection_calendar.next_fixture_kickoff",
                new_callable=AsyncMock,
                return_value=datetime.now(UTC) + timedelta(days=DORMANCY_HORIZON_DAYS - 1),
            ),
            patch.object(
                fetch_betfair_exchange,
                "next_betfair_kickoff",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            await fetch_betfair_exchange.main(JobContext(sport=EPL))

        client.assert_called_once()


class TestPolymarketDormancy:
    @pytest.mark.asyncio
    async def test_a_dormant_run_makes_no_gamma_call_and_writes_a_heartbeat(
        self,
        booked: SimpleNamespace,
        heartbeats: list[str],
        alert_context: Callable[[str], AbstractAsyncContextManager[None]],
    ) -> None:
        with (
            patch.object(fetch_polymarket_odds, "get_settings", return_value=_settings()),
            patch.object(fetch_polymarket_odds, "job_alert_context", alert_context),
            patch.object(fetch_polymarket_odds, "ingest_sport", new_callable=AsyncMock) as ingest,
            patch(
                "odds_lambda.scheduling.decision.get_next_kickoff",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            await fetch_polymarket_odds.main(JobContext(sport=EPL))

        ingest.assert_not_called()
        assert heartbeats == ["fetch-polymarket-odds-epl"]
        booked.polymarket.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_an_event_inside_the_horizon_ingests(
        self,
        booked: SimpleNamespace,
        heartbeats: list[str],
        alert_context: Callable[[str], AbstractAsyncContextManager[None]],
    ) -> None:
        with (
            patch.object(fetch_polymarket_odds, "get_settings", return_value=_settings()),
            patch.object(fetch_polymarket_odds, "job_alert_context", alert_context),
            patch.object(
                fetch_polymarket_odds,
                "ingest_sport",
                new_callable=AsyncMock,
                return_value=fetch_polymarket_odds.IngestionStats(sport=EPL),
            ) as ingest,
            patch(
                "odds_lambda.scheduling.decision.get_next_kickoff",
                new_callable=AsyncMock,
                return_value=datetime.now(UTC) + timedelta(days=DORMANCY_HORIZON_DAYS - 1),
            ),
        ):
            await fetch_polymarket_odds.main(JobContext(sport=EPL))

        ingest.assert_awaited_once()
