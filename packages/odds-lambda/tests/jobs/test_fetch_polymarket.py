"""Tests for Polymarket live polling job."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from odds_core.polymarket_models import PolymarketMarketType
from odds_lambda.jobs import fetch_polymarket
from odds_lambda.jobs.fetch_polymarket import should_fetch_orderbooks


def _make_settings(
    enabled: bool = True,
    price_poll_interval: int = 300,
    orderbook_tiers: list[str] | None = None,
    collect_moneyline: bool = True,
    collect_spreads: bool = False,
    collect_totals: bool = False,
    collect_player_props: bool = False,
    dry_run: bool = False,
    alert_enabled: bool = False,
) -> MagicMock:
    s = MagicMock()
    s.polymarket.enabled = enabled
    s.polymarket.price_poll_interval = price_poll_interval
    s.polymarket.orderbook_tiers = (
        orderbook_tiers if orderbook_tiers is not None else ["closing", "pregame"]
    )
    s.polymarket.collect_moneyline = collect_moneyline
    s.polymarket.collect_spreads = collect_spreads
    s.polymarket.collect_totals = collect_totals
    s.polymarket.collect_player_props = collect_player_props
    s.scheduler.dry_run = dry_run
    s.alerts.alert_enabled = alert_enabled
    s.scheduler.backend = "local"
    return s


def _make_active_event(
    hours_until: float,
    pm_event_id: str = "event-123",
    title: str = "Lakers vs Celtics",
    event_id: int = 1,
) -> MagicMock:
    return MagicMock(
        id=event_id,
        pm_event_id=pm_event_id,
        title=title,
        start_date=datetime.now(UTC) + timedelta(hours=hours_until),
    )


def _make_market(
    token_id_0: str = "token-0",
    token_id_1: str = "token-1",
    market_type: PolymarketMarketType = PolymarketMarketType.MONEYLINE,
    market_id: int = 1,
) -> MagicMock:
    return MagicMock(
        id=market_id,
        question="Test market question",
        clob_token_ids=[token_id_0, token_id_1],
        market_type=market_type,
    )


@pytest.fixture
def mock_backend() -> MagicMock:
    backend = MagicMock()
    backend.schedule_next_execution = AsyncMock()
    backend.get_backend_name.return_value = "local"
    return backend


@pytest.fixture
def sample_active_event_data() -> dict:
    """Gamma API response for a single active NBA event."""
    return {
        "id": "event-123",
        "ticker": "nba-lal-bos-2024-01-15",
        "slug": "lakers-vs-celtics",
        "title": "Lakers vs Celtics",
        "startDate": "2024-01-15T19:00:00Z",
        "endDate": "2024-01-15T22:00:00Z",
        "active": True,
        "closed": False,
        "volume": "10000",
        "liquidity": "5000",
        "markets": [],
    }


class TestShouldFetchOrderbooks:
    """Unit tests for the tier-gating helper."""

    def test_returns_true_for_tier_in_orderbook_tiers(self):
        settings = _make_settings(orderbook_tiers=["closing", "pregame"])
        assert should_fetch_orderbooks("closing", settings) is True
        assert should_fetch_orderbooks("pregame", settings) is True

    def test_returns_false_for_tier_not_in_orderbook_tiers(self):
        settings = _make_settings(orderbook_tiers=["closing", "pregame"])
        assert should_fetch_orderbooks("early", settings) is False
        assert should_fetch_orderbooks("sharp", settings) is False
        assert should_fetch_orderbooks("opening", settings) is False


class TestFetchPolymarketMain:
    """Integration tests for the main fetch_polymarket job function."""

    @pytest.mark.asyncio
    async def test_early_return_when_polymarket_disabled(self, mock_settings):
        mock_settings.polymarket.enabled = False

        with patch("odds_lambda.jobs.fetch_polymarket.get_settings", return_value=mock_settings):
            await fetch_polymarket.main()

    @pytest.mark.asyncio
    async def test_no_api_events_schedules_from_db(
        self,
        mock_polymarket_client,
        mock_async_session,
        mock_backend,
    ):
        """When API returns no events, scheduling uses DB active events."""
        mock_settings = _make_settings()
        db_event = _make_active_event(hours_until=6)

        mock_polymarket_client.get_nba_events.return_value = []

        mock_reader = AsyncMock()
        mock_reader.get_active_events.return_value = [db_event]

        with (
            patch("odds_lambda.jobs.fetch_polymarket.get_settings", return_value=mock_settings),
            patch(
                "odds_lambda.jobs.fetch_polymarket.PolymarketClient",
                return_value=mock_polymarket_client,
            ),
            patch(
                "odds_lambda.jobs.fetch_polymarket.async_session_maker",
                return_value=mock_async_session,
            ),
            patch(
                "odds_lambda.jobs.fetch_polymarket.PolymarketReader",
                return_value=mock_reader,
            ),
            patch(
                "odds_lambda.jobs.fetch_polymarket.PolymarketWriter",
                return_value=AsyncMock(),
            ),
            patch(
                "odds_lambda.jobs.fetch_polymarket.get_scheduler_backend",
                return_value=mock_backend,
            ),
        ):
            await fetch_polymarket.main()

        mock_backend.schedule_next_execution.assert_called_once()
        next_time = mock_backend.schedule_next_execution.call_args.kwargs["next_time"]
        expected = datetime.now(UTC) + timedelta(seconds=300)
        assert abs((next_time - expected).total_seconds()) < 5

    @pytest.mark.asyncio
    async def test_no_api_events_no_db_events_schedules_daily(
        self,
        mock_polymarket_client,
        mock_async_session,
        mock_backend,
    ):
        """When both API and DB have no events, schedules a daily check."""
        mock_settings = _make_settings()

        mock_polymarket_client.get_nba_events.return_value = []

        mock_reader = AsyncMock()
        mock_reader.get_active_events.return_value = []

        with (
            patch("odds_lambda.jobs.fetch_polymarket.get_settings", return_value=mock_settings),
            patch(
                "odds_lambda.jobs.fetch_polymarket.PolymarketClient",
                return_value=mock_polymarket_client,
            ),
            patch(
                "odds_lambda.jobs.fetch_polymarket.async_session_maker",
                return_value=mock_async_session,
            ),
            patch(
                "odds_lambda.jobs.fetch_polymarket.PolymarketReader",
                return_value=mock_reader,
            ),
            patch(
                "odds_lambda.jobs.fetch_polymarket.PolymarketWriter",
                return_value=AsyncMock(),
            ),
            patch(
                "odds_lambda.jobs.fetch_polymarket.get_scheduler_backend",
                return_value=mock_backend,
            ),
        ):
            await fetch_polymarket.main()

        mock_backend.schedule_next_execution.assert_called_once()
        next_time = mock_backend.schedule_next_execution.call_args.kwargs["next_time"]
        expected = datetime.now(UTC) + timedelta(hours=24)
        assert abs((next_time - expected).total_seconds()) < 5

    @pytest.mark.asyncio
    async def test_happy_path_stores_prices(
        self,
        mock_polymarket_client,
        mock_async_session,
        mock_backend,
        sample_active_event_data,
    ):
        """Happy path: prices fetched and stored for active markets."""
        mock_settings = _make_settings()
        mock_event = _make_active_event(hours_until=6)  # pregame tier
        market = _make_market()

        mock_polymarket_client.get_nba_events.return_value = [sample_active_event_data]
        mock_polymarket_client.get_prices_batch.return_value = {"token-0": 0.55, "token-1": 0.45}

        mock_reader = AsyncMock()
        mock_reader.get_active_events.return_value = [mock_event]
        mock_reader.get_markets_by_type.return_value = [market]

        mock_writer = AsyncMock()
        mock_writer.upsert_event.return_value = mock_event

        with (
            patch("odds_lambda.jobs.fetch_polymarket.get_settings", return_value=mock_settings),
            patch(
                "odds_lambda.jobs.fetch_polymarket.PolymarketClient",
                return_value=mock_polymarket_client,
            ),
            patch(
                "odds_lambda.jobs.fetch_polymarket.async_session_maker",
                return_value=mock_async_session,
            ),
            patch("odds_lambda.jobs.fetch_polymarket.PolymarketReader", return_value=mock_reader),
            patch("odds_lambda.jobs.fetch_polymarket.PolymarketWriter", return_value=mock_writer),
            patch(
                "odds_lambda.jobs.fetch_polymarket.get_scheduler_backend",
                return_value=mock_backend,
            ),
            patch("odds_lambda.jobs.fetch_polymarket.asyncio.sleep"),
        ):
            await fetch_polymarket.main()

        mock_writer.store_price_snapshot.assert_called_once()
        prices_arg = mock_writer.store_price_snapshot.call_args.kwargs["prices"]
        assert prices_arg["outcome_0_price"] == 0.55
        assert prices_arg["outcome_1_price"] == 0.45
        mock_backend.schedule_next_execution.assert_called_once()

    @pytest.mark.asyncio
    async def test_happy_path_stores_orderbooks_in_closing_tier(
        self,
        mock_polymarket_client,
        mock_async_session,
        mock_backend,
        sample_active_event_data,
    ):
        """Order books captured when tier is in orderbook_tiers (closing)."""
        mock_settings = _make_settings(orderbook_tiers=["closing"])
        mock_event = _make_active_event(hours_until=1)  # closing tier (< 3h)
        market = _make_market()

        mock_polymarket_client.get_nba_events.return_value = [sample_active_event_data]
        mock_polymarket_client.get_prices_batch.return_value = {"token-0": 0.55, "token-1": 0.45}
        mock_polymarket_client.get_order_book.return_value = {
            "bids": [{"price": "0.54", "size": "100"}],
            "asks": [{"price": "0.56", "size": "100"}],
        }

        mock_reader = AsyncMock()
        mock_reader.get_active_events.return_value = [mock_event]
        mock_reader.get_markets_by_type.return_value = [market]

        mock_writer = AsyncMock()
        mock_writer.upsert_event.return_value = mock_event
        mock_writer.store_orderbook_snapshot.return_value = MagicMock()  # non-None = stored

        with (
            patch("odds_lambda.jobs.fetch_polymarket.get_settings", return_value=mock_settings),
            patch(
                "odds_lambda.jobs.fetch_polymarket.PolymarketClient",
                return_value=mock_polymarket_client,
            ),
            patch(
                "odds_lambda.jobs.fetch_polymarket.async_session_maker",
                return_value=mock_async_session,
            ),
            patch("odds_lambda.jobs.fetch_polymarket.PolymarketReader", return_value=mock_reader),
            patch("odds_lambda.jobs.fetch_polymarket.PolymarketWriter", return_value=mock_writer),
            patch(
                "odds_lambda.jobs.fetch_polymarket.get_scheduler_backend",
                return_value=mock_backend,
            ),
            patch("odds_lambda.jobs.fetch_polymarket.asyncio.sleep"),
        ):
            await fetch_polymarket.main()

        mock_writer.store_orderbook_snapshot.assert_called_once()
        mock_polymarket_client.get_order_book.assert_called_once_with("token-0")

    @pytest.mark.asyncio
    async def test_skips_orderbooks_in_early_tier(
        self,
        mock_polymarket_client,
        mock_async_session,
        mock_backend,
        sample_active_event_data,
    ):
        """Order books NOT fetched when tier is outside orderbook_tiers."""
        mock_settings = _make_settings(orderbook_tiers=["closing", "pregame"])
        mock_event = _make_active_event(hours_until=48)  # early tier (1-3 days)
        market = _make_market()

        mock_polymarket_client.get_nba_events.return_value = [sample_active_event_data]
        mock_polymarket_client.get_prices_batch.return_value = {"token-0": 0.55, "token-1": 0.45}

        mock_reader = AsyncMock()
        mock_reader.get_active_events.return_value = [mock_event]
        mock_reader.get_markets_by_type.return_value = [market]

        mock_writer = AsyncMock()
        mock_writer.upsert_event.return_value = mock_event

        with (
            patch("odds_lambda.jobs.fetch_polymarket.get_settings", return_value=mock_settings),
            patch(
                "odds_lambda.jobs.fetch_polymarket.PolymarketClient",
                return_value=mock_polymarket_client,
            ),
            patch(
                "odds_lambda.jobs.fetch_polymarket.async_session_maker",
                return_value=mock_async_session,
            ),
            patch("odds_lambda.jobs.fetch_polymarket.PolymarketReader", return_value=mock_reader),
            patch("odds_lambda.jobs.fetch_polymarket.PolymarketWriter", return_value=mock_writer),
            patch(
                "odds_lambda.jobs.fetch_polymarket.get_scheduler_backend",
                return_value=mock_backend,
            ),
            patch("odds_lambda.jobs.fetch_polymarket.asyncio.sleep"),
        ):
            await fetch_polymarket.main()

        mock_polymarket_client.get_order_book.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_snapshot_when_price_is_none(
        self,
        mock_polymarket_client,
        mock_async_session,
        mock_backend,
        sample_active_event_data,
    ):
        """Snapshot skipped when a fetched price is None (failed fetch)."""
        mock_settings = _make_settings()
        mock_event = _make_active_event(hours_until=6)
        market = _make_market()

        mock_polymarket_client.get_nba_events.return_value = [sample_active_event_data]
        mock_polymarket_client.get_prices_batch.return_value = {"token-0": 0.55, "token-1": None}

        mock_reader = AsyncMock()
        mock_reader.get_active_events.return_value = [mock_event]
        mock_reader.get_markets_by_type.return_value = [market]

        mock_writer = AsyncMock()
        mock_writer.upsert_event.return_value = mock_event

        with (
            patch("odds_lambda.jobs.fetch_polymarket.get_settings", return_value=mock_settings),
            patch(
                "odds_lambda.jobs.fetch_polymarket.PolymarketClient",
                return_value=mock_polymarket_client,
            ),
            patch(
                "odds_lambda.jobs.fetch_polymarket.async_session_maker",
                return_value=mock_async_session,
            ),
            patch("odds_lambda.jobs.fetch_polymarket.PolymarketReader", return_value=mock_reader),
            patch("odds_lambda.jobs.fetch_polymarket.PolymarketWriter", return_value=mock_writer),
            patch(
                "odds_lambda.jobs.fetch_polymarket.get_scheduler_backend",
                return_value=mock_backend,
            ),
            patch("odds_lambda.jobs.fetch_polymarket.asyncio.sleep"),
        ):
            await fetch_polymarket.main()

        mock_writer.store_price_snapshot.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_snapshot_when_token_missing_from_response(
        self,
        mock_polymarket_client,
        mock_async_session,
        mock_backend,
        sample_active_event_data,
    ):
        """Snapshot skipped when a token ID is absent from prices response."""
        mock_settings = _make_settings()
        mock_event = _make_active_event(hours_until=6)
        market = _make_market()

        mock_polymarket_client.get_nba_events.return_value = [sample_active_event_data]
        # token-1 entirely missing from result
        mock_polymarket_client.get_prices_batch.return_value = {"token-0": 0.55}

        mock_reader = AsyncMock()
        mock_reader.get_active_events.return_value = [mock_event]
        mock_reader.get_markets_by_type.return_value = [market]

        mock_writer = AsyncMock()
        mock_writer.upsert_event.return_value = mock_event

        with (
            patch("odds_lambda.jobs.fetch_polymarket.get_settings", return_value=mock_settings),
            patch(
                "odds_lambda.jobs.fetch_polymarket.PolymarketClient",
                return_value=mock_polymarket_client,
            ),
            patch(
                "odds_lambda.jobs.fetch_polymarket.async_session_maker",
                return_value=mock_async_session,
            ),
            patch("odds_lambda.jobs.fetch_polymarket.PolymarketReader", return_value=mock_reader),
            patch("odds_lambda.jobs.fetch_polymarket.PolymarketWriter", return_value=mock_writer),
            patch(
                "odds_lambda.jobs.fetch_polymarket.get_scheduler_backend",
                return_value=mock_backend,
            ),
            patch("odds_lambda.jobs.fetch_polymarket.asyncio.sleep"),
        ):
            await fetch_polymarket.main()

        mock_writer.store_price_snapshot.assert_not_called()

    @pytest.mark.asyncio
    async def test_per_event_error_resilience(
        self,
        mock_polymarket_client,
        mock_async_session,
        mock_backend,
    ):
        """One event failing during upsert does not prevent other events."""
        mock_settings = _make_settings()

        events_data = [
            {
                "id": f"event-{i}",
                "ticker": f"nba-game-{i}",
                "slug": f"game-{i}",
                "title": f"Team {i} vs Team {i+1}",
                "startDate": "2024-01-15T19:00:00Z",
                "endDate": "2024-01-15T22:00:00Z",
                "active": True,
                "closed": False,
                "markets": [],
            }
            for i in range(2)
        ]

        mock_polymarket_client.get_nba_events.return_value = events_data

        mock_reader = AsyncMock()
        mock_reader.get_active_events.return_value = []  # empty → schedules daily and returns

        mock_writer = AsyncMock()
        mock_writer.upsert_event.side_effect = [
            MagicMock(id=1, pm_event_id="event-0"),
            Exception("DB error"),
        ]

        with (
            patch("odds_lambda.jobs.fetch_polymarket.get_settings", return_value=mock_settings),
            patch(
                "odds_lambda.jobs.fetch_polymarket.PolymarketClient",
                return_value=mock_polymarket_client,
            ),
            patch(
                "odds_lambda.jobs.fetch_polymarket.async_session_maker",
                return_value=mock_async_session,
            ),
            patch("odds_lambda.jobs.fetch_polymarket.PolymarketReader", return_value=mock_reader),
            patch("odds_lambda.jobs.fetch_polymarket.PolymarketWriter", return_value=mock_writer),
            patch(
                "odds_lambda.jobs.fetch_polymarket.get_scheduler_backend",
                return_value=mock_backend,
            ),
        ):
            await fetch_polymarket.main()  # must not raise

        assert mock_writer.upsert_event.call_count == 2

    @pytest.mark.asyncio
    async def test_rate_limit_delay_between_clob_calls(
        self,
        mock_polymarket_client,
        mock_async_session,
        mock_backend,
        sample_active_event_data,
    ):
        """asyncio.sleep(0.1) called after prices batch and after order book."""
        mock_settings = _make_settings(orderbook_tiers=["closing"])
        mock_event = _make_active_event(hours_until=1)  # closing tier
        market = _make_market()

        mock_polymarket_client.get_nba_events.return_value = [sample_active_event_data]
        mock_polymarket_client.get_prices_batch.return_value = {"token-0": 0.55, "token-1": 0.45}
        mock_polymarket_client.get_order_book.return_value = {
            "bids": [{"price": "0.54", "size": "100"}],
            "asks": [{"price": "0.56", "size": "100"}],
        }

        mock_reader = AsyncMock()
        mock_reader.get_active_events.return_value = [mock_event]
        mock_reader.get_markets_by_type.return_value = [market]

        mock_writer = AsyncMock()
        mock_writer.upsert_event.return_value = mock_event
        mock_writer.store_orderbook_snapshot.return_value = MagicMock()

        with (
            patch("odds_lambda.jobs.fetch_polymarket.get_settings", return_value=mock_settings),
            patch(
                "odds_lambda.jobs.fetch_polymarket.PolymarketClient",
                return_value=mock_polymarket_client,
            ),
            patch(
                "odds_lambda.jobs.fetch_polymarket.async_session_maker",
                return_value=mock_async_session,
            ),
            patch("odds_lambda.jobs.fetch_polymarket.PolymarketReader", return_value=mock_reader),
            patch("odds_lambda.jobs.fetch_polymarket.PolymarketWriter", return_value=mock_writer),
            patch(
                "odds_lambda.jobs.fetch_polymarket.get_scheduler_backend",
                return_value=mock_backend,
            ),
            patch("odds_lambda.jobs.fetch_polymarket.asyncio.sleep") as mock_sleep,
        ):
            await fetch_polymarket.main()

        # Once after get_prices_batch, once after get_order_book
        assert mock_sleep.call_count == 2
        mock_sleep.assert_called_with(0.1)

    @pytest.mark.asyncio
    async def test_fetch_log_written_on_success(
        self,
        mock_polymarket_client,
        mock_async_session,
        mock_backend,
        sample_active_event_data,
    ):
        """PolymarketFetchLog entry created with success=True on happy path."""
        mock_settings = _make_settings()
        mock_event = _make_active_event(hours_until=6)
        market = _make_market()

        mock_polymarket_client.get_nba_events.return_value = [sample_active_event_data]
        mock_polymarket_client.get_prices_batch.return_value = {"token-0": 0.55, "token-1": 0.45}

        mock_reader = AsyncMock()
        mock_reader.get_active_events.return_value = [mock_event]
        mock_reader.get_markets_by_type.return_value = [market]

        mock_writer = AsyncMock()
        mock_writer.upsert_event.return_value = mock_event

        with (
            patch("odds_lambda.jobs.fetch_polymarket.get_settings", return_value=mock_settings),
            patch(
                "odds_lambda.jobs.fetch_polymarket.PolymarketClient",
                return_value=mock_polymarket_client,
            ),
            patch(
                "odds_lambda.jobs.fetch_polymarket.async_session_maker",
                return_value=mock_async_session,
            ),
            patch("odds_lambda.jobs.fetch_polymarket.PolymarketReader", return_value=mock_reader),
            patch("odds_lambda.jobs.fetch_polymarket.PolymarketWriter", return_value=mock_writer),
            patch(
                "odds_lambda.jobs.fetch_polymarket.get_scheduler_backend",
                return_value=mock_backend,
            ),
            patch("odds_lambda.jobs.fetch_polymarket.asyncio.sleep"),
        ):
            await fetch_polymarket.main()

        mock_writer.log_fetch.assert_called_once()
        log_arg = mock_writer.log_fetch.call_args.args[0]
        assert log_arg.success is True
        assert log_arg.error_message is None
        mock_async_session.commit.assert_called()

    @pytest.mark.asyncio
    async def test_fetch_log_written_on_failure(
        self,
        mock_polymarket_client,
        mock_async_session,
        mock_backend,
    ):
        """PolymarketFetchLog entry created with success=False on exception."""
        mock_settings = _make_settings()

        mock_polymarket_client.get_nba_events.side_effect = Exception("API down")

        mock_writer = AsyncMock()

        with (
            patch("odds_lambda.jobs.fetch_polymarket.get_settings", return_value=mock_settings),
            patch(
                "odds_lambda.jobs.fetch_polymarket.PolymarketClient",
                return_value=mock_polymarket_client,
            ),
            patch(
                "odds_lambda.jobs.fetch_polymarket.async_session_maker",
                return_value=mock_async_session,
            ),
            patch("odds_lambda.jobs.fetch_polymarket.PolymarketReader", return_value=AsyncMock()),
            patch("odds_lambda.jobs.fetch_polymarket.PolymarketWriter", return_value=mock_writer),
            patch(
                "odds_lambda.jobs.fetch_polymarket.get_scheduler_backend",
                return_value=mock_backend,
            ),
        ):
            with pytest.raises(Exception, match="API down"):
                await fetch_polymarket.main()

        mock_writer.log_fetch.assert_called_once()
        log_arg = mock_writer.log_fetch.call_args.args[0]
        assert log_arg.success is False
        assert "API down" in log_arg.error_message

    @pytest.mark.asyncio
    async def test_scheduling_failure_does_not_fail_job(
        self,
        mock_polymarket_client,
        mock_async_session,
        mock_backend,
        sample_active_event_data,
    ):
        """Scheduling exception after successful fetch does not propagate."""
        mock_settings = _make_settings()
        mock_event = _make_active_event(hours_until=6)
        market = _make_market()

        mock_polymarket_client.get_nba_events.return_value = [sample_active_event_data]
        mock_polymarket_client.get_prices_batch.return_value = {"token-0": 0.55, "token-1": 0.45}

        mock_reader = AsyncMock()
        mock_reader.get_active_events.return_value = [mock_event]
        mock_reader.get_markets_by_type.return_value = [market]

        mock_writer = AsyncMock()
        mock_writer.upsert_event.return_value = mock_event

        mock_backend.schedule_next_execution.side_effect = Exception("Scheduler unreachable")

        with (
            patch("odds_lambda.jobs.fetch_polymarket.get_settings", return_value=mock_settings),
            patch(
                "odds_lambda.jobs.fetch_polymarket.PolymarketClient",
                return_value=mock_polymarket_client,
            ),
            patch(
                "odds_lambda.jobs.fetch_polymarket.async_session_maker",
                return_value=mock_async_session,
            ),
            patch("odds_lambda.jobs.fetch_polymarket.PolymarketReader", return_value=mock_reader),
            patch("odds_lambda.jobs.fetch_polymarket.PolymarketWriter", return_value=mock_writer),
            patch(
                "odds_lambda.jobs.fetch_polymarket.get_scheduler_backend",
                return_value=mock_backend,
            ),
            patch("odds_lambda.jobs.fetch_polymarket.asyncio.sleep"),
        ):
            await fetch_polymarket.main()  # must not raise
