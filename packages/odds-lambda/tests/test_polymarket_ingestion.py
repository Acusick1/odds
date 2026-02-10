"""Tests for PolymarketIngestionService."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from odds_core.polymarket_models import PolymarketMarketType
from odds_lambda.polymarket_ingestion import (
    PolymarketIngestionService,
    should_fetch_orderbooks,
)

FROZEN_NOW = datetime(2024, 6, 15, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(
    orderbook_tiers: list[str] | None = None,
    collect_moneyline: bool = True,
    collect_spreads: bool = False,
    collect_totals: bool = False,
    collect_player_props: bool = False,
) -> MagicMock:
    s = MagicMock()
    s.polymarket.orderbook_tiers = (
        orderbook_tiers if orderbook_tiers is not None else ["closing", "pregame"]
    )
    s.polymarket.collect_moneyline = collect_moneyline
    s.polymarket.collect_spreads = collect_spreads
    s.polymarket.collect_totals = collect_totals
    s.polymarket.collect_player_props = collect_player_props
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
        start_date=FROZEN_NOW + timedelta(hours=hours_until),
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class ServiceMocks:
    """All mocked dependencies for PolymarketIngestionService."""

    def __init__(self, settings: MagicMock | None = None) -> None:
        self.client: AsyncMock = AsyncMock()
        self.reader: AsyncMock = AsyncMock()
        self.writer: AsyncMock = AsyncMock()
        self.writer.session = AsyncMock()
        self.settings: MagicMock = settings or _make_settings()

    def build_service(self, *, clob_delay_ms: int = 0) -> PolymarketIngestionService:
        return PolymarketIngestionService(
            client=self.client,
            reader=self.reader,
            writer=self.writer,
            settings=self.settings,
            clob_delay_ms=clob_delay_ms,
        )


@pytest.fixture
def mocks() -> ServiceMocks:
    return ServiceMocks()


# ---------------------------------------------------------------------------
# should_fetch_orderbooks
# ---------------------------------------------------------------------------


class TestShouldFetchOrderbooks:
    def test_returns_true_for_tier_in_orderbook_tiers(self) -> None:
        settings = _make_settings(orderbook_tiers=["closing", "pregame"])
        assert should_fetch_orderbooks("closing", settings) is True
        assert should_fetch_orderbooks("pregame", settings) is True

    def test_returns_false_for_tier_not_in_orderbook_tiers(self) -> None:
        settings = _make_settings(orderbook_tiers=["closing", "pregame"])
        assert should_fetch_orderbooks("early", settings) is False
        assert should_fetch_orderbooks("sharp", settings) is False
        assert should_fetch_orderbooks("opening", settings) is False


# ---------------------------------------------------------------------------
# discover_and_upsert_events
# ---------------------------------------------------------------------------


class TestDiscoverAndUpsertEvents:
    @pytest.mark.asyncio
    async def test_upserts_events_and_markets(self, mocks: ServiceMocks) -> None:
        service = mocks.build_service()
        event_mock = MagicMock(id=1, title="Lakers vs Celtics")
        mocks.writer.upsert_event.return_value = event_mock
        mocks.writer.upsert_markets.return_value = [MagicMock(), MagicMock()]

        events_data = [
            {"id": "ev-1", "title": "Lakers vs Celtics", "markets": [{"m": 1}, {"m": 2}]}
        ]

        result = await service.discover_and_upsert_events(events_data)

        assert result.events_processed == 1
        assert result.markets_discovered == 2
        assert result.errors == 0
        mocks.writer.session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_per_event_error_resilience(self, mocks: ServiceMocks) -> None:
        service = mocks.build_service()
        mocks.writer.upsert_event.side_effect = [
            MagicMock(id=1, title="Game 1"),
            Exception("DB error"),
        ]

        events_data = [
            {"id": "ev-0", "title": "Game 0", "markets": []},
            {"id": "ev-1", "title": "Game 1", "markets": []},
        ]

        result = await service.discover_and_upsert_events(events_data)

        assert result.events_processed == 1
        assert result.errors == 1
        assert mocks.writer.upsert_event.call_count == 2


# ---------------------------------------------------------------------------
# collect_snapshots
# ---------------------------------------------------------------------------


class TestCollectSnapshots:
    @pytest.mark.asyncio
    async def test_returns_empty_when_no_active_events(self, mocks: ServiceMocks) -> None:
        service = mocks.build_service()
        mocks.reader.get_active_events.return_value = []

        result = await service.collect_snapshots(now=FROZEN_NOW)

        assert result.price_snapshots_stored == 0
        assert result.fetch_tier is None

    @pytest.mark.asyncio
    async def test_stores_prices(self, mocks: ServiceMocks) -> None:
        service = mocks.build_service()
        event = _make_active_event(hours_until=6)
        market = _make_market()

        mocks.reader.get_active_events.return_value = [event]
        mocks.reader.get_markets_by_type.return_value = [market]
        mocks.client.get_prices_batch.return_value = {"token-0": 0.55, "token-1": 0.45}

        result = await service.collect_snapshots(now=FROZEN_NOW)

        assert result.price_snapshots_stored == 1
        assert result.fetch_tier is not None
        prices_arg = mocks.writer.store_price_snapshot.call_args.kwargs["prices"]
        assert prices_arg["outcome_0_price"] == 0.55
        assert prices_arg["outcome_1_price"] == 0.45

    @pytest.mark.asyncio
    async def test_stores_orderbooks_in_closing_tier(self, mocks: ServiceMocks) -> None:
        settings = _make_settings(orderbook_tiers=["closing"])
        m = ServiceMocks(settings=settings)
        service = m.build_service()

        event = _make_active_event(hours_until=1)  # closing tier (< 3h)
        market = _make_market()

        m.reader.get_active_events.return_value = [event]
        m.reader.get_markets_by_type.return_value = [market]
        m.client.get_prices_batch.return_value = {"token-0": 0.55, "token-1": 0.45}
        m.client.get_order_book.return_value = {
            "bids": [{"price": "0.54", "size": "100"}],
            "asks": [{"price": "0.56", "size": "100"}],
        }
        m.writer.store_orderbook_snapshot.return_value = MagicMock()

        result = await service.collect_snapshots(now=FROZEN_NOW)

        assert result.orderbook_snapshots_stored == 1
        m.client.get_order_book.assert_called_once_with("token-0")

    @pytest.mark.asyncio
    async def test_skips_orderbooks_in_early_tier(self, mocks: ServiceMocks) -> None:
        settings = _make_settings(orderbook_tiers=["closing", "pregame"])
        m = ServiceMocks(settings=settings)
        service = m.build_service()

        event = _make_active_event(hours_until=48)  # early tier (24-72h)
        market = _make_market()

        m.reader.get_active_events.return_value = [event]
        m.reader.get_markets_by_type.return_value = [market]
        m.client.get_prices_batch.return_value = {"token-0": 0.55, "token-1": 0.45}

        result = await service.collect_snapshots(now=FROZEN_NOW)

        assert result.price_snapshots_stored == 1
        m.client.get_order_book.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_snapshot_when_price_is_none(self, mocks: ServiceMocks) -> None:
        service = mocks.build_service()
        event = _make_active_event(hours_until=6)
        market = _make_market()

        mocks.reader.get_active_events.return_value = [event]
        mocks.reader.get_markets_by_type.return_value = [market]
        mocks.client.get_prices_batch.return_value = {"token-0": 0.55, "token-1": None}

        result = await service.collect_snapshots(now=FROZEN_NOW)

        assert result.price_snapshots_stored == 0
        mocks.writer.store_price_snapshot.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_snapshot_when_token_missing(self, mocks: ServiceMocks) -> None:
        service = mocks.build_service()
        event = _make_active_event(hours_until=6)
        market = _make_market()

        mocks.reader.get_active_events.return_value = [event]
        mocks.reader.get_markets_by_type.return_value = [market]
        mocks.client.get_prices_batch.return_value = {"token-0": 0.55}

        result = await service.collect_snapshots(now=FROZEN_NOW)

        assert result.price_snapshots_stored == 0
        mocks.writer.store_price_snapshot.assert_not_called()

    @pytest.mark.asyncio
    async def test_stores_zero_price_as_valid(self, mocks: ServiceMocks) -> None:
        service = mocks.build_service()
        event = _make_active_event(hours_until=6)
        market = _make_market()

        mocks.reader.get_active_events.return_value = [event]
        mocks.reader.get_markets_by_type.return_value = [market]
        mocks.client.get_prices_batch.return_value = {"token-0": 0.0, "token-1": 1.0}

        result = await service.collect_snapshots(now=FROZEN_NOW)

        assert result.price_snapshots_stored == 1
        prices_arg = mocks.writer.store_price_snapshot.call_args.kwargs["prices"]
        assert prices_arg["outcome_0_price"] == 0.0
        assert prices_arg["outcome_1_price"] == 1.0

    @pytest.mark.asyncio
    async def test_skips_market_with_insufficient_tokens(self, mocks: ServiceMocks) -> None:
        service = mocks.build_service()
        event = _make_active_event(hours_until=6)
        market = MagicMock(
            id=1,
            question="Test market",
            clob_token_ids=["only-one"],
            market_type=PolymarketMarketType.MONEYLINE,
        )

        mocks.reader.get_active_events.return_value = [event]
        mocks.reader.get_markets_by_type.return_value = [market]

        result = await service.collect_snapshots(now=FROZEN_NOW)

        mocks.client.get_prices_batch.assert_not_called()
        assert result.price_snapshots_stored == 0

    @pytest.mark.asyncio
    async def test_per_event_error_resilience(self, mocks: ServiceMocks) -> None:
        service = mocks.build_service()
        good_event = _make_active_event(hours_until=6, event_id=1, pm_event_id="ev-1")
        bad_event = _make_active_event(hours_until=6, event_id=2, pm_event_id="ev-2")

        mocks.reader.get_active_events.return_value = [good_event, bad_event]
        mocks.reader.get_markets_by_type.side_effect = [
            [_make_market()],
            Exception("DB read error"),
        ]
        mocks.client.get_prices_batch.return_value = {"token-0": 0.55, "token-1": 0.45}

        result = await service.collect_snapshots(now=FROZEN_NOW)

        assert result.price_snapshots_stored == 1
        assert result.errors == 1

    @pytest.mark.asyncio
    async def test_rate_limit_delay_between_clob_calls(self) -> None:
        """asyncio.sleep called with configured delay between CLOB calls."""
        settings = _make_settings(orderbook_tiers=["closing"])
        m = ServiceMocks(settings=settings)
        # Use real delay so we can assert on it
        service = m.build_service(clob_delay_ms=100)

        event = _make_active_event(hours_until=1)
        market = _make_market()

        m.reader.get_active_events.return_value = [event]
        m.reader.get_markets_by_type.return_value = [market]
        m.client.get_prices_batch.return_value = {"token-0": 0.55, "token-1": 0.45}
        m.client.get_order_book.return_value = {
            "bids": [{"price": "0.54", "size": "100"}],
            "asks": [{"price": "0.56", "size": "100"}],
        }
        m.writer.store_orderbook_snapshot.return_value = MagicMock()

        # Patch asyncio.sleep at the module where it's used
        from unittest.mock import patch

        with patch("odds_lambda.polymarket_ingestion.asyncio.sleep") as mock_sleep:
            await service.collect_snapshots(now=FROZEN_NOW)

            # Once after get_prices_batch, once after get_order_book
            assert mock_sleep.call_count == 2
            mock_sleep.assert_called_with(0.1)
