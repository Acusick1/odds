"""Unit tests for Polymarket database models."""

from datetime import UTC, datetime

from odds_core.polymarket_models import (
    PolymarketEvent,
    PolymarketFetchLog,
    PolymarketMarket,
    PolymarketMarketType,
    PolymarketOrderBookSnapshot,
    PolymarketPriceSnapshot,
)


class TestPolymarketMarketType:
    """Tests for PolymarketMarketType enum."""

    def test_enum_values(self):
        """Test all enum values match expected strings."""
        assert PolymarketMarketType.MONEYLINE.value == "moneyline"
        assert PolymarketMarketType.SPREAD.value == "spread"
        assert PolymarketMarketType.TOTAL.value == "total"
        assert PolymarketMarketType.PLAYER_PROP.value == "player_prop"
        assert PolymarketMarketType.OTHER.value == "other"

    def test_enum_is_str(self):
        """Test enum values are strings for JSON serialization."""
        assert isinstance(PolymarketMarketType.MONEYLINE, str)


class TestPolymarketEvent:
    """Tests for PolymarketEvent model."""

    def test_event_creation(self):
        """Test creating a PolymarketEvent with required fields."""
        now = datetime.now(UTC)
        event = PolymarketEvent(
            pm_event_id="pm-abc123",
            ticker="nba-dal-mil-2026-01-25",
            slug="nba-dal-mil-2026-01-25",
            title="Dallas Mavericks vs Milwaukee Bucks",
            start_date=now,
            end_date=now,
        )

        assert event.pm_event_id == "pm-abc123"
        assert event.ticker == "nba-dal-mil-2026-01-25"
        assert event.title == "Dallas Mavericks vs Milwaukee Bucks"
        assert event.start_date == now
        assert event.end_date == now

    def test_nullable_defaults(self):
        """Test nullable fields default to None."""
        event = PolymarketEvent(
            pm_event_id="pm-abc123",
            ticker="nba-dal-mil-2026-01-25",
            slug="nba-dal-mil-2026-01-25",
            title="Dallas Mavericks vs Milwaukee Bucks",
            start_date=datetime.now(UTC),
            end_date=datetime.now(UTC),
        )

        assert event.event_id is None
        assert event.volume is None
        assert event.liquidity is None
        assert event.markets_count is None

    def test_status_defaults(self):
        """Test active/closed default values."""
        event = PolymarketEvent(
            pm_event_id="pm-abc123",
            ticker="nba-dal-mil-2026-01-25",
            slug="nba-dal-mil-2026-01-25",
            title="Dallas Mavericks vs Milwaukee Bucks",
            start_date=datetime.now(UTC),
            end_date=datetime.now(UTC),
        )

        assert event.active is True
        assert event.closed is False

    def test_timestamps_are_timezone_aware(self):
        """Test created_at and updated_at get timezone-aware defaults."""
        event = PolymarketEvent(
            pm_event_id="pm-abc123",
            ticker="nba-dal-mil-2026-01-25",
            slug="nba-dal-mil-2026-01-25",
            title="Dallas Mavericks vs Milwaukee Bucks",
            start_date=datetime.now(UTC),
            end_date=datetime.now(UTC),
        )

        assert event.created_at.tzinfo is not None
        assert event.updated_at.tzinfo is not None


class TestPolymarketMarket:
    """Tests for PolymarketMarket model."""

    def test_market_creation(self):
        """Test creating a PolymarketMarket with required fields."""
        market = PolymarketMarket(
            polymarket_event_id=1,
            pm_market_id="pm-mkt-456",
            condition_id="0xabc123",
            question="Will the Dallas Mavericks beat the Milwaukee Bucks?",
            clob_token_ids=["token-a", "token-b"],
            outcomes=["Mavericks", "Bucks"],
            market_type=PolymarketMarketType.MONEYLINE,
        )

        assert market.pm_market_id == "pm-mkt-456"
        assert market.condition_id == "0xabc123"
        assert market.question == "Will the Dallas Mavericks beat the Milwaukee Bucks?"
        assert market.market_type == PolymarketMarketType.MONEYLINE

    def test_json_fields(self):
        """Test JSON column fields accept list data."""
        market = PolymarketMarket(
            polymarket_event_id=1,
            pm_market_id="pm-mkt-456",
            condition_id="0xabc123",
            question="Spread -6.5?",
            clob_token_ids=["token-a", "token-b"],
            outcomes=["Mavericks", "Bucks"],
            market_type=PolymarketMarketType.SPREAD,
            group_item_title="Spread -6.5",
            point=-6.5,
        )

        assert market.clob_token_ids == ["token-a", "token-b"]
        assert market.outcomes == ["Mavericks", "Bucks"]
        assert market.group_item_title == "Spread -6.5"
        assert market.point == -6.5

    def test_status_defaults(self):
        """Test market status defaults."""
        market = PolymarketMarket(
            polymarket_event_id=1,
            pm_market_id="pm-mkt-456",
            condition_id="0xabc123",
            question="Will the Mavericks win?",
            clob_token_ids=["token-a", "token-b"],
            outcomes=["Mavericks", "Bucks"],
            market_type=PolymarketMarketType.MONEYLINE,
        )

        assert market.active is True
        assert market.closed is False
        assert market.accepting_orders is True
        assert market.group_item_title is None
        assert market.point is None


class TestPolymarketPriceSnapshot:
    """Tests for PolymarketPriceSnapshot model."""

    def test_snapshot_creation(self):
        """Test creating a price snapshot with required fields."""
        now = datetime.now(UTC)
        snapshot = PolymarketPriceSnapshot(
            polymarket_market_id=1,
            snapshot_time=now,
            outcome_0_price=0.62,
            outcome_1_price=0.38,
        )

        assert snapshot.polymarket_market_id == 1
        assert snapshot.snapshot_time == now
        assert snapshot.outcome_0_price == 0.62
        assert snapshot.outcome_1_price == 0.38

    def test_optional_fields_default_none(self):
        """Test optional book summary and tier fields default to None."""
        snapshot = PolymarketPriceSnapshot(
            polymarket_market_id=1,
            snapshot_time=datetime.now(UTC),
            outcome_0_price=0.55,
            outcome_1_price=0.45,
        )

        assert snapshot.best_bid is None
        assert snapshot.best_ask is None
        assert snapshot.spread is None
        assert snapshot.midpoint is None
        assert snapshot.volume is None
        assert snapshot.liquidity is None
        assert snapshot.fetch_tier is None
        assert snapshot.hours_until_commence is None


class TestPolymarketOrderBookSnapshot:
    """Tests for PolymarketOrderBookSnapshot model."""

    def test_snapshot_creation(self):
        """Test creating an order book snapshot."""
        now = datetime.now(UTC)
        raw = {
            "bids": [{"price": 0.60, "size": 100}],
            "asks": [{"price": 0.62, "size": 80}],
        }
        snapshot = PolymarketOrderBookSnapshot(
            polymarket_market_id=1,
            snapshot_time=now,
            token_id="token-a",
            raw_book=raw,
            best_bid=0.60,
            best_ask=0.62,
            spread=0.02,
            midpoint=0.61,
            bid_levels=1,
            ask_levels=1,
            bid_depth_total=100.0,
            ask_depth_total=80.0,
            imbalance=0.11,
            weighted_mid=0.608,
        )

        assert snapshot.token_id == "token-a"
        assert snapshot.raw_book == raw
        assert snapshot.best_bid == 0.60
        assert snapshot.spread == 0.02
        assert snapshot.imbalance == 0.11
        assert snapshot.weighted_mid == 0.608

    def test_optional_fields_default_none(self):
        """Test derived metrics default to None."""
        snapshot = PolymarketOrderBookSnapshot(
            polymarket_market_id=1,
            snapshot_time=datetime.now(UTC),
            token_id="token-a",
            raw_book={"bids": [], "asks": []},
        )

        assert snapshot.best_bid is None
        assert snapshot.best_ask is None
        assert snapshot.spread is None
        assert snapshot.midpoint is None
        assert snapshot.bid_levels is None
        assert snapshot.ask_levels is None
        assert snapshot.bid_depth_total is None
        assert snapshot.ask_depth_total is None
        assert snapshot.imbalance is None
        assert snapshot.weighted_mid is None
        assert snapshot.fetch_tier is None
        assert snapshot.hours_until_commence is None


class TestPolymarketFetchLog:
    """Tests for PolymarketFetchLog model."""

    def test_successful_fetch(self):
        """Test creating a successful fetch log."""
        log = PolymarketFetchLog(
            job_type="price_snapshot",
            events_count=12,
            markets_count=48,
            snapshots_stored=48,
            success=True,
            response_time_ms=342,
        )

        assert log.job_type == "price_snapshot"
        assert log.events_count == 12
        assert log.markets_count == 48
        assert log.snapshots_stored == 48
        assert log.success is True
        assert log.error_message is None
        assert log.response_time_ms == 342

    def test_failed_fetch(self):
        """Test creating a failed fetch log."""
        log = PolymarketFetchLog(
            job_type="orderbook",
            events_count=0,
            markets_count=0,
            snapshots_stored=0,
            success=False,
            error_message="Gamma API timeout",
        )

        assert log.success is False
        assert log.error_message == "Gamma API timeout"

    def test_fetch_time_is_timezone_aware(self):
        """Test fetch_time gets a timezone-aware default."""
        log = PolymarketFetchLog(
            job_type="price_snapshot",
            events_count=5,
            markets_count=20,
            snapshots_stored=20,
            success=True,
        )

        assert log.fetch_time.tzinfo is not None
