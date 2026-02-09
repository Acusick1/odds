"""Tests for Polymarket storage layer (writer and reader)."""

from datetime import UTC, datetime, timedelta

import pytest
from odds_core.polymarket_models import (
    PolymarketFetchLog,
    PolymarketMarketType,
)
from odds_lambda.storage.polymarket_reader import PolymarketReader
from odds_lambda.storage.polymarket_writer import PolymarketWriter


class TestPolymarketWriter:
    """Tests for PolymarketWriter."""

    @pytest.mark.asyncio
    async def test_upsert_event_creates_new(self, pglite_async_session):
        """Test that upsert_event creates a new event record."""
        writer = PolymarketWriter(pglite_async_session)

        event_data = {
            "id": "test_event_123",
            "ticker": "nba-lal-bos-2026-02-10",
            "slug": "lakers-vs-celtics",
            "title": "Lakers vs Celtics",
            "startDate": "2026-02-10T01:00:00Z",
            "endDate": "2026-02-10T04:00:00Z",
            "active": True,
            "closed": False,
            "volume": 50000.0,
            "liquidity": 10000.0,
            "markets": [{"id": "m1"}, {"id": "m2"}],
        }

        event = await writer.upsert_event(event_data)

        assert event.pm_event_id == "test_event_123"
        assert event.ticker == "nba-lal-bos-2026-02-10"
        assert event.title == "Lakers vs Celtics"
        assert event.active is True
        assert event.closed is False
        assert event.volume == 50000.0
        assert event.markets_count == 2
        assert event.id is not None

    @pytest.mark.asyncio
    async def test_upsert_event_updates_existing(self, pglite_async_session):
        """Test that upsert_event updates an existing event with same pm_event_id."""
        writer = PolymarketWriter(pglite_async_session)

        # First insert
        event_data = {
            "id": "test_event_456",
            "ticker": "nba-gsw-mia-2026-02-11",
            "slug": "warriors-vs-heat",
            "title": "Warriors vs Heat",
            "startDate": "2026-02-11T01:00:00Z",
            "endDate": "2026-02-11T04:00:00Z",
            "active": True,
            "closed": False,
            "volume": 25000.0,
            "liquidity": 5000.0,
            "markets": [{"id": "m1"}],
        }

        event1 = await writer.upsert_event(event_data)
        first_id = event1.id

        # Second upsert with updated volume
        event_data["volume"] = 60000.0
        event_data["markets"] = [{"id": "m1"}, {"id": "m2"}, {"id": "m3"}]

        event2 = await writer.upsert_event(event_data)
        await pglite_async_session.flush()

        assert event2.id == first_id  # Same database ID
        assert event2.volume == 60000.0  # Updated volume
        assert event2.markets_count == 3  # Updated count

    @pytest.mark.asyncio
    async def test_upsert_markets_idempotent(self, pglite_async_session):
        """Test that upsert_markets is idempotent (can run twice)."""
        writer = PolymarketWriter(pglite_async_session)

        # Create event first
        event_data = {
            "id": "event_789",
            "ticker": "nba-dal-mil-2026-02-12",
            "slug": "mavs-vs-bucks",
            "title": "Mavericks vs Bucks",
            "startDate": "2026-02-12T01:00:00Z",
            "endDate": "2026-02-12T04:00:00Z",
            "active": True,
            "closed": False,
            "markets": [],
        }
        event = await writer.upsert_event(event_data)

        markets_data = [
            {
                "id": "market_moneyline",
                "conditionId": "cond_123",
                "question": "Mavericks vs Bucks",
                "clobTokenIds": ["token_0", "token_1"],
                "outcomes": ["Mavericks", "Bucks"],
                "active": True,
                "closed": False,
                "acceptingOrders": True,
            },
            {
                "id": "market_spread",
                "conditionId": "cond_456",
                "question": "Mavericks Spread: -6.5",
                "clobTokenIds": ["token_2", "token_3"],
                "outcomes": ["Mavericks -6.5", "Bucks +6.5"],
                "groupItemTitle": "Spread -6.5",
                "active": True,
                "closed": False,
                "acceptingOrders": True,
            },
        ]

        # First upsert
        markets1 = await writer.upsert_markets(event.id, markets_data, event.title)
        assert len(markets1) == 2

        # Second upsert (should not duplicate)
        markets2 = await writer.upsert_markets(event.id, markets_data, event.title)
        assert len(markets2) == 2

        # Verify IDs are the same
        assert markets1[0].id == markets2[0].id
        assert markets1[1].id == markets2[1].id

    @pytest.mark.asyncio
    async def test_upsert_markets_classifies_types(self, pglite_async_session):
        """Test that upsert_markets correctly classifies market types."""
        writer = PolymarketWriter(pglite_async_session)

        # Create event
        event_data = {
            "id": "event_classify",
            "ticker": "nba-test-2026-02-13",
            "slug": "test-game",
            "title": "Lakers vs Celtics",
            "startDate": "2026-02-13T01:00:00Z",
            "endDate": "2026-02-13T04:00:00Z",
            "active": True,
            "closed": False,
            "markets": [],
        }
        event = await writer.upsert_event(event_data)

        markets_data = [
            {
                "id": "m1",
                "conditionId": "c1",
                "question": "Lakers vs Celtics",  # Exact match = moneyline
                "clobTokenIds": ["t1", "t2"],
                "outcomes": ["Lakers", "Celtics"],
            },
            {
                "id": "m2",
                "conditionId": "c2",
                "question": "Lakers Spread: -5.5",  # Spread pattern
                "clobTokenIds": ["t3", "t4"],
                "outcomes": ["Lakers -5.5", "Celtics +5.5"],
            },
            {
                "id": "m3",
                "conditionId": "c3",
                "question": "O/U 220.5",  # Total pattern
                "clobTokenIds": ["t5", "t6"],
                "outcomes": ["Over 220.5", "Under 220.5"],
            },
            {
                "id": "m4",
                "conditionId": "c4",
                "question": "LeBron James: Points Over 28.5",  # Player prop
                "clobTokenIds": ["t7", "t8"],
                "outcomes": ["Over", "Under"],
            },
        ]

        markets = await writer.upsert_markets(event.id, markets_data, event.title)

        assert markets[0].market_type == PolymarketMarketType.MONEYLINE
        assert markets[0].point is None

        assert markets[1].market_type == PolymarketMarketType.SPREAD
        assert markets[1].point == -5.5

        assert markets[2].market_type == PolymarketMarketType.TOTAL
        assert markets[2].point == 220.5

        assert markets[3].market_type == PolymarketMarketType.PLAYER_PROP
        assert markets[3].point is None

    @pytest.mark.asyncio
    async def test_store_price_snapshot_calculates_tier(self, pglite_async_session):
        """Test that store_price_snapshot calculates tier and hours_until correctly."""
        writer = PolymarketWriter(pglite_async_session)

        # Create event and market
        event_data = {
            "id": "event_tier_test",
            "ticker": "nba-tier-test",
            "slug": "tier-test",
            "title": "Tier Test Game",
            "startDate": "2026-02-14T01:00:00Z",
            "endDate": "2026-02-14T04:00:00Z",
            "active": True,
            "closed": False,
            "markets": [],
        }
        event = await writer.upsert_event(event_data)

        markets_data = [
            {
                "id": "market_tier",
                "conditionId": "cond_tier",
                "question": "Tier Test Game",
                "clobTokenIds": ["token_a", "token_b"],
                "outcomes": ["Team A", "Team B"],
            }
        ]
        markets = await writer.upsert_markets(event.id, markets_data, event.title)
        market = markets[0]

        # Snapshot 2 hours before game (should be CLOSING tier)
        commence_time = datetime(2026, 2, 14, 1, 0, 0, tzinfo=UTC)
        snapshot_time = commence_time - timedelta(hours=2)

        prices = {
            "outcome_0_price": 0.55,
            "outcome_1_price": 0.45,
            "best_bid": 0.54,
            "best_ask": 0.56,
            "volume": 1000.0,
            "liquidity": 500.0,
        }

        snapshot = await writer.store_price_snapshot(market, prices, commence_time, snapshot_time)
        await pglite_async_session.flush()

        assert snapshot.fetch_tier == "closing"
        assert snapshot.hours_until_commence == 2.0
        assert abs(snapshot.spread - 0.02) < 1e-10  # 0.56 - 0.54
        assert snapshot.midpoint == 0.55  # (0.54 + 0.56) / 2

    @pytest.mark.asyncio
    async def test_store_orderbook_invalid_book(self, pglite_async_session):
        """Test that store_orderbook_snapshot returns None for invalid books."""
        writer = PolymarketWriter(pglite_async_session)

        # Create event and market
        event_data = {
            "id": "event_invalid_book",
            "ticker": "nba-invalid",
            "slug": "invalid",
            "title": "Invalid Book Game",
            "startDate": "2026-02-15T01:00:00Z",
            "endDate": "2026-02-15T04:00:00Z",
            "active": True,
            "closed": False,
            "markets": [],
        }
        event = await writer.upsert_event(event_data)

        markets_data = [
            {
                "id": "market_invalid",
                "conditionId": "cond_invalid",
                "question": "Invalid Book Game",
                "clobTokenIds": ["token_x"],
                "outcomes": ["Team X", "Team Y"],
            }
        ]
        markets = await writer.upsert_markets(event.id, markets_data, event.title)
        market = markets[0]

        # Empty bids/asks should return None
        raw_book = {"bids": [], "asks": []}
        commence_time = datetime(2026, 2, 15, 1, 0, 0, tzinfo=UTC)

        result = await writer.store_orderbook_snapshot(market, raw_book, commence_time)

        assert result is None

    @pytest.mark.asyncio
    async def test_bulk_store_price_history_idempotency(self, pglite_async_session):
        """Test that bulk_store_price_history is idempotent."""
        writer = PolymarketWriter(pglite_async_session)

        # Create event and market
        event_data = {
            "id": "event_history",
            "ticker": "nba-history",
            "slug": "history",
            "title": "History Test Game",
            "startDate": "2026-02-16T01:00:00Z",
            "endDate": "2026-02-16T04:00:00Z",
            "active": True,
            "closed": False,
            "markets": [],
        }
        event = await writer.upsert_event(event_data)

        markets_data = [
            {
                "id": "market_history",
                "conditionId": "cond_history",
                "question": "History Test Game",
                "clobTokenIds": ["token_h"],
                "outcomes": ["Team H1", "Team H2"],
            }
        ]
        markets = await writer.upsert_markets(event.id, markets_data, event.title)
        market = markets[0]

        commence_time = datetime(2026, 2, 16, 1, 0, 0, tzinfo=UTC)

        # Create historical data points
        history = [
            {"t": int((commence_time - timedelta(hours=48)).timestamp()), "p": "0.50"},
            {"t": int((commence_time - timedelta(hours=24)).timestamp()), "p": "0.52"},
            {"t": int((commence_time - timedelta(hours=12)).timestamp()), "p": "0.54"},
            {"t": int((commence_time - timedelta(hours=6)).timestamp()), "p": "0.56"},
            {"t": int((commence_time - timedelta(hours=1)).timestamp()), "p": "0.58"},
        ]

        # First insert
        count1 = await writer.bulk_store_price_history(market, history, commence_time)
        await pglite_async_session.flush()
        assert count1 == 5

        # Second insert (should skip all duplicates)
        count2 = await writer.bulk_store_price_history(market, history, commence_time)
        await pglite_async_session.flush()
        assert count2 == 0

    @pytest.mark.asyncio
    async def test_bulk_history_retroactive_tiers(self, pglite_async_session):
        """Test that bulk_store_price_history calculates correct tiers for historical points."""
        writer = PolymarketWriter(pglite_async_session)
        reader = PolymarketReader(pglite_async_session)

        # Create event and market
        event_data = {
            "id": "event_tiers",
            "ticker": "nba-tiers",
            "slug": "tiers",
            "title": "Tiers Test Game",
            "startDate": "2026-02-17T01:00:00Z",
            "endDate": "2026-02-17T04:00:00Z",
            "active": True,
            "closed": False,
            "markets": [],
        }
        event = await writer.upsert_event(event_data)

        markets_data = [
            {
                "id": "market_tiers",
                "conditionId": "cond_tiers",
                "question": "Tiers Test Game",
                "clobTokenIds": ["token_t"],
                "outcomes": ["Team T1", "Team T2"],
            }
        ]
        markets = await writer.upsert_markets(event.id, markets_data, event.title)
        market = markets[0]

        commence_time = datetime(2026, 2, 17, 1, 0, 0, tzinfo=UTC)

        # Create points at different tier boundaries
        history = [
            {
                "t": int((commence_time - timedelta(hours=96)).timestamp()),
                "p": "0.50",
            },  # OPENING (>72h)
            {
                "t": int((commence_time - timedelta(hours=48)).timestamp()),
                "p": "0.51",
            },  # EARLY (24-72h)
            {
                "t": int((commence_time - timedelta(hours=18)).timestamp()),
                "p": "0.52",
            },  # SHARP (12-24h)
            {
                "t": int((commence_time - timedelta(hours=8)).timestamp()),
                "p": "0.53",
            },  # PREGAME (3-12h)
            {
                "t": int((commence_time - timedelta(hours=2)).timestamp()),
                "p": "0.54",
            },  # CLOSING (0-3h)
        ]

        await writer.bulk_store_price_history(market, history, commence_time)
        await pglite_async_session.flush()

        # Fetch snapshots and verify tiers
        snapshots = await reader.get_price_series(
            market.id, commence_time - timedelta(hours=100), commence_time
        )

        assert len(snapshots) == 5
        assert snapshots[0].fetch_tier == "opening"
        assert snapshots[1].fetch_tier == "early"
        assert snapshots[2].fetch_tier == "sharp"
        assert snapshots[3].fetch_tier == "pregame"
        assert snapshots[4].fetch_tier == "closing"

    @pytest.mark.asyncio
    async def test_log_fetch(self, pglite_async_session):
        """Test that log_fetch stores and returns the fetch log."""
        writer = PolymarketWriter(pglite_async_session)

        fetch_log = PolymarketFetchLog(
            job_type="current_events",
            events_count=5,
            markets_count=20,
            snapshots_stored=100,
            success=True,
            error_message=None,
            response_time_ms=250,
        )

        result = await writer.log_fetch(fetch_log)

        assert result.job_type == "current_events"
        assert result.events_count == 5
        assert result.success is True


class TestPolymarketReader:
    """Tests for PolymarketReader."""

    @pytest.mark.asyncio
    async def test_get_active_events(self, pglite_async_session):
        """Test that get_active_events returns only active, non-closed events."""
        writer = PolymarketWriter(pglite_async_session)
        reader = PolymarketReader(pglite_async_session)

        # Create active event
        await writer.upsert_event(
            {
                "id": "active_1",
                "ticker": "nba-active",
                "slug": "active",
                "title": "Active Game",
                "startDate": "2026-02-20T01:00:00Z",
                "endDate": "2026-02-20T04:00:00Z",
                "active": True,
                "closed": False,
                "markets": [],
            }
        )

        # Create closed event (should be filtered out)
        await writer.upsert_event(
            {
                "id": "closed_1",
                "ticker": "nba-closed",
                "slug": "closed",
                "title": "Closed Game",
                "startDate": "2026-02-19T01:00:00Z",
                "endDate": "2026-02-19T04:00:00Z",
                "active": False,
                "closed": True,
                "markets": [],
            }
        )

        active_events = await reader.get_active_events()

        assert len(active_events) == 1
        assert active_events[0].pm_event_id == "active_1"

    @pytest.mark.asyncio
    async def test_get_moneyline_market(self, pglite_async_session):
        """Test that get_moneyline_market returns the moneyline market."""
        writer = PolymarketWriter(pglite_async_session)
        reader = PolymarketReader(pglite_async_session)

        # Create event
        event_data = {
            "id": "event_ml",
            "ticker": "nba-ml-test",
            "slug": "ml-test",
            "title": "Moneyline Test",
            "startDate": "2026-02-21T01:00:00Z",
            "endDate": "2026-02-21T04:00:00Z",
            "active": True,
            "closed": False,
            "markets": [],
        }
        event = await writer.upsert_event(event_data)

        # Create markets including moneyline
        markets_data = [
            {
                "id": "ml_market",
                "conditionId": "ml_cond",
                "question": "Moneyline Test",  # Exact match = moneyline
                "clobTokenIds": ["ml_t1", "ml_t2"],
                "outcomes": ["Team A", "Team B"],
            },
            {
                "id": "spread_market",
                "conditionId": "spread_cond",
                "question": "Team A Spread: -3.5",
                "clobTokenIds": ["sp_t1", "sp_t2"],
                "outcomes": ["Team A -3.5", "Team B +3.5"],
            },
        ]
        await writer.upsert_markets(event.id, markets_data, event.title)

        # Get moneyline market
        ml_market = await reader.get_moneyline_market(event.id)

        assert ml_market is not None
        assert ml_market.market_type == PolymarketMarketType.MONEYLINE
        assert ml_market.question == "Moneyline Test"

    @pytest.mark.asyncio
    async def test_get_price_at_time_no_lookahead(self, pglite_async_session):
        """Test that get_price_at_time never returns future data (critical for backtesting)."""
        writer = PolymarketWriter(pglite_async_session)
        reader = PolymarketReader(pglite_async_session)

        # Create event and market
        event_data = {
            "id": "event_lookahead",
            "ticker": "nba-lookahead",
            "slug": "lookahead",
            "title": "Lookahead Test",
            "startDate": "2026-02-22T01:00:00Z",
            "endDate": "2026-02-22T04:00:00Z",
            "active": True,
            "closed": False,
            "markets": [],
        }
        event = await writer.upsert_event(event_data)

        markets_data = [
            {
                "id": "market_lookahead",
                "conditionId": "cond_lookahead",
                "question": "Lookahead Test",
                "clobTokenIds": ["la_t1", "la_t2"],
                "outcomes": ["Team LA1", "Team LA2"],
            }
        ]
        markets = await writer.upsert_markets(event.id, markets_data, event.title)
        market = markets[0]

        commence_time = datetime(2026, 2, 22, 1, 0, 0, tzinfo=UTC)
        target_time = commence_time - timedelta(hours=5)

        # Create snapshots before and after target time
        prices_before = {
            "outcome_0_price": 0.50,
            "outcome_1_price": 0.50,
            "best_bid": 0.49,
            "best_ask": 0.51,
        }
        await writer.store_price_snapshot(
            market, prices_before, commence_time, target_time - timedelta(minutes=2)
        )

        prices_after = {
            "outcome_0_price": 0.60,
            "outcome_1_price": 0.40,
            "best_bid": 0.59,
            "best_ask": 0.61,
        }
        await writer.store_price_snapshot(
            market, prices_after, commence_time, target_time + timedelta(minutes=2)
        )

        # Get price at target time - should only return the "before" snapshot
        snapshot = await reader.get_price_at_time(market.id, target_time)

        assert snapshot is not None
        assert snapshot.outcome_0_price == 0.50  # Before snapshot
        assert snapshot.snapshot_time < target_time  # Never returns future data

    @pytest.mark.asyncio
    async def test_get_price_at_time_exact_match(self, pglite_async_session):
        """Test that get_price_at_time returns exact match when available."""
        writer = PolymarketWriter(pglite_async_session)
        reader = PolymarketReader(pglite_async_session)

        # Create event and market
        event_data = {
            "id": "event_exact",
            "ticker": "nba-exact",
            "slug": "exact",
            "title": "Exact Match Test",
            "startDate": "2026-02-23T01:00:00Z",
            "endDate": "2026-02-23T04:00:00Z",
            "active": True,
            "closed": False,
            "markets": [],
        }
        event = await writer.upsert_event(event_data)

        markets_data = [
            {
                "id": "market_exact",
                "conditionId": "cond_exact",
                "question": "Exact Match Test",
                "clobTokenIds": ["ex_t1", "ex_t2"],
                "outcomes": ["Team E1", "Team E2"],
            }
        ]
        markets = await writer.upsert_markets(event.id, markets_data, event.title)
        market = markets[0]

        commence_time = datetime(2026, 2, 23, 1, 0, 0, tzinfo=UTC)
        exact_time = commence_time - timedelta(hours=10)

        # Create snapshot at exact target time
        prices = {"outcome_0_price": 0.55, "outcome_1_price": 0.45}
        await writer.store_price_snapshot(market, prices, commence_time, exact_time)

        # Get price at exact time
        snapshot = await reader.get_price_at_time(market.id, exact_time)

        assert snapshot is not None
        assert snapshot.snapshot_time == exact_time
        assert snapshot.outcome_0_price == 0.55

    @pytest.mark.asyncio
    async def test_get_price_at_time_tolerance(self, pglite_async_session):
        """Test that get_price_at_time returns None when no snapshots within tolerance."""
        writer = PolymarketWriter(pglite_async_session)
        reader = PolymarketReader(pglite_async_session)

        # Create event and market
        event_data = {
            "id": "event_tolerance",
            "ticker": "nba-tolerance",
            "slug": "tolerance",
            "title": "Tolerance Test",
            "startDate": "2026-02-24T01:00:00Z",
            "endDate": "2026-02-24T04:00:00Z",
            "active": True,
            "closed": False,
            "markets": [],
        }
        event = await writer.upsert_event(event_data)

        markets_data = [
            {
                "id": "market_tolerance",
                "conditionId": "cond_tolerance",
                "question": "Tolerance Test",
                "clobTokenIds": ["tol_t1", "tol_t2"],
                "outcomes": ["Team T1", "Team T2"],
            }
        ]
        markets = await writer.upsert_markets(event.id, markets_data, event.title)
        market = markets[0]

        commence_time = datetime(2026, 2, 24, 1, 0, 0, tzinfo=UTC)
        target_time = commence_time - timedelta(hours=6)

        # Create snapshot outside tolerance window (10 minutes before target)
        prices = {"outcome_0_price": 0.50, "outcome_1_price": 0.50}
        await writer.store_price_snapshot(
            market, prices, commence_time, target_time - timedelta(minutes=10)
        )

        # Get price with default 5-minute tolerance (should return None)
        snapshot = await reader.get_price_at_time(market.id, target_time, tolerance_minutes=5)

        assert snapshot is None

    @pytest.mark.asyncio
    async def test_get_price_series(self, pglite_async_session):
        """Test that get_price_series returns snapshots in time order."""
        writer = PolymarketWriter(pglite_async_session)
        reader = PolymarketReader(pglite_async_session)

        # Create event and market
        event_data = {
            "id": "event_series",
            "ticker": "nba-series",
            "slug": "series",
            "title": "Series Test",
            "startDate": "2026-02-25T01:00:00Z",
            "endDate": "2026-02-25T04:00:00Z",
            "active": True,
            "closed": False,
            "markets": [],
        }
        event = await writer.upsert_event(event_data)

        markets_data = [
            {
                "id": "market_series",
                "conditionId": "cond_series",
                "question": "Series Test",
                "clobTokenIds": ["ser_t1", "ser_t2"],
                "outcomes": ["Team S1", "Team S2"],
            }
        ]
        markets = await writer.upsert_markets(event.id, markets_data, event.title)
        market = markets[0]

        commence_time = datetime(2026, 2, 25, 1, 0, 0, tzinfo=UTC)

        # Create multiple snapshots
        for i in range(5):
            prices = {"outcome_0_price": 0.5 + (i * 0.01), "outcome_1_price": 0.5 - (i * 0.01)}
            snapshot_time = commence_time - timedelta(hours=20 - (i * 4))
            await writer.store_price_snapshot(market, prices, commence_time, snapshot_time)

        # Get series
        start = commence_time - timedelta(hours=24)
        end = commence_time
        series = await reader.get_price_series(market.id, start, end)

        assert len(series) == 5
        # Verify ordered by time
        for i in range(len(series) - 1):
            assert series[i].snapshot_time < series[i + 1].snapshot_time

    @pytest.mark.asyncio
    async def test_get_orderbook_at_time_no_lookahead(self, pglite_async_session):
        """Test that get_orderbook_at_time never returns future data (prevents look-ahead bias)."""
        writer = PolymarketWriter(pglite_async_session)
        reader = PolymarketReader(pglite_async_session)

        # Create event and market
        event_data = {
            "id": "event_ob_lookahead",
            "ticker": "nba-ob-lookahead",
            "slug": "ob-lookahead",
            "title": "OrderBook Lookahead Test",
            "startDate": "2026-02-27T01:00:00Z",
            "endDate": "2026-02-27T04:00:00Z",
            "active": True,
            "closed": False,
            "markets": [],
        }
        event = await writer.upsert_event(event_data)

        markets_data = [
            {
                "id": "market_ob_lookahead",
                "conditionId": "cond_ob_lookahead",
                "question": "OrderBook Lookahead Test",
                "clobTokenIds": ["ob_t1", "ob_t2"],
                "outcomes": ["Team OB1", "Team OB2"],
            }
        ]
        markets = await writer.upsert_markets(event.id, markets_data, event.title)
        market = markets[0]

        commence_time = datetime(2026, 2, 27, 1, 0, 0, tzinfo=UTC)
        target_time = commence_time - timedelta(hours=4)

        # Create order book snapshots before and after target time
        raw_book_before = {
            "bids": [{"price": "0.50", "size": "100"}],
            "asks": [{"price": "0.52", "size": "100"}],
        }
        await writer.store_orderbook_snapshot(
            market, raw_book_before, commence_time, target_time - timedelta(minutes=2)
        )

        raw_book_after = {
            "bids": [{"price": "0.60", "size": "100"}],
            "asks": [{"price": "0.62", "size": "100"}],
        }
        await writer.store_orderbook_snapshot(
            market, raw_book_after, commence_time, target_time + timedelta(minutes=2)
        )
        await pglite_async_session.flush()

        # Get order book at target time - should only return the "before" snapshot
        snapshot = await reader.get_orderbook_at_time(market.id, target_time)

        assert snapshot is not None
        assert snapshot.best_bid == 0.50  # Before snapshot
        assert snapshot.snapshot_time < target_time  # Never returns future data

    @pytest.mark.asyncio
    async def test_get_linked_events(self, pglite_async_session):
        """Test that get_linked_events returns events linked to internal NBA events."""
        from odds_core.models import Event, EventStatus

        writer = PolymarketWriter(pglite_async_session)
        reader = PolymarketReader(pglite_async_session)

        # Create internal NBA event first
        nba_event = Event(
            id="nba_game_123",
            sport_key="basketball_nba",
            sport_title="NBA",
            home_team="Lakers",
            away_team="Celtics",
            commence_time=datetime(2026, 2, 28, 1, 0, 0, tzinfo=UTC),
            status=EventStatus.SCHEDULED,
        )
        pglite_async_session.add(nba_event)
        await pglite_async_session.flush()

        # Create Polymarket event linked to NBA event
        linked_event_data = {
            "id": "pm_linked_event",
            "ticker": "nba-lal-bos-2026-02-28",
            "slug": "lakers-vs-celtics",
            "title": "Lakers vs Celtics",
            "startDate": "2026-02-28T01:00:00Z",
            "endDate": "2026-02-28T04:00:00Z",
            "active": True,
            "closed": False,
            "markets": [],
        }
        pm_event = await writer.upsert_event(linked_event_data)

        # Link the Polymarket event to the NBA event
        pm_event.event_id = nba_event.id
        await pglite_async_session.flush()

        # Create unlinked Polymarket event (should not be returned)
        unlinked_event_data = {
            "id": "pm_unlinked_event",
            "ticker": "nba-unlinked",
            "slug": "unlinked",
            "title": "Unlinked Event",
            "startDate": "2026-03-01T01:00:00Z",
            "endDate": "2026-03-01T04:00:00Z",
            "active": True,
            "closed": False,
            "markets": [],
        }
        await writer.upsert_event(unlinked_event_data)
        await pglite_async_session.flush()

        # Get linked events
        linked_events = await reader.get_linked_events()

        assert len(linked_events) == 1
        pm_evt, nba_evt = linked_events[0]
        assert pm_evt.pm_event_id == "pm_linked_event"
        assert nba_evt.id == "nba_game_123"
        assert pm_evt.event_id == nba_evt.id

    @pytest.mark.asyncio
    async def test_get_backfilled_market_ids(self, pglite_async_session):
        """Test that get_backfilled_market_ids correctly identifies markets with sufficient history."""
        writer = PolymarketWriter(pglite_async_session)
        reader = PolymarketReader(pglite_async_session)

        # Create event
        event_data = {
            "id": "event_backfill",
            "ticker": "nba-backfill",
            "slug": "backfill",
            "title": "Backfill Test",
            "startDate": "2026-02-26T01:00:00Z",
            "endDate": "2026-02-26T04:00:00Z",
            "active": True,
            "closed": False,
            "markets": [],
        }
        event = await writer.upsert_event(event_data)

        # Create two markets
        markets_data = [
            {
                "id": "market_backfilled",
                "conditionId": "cond_bf1",
                "question": "Backfill Test",
                "clobTokenIds": ["bf_t1", "bf_t2"],
                "outcomes": ["Team BF1", "Team BF2"],
            },
            {
                "id": "market_not_backfilled",
                "conditionId": "cond_bf2",
                "question": "Not Backfilled Test",
                "clobTokenIds": ["bf_t3", "bf_t4"],
                "outcomes": ["Team BF3", "Team BF4"],
            },
        ]
        markets = await writer.upsert_markets(event.id, markets_data, event.title)

        commence_time = datetime(2026, 2, 26, 1, 0, 0, tzinfo=UTC)

        # Add 15 snapshots to first market (above threshold)
        history_1 = [
            {"t": int((commence_time - timedelta(hours=i)).timestamp()), "p": str(0.5 + (i * 0.01))}
            for i in range(15)
        ]
        await writer.bulk_store_price_history(markets[0], history_1, commence_time)
        await pglite_async_session.flush()

        # Add 5 snapshots to second market (below threshold)
        history_2 = [
            {"t": int((commence_time - timedelta(hours=i)).timestamp()), "p": str(0.5 + (i * 0.01))}
            for i in range(5)
        ]
        await writer.bulk_store_price_history(markets[1], history_2, commence_time)
        await pglite_async_session.flush()

        # Get backfilled markets with min_snapshots=10
        backfilled_ids = await reader.get_backfilled_market_ids(min_snapshots=10)

        assert markets[0].id in backfilled_ids
        assert markets[1].id not in backfilled_ids
