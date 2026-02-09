"""Shared fixtures for odds-lambda tests."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from odds_core.polymarket_models import PolymarketEvent, PolymarketMarket, PolymarketMarketType


@pytest.fixture
def mock_polymarket_client():
    """Create mock PolymarketClient for testing."""
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    return mock_client


@pytest.fixture
def mock_async_session():
    """Create mock async database session for testing."""
    mock_session = AsyncMock()
    mock_session.__aenter__.return_value = mock_session
    mock_session.__aexit__.return_value = None
    mock_session.commit = AsyncMock()
    mock_session.rollback = AsyncMock()
    return mock_session


@pytest.fixture
def mock_polymarket_reader():
    """Create mock PolymarketReader for testing."""
    mock_reader = AsyncMock()
    mock_reader.get_backfilled_market_ids.return_value = set()
    return mock_reader


@pytest.fixture
def mock_polymarket_writer():
    """Create mock PolymarketWriter for testing."""
    return AsyncMock()


@pytest.fixture
def sample_polymarket_event():
    """Create sample PolymarketEvent for testing."""
    return PolymarketEvent(
        id=1,
        pm_event_id="test-event-123",
        ticker="nba-test-game",
        slug="test-game-slug",
        title="Test Team vs Test Team",
        start_date=datetime(2024, 1, 15, 19, 0, 0, tzinfo=UTC),
        end_date=datetime(2024, 1, 15, 22, 0, 0, tzinfo=UTC),
        active=False,
        closed=True,
        volume=10000.0,
        liquidity=5000.0,
        markets_count=3,
    )


@pytest.fixture
def sample_polymarket_market():
    """Create sample PolymarketMarket for testing."""
    return PolymarketMarket(
        id=1,
        polymarket_event_id=1,
        pm_market_id="test-market-123",
        condition_id="test-condition",
        question="Test Team vs Test Team",
        clob_token_ids=["test-token-id-1", "test-token-id-2"],
        outcomes=["Test Team", "Test Team"],
        market_type=PolymarketMarketType.MONEYLINE,
        group_item_title=None,
        point=None,
        active=False,
        closed=True,
        accepting_orders=False,
    )


@pytest.fixture
def sample_event_data():
    """Sample Gamma API event response data."""
    return {
        "id": "event-123",
        "ticker": "nba-lal-bos-2024-01-15",
        "slug": "lakers-vs-celtics",
        "title": "Lakers vs Celtics",
        "startDate": "2024-01-15T19:00:00Z",
        "endDate": "2024-01-15T22:00:00Z",
        "active": False,
        "closed": True,
        "volume": "10000",
        "liquidity": "5000",
        "markets": [
            {
                "id": "market-ml-123",
                "conditionId": "condition-ml-123",
                "question": "Lakers vs Celtics",
                "clobTokenIds": ["token-lal", "token-bos"],
                "outcomes": ["Lakers", "Celtics"],
                "active": False,
                "closed": True,
                "acceptingOrders": False,
            }
        ],
    }


@pytest.fixture
def sample_price_history():
    """Sample CLOB API price history response."""
    return [
        {"t": 1705339200, "p": "0.48"},  # 2024-01-15 14:00:00 UTC
        {"t": 1705339500, "p": "0.49"},  # 2024-01-15 14:05:00 UTC
        {"t": 1705339800, "p": "0.50"},  # 2024-01-15 14:10:00 UTC
        {"t": 1705340100, "p": "0.51"},  # 2024-01-15 14:15:00 UTC
        {"t": 1705340400, "p": "0.52"},  # 2024-01-15 14:20:00 UTC
    ]


@pytest.fixture
def mock_settings():
    """Create mock Settings for testing."""
    mock = MagicMock()
    mock.polymarket.enabled = True
    return mock
