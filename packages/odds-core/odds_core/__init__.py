"""
Core foundation layer for betting odds pipeline.

Provides models, database connection, and configuration.
"""

from odds_core.api_models import (
    HistoricalOddsResponse,
    OddsResponse,
    ScoresResponse,
    api_dict_to_event,
    create_completed_event,
    create_scheduled_event,
    parse_scores_from_api_dict,
)
from odds_core.config import Settings, get_settings
from odds_core.database import async_session_maker, engine
from odds_core.models import (
    DataQualityLog,
    Event,
    EventStatus,
    FetchLog,
    Odds,
    OddsSnapshot,
)
from odds_core.paper_trade_models import PaperTrade, TradeResult
from odds_core.polymarket_models import (
    PolymarketEvent,
    PolymarketFetchLog,
    PolymarketMarket,
    PolymarketMarketType,
    PolymarketOrderBookSnapshot,
    PolymarketPriceSnapshot,
)
from odds_core.prediction_models import Prediction
from odds_core.team import normalize_team_name

__all__ = [
    # Models
    "Event",
    "EventStatus",
    "Odds",
    "OddsSnapshot",
    "DataQualityLog",
    "FetchLog",
    # Prediction
    "Prediction",
    # Paper Trading
    "PaperTrade",
    "TradeResult",
    # Polymarket Models
    "PolymarketMarketType",
    "PolymarketEvent",
    "PolymarketMarket",
    "PolymarketPriceSnapshot",
    "PolymarketOrderBookSnapshot",
    "PolymarketFetchLog",
    # Database
    "engine",
    "async_session_maker",
    # Config
    "Settings",
    "get_settings",
    # Team normalization
    "normalize_team_name",
    # API Models
    "OddsResponse",
    "ScoresResponse",
    "HistoricalOddsResponse",
    "api_dict_to_event",
    "create_scheduled_event",
    "create_completed_event",
    "parse_scores_from_api_dict",
]
