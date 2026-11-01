"""
Core foundation layer for betting odds pipeline.

Provides models, database connection, and configuration.
"""

from odds_core.api_models import (
    ApiEvent,
    HistoricalOddsResponse,
    OddsResponse,
    ScoresResponse,
    parse_api_event,
    parse_scores_from_api_dict,
)
from odds_core.config import Settings, get_settings
from odds_core.database import async_session_maker, get_engine
from odds_core.models import (
    DataQualityLog,
    Event,
    EventStatus,
    FetchLog,
    OddsSnapshot,
)
from odds_core.paper_trade_models import PaperTrade, TradeResult
from odds_core.team import normalize_team_name, team_abbrev

__all__ = [
    # Models
    "Event",
    "EventStatus",
    "OddsSnapshot",
    "DataQualityLog",
    "FetchLog",
    # Paper Trading
    "PaperTrade",
    "TradeResult",
    # Database
    "get_engine",
    "async_session_maker",
    # Config
    "Settings",
    "get_settings",
    # Team normalization
    "normalize_team_name",
    "team_abbrev",
    # API Models
    "OddsResponse",
    "ScoresResponse",
    "HistoricalOddsResponse",
    "ApiEvent",
    "parse_api_event",
    "parse_scores_from_api_dict",
]
