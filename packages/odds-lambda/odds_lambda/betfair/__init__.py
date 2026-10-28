"""Betfair Exchange API integration.

Direct ingestion of each sport's markets via the delayed application key, stored as
``betfair`` snapshots whose price rows are under the bookmaker ``betfair_exchange``.
"""

from odds_lambda.betfair.adapter import (
    betfair_book_to_bookmaker_entry,
    doubleheader_game_number,
    resolve_teams,
)
from odds_lambda.betfair.client import BetfairBook, BetfairEvent, BetfairExchangeClient
from odds_lambda.betfair.constants import (
    EPL_COMPETITION_ID,
    SPORT_CONFIG,
    SportBetfairConfig,
)

__all__ = [
    "BetfairBook",
    "BetfairEvent",
    "BetfairExchangeClient",
    "EPL_COMPETITION_ID",
    "SPORT_CONFIG",
    "SportBetfairConfig",
    "betfair_book_to_bookmaker_entry",
    "doubleheader_game_number",
    "resolve_teams",
]
