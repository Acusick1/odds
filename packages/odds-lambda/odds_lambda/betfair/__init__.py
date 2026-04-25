"""Betfair Exchange API integration.

Direct ingestion of Match Odds via the delayed application key, written into
``odds_snapshots.raw_data`` with ``key="betfair_exchange"`` so the existing
agent / analytics / sharp-fallback consumers pick it up unchanged.
"""

from odds_lambda.betfair.adapter import betfair_book_to_bookmaker_entry, resolve_teams
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
    "resolve_teams",
]
