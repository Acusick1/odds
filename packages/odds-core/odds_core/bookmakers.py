"""Bookmaker roles shared by validators, feature selection and the MCP tools."""

from __future__ import annotations

from typing import Any

EXCHANGE_BOOKMAKERS: frozenset[str] = frozenset(
    {"betfair_exchange", "betdaq", "smarkets", "matchbook"}
)
PREDICTION_MARKET_BOOKMAKERS: frozenset[str] = frozenset({"polymarket"})
# Venues that quote no meaningful overround.
ZERO_MARGIN_BOOKMAKERS: frozenset[str] = EXCHANGE_BOOKMAKERS | PREDICTION_MARKET_BOOKMAKERS
# Venues a UK bettor cannot bet on, so never a retail price to find edges against.
NON_BETTABLE_BOOKMAKERS: frozenset[str] = frozenset({"pinnacle", "polymarket"})


def is_reference_only_snapshot(raw_data: dict[str, Any] | None) -> bool:
    """True when every bookmaker in the snapshot is a prediction market."""
    keys = {bm.get("key") for bm in (raw_data or {}).get("bookmakers", [])}
    return bool(keys) and keys <= PREDICTION_MARKET_BOOKMAKERS
