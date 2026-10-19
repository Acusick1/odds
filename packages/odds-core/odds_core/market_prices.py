"""Outcome names and per-book decimal prices read from a snapshot's ``raw_data``."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from odds_core.bookmakers import canonical_bookmaker_key
from odds_core.models import Event
from odds_core.odds_math import outcome_price_decimal
from odds_core.snapshot_sources import SnapshotKind

DRAW_OUTCOME_NAME = "Draw"
THREE_WAY_MARKET = "1x2"
TWO_WAY_MARKET = "h2h"


class Outcome(StrEnum):
    HOME = "home"
    DRAW = "draw"
    AWAY = "away"


def market_outcomes(market: str) -> tuple[Outcome, ...]:
    if market == THREE_WAY_MARKET:
        return (Outcome.HOME, Outcome.DRAW, Outcome.AWAY)
    if market == TWO_WAY_MARKET:
        return (Outcome.HOME, Outcome.AWAY)
    raise ValueError(f"No outcomes defined for market {market!r}")


def outcome_names(event: Event, market: str) -> dict[str, Outcome]:
    names = {event.home_team: Outcome.HOME, event.away_team: Outcome.AWAY}
    if Outcome.DRAW in market_outcomes(market):
        names[DRAW_OUTCOME_NAME] = Outcome.DRAW
    return names


def book_prices(
    raw_data: dict[str, Any], kind: SnapshotKind, market: str, names: dict[str, Outcome]
) -> dict[str, dict[Outcome, float]]:
    """Valid decimal prices per canonical book key; the first entry wins for an aliased book."""
    books: dict[str, dict[Outcome, float]] = {}
    for bookmaker in raw_data.get("bookmakers", []):
        key = canonical_bookmaker_key(bookmaker.get("key", ""), kind)
        if key in books:
            continue
        for mkt in bookmaker.get("markets", []):
            if mkt.get("key") != market:
                continue
            prices: dict[Outcome, float] = {}
            for outcome in mkt.get("outcomes", []):
                side = names.get(outcome.get("name"))
                if side is None:
                    continue
                price = outcome_price_decimal(outcome)
                if price > 1.0:
                    prices[side] = price
            if prices:
                books[key] = prices
            break
    return books
