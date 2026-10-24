"""The live Betfair snapshot that ADR-0012 takes as the close of an event."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from odds_core.bookmakers import canonical_bookmaker_key
from odds_core.market_prices import Outcome, book_prices, market_outcomes, outcome_names
from odds_core.models import Event, OddsSnapshot
from odds_core.odds_math import devig_probabilities
from odds_core.snapshot_sources import SnapshotKind, snapshot_kind

LIVE_CLOSE_WINDOW = timedelta(minutes=30)
EXCHANGE_BOOKMAKER = "betfair_exchange"
# A wider back book is a thin market whose best backs are not a price (ADR-0014).
MAX_EXCHANGE_OVERROUND = 0.05


@dataclass(frozen=True)
class ExchangePrice:
    """Devigged Betfair Exchange back prices from one snapshot."""

    time: datetime
    kind: SnapshotKind
    fair: dict[Outcome, float]
    overround: float


def is_in_play(raw_data: dict[str, Any]) -> bool:
    return any(
        bool((bm.get("betfair_meta") or {}).get("inplay"))
        for bm in raw_data.get("bookmakers", [])
        if canonical_bookmaker_key(bm.get("key", ""), SnapshotKind.BETFAIR_LIVE)
        == EXCHANGE_BOOKMAKER
    )


def complete_exchange_prices(
    raw_data: dict[str, Any], kind: SnapshotKind, market: str, names: dict[str, Outcome]
) -> dict[Outcome, float] | None:
    """The exchange's back price for every outcome of ``market``, or None when any is missing."""
    prices = book_prices(raw_data, kind, market, names).get(EXCHANGE_BOOKMAKER)
    if prices is None or set(prices) != set(market_outcomes(market)):
        return None
    return prices


def devig(prices: dict[Outcome, float], outcomes: tuple[Outcome, ...]) -> dict[Outcome, float]:
    fair = devig_probabilities(*(1 / prices[o] for o in outcomes))
    return dict(zip(outcomes, fair, strict=True))


def overround(prices: dict[Outcome, float], outcomes: tuple[Outcome, ...]) -> float:
    return sum(1 / prices[o] for o in outcomes) - 1


def exchange_price(
    snapshot: OddsSnapshot,
    kind: SnapshotKind,
    market: str,
    names: dict[str, Outcome],
    outcomes: tuple[Outcome, ...],
) -> ExchangePrice | None:
    prices = complete_exchange_prices(snapshot.raw_data, kind, market, names)
    if prices is None:
        return None
    return ExchangePrice(
        time=snapshot.snapshot_time,
        kind=kind,
        fair=devig(prices, outcomes),
        overround=overround(prices, outcomes),
    )


def is_live_close(price: ExchangePrice, kickoff: datetime) -> bool:
    """Whether a live Betfair price falls in the window that can be the close."""
    return (
        price.kind is SnapshotKind.BETFAIR_LIVE
        and kickoff - LIVE_CLOSE_WINDOW <= price.time <= kickoff
    )


def event_close(
    event: Event,
    snapshots: Sequence[OddsSnapshot],
    market: str,
    max_exchange_overround: float = MAX_EXCHANGE_OVERROUND,
) -> ExchangePrice | None:
    """The devigged Betfair Exchange close of an event, or None when it has none.

    The close is the last live Betfair snapshot that is not in-play within
    ``LIVE_CLOSE_WINDOW`` before kick-off, else the FDUK closing price (stamped at
    kick-off). Exchange prices whose back overround exceeds ``max_exchange_overround``
    are not prices.
    """
    outcomes = market_outcomes(market)
    names = outcome_names(event, market)
    live_close: ExchangePrice | None = None
    fduk_close: ExchangePrice | None = None
    for snapshot in sorted(snapshots, key=lambda s: s.snapshot_time):
        kind = snapshot_kind(snapshot.raw_data)
        if kind is SnapshotKind.BETFAIR_LIVE:
            if is_in_play(snapshot.raw_data):
                continue
        elif kind is not SnapshotKind.FDUK_CLOSING:
            continue
        price = exchange_price(snapshot, kind, market, names, outcomes)
        if price is None or price.overround > max_exchange_overround:
            continue
        if kind is SnapshotKind.FDUK_CLOSING:
            fduk_close = price
        elif is_live_close(price, event.commence_time):
            live_close = price
    return live_close or fduk_close
