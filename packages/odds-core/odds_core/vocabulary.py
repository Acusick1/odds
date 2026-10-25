"""Closed sets stored as CHECKed text in the snapshot, price and event-identity tables."""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from typing import Any

from sqlalchemy import CheckConstraint, Dialect, Text
from sqlalchemy.types import TypeDecorator


class Source(StrEnum):
    """A provider of prices, event ids or kick-off times."""

    ODDS_API = "odds_api"
    ODDSPORTAL = "oddsportal"
    BETFAIR = "betfair"
    FOOTBALL_DATA_UK = "football_data_uk"
    POLYMARKET = "polymarket"
    MLB_STATS_API = "mlb_stats_api"
    ESPN = "espn"


# Providers whose observations become snapshots; historical and live OddsPortal are one source.
PRICE_SOURCES: frozenset[Source] = frozenset(
    {
        Source.ODDS_API,
        Source.ODDSPORTAL,
        Source.BETFAIR,
        Source.FOOTBALL_DATA_UK,
        Source.POLYMARKET,
    }
)
# Providers that issue their own event ids; football-data.co.uk issues none.
EXTERNAL_ID_SOURCES: frozenset[Source] = frozenset(
    {
        Source.ODDS_API,
        Source.ODDSPORTAL,
        Source.BETFAIR,
        Source.POLYMARKET,
        Source.MLB_STATS_API,
        Source.ESPN,
    }
)


class TimeBasis(StrEnum):
    """What a snapshot's ``snapshot_time`` means."""

    OBSERVED = "observed"
    SOURCE_STAMPED = "source_stamped"
    UPPER_BOUND = "upper_bound"
    CLOSE = "close"


class Market(StrEnum):
    ONE_X_TWO = "1x2"
    H2H = "h2h"
    SPREADS = "spreads"
    TOTALS = "totals"


class PriceOutcome(StrEnum):
    HOME = "home"
    DRAW = "draw"
    AWAY = "away"
    OVER = "over"
    UNDER = "under"


class Side(StrEnum):
    BACK = "back"
    LAY = "lay"
    MID = "mid"
    TRADED = "traded"


class HealthCheck(StrEnum):
    """A health check an event can be acknowledged as failing for a known reason."""

    MISSING_MLB_GAME_PK = "missing_mlb_game_pk"


class StrEnumText[E: StrEnum](TypeDecorator[E]):
    """A ``StrEnum`` stored as its value in a TEXT column; pair it with ``closed_set_check``."""

    impl = Text
    cache_ok = True

    def __init__(self, enum_type: type[E]) -> None:
        super().__init__()
        self.enum_type = enum_type

    def process_bind_param(self, value: Any, dialect: Dialect) -> str | None:
        return None if value is None else self.enum_type(value).value

    def process_result_value(self, value: Any, dialect: Dialect) -> E | None:
        return None if value is None else self.enum_type(value)


def closed_set_check(column: str, members: Iterable[StrEnum], name: str) -> CheckConstraint:
    listed = ", ".join(f"'{value}'" for value in sorted(member.value for member in members))
    return CheckConstraint(f"{column} IN ({listed})", name=name)
