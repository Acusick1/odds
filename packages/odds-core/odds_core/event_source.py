from __future__ import annotations

from enum import StrEnum

from odds_core.event_ids import LIVE_EVENT_ID_PREFIX


class EventSource(StrEnum):
    """Which pipeline minted an event id, in canonical-merge order (first wins)."""

    ODDS_API = ""
    ODDSPORTAL_HISTORICAL = "op_"
    ODDSPORTAL_LIVE = LIVE_EVENT_ID_PREFIX
    FOOTBALL_DATA_UK = "fduk_"

    @classmethod
    def of(cls, event_id: str) -> EventSource:
        if event_id.startswith(cls.ODDSPORTAL_LIVE.value):
            return cls.ODDSPORTAL_LIVE
        if event_id.startswith(cls.ODDSPORTAL_HISTORICAL.value):
            return cls.ODDSPORTAL_HISTORICAL
        if event_id.startswith(cls.FOOTBALL_DATA_UK.value):
            return cls.FOOTBALL_DATA_UK
        return cls.ODDS_API

    @property
    def rank(self) -> int:
        return tuple(type(self)).index(self)


def canonical_event_id(event_ids: list[str]) -> str:
    """The event kept when ids name one match: best source, then smallest id."""
    return min(event_ids, key=lambda event_id: (EventSource.of(event_id).rank, event_id))
