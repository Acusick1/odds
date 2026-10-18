"""What kind of price an ``OddsSnapshot`` holds, read from its ``raw_data`` tags."""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class SnapshotSource(StrEnum):
    """Values producers write to ``raw_data["source"]``; the Odds API writes none."""

    ODDSPORTAL = "oddsportal"
    ODDSPORTAL_LIVE = "oddsportal_live"
    BETFAIR_API = "betfair_api"
    FOOTBALL_DATA_UK = "football_data_uk"
    POLYMARKET = "polymarket"


class FdukTimeBasis(StrEnum):
    """``raw_data["timing"]["basis"]`` of a football-data.co.uk snapshot."""

    COLLECTION_DEADLINE = "fduk_collection_deadline"
    KICKOFF = "kickoff"


class SnapshotKind(StrEnum):
    """A snapshot's source, with FDUK split into its opening and closing prices."""

    ODDS_API = "odds_api"
    ODDSPORTAL = "oddsportal"
    ODDSPORTAL_LIVE = "oddsportal_live"
    BETFAIR_LIVE = "betfair_live"
    FDUK_OPENING = "fduk_opening"
    FDUK_CLOSING = "fduk_closing"
    POLYMARKET = "polymarket"


_KIND_BY_SOURCE: dict[SnapshotSource, SnapshotKind] = {
    SnapshotSource.ODDSPORTAL: SnapshotKind.ODDSPORTAL,
    SnapshotSource.ODDSPORTAL_LIVE: SnapshotKind.ODDSPORTAL_LIVE,
    SnapshotSource.BETFAIR_API: SnapshotKind.BETFAIR_LIVE,
    SnapshotSource.POLYMARKET: SnapshotKind.POLYMARKET,
}

_FDUK_KIND_BY_BASIS: dict[FdukTimeBasis, SnapshotKind] = {
    FdukTimeBasis.COLLECTION_DEADLINE: SnapshotKind.FDUK_OPENING,
    FdukTimeBasis.KICKOFF: SnapshotKind.FDUK_CLOSING,
}


def snapshot_kind(raw_data: dict[str, Any]) -> SnapshotKind:
    """Classify a snapshot by its ``raw_data`` tags.

    Raises:
        ValueError: for a source tag or FDUK timing basis this module does not know.
    """
    tag = raw_data.get("source")
    if tag is None:
        return SnapshotKind.ODDS_API
    source = SnapshotSource(tag)
    if source is SnapshotSource.FOOTBALL_DATA_UK:
        return _FDUK_KIND_BY_BASIS[FdukTimeBasis(raw_data["timing"]["basis"])]
    return _KIND_BY_SOURCE[source]
