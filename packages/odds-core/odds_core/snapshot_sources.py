"""The source and FDUK timing tags producers write to an ``OddsSnapshot``'s ``raw_data``."""

from __future__ import annotations

from enum import StrEnum


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
