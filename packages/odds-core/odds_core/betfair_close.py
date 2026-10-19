"""The live Betfair snapshot that ADR-0012 takes as the close of an event."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from odds_core.bookmakers import canonical_bookmaker_key
from odds_core.snapshot_sources import SnapshotKind

LIVE_CLOSE_WINDOW = timedelta(minutes=30)
EXCHANGE_BOOKMAKER = "betfair_exchange"


def is_in_play(raw_data: dict[str, Any]) -> bool:
    return any(
        bool((bm.get("betfair_meta") or {}).get("inplay"))
        for bm in raw_data.get("bookmakers", [])
        if canonical_bookmaker_key(bm.get("key", ""), SnapshotKind.BETFAIR_LIVE)
        == EXCHANGE_BOOKMAKER
    )
