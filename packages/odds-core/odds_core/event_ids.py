"""Event id conventions shared by the writers, training filters and pairing."""

from __future__ import annotations

LIVE_EVENT_ID_PREFIX = "op_live_"
# Ids minted by the OddsPortal and football-data.co.uk loaders; any other id is The Odds API's.
SCRAPED_EVENT_ID_PREFIXES: tuple[str, ...] = ("op_", "fduk_")


def is_odds_api_event_id(event_id: str) -> bool:
    return not event_id.startswith(SCRAPED_EVENT_ID_PREFIXES)
