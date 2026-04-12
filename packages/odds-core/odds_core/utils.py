"""Shared utility functions for odds pipeline."""

from typing import Any


def raw_data_has_market(raw_data: dict[str, Any] | None, market: str) -> bool:
    """Check if raw_data JSON contains data for the given market key.

    Iterates bookmakers -> markets -> key to find a match.
    """
    if not raw_data:
        return False
    for bookmaker in raw_data.get("bookmakers", []):
        for m in bookmaker.get("markets", []):
            if m.get("key") == market:
                return True
    return False
