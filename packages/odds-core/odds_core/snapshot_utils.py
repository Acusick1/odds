"""Helpers for parsing ``OddsSnapshot.raw_data`` into ``Odds`` rows.

Lives in ``odds-core`` (rather than ``odds-analytics``) so that Lambda code
which only needs to read snapshots does not have to depend on the analytics
package (numpy, pandas, etc.).
"""

from __future__ import annotations

from datetime import datetime

from odds_core.models import Odds, OddsSnapshot


def extract_odds_from_snapshot(
    snapshot: OddsSnapshot,
    event_id: str,
    market: str | None = None,
    outcome: str | None = None,
) -> list[Odds]:
    """Extract ``Odds`` rows from a snapshot's ``raw_data`` JSON.

    Args:
        snapshot: OddsSnapshot with raw_data JSON
        event_id: Event identifier
        market: Optional market key to filter (h2h, spreads, totals)
        outcome: Optional outcome name to filter (team name or Over/Under)

    Returns:
        List of ``Odds`` objects, optionally filtered.
    """
    raw_data = snapshot.raw_data
    if not raw_data or "bookmakers" not in raw_data:
        return []

    odds_list: list[Odds] = []
    for bookmaker_data in raw_data.get("bookmakers", []):
        bookmaker_key = bookmaker_data.get("key")
        bookmaker_title = bookmaker_data.get("title")
        last_update_str = bookmaker_data.get("last_update")

        if last_update_str:
            try:
                last_update = datetime.fromisoformat(last_update_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                last_update = snapshot.snapshot_time
        else:
            last_update = snapshot.snapshot_time

        for market_data in bookmaker_data.get("markets", []):
            market_key = market_data.get("key")
            if market is not None and market_key != market:
                continue

            for outcome_data in market_data.get("outcomes", []):
                outcome_name = outcome_data.get("name")
                if outcome is not None and outcome_name != outcome:
                    continue

                odds_list.append(
                    Odds(
                        event_id=event_id,
                        bookmaker_key=bookmaker_key,
                        bookmaker_title=bookmaker_title,
                        market_key=market_key,
                        outcome_name=outcome_name,
                        price=outcome_data.get("price"),
                        point=outcome_data.get("point"),
                        odds_timestamp=snapshot.snapshot_time,
                        last_update=last_update,
                        is_valid=True,
                    )
                )

    return odds_list
