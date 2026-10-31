"""Convert OddsPortal feed prices to the envelope shape the writers store.

Pure conversion module: no database access, no network calls.
Produces envelopes for ``OddsWriter.store_envelope``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from odds_core.odds_math import PRICE_DECIMAL_KEY

from odds_lambda.oddsportal_client import (
    FEED_MARKETS,
    BookmakerPrices,
    FeedMarketSpec,
    ListingEvent,
)
from odds_lambda.oddsportal_common import (
    DRAW_OUTCOME,
    normalize_bookmaker_key,
)


@dataclass
class MatchOdds:
    """Converted match ready for ingestion."""

    market: str
    home_team: str
    away_team: str
    match_date: datetime
    match_link: str
    scraped_date: datetime
    envelope: dict[str, Any]
    bookmaker_count: int


def _outcomes(
    event: ListingEvent, spec: FeedMarketSpec, bk: BookmakerPrices
) -> list[dict[str, Any]]:
    names = {
        "home": event.home_team,
        "away": event.away_team,
        "draw": DRAW_OUTCOME,
        "over": "Over",
        "under": "Under",
    }
    outcomes: list[dict[str, Any]] = []
    for label, price in zip(spec.outcomes, bk.prices, strict=True):
        outcome: dict[str, Any] = {"name": names[label], PRICE_DECIMAL_KEY: price}
        line = bk.line if bk.line is not None else spec.point
        if line is not None:
            # A spread's feed line is the home side's; each side carries its own sign.
            mirrored = spec.mirror_line and label == "away"
            outcome["point"] = -line if mirrored else line
        outcomes.append(outcome)
    return outcomes


def _iso_z(moment: datetime) -> str:
    return moment.isoformat().replace("+00:00", "Z")


def convert_market(
    event: ListingEvent,
    market: str,
    bookmakers: list[BookmakerPrices],
    scraped_at: datetime,
) -> MatchOdds | None:
    """Convert one event's feed prices for a market to ``MatchOdds``.

    Args:
        event: Listing entry the prices belong to.
        market: LeagueSpec market key: a ``FEED_MARKETS`` key.
        bookmakers: Prices from ``extract_market``.
        scraped_at: Time the feed was read; stamps the snapshot.

    Returns:
        ``MatchOdds``, or ``None`` when no bookmaker has prices.
    """
    spec = FEED_MARKETS.get(market)
    if spec is None:
        raise ValueError(f"Unsupported market: {market}")

    by_book: dict[tuple[str, str], list[BookmakerPrices]] = {}
    for bk in bookmakers:
        by_book.setdefault((bk.bookmaker_id, bk.name), []).append(bk)

    entries: list[dict[str, Any]] = []
    for rows in by_book.values():
        first = rows[0]
        # The bookmaker's own change time is its update time; the validator rejects
        # future timestamps, so cap it at the scrape time.
        changes = [r.last_change for r in rows if r.last_change]
        last_update = min(max(changes), scraped_at) if changes else scraped_at
        entry: dict[str, Any] = {
            "key": normalize_bookmaker_key(first.name),
            "title": first.name,
            "last_update": _iso_z(last_update),
            "markets": [
                {
                    "key": spec.pipeline_key,
                    "outcomes": [o for r in rows for o in _outcomes(event, spec, r)],
                }
            ],
        }
        # Volumes are per outcome and the entry's rows would all take them, so a market
        # quoting several lines carries none.
        if first.volumes is not None and not spec.ladder:
            entry["betfair_matched"] = dict(zip(spec.outcomes, first.volumes, strict=True))
        entries.append(entry)

    if not entries:
        return None

    return MatchOdds(
        market=market,
        home_team=event.home_team,
        away_team=event.away_team,
        match_date=event.start_time,
        match_link=event.url,
        scraped_date=scraped_at,
        envelope={
            "bookmakers": entries,
            "source": "oddsportal_live",
            "oddsportal_event_id": event.event_id,
        },
        bookmaker_count=len(entries),
    )
