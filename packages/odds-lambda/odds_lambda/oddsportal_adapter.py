"""Convert OddsPortal feed prices to pipeline raw_data format.

Pure conversion module: no database access, no network calls.
Produces dicts matching the OddsWriter.store_odds_snapshot() contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from odds_lambda.oddsportal_client import (
    FEED_MARKETS,
    BookmakerPrices,
    FeedMarketSpec,
    ListingEvent,
)
from odds_lambda.oddsportal_common import (
    DRAW_OUTCOME,
    decimal_to_american,
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
    raw_data: dict[str, Any]
    bookmaker_count: int


def _outcomes(
    event: ListingEvent, spec: FeedMarketSpec, prices: tuple[int, ...]
) -> list[dict[str, Any]]:
    names = {
        "home": event.home_team,
        "away": event.away_team,
        "draw": DRAW_OUTCOME,
        "over": "Over",
        "under": "Under",
    }
    outcomes: list[dict[str, Any]] = []
    for label, price in zip(spec.outcomes, prices, strict=True):
        outcome: dict[str, Any] = {"name": names[label], "price": price}
        if spec.point is not None:
            outcome["point"] = spec.point
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
        market: LeagueSpec market key: "1x2", "over_under_2_5" or "home_away".
        bookmakers: Prices from ``extract_market``.
        scraped_at: Time the feed was read; stamps the snapshot.

    Returns:
        ``MatchOdds``, or ``None`` when no bookmaker has prices.
    """
    spec = FEED_MARKETS.get(market)
    if spec is None:
        raise ValueError(f"Unsupported market: {market}")

    entries: list[dict[str, Any]] = []
    for bk in bookmakers:
        american = tuple(decimal_to_american(p) for p in bk.prices)
        # The bookmaker's own change time is its update time; the validator rejects
        # future timestamps, so cap it at the scrape time.
        last_update = min(bk.last_change, scraped_at) if bk.last_change else scraped_at
        entry: dict[str, Any] = {
            "key": normalize_bookmaker_key(bk.name),
            "title": bk.name,
            "last_update": _iso_z(last_update),
            "markets": [
                {
                    "key": spec.pipeline_key,
                    "outcomes": _outcomes(event, spec, american),
                }
            ],
        }
        if bk.volumes is not None:
            entry["betfair_matched"] = dict(zip(spec.outcomes, bk.volumes, strict=True))
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
        raw_data={"bookmakers": entries, "source": "oddsportal_live"},
        bookmaker_count=len(entries),
    )
