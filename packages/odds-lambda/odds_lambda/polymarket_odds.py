"""Polymarket match markets as ``polymarket`` odds snapshots."""

from __future__ import annotations

import json
import re
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal

import structlog
from odds_core.event_ids import is_odds_api_event_id
from odds_core.models import Event
from odds_core.odds_math import PRICE_DECIMAL_KEY
from odds_core.team import normalize_team_name

from odds_lambda.fetch_tier import FetchTier
from odds_lambda.polymarket_sports import PolymarketSportConfig

logger = structlog.get_logger()

Side = Literal["home", "draw", "away"]
SIDES: tuple[Side, ...] = ("home", "draw", "away")
Observation = tuple[datetime, float]

BOOKMAKER_KEY = "polymarket"
BOOKMAKER_TITLE = "Polymarket"
SNAPSHOT_SOURCE = "polymarket"
HISTORY_REQUEST_ID = "polymarket_history"
LIVE_REQUEST_ID_PREFIX = "polymarket_live_"
DRAW_OUTCOME = "Draw"
DRAW_SLUG_SUFFIX = "draw"

_MAX_HORIZON_HOURS = 168.0
MAX_HORIZON = timedelta(hours=_MAX_HORIZON_HOURS)
# Grid spacing per fetch tier, shared by the backfill and the live job cadence.
TIER_STEPS: dict[FetchTier, timedelta] = {
    FetchTier.CLOSING: timedelta(minutes=5),
    FetchTier.PREGAME: timedelta(minutes=15),
    FetchTier.SHARP: timedelta(minutes=15),
    FetchTier.EARLY: timedelta(minutes=30),
    FetchTier.OPENING: timedelta(hours=1),
}
# The three Yes prices sum to 1 within 0.03 from three days out; wider gaps are placeholder quotes.
SUM_TOLERANCE = 0.03
MAX_STALENESS = timedelta(minutes=30)
# A live midpoint is only quoted when the book is tight; a wider spread is an empty book.
MAX_QUOTE_SPREAD = 0.03
EVENT_MATCH_WINDOW = timedelta(hours=3)

_CLUB_SUFFIX = re.compile(r"\s+A?FC$")


@dataclass(frozen=True)
class Prices:
    home: float
    draw: float
    away: float

    @property
    def total(self) -> float:
        return self.home + self.draw + self.away

    def by_side(self) -> dict[Side, float]:
        return {"home": self.home, "draw": self.draw, "away": self.away}


@dataclass(frozen=True)
class OutcomeMarket:
    pm_market_id: str
    yes_token_id: str
    best_bid: float | None
    best_ask: float | None

    @property
    def spread(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid

    @property
    def midpoint(self) -> float | None:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2


@dataclass(frozen=True)
class PolymarketMatch:
    pm_event_id: str
    ticker: str
    start_time: datetime
    home_team: str
    away_team: str
    home: OutcomeMarket
    draw: OutcomeMarket
    away: OutcomeMarket

    def outcome_market(self, side: Side) -> OutcomeMarket:
        return {"home": self.home, "draw": self.draw, "away": self.away}[side]

    def max_spread(self) -> float | None:
        spreads = [self.outcome_market(side).spread for side in SIDES]
        return None if any(spread is None for spread in spreads) else max(spreads)  # type: ignore[type-var]

    def midpoints(self) -> Prices | None:
        mids = {side: self.outcome_market(side).midpoint for side in SIDES}
        if any(mid is None or not 0.0 < mid < 1.0 for mid in mids.values()):
            return None
        return Prices(**mids)  # type: ignore[arg-type]


@dataclass(frozen=True)
class GridPoint:
    time: datetime
    prices: Prices


@dataclass
class Resampled:
    points: list[GridPoint] = field(default_factory=list)
    missing: int = 0
    off_sum: int = 0


@dataclass
class Pairing:
    pairs: list[tuple[Event, PolymarketMatch]] = field(default_factory=list)
    twins: list[tuple[PolymarketMatch, list[Event]]] = field(default_factory=list)
    unmatched_events: list[Event] = field(default_factory=list)
    ambiguous_events: list[Event] = field(default_factory=list)
    unmatched_matches: list[PolymarketMatch] = field(default_factory=list)


def canonical_team(name: str) -> str:
    return normalize_team_name(_CLUB_SUFFIX.sub("", name))


def _outcome_market(market: dict[str, Any]) -> OutcomeMarket:
    bid, ask = market.get("bestBid"), market.get("bestAsk")
    return OutcomeMarket(
        pm_market_id=str(market["id"]),
        yes_token_id=json.loads(market["clobTokenIds"])[0],
        best_bid=None if bid is None else float(bid),
        best_ask=None if ask is None else float(ask),
    )


def parse_match_event(
    event: dict[str, Any], sport: PolymarketSportConfig
) -> PolymarketMatch | None:
    slug_match = sport.match_slug.match(event.get("slug", ""))
    if slug_match is None:
        return None
    names = {team["ordering"]: team["name"] for team in event.get("teams", [])}
    if "home" not in names or "away" not in names:
        return None

    slug_suffix: dict[str, Side] = {
        slug_match["home"]: "home",
        DRAW_SLUG_SUFFIX: "draw",
        slug_match["away"]: "away",
    }
    by_side: dict[Side, dict[str, Any]] = {}
    for market in event.get("markets", []):
        side = slug_suffix.get(market.get("slug", "").removeprefix(f"{event['slug']}-"))
        if side is None or side in by_side:
            logger.warning(
                "polymarket_unclassified_market", ticker=event["slug"], market=market.get("slug")
            )
            return None
        by_side[side] = market
    if len(by_side) != len(SIDES):
        logger.warning("polymarket_incomplete_match_markets", ticker=event["slug"])
        return None

    return PolymarketMatch(
        pm_event_id=str(event["id"]),
        ticker=event["slug"],
        start_time=datetime.fromisoformat(event["startTime"]),
        home_team=canonical_team(names["home"]),
        away_team=canonical_team(names["away"]),
        home=_outcome_market(by_side["home"]),
        draw=_outcome_market(by_side["draw"]),
        away=_outcome_market(by_side["away"]),
    )


def preferred_event(events: Sequence[Event]) -> Event:
    """The Odds API event when there is one, as ``OddsWriter.find_or_create_event`` prefers."""
    api = [e for e in events if is_odds_api_event_id(e.id)]
    return min(api or events, key=lambda e: e.id)


def pair_events(events: Sequence[Event], matches: Sequence[PolymarketMatch]) -> Pairing:
    """Pair events with matches, one event per match.

    Several events can match one Polymarket match when one fixture exists under
    several ids, a data defect: only the preferred event is paired and the rest
    are listed in ``twins``.
    """
    by_teams: dict[tuple[str, str], list[PolymarketMatch]] = defaultdict(list)
    for match in matches:
        by_teams[(match.home_team, match.away_team)].append(match)

    pairing = Pairing()
    claimed: dict[str, list[Event]] = defaultdict(list)
    by_id = {m.pm_event_id: m for m in matches}
    for event in events:
        candidates = [
            match
            for match in by_teams[(event.home_team, event.away_team)]
            if abs(match.start_time - event.commence_time) <= EVENT_MATCH_WINDOW
        ]
        if len(candidates) == 1:
            claimed[candidates[0].pm_event_id].append(event)
        elif candidates:
            pairing.ambiguous_events.append(event)
        else:
            pairing.unmatched_events.append(event)

    for pm_event_id, claimants in claimed.items():
        match = by_id[pm_event_id]
        chosen = preferred_event(claimants)
        pairing.pairs.append((chosen, match))
        if len(claimants) > 1:
            pairing.twins.append((match, claimants))
            logger.warning(
                "polymarket_event_twins",
                ticker=match.ticker,
                event_ids=sorted(e.id for e in claimants),
                chosen=chosen.id,
            )
    pairing.pairs.sort(key=lambda pair: pair[0].commence_time)
    pairing.unmatched_matches = [m for m in matches if m.pm_event_id not in claimed]
    return pairing


def grid_offsets() -> list[timedelta]:
    """Times before kick-off on the snapshot grid, ascending from 0.

    Each tier's range holds the multiples of its step, so the tier boundaries
    (3 h, 12 h, 24 h, 72 h) are grid points at every spacing.
    """
    offsets = [timedelta(0)]
    lower = timedelta(0)
    for tier in FetchTier.get_priority_order():
        if tier is FetchTier.IN_PLAY:
            continue
        upper = timedelta(hours=min(tier.max_hours, _MAX_HORIZON_HOURS))
        step = TIER_STEPS[tier]
        multiple = lower // step + 1
        while multiple * step <= upper:
            offsets.append(multiple * step)
            multiple += 1
        lower = upper
    return offsets


def sum_is_valid(prices: Prices) -> bool:
    return abs(prices.total - 1.0) <= SUM_TOLERANCE


def resample(
    series: Mapping[Side, Sequence[Observation]], kickoff: datetime, cutoff: datetime
) -> Resampled:
    """As-of prices on the snapshot grid, at or before ``cutoff``.

    A grid time needs an observation for every outcome no older than
    ``MAX_STALENESS``, with prices inside (0, 1) and a total within
    ``SUM_TOLERANCE`` of 1; other grid times are counted and dropped.
    """
    times = {side: [t for t, _ in series[side]] for side in SIDES}
    result = Resampled()
    for offset in reversed(grid_offsets()):
        at = kickoff - offset
        if at > cutoff:
            continue
        values: dict[Side, float] = {}
        for side in SIDES:
            index = bisect_right(times[side], at) - 1
            if index < 0 or at - times[side][index] > MAX_STALENESS:
                break
            values[side] = series[side][index][1]
        if len(values) < len(SIDES) or not all(0.0 < p < 1.0 for p in values.values()):
            result.missing += 1
            continue
        prices = Prices(**values)
        if not sum_is_valid(prices):
            result.off_sum += 1
            continue
        result.points.append(GridPoint(time=at, prices=prices))
    return result


def build_raw_data(
    match: PolymarketMatch,
    sport: PolymarketSportConfig,
    event: Event,
    prices: Prices,
    snapshot_time: datetime,
    include_quotes: bool = False,
) -> dict[str, Any]:
    names: dict[Side, str] = {
        "home": event.home_team,
        "draw": DRAW_OUTCOME,
        "away": event.away_team,
    }
    return {
        "bookmakers": [
            {
                "key": BOOKMAKER_KEY,
                "title": BOOKMAKER_TITLE,
                "last_update": snapshot_time.isoformat(),
                "markets": [
                    {
                        "key": sport.market_key,
                        "outcomes": [
                            {"name": names[side], PRICE_DECIMAL_KEY: 1 / p}
                            for side, p in prices.by_side().items()
                        ],
                    }
                ],
                "polymarket_meta": {
                    "pm_event_id": match.pm_event_id,
                    "ticker": match.ticker,
                    "yes_probabilities": prices.by_side(),
                    "probability_sum": prices.total,
                    **(
                        {
                            "best_bid": {s: match.outcome_market(s).best_bid for s in SIDES},
                            "best_ask": {s: match.outcome_market(s).best_ask for s in SIDES},
                        }
                        if include_quotes
                        else {}
                    ),
                },
            }
        ],
        "source": SNAPSHOT_SOURCE,
    }
