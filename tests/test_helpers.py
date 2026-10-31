"""Reusable test helpers and stub implementations for API clients."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from odds_core.api_models import EventsResponse, HistoricalOddsResponse, OddsResponse
from odds_core.models import Event, EventStatus, OddsSnapshot
from odds_core.quotes import Quote
from odds_core.vocabulary import Source, TimeBasis
from odds_lambda.fetch_tier import FetchTier
from odds_lambda.price_rows import expand_envelope
from odds_lambda.storage.game_pks import game_pks_of
from odds_lambda.storage.shaped_snapshots import SnapshotShape, store_shaped_snapshot
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

# Hours before kick-off that odds_lambda.tier_utils.calculate_tier puts in each fetch tier, at the
# Odds API backfill's snapshot time where one lies in the tier
# (odds_analytics.gap_backfill_planner.TIER_SNAPSHOT_LEADS); IN_PLAY is after kick-off.
TIER_HOURS_BEFORE: dict[FetchTier, float] = {
    FetchTier.OPENING: 100,
    FetchTier.EARLY: 72,
    FetchTier.SHARP: 24,
    FetchTier.PREGAME: 12,
    FetchTier.CLOSING: 0.5,
    FetchTier.IN_PLAY: -1,
}


async def add_event(session: AsyncSession, event: Event) -> Event:
    """Store ``event`` as it is, without resolving it against other events."""
    session.add(event)
    await session.flush()
    return event


# Events given a gamePk by ``with_game_pk`` that ``hold_game_pks`` has not recorded yet.
_PENDING_GAME_PKS: dict[int, tuple[Event, int]] = {}


def with_game_pk(event: Event, game_pk: int | None) -> Event:
    """``event``, to hold ``game_pk`` as its ``mlb_stats_api`` external id once
    ``hold_game_pks`` runs after it is added to a session."""
    if game_pk is not None:
        _PENDING_GAME_PKS[id(event)] = (event, game_pk)
    return event


async def hold_game_pks(session: AsyncSession) -> None:
    """Record the gamePk ``with_game_pk`` gave each event in ``session`` as its
    ``mlb_stats_api`` external id, where readers find it. Flushes the session first."""
    await session.flush()
    for key, (event, game_pk) in list(_PENDING_GAME_PKS.items()):
        if event not in session:
            continue
        del _PENDING_GAME_PKS[key]
        await session.execute(
            text(
                "INSERT INTO event_external_ids (source, external_id, event_id, created_at) "
                "VALUES ('mlb_stats_api', :game_pk, :event_id, now()) ON CONFLICT DO NOTHING"
            ),
            {"game_pk": str(game_pk), "event_id": event.id},
        )


async def game_pk_of(session: AsyncSession, event: Event) -> int | None:
    """The gamePk ``event`` holds, read from its ``mlb_stats_api`` external id."""
    return (await game_pks_of(session, [event])).get(event.id)


class StubOddsClient:
    """
    Reusable stub for TheOddsAPIClient with configurable current odds responses.

    This stub implements the async context manager protocol and provides a
    simple, explicit way to mock API client behavior for current odds fetching.

    Example:
        >>> response = OddsResponse(events=[...], raw_events_data=[...], ...)
        >>> client = StubOddsClient(response)
        >>> async with client as c:
        ...     result = await c.get_odds(sport="basketball_nba")
        ...     assert result == response
    """

    def __init__(
        self,
        odds_response: OddsResponse,
        events_response: EventsResponse | None = None,
    ):
        """
        Initialize stub with responses to return.

        Args:
            odds_response: The OddsResponse to return from get_odds()
            events_response: The EventsResponse to return from get_events(). Defaults to empty.
        """
        self._odds_response = odds_response
        self.get_odds_calls: list[dict[str, Any]] = []
        self._events_response = events_response or EventsResponse(
            events=[],
            response_time_ms=0,
            quota_remaining=None,
            timestamp=datetime.now(UTC),
        )

    async def __aenter__(self):
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """Exit async context manager."""
        return False

    async def get_odds(self, *args, **kwargs) -> OddsResponse:
        """Return the configured odds response."""
        self.get_odds_calls.append(kwargs)
        return self._odds_response

    async def get_events(self, *args, **kwargs) -> EventsResponse:
        """Return the configured events response."""
        return self._events_response


class StubHistoricalOddsClient:
    """
    Stub for historical odds API client with call-based response variation.

    This stub cycles through a list of responses based on call count, useful
    for testing pagination or multiple API calls with different results.

    Example:
        >>> responses = [
        ...     HistoricalOddsResponse(events=[event1], ...),
        ...     HistoricalOddsResponse(events=[event2], ...),
        ... ]
        >>> client = StubHistoricalOddsClient(responses)
        >>> async with client as c:
        ...     result1 = await c.get_historical_odds(...)  # Returns responses[0]
        ...     result2 = await c.get_historical_odds(...)  # Returns responses[1]
        ...     result3 = await c.get_historical_odds(...)  # Returns responses[1] again
    """

    def __init__(self, responses: list[HistoricalOddsResponse]):
        """
        Initialize stub with list of responses.

        Args:
            responses: List of HistoricalOddsResponse objects to return.
                       Cycles through based on call count.
        """
        if not responses:
            raise ValueError("Must provide at least one response")
        self._responses = responses
        self._call_count = 0

    async def __aenter__(self):
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """Exit async context manager."""
        return False

    async def get_historical_odds(self, *args, **kwargs) -> HistoricalOddsResponse:
        """
        Return a response based on call count.

        First call returns responses[0], second call returns responses[1], etc.
        If call count exceeds list length, returns the last response repeatedly.

        Args and kwargs are accepted but ignored for stub simplicity.
        """
        # Use min to avoid index out of bounds - return last response if exceeded
        response = self._responses[min(self._call_count, len(self._responses) - 1)]
        self._call_count += 1
        return response

    def reset_call_count(self):
        """Reset the call counter to 0 (useful for test isolation)."""
        self._call_count = 0


TWIN_KICKOFF = datetime(2026, 4, 15, 15, 0, tzinfo=UTC)
TWIN_EVENT_ID = 1
API_EVENT_ID = 2


def make_epl_event(
    event_id: int,
    commence_time: datetime = TWIN_KICKOFF,
    home_team: str = "Arsenal",
    away_team: str = "Chelsea",
    **fields,
) -> Event:
    fields.setdefault("status", EventStatus.SCHEDULED)
    return Event(
        id=event_id,
        sport_key="soccer_epl",
        sport_title="EPL",
        commence_time=commence_time,
        home_team=home_team,
        away_team=away_team,
        **fields,
    )


def polymarket_bookmaker(
    probabilities: tuple[float, float, float], spread: float = 0.02
) -> dict[str, Any]:
    """A Polymarket envelope entry whose 1x2 yes mids are ``probabilities`` (home, draw,
    away), with best ask and bid ``spread / 2`` either side."""
    labels = ("home", "draw", "away")

    def by_label(shift: float) -> dict[str, float]:
        return {label: p + shift for label, p in zip(labels, probabilities, strict=True)}

    return {
        "key": "polymarket",
        "title": "Polymarket",
        "markets": [{"key": "1x2", "outcomes": []}],
        "polymarket_meta": {
            "yes_probabilities": by_label(0),
            "best_ask": by_label(spread / 2),
            "best_bid": by_label(-spread / 2),
        },
    }


def three_way_bookmaker(
    key: str,
    prices: tuple[float, float, float],
    home_team: str = "Arsenal",
    away_team: str = "Chelsea",
    **extra: Any,
) -> dict[str, Any]:
    """An envelope bookmaker entry with 1x2 decimal prices (home, draw, away)."""
    names = (home_team, "Draw", away_team)
    return {
        "key": key,
        "title": key,
        "markets": [
            {
                "key": "1x2",
                "outcomes": [
                    {"name": name, "price_decimal": price}
                    for name, price in zip(names, prices, strict=True)
                ],
            }
        ],
        **extra,
    }


def totals_book(
    key: str, lines: dict[float, tuple[float, float]], betfair_runners: bool = False
) -> dict[str, Any]:
    """An envelope bookmaker entry with totals (over, under) prices per line. With
    ``betfair_runners`` it has the Betfair job's shape: back prices as ``betfair_meta`` runners."""
    if betfair_runners:
        runners = [
            {"runner_name": f"{side} {line:g} Goals", "best_back": price}
            for line, prices in lines.items()
            for side, price in zip(("Over", "Under"), prices, strict=True)
        ]
        return {
            "key": key,
            "markets": [{"key": "totals", "outcomes": []}],
            "betfair_meta": {"inplay": False, "market_status": "OPEN", "runners": runners},
        }
    outcomes = [
        outcome
        for line, (over, under) in lines.items()
        for outcome in (
            {"name": "Over", "price_decimal": over, "point": line},
            {"name": "Under", "price_decimal": under, "point": line},
        )
    ]
    return {"key": key, "markets": [{"key": "totals", "outcomes": outcomes}]}


@dataclass
class TaggedSnapshot:
    """An envelope of ``bookmakers`` for one event and time, with the source tag and, for FDUK,
    the timing basis that pick the source and time basis it is stored under
    (``snapshot_shape``)."""

    event_id: int
    snapshot_time: datetime
    bookmakers: list[dict[str, Any]]
    source: str | None
    fduk_basis: str | None = None
    created_at: datetime | None = None
    # The stored snapshot of the envelope's first market, once ``store_tagged`` stored it.
    id: int | None = None

    @property
    def envelope(self) -> dict[str, Any]:
        return {"bookmakers": self.bookmakers}


def make_tagged_snapshot(
    event_id: int,
    snapshot_time: datetime,
    bookmakers: list[dict[str, Any]],
    source: str | None,
    fduk_basis: str | None = None,
) -> TaggedSnapshot:
    """A snapshot to store with ``store_tagged``: ``source`` is a tag (``None`` is the Odds API,
    ``"oddsportal"`` OddsPortal history, ``"oddsportal_live"``, ``"betfair_api"``,
    ``"polymarket"``, ``"football_data_uk"``), and ``fduk_basis`` FDUK's ``"kickoff"`` (the
    close) or ``"fduk_collection_deadline"``."""
    return TaggedSnapshot(event_id, snapshot_time, bookmakers, source, fduk_basis)


def betfair_exchange_bookmaker(
    prices: tuple[float, float, float],
    inplay: bool = False,
    home_team: str = "Arsenal",
    away_team: str = "Chelsea",
    **meta: Any,
) -> dict[str, Any]:
    """A Betfair Exchange entry with 1x2 back prices, both as outcomes (what OddsPortal and the
    Odds API carry) and as Betfair runners (what the Betfair job carries)."""
    names = (home_team, "The Draw", away_team)
    runners = [
        {"runner_name": name, "best_back": price} for name, price in zip(names, prices, strict=True)
    ]
    return three_way_bookmaker(
        "betfair_exchange",
        prices,
        home_team,
        away_team,
        betfair_meta={"inplay": inplay, "market_status": "OPEN", "runners": runners, **meta},
    )


def snapshot_shape(source: str | None, fduk_basis: str | None = None) -> tuple[Source, TimeBasis]:
    """The source and time basis a ``make_tagged_snapshot`` tag is stored under."""
    if source == "football_data_uk":
        return Source.FOOTBALL_DATA_UK, (
            TimeBasis.CLOSE if fduk_basis == "kickoff" else TimeBasis.UPPER_BOUND
        )
    return {
        None: (Source.ODDS_API, TimeBasis.OBSERVED),
        "oddsportal": (Source.ODDSPORTAL, TimeBasis.SOURCE_STAMPED),
        "oddsportal_live": (Source.ODDSPORTAL, TimeBasis.OBSERVED),
        "betfair_api": (Source.BETFAIR, TimeBasis.OBSERVED),
        "polymarket": (Source.POLYMARKET, TimeBasis.OBSERVED),
    }[source]


def make_quotes(
    event: Event,
    snapshot_id: int,
    snapshot_time: datetime,
    bookmakers: list[dict[str, Any]],
    source: str | None,
    fduk_basis: str | None = None,
    created_at: datetime | None = None,
) -> list[Quote]:
    """The quotes the writers would store for a ``make_tagged_snapshot`` of ``event``, as one
    snapshot per market numbered from ``snapshot_id``."""
    shape_source, time_basis = snapshot_shape(source, fduk_basis)
    expanded = expand_envelope(
        {"bookmakers": bookmakers}, shape_source, event.home_team, event.away_team
    )
    return [
        Quote(
            snapshot_id=snapshot_id + i,
            event_id=event.id,
            snapshot_time=snapshot_time,
            created_at=created_at or snapshot_time,
            source=shape_source,
            market=market.market,
            time_basis=time_basis,
            in_play=market.meta.in_play,
            market_status=market.meta.market_status,
            total_matched=None,
            market_start_time=market.meta.market_start_time,
            bookmaker=spec.bookmaker,
            outcome=spec.outcome,
            line=None if spec.line is None else float(spec.line),
            side=spec.side,
            price=float(spec.price),
            size=None if spec.size is None else float(spec.size),
            matched=None if spec.matched is None else float(spec.matched),
            changed_at=spec.changed_at,
        )
        for i, market in enumerate(expanded.markets)
        for spec in market.prices
    ]


async def store_tagged(session: AsyncSession, snapshot: TaggedSnapshot) -> OddsSnapshot:
    """Store a ``make_tagged_snapshot`` with its price rows, as the writers do, and return the
    stored snapshot of its first market; its event must be in the session."""
    await session.flush()
    event = await session.get(Event, snapshot.event_id)
    assert event is not None
    source, time_basis = snapshot_shape(snapshot.source, snapshot.fduk_basis)
    written = await store_shaped_snapshot(
        session,
        event,
        snapshot.envelope,
        snapshot.snapshot_time,
        SnapshotShape(source, time_basis, None),
        created_at=snapshot.created_at,
    )
    snapshot.id = written.snapshot.id
    return written.snapshot


class FakeBetfairClient:
    def __init__(self, books: list[Any]) -> None:
        self.books = books
        self.lookback_hours: int | None = None

    def list_events(
        self, sport_cfg: Any, lookahead_hours: int, lookback_hours: int = 0
    ) -> list[str]:
        self.lookback_hours = lookback_hours
        return [book.betfair_event_id for book in self.books]

    def list_books(self, sport_cfg: Any, events: list[str]) -> list[Any]:
        return self.books
