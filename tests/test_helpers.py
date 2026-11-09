"""Reusable test helpers and stub implementations for API clients."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from odds_core.api_models import EventsResponse, OddsResponse
from odds_core.models import Event, EventStatus, OddsSnapshot
from odds_core.quotes import Quote
from odds_core.sports import SportKey
from odds_core.vocabulary import Source, TimeBasis
from odds_lambda.fetch_tier import FetchTier
from odds_lambda.mlb_stats_fetcher import MlbGameResult, MlbGameState
from odds_lambda.price_rows import expand_envelope
from odds_lambda.storage.event_resolver import ListedGame, MatchWindow, resolve_event
from odds_lambda.storage.game_pks import game_pks_of
from odds_lambda.storage.shaped_snapshots import SnapshotShape, store_shaped_snapshot
from sqlalchemy import Connection, text
from sqlalchemy import event as sqlalchemy_event
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

# Hours before kick-off that odds_lambda.tier_utils.calculate_tier puts in each fetch tier;
# IN_PLAY is after kick-off.
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


def mlb_game(
    pk: int,
    start: datetime,
    home: str,
    away: str,
    game_number: int = 1,
    *,
    tbd: bool = False,
    state: MlbGameState = MlbGameState.SCHEDULED,
) -> MlbGameResult:
    return MlbGameResult(
        game_pk=pk,
        start_time=start,
        home_team=home,
        away_team=away,
        is_final=False,
        home_score=None,
        away_score=None,
        game_number=game_number,
        start_time_tbd=tbd,
        official_date=start.date(),
        state=state,
        schedule_date=start.date(),
    )


class FakeMlbStatsFetcher:
    """Stands in for MlbStatsFetcher: a scripted schedule, or an outage."""

    schedule: list[MlbGameResult] = []
    down: bool = False
    requests: int = 0

    async def __aenter__(self) -> FakeMlbStatsFetcher:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def fetch_results(self, start: date, end: date) -> list[MlbGameResult]:
        type(self).requests += 1
        if self.down:
            raise httpx.ConnectError("statsapi.mlb.com unreachable")
        return list(self.schedule)


@dataclass(frozen=True)
class LeagueFixture:
    """One 2024-25 match as football-data.co.uk and the Odds API spell it."""

    sport: SportKey
    sport_title: str
    fduk_home: str
    fduk_away: str
    api_home: str
    api_away: str
    kickoff: datetime


# Kick-offs are the FDUK files' UK times in UTC.
LEAGUE_FIXTURES = [
    LeagueFixture(
        "soccer_efl_champ",
        "Championship",
        "Sheffield Weds",
        "QPR",
        "Sheffield Wednesday",
        "Queens Park Rangers",
        datetime(2024, 9, 14, 14, 0, tzinfo=UTC),
    ),
    LeagueFixture(
        "soccer_spain_la_liga",
        "La Liga - Spain",
        "Ath Madrid",
        "Betis",
        "Atlético Madrid",
        "Real Betis",
        datetime(2025, 5, 18, 17, 0, tzinfo=UTC),
    ),
    LeagueFixture(
        "soccer_italy_serie_a",
        "Serie A - Italy",
        "Inter",
        "Milan",
        "Inter Milan",
        "AC Milan",
        datetime(2024, 9, 22, 18, 45, tzinfo=UTC),
    ),
    LeagueFixture(
        "soccer_germany_bundesliga",
        "Bundesliga - Germany",
        "M'gladbach",
        "Ein Frankfurt",
        "Borussia Monchengladbach",
        "Eintracht Frankfurt",
        datetime(2025, 2, 8, 17, 30, tzinfo=UTC),
    ),
    LeagueFixture(
        "soccer_france_ligue_one",
        "Ligue 1 - France",
        "Paris SG",
        "St Etienne",
        "Paris Saint Germain",
        "Saint Etienne",
        datetime(2025, 1, 12, 19, 45, tzinfo=UTC),
    ),
    LeagueFixture(
        "soccer_epl",
        "EPL",
        "Man United",
        "Newcastle",
        "Manchester United",
        "Newcastle United",
        datetime(2025, 4, 13, 15, 30, tzinfo=UTC),
    ),
]


async def load_fduk_fixture(session: AsyncSession, fixture: LeagueFixture) -> Event:
    """The fixture as the football-data.co.uk loader stores a played match: FINAL with scores."""
    resolved = await resolve_event(
        session,
        ListedGame(
            source=Source.FOOTBALL_DATA_UK,
            sport_key=fixture.sport,
            sport_title=fixture.sport_title,
            home_team=fixture.fduk_home,
            away_team=fixture.fduk_away,
            commence_time=fixture.kickoff,
            match_window=MatchWindow.HISTORICAL,
        ),
    )
    event = resolved.event
    event.status = EventStatus.FINAL
    event.home_score, event.away_score = 1, 1
    event.completed_at = fixture.kickoff + timedelta(hours=2)
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
            quota_used=None,
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


def make_nba_event(
    event_id: int,
    commence_time: datetime = TWIN_KICKOFF,
    home_team: str = "Boston Celtics",
    away_team: str = "New York Knicks",
    **fields,
) -> Event:
    fields.setdefault("status", EventStatus.SCHEDULED)
    fields.setdefault("sport_key", "basketball_nba")
    return Event(
        id=event_id,
        sport_title="NBA",
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


class SessionFactory:
    """A session factory that hands every caller the test session."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    def __call__(self) -> SessionFactory:
        return self

    async def __aenter__(self) -> AsyncSession:
        return self._session

    async def __aexit__(self, *args: object) -> bool:
        return False


class RecordedHttpSession:
    """Serves one recorded historical response through the client's HTTP layer."""

    def __init__(self, payload: dict[str, Any], headers: dict[str, str]) -> None:
        self._payload = payload
        self._headers = headers
        self.status = 200
        self.requests: list[dict[str, Any]] = []

    def get(self, url: str, params: dict[str, Any], timeout: Any) -> RecordedHttpSession:
        self.requests.append({"url": url, **params})
        return self

    @property
    def headers(self) -> dict[str, str]:
        return self._headers

    def raise_for_status(self) -> None:
        return None

    async def json(self) -> Any:
        return self._payload

    async def __aenter__(self) -> RecordedHttpSession:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


class ReplayHttpSession(RecordedHttpSession):
    """Serves the recorded response of each requested ``date``; a date with none is a call the
    test did not expect to spend."""

    def __init__(
        self,
        responses: dict[str, dict[str, Any]],
        headers: dict[str, str],
        headers_by_date: dict[str, dict[str, str]] | None = None,
    ) -> None:
        super().__init__({}, headers)
        self._responses = responses
        self._base_headers = headers
        self._headers_by_date = headers_by_date or {}

    def get(self, url: str, params: dict[str, Any], timeout: Any) -> ReplayHttpSession:
        if params["date"] not in self._responses:
            raise AssertionError(f"no recorded response for {params['date']}")
        self._payload = self._responses[params["date"]]
        self._headers = self._base_headers | self._headers_by_date.get(params["date"], {})
        return super().get(url, params, timeout)  # type: ignore[return-value]


def odds_api_quote(key: str, outcomes: list[dict[str, Any]], stamp: str) -> dict[str, Any]:
    return {"key": key, "last_update": stamp, "outcomes": outcomes}


def odds_api_h2h(prices: dict[str, int], stamp: str) -> dict[str, Any]:
    """The h2h market in the API's shape: outcome names to American prices."""
    return odds_api_quote(
        "h2h", [{"name": name, "price": price} for name, price in prices.items()], stamp
    )


def odds_api_totals(line: float, over: int, under: int, stamp: str) -> dict[str, Any]:
    """The totals market at one line, as a book returns it."""
    return odds_api_quote(
        "totals",
        [
            {"name": "Over", "price": over, "point": line},
            {"name": "Under", "price": under, "point": line},
        ],
        stamp,
    )


def odds_api_game(
    api_id: str,
    sport: SportKey,
    sport_title: str,
    home: str,
    away: str,
    start: datetime,
    books: dict[str, list[dict[str, Any]]],
    stamp: str,
) -> dict[str, Any]:
    """One game of a historical response: the book keys the API uses, each with its markets."""
    return {
        "id": api_id,
        "sport_key": sport,
        "sport_title": sport_title,
        "commence_time": start.isoformat().replace("+00:00", "Z"),
        "home_team": home,
        "away_team": away,
        "bookmakers": [
            {"key": key, "title": key, "last_update": stamp, "markets": markets}
            for key, markets in books.items()
        ],
    }


async def create_scheduler_tables(session: AsyncSession) -> None:
    """Create the APScheduler data store's tables as the scheduler does on its first start."""
    from apscheduler.datastores.sqlalchemy import SQLAlchemyDataStore

    assert session.bind is not None
    metadata = SQLAlchemyDataStore(session.bind).get_table_definitions()
    connection = await session.connection()
    await connection.run_sync(metadata.create_all)
    await session.commit()


ODDS_WEB_GRANTS = Path(__file__).parents[1] / "scripts" / "odds_web_grants.sql"


async def apply_odds_web_role(session: AsyncSession) -> None:
    """Create the ``odds_web`` role and apply the grants file, as the host wizard does."""
    # The APScheduler tables exist only once the scheduler has booted; the grants file names them.
    await create_scheduler_tables(session)
    await session.execute(
        text(
            "DO $$ BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'odds_web') "
            "THEN CREATE ROLE odds_web NOLOGIN; END IF; END $$"
        )
    )
    for statement in ODDS_WEB_GRANTS.read_text().split(";"):
        if statement.strip():
            await session.execute(text(statement))
    await session.commit()


def client_of(app: FastAPI) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@contextmanager
def as_odds_web(engine: AsyncEngine) -> Iterator[None]:
    """Every transaction on ``engine`` runs as the ``odds_web`` role."""

    def set_role(connection: Connection) -> None:
        connection.exec_driver_sql("SET LOCAL ROLE odds_web")

    sqlalchemy_event.listen(engine.sync_engine, "begin", set_role)
    try:
        yield
    finally:
        sqlalchemy_event.remove(engine.sync_engine, "begin", set_role)


@contextmanager
def as_odds_web_login(engine: AsyncEngine) -> Iterator[None]:
    """Like ``as_odds_web``, with the role-level ``default_transaction_read_only`` a login applies.

    ``SET ROLE`` does not apply ``ALTER ROLE ... SET``, so the setting is read back from
    ``pg_db_role_setting`` and applied to each transaction as the server does for a new session.
    """

    def set_role(connection: Connection) -> None:
        defaults = connection.exec_driver_sql(
            "SELECT s.setconfig FROM pg_db_role_setting s JOIN pg_roles r ON r.oid = s.setrole "
            "WHERE r.rolname = 'odds_web'"
        ).scalar()
        connection.exec_driver_sql("SET LOCAL ROLE odds_web")
        if defaults and "default_transaction_read_only=on" in defaults:
            connection.exec_driver_sql("SET TRANSACTION READ ONLY")

    sqlalchemy_event.listen(engine.sync_engine, "begin", set_role)
    try:
        yield
    finally:
        sqlalchemy_event.remove(engine.sync_engine, "begin", set_role)
