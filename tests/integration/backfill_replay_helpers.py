"""Recorded-response replay through the real backfill executor, and the loaders its tests share."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

from odds_analytics import backfill_executor
from odds_analytics.backfill_checks import ResponseCheck
from odds_analytics.backfill_executor import BackfillExecutor, BackfillResult
from odds_analytics.backfill_plan import Call, Fixture, Manifest, Target
from odds_core.config import APIConfig, Settings
from odds_core.models import Event, EventStatus
from odds_core.sports import SportKey
from odds_core.time import utc_isoformat
from odds_core.utils import is_three_way_sport
from odds_lambda.data_fetcher import TheOddsAPIClient
from odds_lambda.mlb_stats_fetcher import MlbGameState
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from tests.test_helpers import (
    FakeMlbStatsFetcher,
    LeagueFixture,
    ReplayHttpSession,
    add_event,
    hold_game_pks,
    load_fduk_fixture,
    make_nba_event,
    mlb_game,
    odds_api_game,
    odds_api_h2h,
    odds_api_totals,
    with_game_pk,
)

ACQUISITION_KEY = "purchased"


class _ReplayClient(TheOddsAPIClient):
    """The real client over a recorded HTTP session."""

    def __init__(self, api_key: str, http: ReplayHttpSession) -> None:
        super().__init__(api_key=api_key, base_url="http://example.invalid")
        self._http = http

    async def __aenter__(self) -> _ReplayClient:
        self.session = self._http  # type: ignore[assignment]
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


class BackfillReplay:
    """Replays recorded historical responses, one per call time, through the real client and
    ``BackfillExecutor``, each call in its own committed session as production runs it."""

    def __init__(self, engine: AsyncEngine, session: AsyncSession) -> None:
        self._engine = engine
        self._session = session
        self._responses: dict[str, dict[str, Any]] = {}
        self.requests: list[dict[str, Any]] = []
        self._billing: dict[str, tuple[int | None, bool]] = {}

    def record(
        self,
        at: datetime,
        games: list[dict[str, Any]],
        *,
        snapshot: datetime | None = None,
        previous_gap: timedelta | None = timedelta(minutes=5),
        next_gap: timedelta | None = timedelta(minutes=5),
        billed: int | None = None,
        bill_header: bool = True,
    ) -> None:
        """The response a call at ``at`` gets: the provider's snapshot, by default 5 minutes
        earlier, with ``games`` and the neighbouring snapshots ``previous_gap`` before and
        ``next_gap`` after it (None leaves that timestamp out). The call is billed ``billed`` units
        in ``x-requests-last``: by default the manifest's for a response with games and nothing
        for an empty one, which the API does not count; ``bill_header=False`` sends no header."""
        taken = snapshot or at - timedelta(minutes=5)
        response: dict[str, Any] = {"timestamp": utc_isoformat(taken), "data": games}
        if previous_gap is not None:
            response["previous_timestamp"] = utc_isoformat(taken - previous_gap)
        if next_gap is not None:
            response["next_timestamp"] = utc_isoformat(taken + next_gap)
        self._responses[utc_isoformat(at)] = response
        self._billing[utc_isoformat(at)] = (billed, bill_header)

    @staticmethod
    def call(manifest: Manifest, at: datetime, *serves: Event) -> Call:
        """A planned call at ``at`` serving the first band of ``manifest`` for ``serves``."""
        band = manifest.coverage[0]
        return Call(
            at,
            tuple(
                Target(Fixture(event.id, event.commence_time), band, at, at)
                for event in serves
                if event.id is not None
            ),
        )

    @asynccontextmanager
    async def _executor(
        self, manifest: Manifest, **options: Any
    ) -> AsyncIterator[BackfillExecutor]:
        """An executor building its client from the acquisition key, as a real run does."""
        headers = {"x-requests-remaining": "4900000", "x-requests-used": "100000"}
        by_date: dict[str, dict[str, str]] = {}
        for date, (billed, bill_header) in self._billing.items():
            units = billed
            if units is None:
                units = manifest.units_per_call if self._responses[date]["data"] else 0
            by_date[date] = {"x-requests-last": str(units)} if bill_header else {}
        http = ReplayHttpSession(self._responses, headers, by_date)
        settings = Settings(
            api=APIConfig(
                keys=f"live,nba-live,{ACQUISITION_KEY}", acquisition_key_index=2, _env_file=None
            ),
            database={"url": "postgresql+asyncpg://localhost/test"},
        )
        try:
            with (
                patch.object(backfill_executor, "get_settings", return_value=settings),
                patch("odds_lambda.data_fetcher.get_settings", return_value=settings),
                patch.object(
                    backfill_executor,
                    "TheOddsAPIClient",
                    lambda api_key: _ReplayClient(api_key, http),
                ),
            ):
                async with BackfillExecutor(
                    manifest,
                    session_factory=lambda: AsyncSession(self._engine, expire_on_commit=False),
                    rate_limit_seconds=0,
                    **options,
                ) as executor:
                    yield executor
        finally:
            self.requests += http.requests
            self._session.expunge_all()

    async def run(
        self, manifest: Manifest, calls: list[Call], *, skip_existing: bool = True, **options: Any
    ) -> BackfillResult:
        """Run ``calls``. Reports each request in ``requests``."""
        async with self._executor(manifest, skip_existing=skip_existing, **options) as executor:
            return await executor.execute(calls)

    async def probe(self, manifest: Manifest) -> list[ResponseCheck]:
        async with self._executor(manifest) as executor:
            return await executor.probe()


# Shared by the backfill replay and guard tests.
BACKFILLS = Path(__file__).resolve().parents[2] / "backfills"
MANIFEST_FILES: dict[SportKey, str] = {
    "soccer_epl": "odds-api-epl-2020-26.yaml",
    "soccer_spain_la_liga": "odds-api-la-liga-2024-26.yaml",
    "soccer_italy_serie_a": "odds-api-serie-a-2024-26.yaml",
    "soccer_germany_bundesliga": "odds-api-bundesliga-2024-26.yaml",
    "soccer_france_ligue_one": "odds-api-ligue-1-2024-26.yaml",
    "soccer_efl_champ": "odds-api-championship-2024-26.yaml",
    "basketball_nba": "odds-api-nba-2024-26.yaml",
    "baseball_mlb": "odds-api-mlb-2024-26.yaml",
}

NBA = "basketball_nba"
MLB = "baseball_mlb"
CUBS, CARDINALS = "Chicago Cubs", "St. Louis Cardinals"
YANKEES, RED_SOX = "New York Yankees", "Boston Red Sox"

# Every game's quotes: Pinnacle and William Hill each quote their own totals line, and two books
# quote only h2h. The totals lines are (Pinnacle, William Hill, Pinnacle after it moves).
TOTALS_LINES: dict[str, tuple[float, float, float]] = {
    "soccer": (2.5, 2.75, 3.0),
    NBA: (224.5, 225.0, 226.5),
    MLB: (8.5, 9.0, 9.5),
}
HOME_PRICE = -120
STORED_BOOKS = {"pinnacle", "williamhill", "ladbrokes", "betfair_exchange"}


def load_manifest(sport: SportKey) -> Manifest:
    return Manifest.load(BACKFILLS / MANIFEST_FILES[sport])


def totals_lines(sport: SportKey) -> tuple[float, float, float]:
    return TOTALS_LINES["soccer" if is_three_way_sport(sport) else sport]


@dataclass(frozen=True)
class Listed:
    """A loaded FINAL event and the game the API lists for it, in the API's spellings."""

    event: Event
    api_id: str
    home: str
    away: str
    start: datetime
    sport_title: str

    def listing(self, stamp: datetime, *, totals_moved: bool = False) -> dict[str, Any]:
        sport: SportKey = self.event.sport_key
        pinnacle, william_hill, moved = totals_lines(sport)
        at = stamp.isoformat().replace("+00:00", "Z")
        prices = {self.home: HOME_PRICE, self.away: 100}
        if is_three_way_sport(sport):
            prices = {self.home: HOME_PRICE, self.away: 330, "Draw": 260}
        h2h = odds_api_h2h(prices, at)
        return odds_api_game(
            self.api_id,
            sport,
            self.sport_title,
            self.home,
            self.away,
            self.start,
            {
                "pinnacle": [
                    h2h,
                    odds_api_totals(moved if totals_moved else pinnacle, -110, -105, at),
                ],
                "williamhill": [h2h, odds_api_totals(william_hill, -115, -105, at)],
                "ladbrokes_uk": [h2h],
                "betfair_ex_uk": [h2h],
            },
            at,
        )


def final_event(
    event_id: int, sport: SportKey, home: str, away: str, start: datetime, **fields: Any
) -> Event:
    return Event(
        id=event_id,
        sport_key=sport,
        sport_title="MLB" if sport == MLB else sport,
        commence_time=start,
        home_team=home,
        away_team=away,
        status=EventStatus.FINAL,
        home_score=5,
        away_score=3,
        completed_at=start + timedelta(hours=3),
        **fields,
    )


async def load_football(
    session: AsyncSession, mlb_schedule: type[FakeMlbStatsFetcher], league: LeagueFixture
) -> list[Listed]:
    event = await load_fduk_fixture(session, league)
    await session.commit()
    return [
        Listed(event, "api-1", league.api_home, league.api_away, league.kickoff, league.sport_title)
    ]


async def load_nba(session: AsyncSession, mlb_schedule: type[FakeMlbStatsFetcher]) -> list[Listed]:
    night = datetime(2025, 1, 16, 0, 30, tzinfo=UTC)
    late = night + timedelta(hours=2, minutes=30)
    games = [
        (make_nba_event(1, night, "Boston Celtics", "New York Knicks"), "nba-1"),
        (make_nba_event(2, late, "Los Angeles Lakers", "Golden State Warriors"), "nba-2"),
    ]
    listed = []
    for event, api_id in games:
        event.status, event.home_score, event.away_score = EventStatus.FINAL, 112, 104
        event.completed_at = event.commence_time + timedelta(hours=2, minutes=20)
        await add_event(session, event)
        listed.append(
            Listed(event, api_id, event.home_team, event.away_team, event.commence_time, "NBA")
        )
    await session.commit()
    return listed


AFTERNOON = datetime(2025, 6, 2, 17, 5, tzinfo=UTC)
EVENING = datetime(2025, 6, 2, 23, 10, tzinfo=UTC)


async def load_mlb(session: AsyncSession, mlb_schedule: type[FakeMlbStatsFetcher]) -> list[Listed]:
    """A Cubs-Cardinals doubleheader and a Yankees-Red Sox game, each holding its gamePk."""
    games = [
        (901, AFTERNOON, CUBS, CARDINALS, 1),
        (902, EVENING, CUBS, CARDINALS, 2),
        (777, EVENING, YANKEES, RED_SOX, 1),
    ]
    listed = []
    for event_id, (game_pk, start, home, away, number) in enumerate(games, start=1):
        event = with_game_pk(final_event(event_id, MLB, home, away, start), game_pk)
        await add_event(session, event)
        mlb_schedule.schedule.append(
            mlb_game(game_pk, start, home, away, number, state=MlbGameState.FINAL)
        )
        listed.append(Listed(event, f"mlb-{game_pk}", home, away, start, "MLB"))
    await hold_game_pks(session)
    await session.commit()
    return listed


Loader = Callable[[AsyncSession, type[FakeMlbStatsFetcher]], Awaitable[list[Listed]]]


async def count(session: AsyncSession, model: type[Any]) -> int:
    return (await session.execute(select(func.count()).select_from(model))).scalar_one()
