"""NBA's schedule from ESPN's public scoreboard: events for upcoming and played games.

The NBA has no published schedule the live sources can read: OddsPortal, Betfair and the Odds API
list only games that already have prices, and Polymarket attaches to existing events and creates
none. ESPN's scoreboard lists every game of a day with its kick-off, final score and an id of its
own (``odds_core.vocabulary.Source.ESPN``). The events it creates are the schedule that wakes the
dormant NBA collectors. It is not a kick-off authority for NBA
(``odds_lambda.storage.event_kickoff.KICKOFF_AUTHORITY``), so it never moves an event another
source created.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx
import structlog
from odds_core.database import async_session_maker
from odds_core.models import EventStatus
from odds_core.vocabulary import Source

from odds_lambda.polymarket_matching import canonical_nba_team
from odds_lambda.storage.event_resolver import ListedGame, Resolution, resolve_event
from odds_lambda.storage.writers import OddsWriter

logger = structlog.get_logger()

SPORT_KEY = "basketball_nba"
SPORT_TITLE = "NBA"
SCOREBOARD_URL = "http://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
REQUEST_DELAY_SECONDS = 0.5
_MAX_ATTEMPTS = 3
_REQUEST_TIMEOUT_SECONDS = 15.0
# ESPN's ``season.type`` of preseason games, which no collector prices.
_PRESEASON = 1
_STATUSES: dict[str, EventStatus] = {
    "STATUS_FINAL": EventStatus.FINAL,
    "STATUS_POSTPONED": EventStatus.POSTPONED,
    "STATUS_CANCELED": EventStatus.CANCELLED,
}


@dataclass(frozen=True)
class EspnNbaGame:
    espn_id: str
    commence_time: datetime
    home_team: str
    away_team: str
    # FINAL, POSTPONED or CANCELLED when ESPN says so, else None (scheduled or in play).
    status: EventStatus | None
    home_score: int | None
    away_score: int | None

    def listed(self) -> ListedGame:
        return ListedGame(
            source=Source.ESPN,
            sport_key=SPORT_KEY,
            sport_title=SPORT_TITLE,
            home_team=self.home_team,
            away_team=self.away_team,
            commence_time=self.commence_time,
            external_id=self.espn_id,
        )


@dataclass
class SyncReport:
    games: int = 0
    created: int = 0
    matched: int = 0
    finalised: int = 0
    rescheduled: int = 0
    errors: list[str] = field(default_factory=list)


def parse_scoreboard(data: dict[str, Any]) -> list[EspnNbaGame]:
    """The regular-season and playoff games of one scoreboard response between two NBA teams.

    Skipped: events without two competitors, preseason games, games whose start ESPN marks
    invalid (NBA Cup "TBD @ TBD" placeholders) and games with a side that is not one of the 30
    teams (placeholders, All-Star teams). A placeholder would otherwise take the real game's
    ESPN id and keep the event it created.
    """
    games: list[EspnNbaGame] = []
    for event in data.get("events", []):
        competition = (event.get("competitions") or [{}])[0]
        sides = {c.get("homeAway"): c for c in competition.get("competitors", [])}
        if "home" not in sides or "away" not in sides or not event.get("date"):
            continue
        if event.get("season", {}).get("type") == _PRESEASON:
            continue
        if competition.get("timeValid") is False or event.get("timeValid") is False:
            continue
        home = canonical_nba_team(sides["home"]["team"].get("displayName", ""))
        away = canonical_nba_team(sides["away"]["team"].get("displayName", ""))
        if home is None or away is None:
            continue
        status = _STATUSES.get(competition.get("status", {}).get("type", {}).get("name", ""))
        scores = (sides["home"].get("score"), sides["away"].get("score"))
        if status is EventStatus.FINAL and None in scores:
            status = None
        scored = status is EventStatus.FINAL
        games.append(
            EspnNbaGame(
                espn_id=str(event["id"]),
                commence_time=datetime.fromisoformat(
                    event["date"].replace("Z", "+00:00")
                ).astimezone(UTC),
                home_team=home,
                away_team=away,
                status=status,
                home_score=int(scores[0]) if scored else None,
                away_score=int(scores[1]) if scored else None,
            )
        )
    return games


async def fetch_games(day: date, client: httpx.AsyncClient) -> list[EspnNbaGame]:
    """The games ESPN lists for a US calendar day (``dates=YYYYMMDD``), with retry."""
    last_exc: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            response = await client.get(
                SCOREBOARD_URL,
                params={"dates": day.strftime("%Y%m%d"), "limit": 100},
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            return parse_scoreboard(response.json())
        except httpx.HTTPError as e:
            last_exc = e
            logger.warning("espn_nba_fetch_retry", day=day.isoformat(), attempt=attempt + 1)
            await asyncio.sleep(1.0)
    assert last_exc is not None
    raise last_exc


async def sync_games(games: list[EspnNbaGame], report: SyncReport, *, apply: bool) -> None:
    """Resolve each game to its event (creating it when none matches) and set the event of a finished game
    FINAL with its score, and of a postponed or cancelled one POSTPONED or CANCELLED. One transaction per game, rolled back unless ``apply``."""
    for game in games:
        report.games += 1
        try:
            async with async_session_maker() as session:
                resolved = await resolve_event(session, game.listed())
                event = resolved.event
                created = resolved.resolution is Resolution.CREATED
                report.created += int(created)
                report.matched += int(not created)
                if (
                    game.status is not None
                    and event.status is not game.status
                    and event.status.can_become(game.status)
                ):
                    await OddsWriter(session).update_event_status(
                        event.id, game.status, game.home_score, game.away_score
                    )
                    report.finalised += int(game.status is EventStatus.FINAL)
                    report.rescheduled += int(game.status is not EventStatus.FINAL)
                if apply:
                    await session.commit()
                else:
                    await session.rollback()
        except Exception as e:
            report.errors.append(
                f"{game.away_team} at {game.home_team} {game.commence_time:%F}: {e}"
            )
            logger.error("espn_nba_sync_failed", espn_id=game.espn_id, exc_info=True)


UPCOMING_DAYS = 21
SETTLE_DAYS = 3


async def sync_upcoming(now: datetime) -> SyncReport:
    """Create the events of the next ``UPCOMING_DAYS`` and settle those of the last
    ``SETTLE_DAYS``; the scheduled job's window."""
    today = now.astimezone(UTC).date()
    return await sync_schedule(
        today - timedelta(days=SETTLE_DAYS), today + timedelta(days=UPCOMING_DAYS), apply=True
    )


async def sync_schedule(since: date, until: date, *, apply: bool) -> SyncReport:
    """Sync the NBA games ESPN lists for each day from ``since`` to ``until`` inclusive."""
    report = SyncReport()
    seen: set[str] = set()
    async with httpx.AsyncClient() as client:
        day = since
        while day <= until:
            games = [g for g in await fetch_games(day, client) if g.espn_id not in seen]
            seen.update(g.espn_id for g in games)
            await sync_games(games, report, apply=apply)
            day += timedelta(days=1)
            await asyncio.sleep(REQUEST_DELAY_SECONDS)
    return report
