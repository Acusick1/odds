"""A football league's schedule from ESPN's public scoreboard: the events of its upcoming matches.

OddsPortal and Betfair list only matches that already have prices, so a league with no schedule
feed of its own leaves their collectors dormant, and collection is what creates the events
([dormancy](../../wiki/scheduling/decision-engine.md#dormancy)). ESPN's scoreboard lists every
match of a league with its kick-off and an id of its own (``odds_core.vocabulary.Source.ESPN``);
the events it creates are the schedule that wakes those collectors. A league is one
``FootballLeague`` in ``LEAGUES``: its ESPN slug and the clubs ESPN may list for it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from types import MappingProxyType
from typing import Any

import structlog
from odds_core.database import async_session_maker
from odds_core.models import EventStatus
from odds_core.sports import SportKey
from odds_core.team import normalize_team_name
from odds_core.vocabulary import Source

from odds_lambda.espn_fixture_fetcher import EspnFixtureFetcher
from odds_lambda.live_events import DroppedGame, SkipReason
from odds_lambda.mlb_game_resolver import GameRow
from odds_lambda.storage.event_resolver import (
    ListedGame,
    Resolution,
    attach_event,
    find_event,
    resolve_event,
)
from odds_lambda.storage.writers import OddsWriter

logger = structlog.get_logger()

# Matches this far ahead are created; the football dormancy horizon is 28 days.
UPCOMING_DAYS = 21
# Matches this far back may still be postponed or cancelled by ESPN's word.
RECENT_DAYS = 3

_ESPN_STATUSES: dict[str, EventStatus] = {
    "STATUS_POSTPONED": EventStatus.POSTPONED,
    "STATUS_CANCELED": EventStatus.CANCELLED,
}
# ESPN's ``status.type.state`` of a match not yet started.
_PRE = "pre"


@dataclass(frozen=True)
class FootballLeague:
    sport_key: SportKey
    sport_title: str
    # ESPN's scoreboard slug under ``soccer/``.
    slug: str
    # Canonical names (``odds_core.team.normalize_team_name``) of the clubs in the league this
    # season. A club outside it, listed by ESPN, is a promotion, or a spelling with no alias.
    clubs: frozenset[str]


LEAGUES: Mapping[SportKey, FootballLeague] = MappingProxyType(
    {
        league.sport_key: league
        for league in (
            FootballLeague(
                "soccer_efl_champ",
                "Championship",
                "eng.2",
                frozenset(
                    {
                        "Birmingham City",
                        "Blackburn Rovers",
                        "Bolton",
                        "Bristol City",
                        "Burnley",
                        "Cardiff",
                        "Charlton Athletic",
                        "Derby County",
                        "Lincoln",
                        "Middlesbrough",
                        "Millwall",
                        "Norwich",
                        "Portsmouth",
                        "Preston North End",
                        "Queens Park Rangers",
                        "Sheffield Utd",
                        "Southampton",
                        "Stoke",
                        "Swansea",
                        "Watford",
                        "West Brom",
                        "West Ham",
                        "Wolves",
                        "Wrexham AFC",
                    }
                ),
            ),
            FootballLeague(
                "soccer_spain_la_liga",
                "La Liga",
                "esp.1",
                frozenset(
                    {
                        "Alavés",
                        "Athletic Bilbao",
                        "Atlético Madrid",
                        "Barcelona",
                        "CA Osasuna",
                        "Celta Vigo",
                        "Deportivo La Coruña",
                        "Elche CF",
                        "Espanyol",
                        "Getafe",
                        "Levante",
                        "Málaga",
                        "Racing Santander",
                        "Rayo Vallecano",
                        "Real Betis",
                        "Real Madrid",
                        "Real Sociedad",
                        "Sevilla",
                        "Valencia",
                        "Villarreal",
                    }
                ),
            ),
            FootballLeague(
                "soccer_italy_serie_a",
                "Serie A",
                "ita.1",
                frozenset(
                    {
                        "AC Milan",
                        "AS Roma",
                        "Atalanta BC",
                        "Bologna",
                        "Cagliari",
                        "Como",
                        "Fiorentina",
                        "Frosinone",
                        "Genoa",
                        "Inter Milan",
                        "Juventus",
                        "Lazio",
                        "Lecce",
                        "Monza",
                        "Napoli",
                        "Parma",
                        "Sassuolo",
                        "Torino",
                        "Udinese",
                        "Venezia",
                    }
                ),
            ),
            FootballLeague(
                "soccer_germany_bundesliga",
                "Bundesliga",
                "ger.1",
                frozenset(
                    {
                        "1. FC Köln",
                        "Augsburg",
                        "Bayer Leverkusen",
                        "Bayern Munich",
                        "Borussia Dortmund",
                        "Borussia Monchengladbach",
                        "Eintracht Frankfurt",
                        "Elversberg",
                        "FSV Mainz 05",
                        "Hamburger SV",
                        "Paderborn",
                        "RB Leipzig",
                        "SC Freiburg",
                        "Schalke",
                        "TSG Hoffenheim",
                        "Union Berlin",
                        "VfB Stuttgart",
                        "Werder Bremen",
                    }
                ),
            ),
            FootballLeague(
                "soccer_france_ligue_one",
                "Ligue 1",
                "fra.1",
                frozenset(
                    {
                        "AS Monaco",
                        "Angers",
                        "Auxerre",
                        "Brest",
                        "Le Havre",
                        "Le Mans",
                        "Lille",
                        "Lorient",
                        "Lyon",
                        "Marseille",
                        "Nice",
                        "Paris FC",
                        "Paris Saint Germain",
                        "RC Lens",
                        "Rennes",
                        "Strasbourg",
                        "Toulouse",
                        "Troyes",
                    }
                ),
            ),
        )
    }
)


@dataclass(frozen=True)
class EspnFootballGame:
    espn_id: str
    commence_time: datetime
    home_team: str
    away_team: str
    # POSTPONED or CANCELLED when ESPN says so, else None.
    status: EventStatus | None
    # Whether the match has not started and ESPN gives it a valid start: only such a match is
    # created or moved. Any other match only has its status applied to the event it has.
    upcoming: bool

    def listed(self, league: FootballLeague) -> ListedGame:
        return ListedGame(
            source=Source.ESPN,
            sport_key=league.sport_key,
            sport_title=league.sport_title,
            home_team=self.home_team,
            away_team=self.away_team,
            commence_time=self.commence_time,
            external_id=self.espn_id,
        )


@dataclass
class ParsedScoreboard:
    games: list[EspnFootballGame] = field(default_factory=list)
    # Matches with a side outside the league's clubs, which are not stored.
    dropped: list[DroppedGame] = field(default_factory=list)


@dataclass
class SyncReport:
    games: int = 0
    created: int = 0
    matched: int = 0
    # Matches already started and finished or in play, which neither create nor move an event.
    skipped: int = 0
    rescheduled: int = 0
    dropped: list[DroppedGame] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def merge(self, other: SyncReport) -> None:
        self.games += other.games
        self.created += other.created
        self.matched += other.matched
        self.skipped += other.skipped
        self.rescheduled += other.rescheduled
        self.dropped.extend(other.dropped)
        self.errors.extend(other.errors)


def parse_scoreboard(data: dict[str, Any], league: FootballLeague) -> ParsedScoreboard:
    """The matches of one scoreboard response between two of the league's clubs.

    Skipped: events without two competitors and, unless ESPN postpones or cancels them, matches
    whose start ESPN marks invalid (those only have their status applied). A match
    with a side whose normalised name is not in ``league.clubs`` is dropped, and reported in
    ``dropped``: stored under a spelling OddsPortal and Betfair do not use, it would split the
    fixture into two events.
    """
    parsed = ParsedScoreboard()
    for event in data.get("events", []):
        competition = (event.get("competitions") or [{}])[0]
        sides = {c.get("homeAway"): c for c in competition.get("competitors", [])}
        if "home" not in sides or "away" not in sides or not event.get("date"):
            continue
        start_valid = (
            competition.get("timeValid") is not False and event.get("timeValid") is not False
        )
        status = _ESPN_STATUSES.get(competition.get("status", {}).get("type", {}).get("name", ""))
        if not start_valid and status is None:
            continue
        start = datetime.fromisoformat(event["date"].replace("Z", "+00:00")).astimezone(UTC)
        raw = (
            sides["home"]["team"].get("displayName", ""),
            sides["away"]["team"].get("displayName", ""),
        )
        home, away = normalize_team_name(raw[0]), normalize_team_name(raw[1])
        unknown = [
            name
            for name, canonical in zip(raw, (home, away), strict=True)
            if canonical not in league.clubs
        ]
        if unknown:
            parsed.dropped.append(
                DroppedGame(
                    GameRow(raw[0], raw[1], start),
                    SkipReason.UNKNOWN_TEAM,
                    f"{', '.join(repr(n) for n in unknown)} not a {league.sport_title} club",
                )
            )
            continue
        status_type = competition.get("status", {}).get("type", {})
        parsed.games.append(
            EspnFootballGame(
                espn_id=str(event["id"]),
                commence_time=start,
                home_team=home,
                away_team=away,
                status=status,
                upcoming=start_valid and status_type.get("state") == _PRE,
            )
        )
    return parsed


def _months(since: date, until: date) -> list[tuple[int, int]]:
    months: list[tuple[int, int]] = []
    year, month = since.year, since.month
    while (year, month) <= (until.year, until.month):
        months.append((year, month))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return months


async def fetch_games(
    league: FootballLeague, since: date, until: date, fetcher: EspnFixtureFetcher
) -> ParsedScoreboard:
    """The matches ESPN lists for ``league`` from ``since`` to ``until`` inclusive (UTC days), one
    request per calendar month. Raises ``httpx.HTTPError`` when a request fails."""
    parsed = ParsedScoreboard()
    seen: set[str] = set()
    for year, month in _months(since, until):
        month_scoreboard = parse_scoreboard(
            await fetcher.fetch_month_scoreboard_json(league.slug, year, month), league
        )
        parsed.games.extend(
            g
            for g in month_scoreboard.games
            if since <= g.commence_time.date() <= until and g.espn_id not in seen
        )
        seen.update(g.espn_id for g in month_scoreboard.games)
        parsed.dropped.extend(
            d for d in month_scoreboard.dropped if since <= d.game.start.date() <= until
        )
    return parsed


async def sync_games(
    league: FootballLeague, games: list[EspnFootballGame], report: SyncReport, *, apply: bool
) -> None:
    """Resolve each match to its event, creating it when none matches, and set the event of a
    postponed or cancelled one POSTPONED or CANCELLED. One transaction per match.

    A match already started creates and moves nothing; ESPN's postponement or cancellation of it
    is applied to the event it already has. Without ``apply`` nothing is written: the lookup is
    read-only and ``created`` counts the matches that have no event.
    """
    for game in games:
        report.games += 1
        try:
            async with async_session_maker() as session:
                listed = game.listed(league)
                if apply and game.upcoming:
                    resolved = await resolve_event(session, listed)
                elif apply and game.status is not None:
                    # Records ESPN's id on the event, so a later reschedule finds it, without
                    # creating or moving one.
                    resolved = await attach_event(session, listed)
                else:
                    resolved = await find_event(session, listed)
                    if resolved is None and game.upcoming:
                        report.created += 1
                        continue
                if resolved is None or not (game.upcoming or game.status is not None):
                    report.skipped += 1
                    continue
                event = resolved.event
                if resolved.resolution is Resolution.CREATED:
                    report.created += 1
                else:
                    report.matched += 1
                if (
                    game.status is not None
                    and event.status is not game.status
                    and event.status.can_become(game.status)
                ):
                    report.rescheduled += 1
                    if apply:
                        await OddsWriter(session).update_event_status(event.id, game.status)
                if apply:
                    await session.commit()
        except Exception as e:
            report.errors.append(
                f"{game.away_team} at {game.home_team} {game.commence_time:%F}: {e}"
            )
            logger.error("espn_football_sync_failed", espn_id=game.espn_id, exc_info=True)


async def sync_schedule(
    league: FootballLeague, since: date, until: date, *, apply: bool
) -> SyncReport:
    """Sync the matches ESPN lists for ``league`` from ``since`` to ``until`` inclusive.

    Raises ``httpx.HTTPError`` when ESPN cannot be read.
    """
    async with EspnFixtureFetcher() as fetcher:
        parsed = await fetch_games(league, since, until, fetcher)
    report = SyncReport(dropped=parsed.dropped)
    await sync_games(league, parsed.games, report, apply=apply)
    return report


async def sync_upcoming(league: FootballLeague, now: datetime) -> SyncReport:
    """Create the events of the next ``UPCOMING_DAYS`` and apply ESPN's postponements of the last
    ``RECENT_DAYS``; the scheduled job's window."""
    today = now.astimezone(UTC).date()
    return await sync_schedule(
        league,
        today - timedelta(days=RECENT_DAYS),
        today + timedelta(days=UPCOMING_DAYS),
        apply=True,
    )
