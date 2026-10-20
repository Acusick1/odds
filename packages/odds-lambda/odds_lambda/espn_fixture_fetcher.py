"""Async fetcher for ESPN all-competition EPL fixture schedules.

Wraps the ESPN Site API. Given a season start year, returns a list of
``EspnFixtureRecord`` with deduplicated fixtures across Premier League,
FA Cup, League Cup, and European competitions.

The ESPN Site API is free and unauthenticated.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import structlog
from odds_core.epl_data_models import EspnFixtureRecord
from odds_core.team import normalize_team_name

logger = structlog.get_logger()

BASE_URL = "http://site.api.espn.com/apis/site/v2/sports/soccer"

# ESPN league slugs for all competitions an EPL team might play in.
LEAGUE_SLUGS: dict[str, str] = {
    "eng.1": "Premier League",
    "eng.fa": "FA Cup",
    "eng.league_cup": "League Cup",
    "uefa.champions": "Champions League",
    "uefa.europa": "Europa League",
    "uefa.europa.conf": "Conference League",
}

# Seasons: start year -> display label (matches FDUK convention).
SEASONS: dict[int, str] = {
    2015: "2015-16",
    2016: "2016-17",
    2017: "2017-18",
    2018: "2018-19",
    2019: "2019-20",
    2020: "2020-21",
    2021: "2021-22",
    2022: "2022-23",
    2023: "2023-24",
    2024: "2024-25",
    2025: "2025-26",
    2026: "2026-27",
}

# Month at which the EPL season start year flips (August).
_SEASON_FLIP_MONTH = 8

# Default pacing between ESPN HTTP requests. ESPN is lenient but we stay polite.
DEFAULT_REQUEST_DELAY_SECONDS = 0.5

# HTTP retry policy for ESPN endpoints.
_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 1.0
_REQUEST_TIMEOUT_SECONDS = 15.0

# ESPN's scoreboard default page is 100 events; a month of any one competition
# is below this limit, which is set explicitly so a busy month is never truncated.
_SCOREBOARD_LIMIT = 300


PRIMARY_LEAGUE_SLUG = "eng.1"


class EspnUpcomingFetchError(RuntimeError):
    """The Premier League's upcoming-fixture request to ESPN failed."""


class EspnSeasonFetchError(RuntimeError):
    """The Premier League schedule request failed for every team in the season."""


@dataclass(frozen=True, slots=True)
class UpcomingFixtures:
    """Upcoming records plus the competitions whose request failed."""

    records: list[EspnFixtureRecord]
    failed_leagues: list[str]


def current_season(now: datetime | None = None) -> int:
    """Return the EPL season start year for the given instant.

    The season flips on August 1: dates in Aug-Dec resolve to that calendar
    year, Jan-Jul resolve to the previous calendar year. ``now`` defaults to
    the current UTC datetime.
    """
    if now is None:
        now = datetime.now(UTC)
    if now.month >= _SEASON_FLIP_MONTH:
        return now.year
    return now.year - 1


def _months_between(start: datetime, end: datetime) -> list[tuple[int, int]]:
    """Return the ``(year, month)`` pairs from ``start``'s month through ``end``'s."""
    months: list[tuple[int, int]] = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        months.append((year, month))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return months


def season_label(season: int) -> str:
    """Return the display label (e.g. ``"2025-26"``) for a season start year."""
    if season in SEASONS:
        return SEASONS[season]
    # Fall back to deriving the label for seasons beyond the static table.
    return f"{season}-{(season + 1) % 100:02d}"


def _extract_score(competitor: dict[str, Any]) -> str:
    """Extract score from a competitor entry. Returns empty string if unavailable."""
    score = competitor.get("score")
    if score is None:
        return ""
    if isinstance(score, dict):
        return score.get("displayValue", "")
    return str(score)


class EspnFixtureFetcher:
    """Async fetcher for ESPN all-competition EPL fixture schedules.

    Usage::

        async with EspnFixtureFetcher() as fetcher:
            records = await fetcher.fetch_season(2025)
    """

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        request_delay_seconds: float = DEFAULT_REQUEST_DELAY_SECONDS,
    ) -> None:
        self._owns_client = client is None
        self._client = client
        self.request_delay_seconds = request_delay_seconds

    async def __aenter__(self) -> EspnFixtureFetcher:
        if self._client is None:
            self._client = httpx.AsyncClient()
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    def _require_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError(
                "EspnFixtureFetcher used outside async context manager; "
                "wrap calls in `async with EspnFixtureFetcher()`."
            )
        return self._client

    async def _fetch_json(self, url: str) -> dict[str, Any]:
        """Fetch JSON from ESPN API with retry."""
        client = self._require_client()
        last_exc: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                resp = await client.get(url, timeout=_REQUEST_TIMEOUT_SECONDS)
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPError as e:
                last_exc = e
                if attempt == _MAX_ATTEMPTS - 1:
                    break
                logger.warning("espn_fetch_retry", url=url, attempt=attempt + 1, error=str(e))
                await asyncio.sleep(_RETRY_BACKOFF_SECONDS)
        assert last_exc is not None
        raise last_exc

    async def fetch_teams(self, season: int) -> list[dict[str, str]]:
        """Fetch EPL teams for a season. Returns list of {id, name}."""
        url = f"{BASE_URL}/eng.1/teams?season={season}"
        data = await self._fetch_json(url)
        teams = data["sports"][0]["leagues"][0]["teams"]
        return [{"id": t["team"]["id"], "name": t["team"]["displayName"]} for t in teams]

    async def fetch_team_schedule(
        self,
        league_slug: str,
        team_id: str,
        season: int,
    ) -> list[EspnFixtureRecord]:
        """Fetch a team's schedule for one competition/season.

        Returns a list of ``EspnFixtureRecord`` entries. ``season`` is stored as
        the display label (e.g. ``"2025-26"``) on each record.
        """
        url = f"{BASE_URL}/{league_slug}/teams/{team_id}/schedule?season={season}"
        data = await self._fetch_json(url)
        events = data.get("events", [])
        competition = LEAGUE_SLUGS[league_slug]
        label = season_label(season)
        records: list[EspnFixtureRecord] = []

        for event in events:
            date_str = event.get("date", "")
            if not date_str:
                continue

            comps = event.get("competitions", [])
            if not comps:
                continue

            comp = comps[0]
            competitors = comp.get("competitors", [])
            if len(competitors) != 2:
                continue

            team_entry = None
            opponent_entry = None
            for c in competitors:
                if c["team"]["id"] == team_id:
                    team_entry = c
                else:
                    opponent_entry = c

            if team_entry is None or opponent_entry is None:
                continue

            team_name = normalize_team_name(team_entry["team"]["displayName"])
            opponent_name = normalize_team_name(opponent_entry["team"]["displayName"])

            status_type = comp.get("status", {}).get("type", {})
            status = status_type.get("description", "")
            state = status_type.get("state") or None
            round_name = event.get("seasonType", {}).get("name", "")

            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)

            records.append(
                EspnFixtureRecord(
                    date=dt,
                    team=team_name,
                    opponent=opponent_name,
                    competition=competition,
                    match_round=round_name,
                    home_away=team_entry["homeAway"],
                    score_team=_extract_score(team_entry),
                    score_opponent=_extract_score(opponent_entry),
                    status=status,
                    state=state,
                    season=label,
                )
            )

        return records

    async def fetch_month_scoreboard(
        self,
        league_slug: str,
        year: int,
        month: int,
    ) -> list[EspnFixtureRecord]:
        """Fetch one calendar month's scoreboard for one competition.

        The scoreboard endpoint returns one event per match (rather than one
        per team per match, like ``teams/{id}/schedule``). For each event we
        emit two records, one anchored on each team, to match the schema
        used by ``fetch_team_schedule``.

        ESPN rejects ``dates=`` ranges with HTTP 400; it accepts a single day
        (``YYYYMMDD``), a month (``YYYYMM``) or a year. Raises
        ``httpx.HTTPError`` when the request fails.
        """
        competition = LEAGUE_SLUGS[league_slug]
        url = (
            f"{BASE_URL}/{league_slug}/scoreboard"
            f"?dates={year:04d}{month:02d}&limit={_SCOREBOARD_LIMIT}"
        )
        data = await self._fetch_json(url)

        events = data.get("events", [])
        records: list[EspnFixtureRecord] = []
        for event in events:
            date_str = event.get("date", "")
            if not date_str:
                continue
            comps = event.get("competitions", [])
            if not comps:
                continue
            comp = comps[0]
            competitors = comp.get("competitors", [])
            if len(competitors) != 2:
                continue

            home_entry = None
            away_entry = None
            for c in competitors:
                if c.get("homeAway") == "home":
                    home_entry = c
                elif c.get("homeAway") == "away":
                    away_entry = c
            if home_entry is None or away_entry is None:
                continue

            status_type = comp.get("status", {}).get("type", {})
            status = status_type.get("description", "")
            state = status_type.get("state") or None
            round_name = event.get("seasonType", {}).get("name", "")

            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)

            home_name = normalize_team_name(home_entry["team"]["displayName"])
            away_name = normalize_team_name(away_entry["team"]["displayName"])
            home_score = _extract_score(home_entry)
            away_score = _extract_score(away_entry)

            label = season_label(current_season(dt))

            # Home-anchored row
            records.append(
                EspnFixtureRecord(
                    date=dt,
                    team=home_name,
                    opponent=away_name,
                    competition=competition,
                    match_round=round_name,
                    home_away="home",
                    score_team=home_score,
                    score_opponent=away_score,
                    status=status,
                    state=state,
                    season=label,
                )
            )
            # Away-anchored row (mirror)
            records.append(
                EspnFixtureRecord(
                    date=dt,
                    team=away_name,
                    opponent=home_name,
                    competition=competition,
                    match_round=round_name,
                    home_away="away",
                    score_team=away_score,
                    score_opponent=home_score,
                    status=status,
                    state=state,
                    season=label,
                )
            )

        return records

    async def fetch_upcoming(
        self, days_ahead: int = 30, *, now: datetime | None = None
    ) -> UpcomingFixtures:
        """Fetch near-future fixtures across all configured competitions.

        The ``teams/{id}/schedule`` endpoint only returns completed matches,
        so upcoming fixtures come from the scoreboard endpoint. One request
        per competition per calendar month covers the window; records are
        filtered to ``start of today → today+days_ahead`` and deduplicated on
        ``(date, team, opponent)``.

        A cup or European competition whose request fails is logged and listed
        in ``failed_leagues``. Raises ``EspnUpcomingFetchError`` when the
        Premier League request failed or every competition failed. ``now``
        defaults to the current UTC time.
        """
        now = now or datetime.now(UTC)
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end = now + timedelta(days=days_ahead)
        all_records: list[EspnFixtureRecord] = []
        seen: set[tuple[str, str, str]] = set()
        failed_leagues: list[str] = []

        for league_slug in LEAGUE_SLUGS:
            new_count = 0
            try:
                for year, month in _months_between(start, end):
                    if self.request_delay_seconds > 0:
                        await asyncio.sleep(self.request_delay_seconds)
                    fixtures = await self.fetch_month_scoreboard(league_slug, year, month)
                    for record in fixtures:
                        if not start <= record.date <= end:
                            continue
                        key = (record.date.isoformat(), record.team, record.opponent)
                        if key in seen:
                            continue
                        seen.add(key)
                        all_records.append(record)
                        new_count += 1
            except httpx.HTTPError as e:
                failed_leagues.append(league_slug)
                logger.warning("espn_scoreboard_fetch_failed", league=league_slug, error=str(e))
                continue
            logger.info(
                "espn_scoreboard_loaded",
                league=league_slug,
                new_fixtures=new_count,
            )

        if PRIMARY_LEAGUE_SLUG in failed_leagues:
            raise EspnUpcomingFetchError(
                f"ESPN scoreboard request failed for: {', '.join(failed_leagues)}"
            )

        all_records.sort(key=lambda r: r.date)
        return UpcomingFixtures(records=all_records, failed_leagues=failed_leagues)

    async def fetch_season(self, season: int) -> list[EspnFixtureRecord]:
        """Fetch all fixtures for all EPL teams in a season across all competitions.

        Each match is represented once per team; duplicates that would otherwise
        appear (because a match shows up on both teams' schedules) are collapsed
        using the ``(date, team, opponent)`` tuple so each row is unique per team.
        """
        teams = await self.fetch_teams(season)
        logger.info("espn_teams_loaded", season=season, count=len(teams))

        all_records: list[EspnFixtureRecord] = []
        seen: set[tuple[str, str, str]] = set()
        primary_failures = 0

        for idx, team in enumerate(teams):
            team_id = team["id"]
            team_name = normalize_team_name(team["name"])
            team_records = 0

            for league_slug in LEAGUE_SLUGS:
                if self.request_delay_seconds > 0:
                    await asyncio.sleep(self.request_delay_seconds)
                try:
                    fixtures = await self.fetch_team_schedule(league_slug, team_id, season)
                except httpx.HTTPError as e:
                    if league_slug == PRIMARY_LEAGUE_SLUG:
                        primary_failures += 1
                    logger.warning(
                        "espn_schedule_fetch_failed",
                        team=team_name,
                        league=league_slug,
                        error=str(e),
                    )
                    continue

                for record in fixtures:
                    key = (record.date.isoformat(), record.team, record.opponent)
                    if key in seen:
                        continue
                    seen.add(key)
                    all_records.append(record)
                    team_records += 1

            logger.info(
                "espn_team_schedule_loaded",
                season=season,
                team=team_name,
                new_fixtures=team_records,
                index=idx + 1,
                total=len(teams),
            )

        if teams and primary_failures == len(teams):
            raise EspnSeasonFetchError(
                f"ESPN Premier League schedule request failed for all {len(teams)} teams in season {season}"
            )

        all_records.sort(key=lambda r: r.date)
        return all_records
