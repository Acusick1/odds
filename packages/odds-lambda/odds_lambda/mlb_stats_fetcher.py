"""Async fetcher for MLB Stats API probable-pitcher snapshots.

Wraps ``https://statsapi.mlb.com/api/v1/schedule`` with the
``probablePitcher`` hydration. Given a list of UTC dates, returns one
``MlbProbablePitchersRecord`` per scheduled game per date — even when the
``probablePitcher`` object is absent (the null itself is the signal we want
to snapshot point-in-time).

The MLB Stats API is free and unauthenticated.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from typing import Any

import httpx
import structlog
from odds_core.mlb_data_models import IF_NECESSARY_STATUS, MlbProbablePitchersRecord

logger = structlog.get_logger()

BASE_URL = "https://statsapi.mlb.com/api/v1"

# sportId=1 scopes results to MLB (regular season + post-season). Spring
# training and minor-league sports use other ids and are out of scope.
_SPORT_ID = 1

# Default pacing between MLB API requests. The endpoint is unauthenticated
# and undocumented for rate limits, so we stay polite.
DEFAULT_REQUEST_DELAY_SECONDS = 0.25

# HTTP retry policy.
_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 1.0
_REQUEST_TIMEOUT_SECONDS = 15.0


# Longest span requested in one schedule call, so a season-long backfill is a handful of calls.
_RESULTS_CHUNK_DAYS = 30


class MlbGameState(StrEnum):
    SCHEDULED = "scheduled"
    LIVE = "live"
    FINAL = "final"
    POSTPONED = "postponed"
    CANCELLED = "cancelled"
    OTHER = "other"


@dataclass(frozen=True)
class MlbGameResult:
    """One MLB schedule slot: a game as listed under one schedule date, with its score once Final.

    A postponed or resumed game is listed under several dates, one slot each, sharing its ``game_pk``.
    """

    game_pk: int
    start_time: datetime
    home_team: str
    away_team: str
    is_final: bool
    home_score: int | None
    away_score: int | None
    game_number: int = 1
    start_time_tbd: bool = False
    official_date: date | None = None
    state: MlbGameState = MlbGameState.SCHEDULED
    schedule_date: date | None = None
    resumed: bool = False

    @property
    def day(self) -> date:
        return self.official_date or self.start_time.date()


def _parse_utc(value: str) -> datetime:
    """Parse an ISO 8601 timestamp from MLBAM into a UTC-aware datetime."""
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def dates_for_window(now: datetime, lookahead_hours: float) -> list[date]:
    """Return the set of dates covering ``[now, now + lookahead_hours]`` plus a 1-day pad on each side.

    The MLB Stats API ``/schedule?date=YYYY-MM-DD`` parameter resolves the date
    in MLBAM's reference frame (effectively US Eastern / stadium-local), not
    UTC. A late-night ET game on April 26 carries ``gameDate=2026-04-27T03:05Z``
    but appears in the MLBAM response under ``date=2026-04-26``. Conversely, an
    early-UTC game (before ~4 AM UTC) corresponds to the previous ET calendar
    date.

    To avoid missing games whose ET-local date differs from their UTC date,
    we over-fetch by one calendar day on each side of the UTC window. The
    downstream reader/tool already filter by ``commence_time`` window and the
    writer is idempotent on ``(game_pk, fetched_at)``, so over-fetching is
    safe.

    Includes both padded endpoints; results are ordered ascending.
    """
    if lookahead_hours < 0:
        raise ValueError("lookahead_hours must be non-negative")
    end = now + timedelta(hours=lookahead_hours)
    start_d = now.astimezone(UTC).date() - timedelta(days=1)
    end_d = end.astimezone(UTC).date() + timedelta(days=1)
    days: list[date] = []
    cur = start_d
    while cur <= end_d:
        days.append(cur)
        cur = cur + timedelta(days=1)
    return days


class MlbStatsFetcher:
    """Async fetcher for MLB Stats API probable-pitcher snapshots.

    Usage::

        async with MlbStatsFetcher() as fetcher:
            records = await fetcher.fetch_dates([date(2026, 4, 26)])
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

    async def __aenter__(self) -> MlbStatsFetcher:
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
                "MlbStatsFetcher used outside async context manager; "
                "wrap calls in `async with MlbStatsFetcher()`."
            )
        return self._client

    async def _fetch_json(self, url: str, params: dict[str, str]) -> dict[str, Any]:
        """Fetch JSON from MLB Stats API with retry."""
        client = self._require_client()
        last_exc: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                resp = await client.get(
                    url,
                    params=params,
                    timeout=_REQUEST_TIMEOUT_SECONDS,
                )
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPError as e:
                last_exc = e
                if attempt == _MAX_ATTEMPTS - 1:
                    break
                logger.warning(
                    "mlb_stats_fetch_retry",
                    url=url,
                    params=params,
                    attempt=attempt + 1,
                    error=str(e),
                )
                await asyncio.sleep(_RETRY_BACKOFF_SECONDS)
        assert last_exc is not None
        raise last_exc

    async def fetch_date(
        self,
        target_date: date,
        *,
        fetched_at: datetime | None = None,
    ) -> list[MlbProbablePitchersRecord]:
        """Fetch probable pitchers for one UTC date.

        Returns one record per scheduled MLB game on the date. When the
        ``probablePitcher`` object is missing for a side, the corresponding
        ``*_pitcher_name`` / ``*_pitcher_id`` fields are ``None``.

        Args:
            target_date: UTC date to query MLBAM for.
            fetched_at: Stamp applied to every record from this fetch. Defaults
                to ``datetime.now(UTC)`` when omitted; passing it in lets the
                caller share a single ``fetched_at`` across multiple dates so
                an idempotent re-call writes a coherent snapshot row group.

        Returns:
            List of ``MlbProbablePitchersRecord``, one per scheduled game.
        """
        url = f"{BASE_URL}/schedule"
        params = {
            "sportId": str(_SPORT_ID),
            "date": target_date.strftime("%Y-%m-%d"),
            "hydrate": "probablePitcher",
        }
        data = await self._fetch_json(url, params)
        snapshot_time = fetched_at or datetime.now(UTC)

        records: list[MlbProbablePitchersRecord] = []
        for day in data.get("dates", []):
            for game in day.get("games", []):
                record = _record_from_game(game, fetched_at=snapshot_time)
                if record is not None:
                    records.append(record)
        return records

    async def fetch_results(self, start: date, end: date) -> list[MlbGameResult]:
        """Fetch every MLB game from ``start`` to ``end``, with scores from the linescore hydration.

        ``start`` and ``end`` are inclusive and resolved by MLBAM in US Eastern time, so pad
        them by a day when matching UTC kick-offs. The range is split into calls of at most
        ``_RESULTS_CHUNK_DAYS`` days. A response without a ``dates`` list raises ``ValueError``.
        """
        results: list[MlbGameResult] = []
        chunk_start = start
        while chunk_start <= end:
            chunk_end = min(chunk_start + timedelta(days=_RESULTS_CHUNK_DAYS - 1), end)
            if results and self.request_delay_seconds > 0:
                await asyncio.sleep(self.request_delay_seconds)
            data = await self._fetch_json(
                f"{BASE_URL}/schedule",
                {
                    "sportId": str(_SPORT_ID),
                    "startDate": chunk_start.isoformat(),
                    "endDate": chunk_end.isoformat(),
                    "hydrate": "linescore",
                },
            )
            if not isinstance(data.get("dates"), list):
                raise ValueError("MLB schedule response has no dates list")
            for day in data["dates"]:
                schedule_date = date.fromisoformat(day["date"]) if day.get("date") else None
                for game in day.get("games", []):
                    result = _result_from_game(game, schedule_date)
                    if result is not None:
                        results.append(result)
            chunk_start = chunk_end + timedelta(days=1)
        return results

    async def fetch_dates(
        self,
        target_dates: list[date],
        *,
        fetched_at: datetime | None = None,
    ) -> list[MlbProbablePitchersRecord]:
        """Fetch probable pitchers across a list of UTC dates.

        Iterates ``target_dates`` sequentially with ``request_delay_seconds``
        spacing between calls. Pinning ``fetched_at`` once across the loop
        keeps every row from a single MCP/cron call sharing the same
        ``fetched_at`` value, so consumers reading the latest row per
        ``game_pk`` see a coherent snapshot.
        """
        if not target_dates:
            return []
        snapshot_time = fetched_at or datetime.now(UTC)
        records: list[MlbProbablePitchersRecord] = []
        for idx, d in enumerate(target_dates):
            if idx > 0 and self.request_delay_seconds > 0:
                await asyncio.sleep(self.request_delay_seconds)
            day_records = await self.fetch_date(d, fetched_at=snapshot_time)
            logger.info(
                "mlb_probable_pitchers_loaded",
                date=d.isoformat(),
                count=len(day_records),
            )
            records.extend(day_records)
        return records


def _record_from_game(
    game: dict[str, Any],
    *,
    fetched_at: datetime,
) -> MlbProbablePitchersRecord | None:
    """Convert a raw MLBAM ``games[]`` entry into a record.

    Returns ``None`` when the entry is missing the minimum identifiers
    (``gamePk``, ``gameDate``, both team names and ids; a team without an id is a
    placeholder) needed to write a usable row.
    """
    game_pk = game.get("gamePk")
    game_date = game.get("gameDate")
    teams = game.get("teams") or {}
    home = teams.get("home") or {}
    away = teams.get("away") or {}
    home_team = (home.get("team") or {}).get("name")
    away_team = (away.get("team") or {}).get("name")
    teams_identified = (home.get("team") or {}).get("id") and (away.get("team") or {}).get("id")

    if (
        not isinstance(game_pk, int)
        or not game_date
        or not home_team
        or not away_team
        or not teams_identified
    ):
        logger.warning("mlb_game_missing_core_fields", raw=game)
        return None

    status = game.get("status") or {}
    home_pitcher = home.get("probablePitcher") or {}
    away_pitcher = away.get("probablePitcher") or {}

    return MlbProbablePitchersRecord(
        game_pk=game_pk,
        commence_time=_parse_utc(game_date),
        fetched_at=fetched_at,
        home_team=home_team,
        away_team=away_team,
        game_type=game.get("gameType") or "R",
        home_pitcher_name=home_pitcher.get("fullName"),
        home_pitcher_id=home_pitcher.get("id"),
        away_pitcher_name=away_pitcher.get("fullName"),
        away_pitcher_id=away_pitcher.get("id"),
        game_status=(
            IF_NECESSARY_STATUS if game.get("ifNecessary") == "Y" else status.get("detailedState")
        ),
        start_time_tbd=bool(status.get("startTimeTBD")),
    )


def _game_state(status: dict[str, Any]) -> MlbGameState:
    # A postponed or cancelled slot reports abstractGameState "Final", so the coded state decides.
    coded = status.get("codedGameState")
    if coded == "D":
        return MlbGameState.POSTPONED
    if coded == "C":
        return MlbGameState.CANCELLED
    if coded in ("W", "X"):
        return MlbGameState.OTHER
    return _ABSTRACT_STATES.get(status.get("abstractGameState"), MlbGameState.OTHER)


_ABSTRACT_STATES = {
    "Preview": MlbGameState.SCHEDULED,
    "Live": MlbGameState.LIVE,
    "Final": MlbGameState.FINAL,
}


def _result_from_game(game: dict[str, Any], schedule_date: date | None) -> MlbGameResult | None:
    """Convert a raw ``games[]`` entry of a linescore-hydrated schedule; ``None`` without core fields.

    ``schedule_date`` is the ``dates[].date`` the entry is listed under: a postponed slot is listed
    under its original date while its ``officialDate`` is the date it is played.
    """
    game_pk = game.get("gamePk")
    game_date = game.get("gameDate")
    teams = game.get("teams") or {}
    home_name = ((teams.get("home") or {}).get("team") or {}).get("name")
    away_name = ((teams.get("away") or {}).get("team") or {}).get("name")
    if not isinstance(game_pk, int) or not game_date or not home_name or not away_name:
        return None

    linescore_teams = (game.get("linescore") or {}).get("teams") or {}

    def score(side: str) -> int | None:
        value = (teams.get(side) or {}).get("score")
        if value is None:
            value = (linescore_teams.get(side) or {}).get("runs")
        return value if isinstance(value, int) else None

    home_score, away_score = score("home"), score("away")
    is_final = (game.get("status") or {}).get("abstractGameState") == "Final" and None not in (
        home_score,
        away_score,
    )
    return MlbGameResult(
        game_pk=game_pk,
        start_time=_parse_utc(game_date),
        home_team=home_name,
        away_team=away_name,
        is_final=is_final,
        home_score=home_score,
        away_score=away_score,
        game_number=game["gameNumber"] if isinstance(game.get("gameNumber"), int) else 1,
        start_time_tbd=bool((game.get("status") or {}).get("startTimeTBD")),
        official_date=date.fromisoformat(game["officialDate"])
        if game.get("officialDate")
        else None,
        state=_game_state(game.get("status") or {}),
        schedule_date=schedule_date,
        resumed=bool(game.get("resumedFrom")),
    )
