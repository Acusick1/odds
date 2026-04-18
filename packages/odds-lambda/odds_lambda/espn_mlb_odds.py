"""ESPN undocumented MLB odds lookup — featured Over/Under line per game.

The root ``overUnder`` field on ESPN's competition odds endpoint reflects
DraftKings' current featured total and is populated well before first pitch.
Treated as fragile infrastructure: all network calls are wrapped in
try/except and failed events are silently dropped so a single bad response
can't break the whole lookup.

Not intended for ingestion. Used at scrape-planning time to determine
which ``over_under_X_Y`` markets to request from OddsPortal so we don't
have to sweep the full 6.5–11.5 line range per match.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import aiohttp
import structlog

logger = structlog.get_logger()

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
EVENT_ODDS_URL = (
    "https://sports.core.api.espn.com/v2/sports/baseball/leagues/mlb/"
    "events/{event_id}/competitions/{event_id}/odds"
)
DEFAULT_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True)
class MlbGameTotal:
    """Featured Over/Under line for a single MLB game."""

    event_id: str
    home_team: str
    away_team: str
    commence_time: datetime
    line: float


async def get_mlb_main_totals(
    *,
    target_date: date | None = None,
    session: aiohttp.ClientSession | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> list[MlbGameTotal]:
    """Look up ESPN's featured Over/Under for every MLB game on ``target_date``.

    Args:
        target_date: UTC date to query. ``None`` → ESPN's current scoreboard.
        session: Optional aiohttp session for connection reuse. A fresh session
            is created and closed if omitted.
        timeout_seconds: Per-request timeout.

    Returns:
        List of ``MlbGameTotal`` for every game whose odds endpoint returned a
        valid ``overUnder`` float. Games with network errors, missing totals,
        or malformed responses are omitted.
    """
    owns_session = session is None
    client = session or aiohttp.ClientSession()
    try:
        events = await _fetch_scoreboard(client, target_date, timeout_seconds)
        if not events:
            return []
        tasks = [_fetch_event_total(client, event, timeout_seconds) for event in events]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]
    finally:
        if owns_session:
            await client.close()


async def _fetch_scoreboard(
    client: aiohttp.ClientSession,
    target_date: date | None,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    """Fetch MLB scoreboard for target date and return the events array."""
    params: dict[str, str] = {}
    if target_date is not None:
        params["dates"] = target_date.strftime("%Y%m%d")

    try:
        async with client.get(
            SCOREBOARD_URL,
            params=params,
            timeout=aiohttp.ClientTimeout(total=timeout_seconds),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
    except (aiohttp.ClientError, TimeoutError) as e:
        logger.warning("espn_scoreboard_fetch_failed", error=str(e))
        return []

    events = data.get("events", [])
    if not isinstance(events, list):
        return []
    return events


async def _fetch_event_total(
    client: aiohttp.ClientSession,
    event: dict[str, Any],
    timeout_seconds: float,
) -> MlbGameTotal | None:
    """Fetch the featured Over/Under for a single event.

    Returns ``None`` on any network error, missing field, or malformed shape.
    The root ``items[0].overUnder`` float is authoritative — ``current.total.value``
    is unreliable (often ``0.0``) and is ignored.
    """
    event_id = event.get("id")
    if not event_id:
        return None

    home, away = _extract_teams(event)
    commence_time = _extract_commence_time(event)
    if not home or not away or commence_time is None:
        return None

    url = EVENT_ODDS_URL.format(event_id=event_id)

    try:
        async with client.get(url, timeout=aiohttp.ClientTimeout(total=timeout_seconds)) as resp:
            resp.raise_for_status()
            data = await resp.json()
    except (aiohttp.ClientError, TimeoutError) as e:
        logger.debug("espn_event_odds_fetch_failed", event_id=event_id, error=str(e))
        return None

    items = data.get("items") or []
    if not items:
        return None
    line = items[0].get("overUnder")
    # bool is a subclass of int in Python; exclude it explicitly.
    if isinstance(line, bool) or not isinstance(line, int | float) or line <= 0:
        return None

    return MlbGameTotal(
        event_id=str(event_id),
        home_team=home,
        away_team=away,
        commence_time=commence_time,
        line=float(line),
    )


def _extract_teams(event: dict[str, Any]) -> tuple[str | None, str | None]:
    """Pull home/away display names from an ESPN scoreboard event."""
    competitions = event.get("competitions") or []
    if not competitions:
        return None, None
    competitors = competitions[0].get("competitors") or []

    home: str | None = None
    away: str | None = None
    for c in competitors:
        team = c.get("team") or {}
        name = team.get("displayName") or ""
        side = c.get("homeAway")
        if side == "home":
            home = name or None
        elif side == "away":
            away = name or None
    return home, away


def _extract_commence_time(event: dict[str, Any]) -> datetime | None:
    """Parse the event's ISO8601 ``date`` field to a timezone-aware datetime."""
    raw = event.get("date")
    if not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def line_to_market_key(line: float) -> str:
    """Convert a numeric total to OddsHarvester's ``over_under_X_Y`` market key.

    Rounds to the nearest 0.5 since OddsHarvester's ``BaseballOverUnderMarket``
    enum only defines half-point and whole-number lines. Examples::

        8.5 → "over_under_8_5"
        9.0 → "over_under_9_0"
        8.3 → "over_under_8_5"  (rounded to nearest 0.5)
    """
    rounded = round(line * 2) / 2
    whole = int(rounded)
    frac = int(round((rounded - whole) * 10))
    return f"over_under_{whole}_{frac}"


def distinct_market_keys(totals: list[MlbGameTotal]) -> list[str]:
    """De-duplicated ``over_under_X_Y`` keys for a slate, sorted by numeric line.

    Sorting on the parsed float avoids the lexicographic ordering surprise
    where ``over_under_10_5`` would precede ``over_under_9_5``.
    """
    distinct_lines = sorted({round(t.line * 2) / 2 for t in totals})
    return [line_to_market_key(line) for line in distinct_lines]


def _team_key(home: str, away: str) -> str:
    """Normalise a team-pair into a case-insensitive matching key."""
    return f"{home.strip().lower()}|{away.strip().lower()}"


def group_match_links_by_line(
    totals: list[MlbGameTotal],
    raw_matches: list[dict[str, Any]],
) -> dict[str, list[str]]:
    """Group OddsPortal match URLs by their ESPN-discovered main Over/Under line.

    Args:
        totals: Per-game featured totals from ESPN.
        raw_matches: OddsHarvester output (typically a home_away discovery
            scrape) containing ``home_team`` / ``away_team`` / ``match_link``.

    Returns:
        Mapping of ``over_under_X_Y`` → ``[match_link, ...]``. Matches whose
        teams don't appear in ``totals`` are silently omitted — no main-line
        data means nothing to target for them.
    """
    link_by_teams: dict[str, str] = {}
    for m in raw_matches:
        home = (m.get("home_team") or "").strip()
        away = (m.get("away_team") or "").strip()
        link = (m.get("match_link") or "").strip()
        if not home or not away or not link:
            continue
        link_by_teams[_team_key(home, away)] = link

    groups: dict[str, list[str]] = {}
    for total in totals:
        link = link_by_teams.get(_team_key(total.home_team, total.away_team))
        if link is None:
            continue
        market = line_to_market_key(total.line)
        groups.setdefault(market, []).append(link)
    return groups
