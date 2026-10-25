"""Unified, sport-aware scheduling decision engine.

This module owns the single mapping from *proximity to game* to *polling
cadence* plus the run/skip gate, replacing three previously divergent engines
(``SchedulingIntelligence``, ``ProximitySchedule``, and the agent's hand-rolled
``TIER_*`` constants).

Two decision functions share the query layer, overnight handling, and return
type:

- :func:`decide_forward` keys off the *next* upcoming game for a sport. It maps
  proximity to a per-job cadence via :class:`CadenceConfig` (whose tier
  boundaries are the canonical :class:`FetchTier` boundaries) and gates
  ``should_execute=False`` when no upcoming game exists.
- :func:`decide_backward` keys off *recent* games for a sport. It runs while a
  game finished within ``window`` is missing a final status (active cadence)
  and idles otherwise — the semantics used by scores and status-update jobs.

Both are sport-scoped: callers pass a ``sport_key`` and queries filter on it,
fixing the prior all-sports bug where a per-sport compound job was kept alive by
another sport's fixtures.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol

import structlog
from odds_core.database import async_session_maker
from odds_core.models import EventStatus
from sqlalchemy.ext.asyncio import AsyncSession

from odds_lambda.fetch_tier import FetchTier
from odds_lambda.scheduling.helpers import apply_overnight_skip, get_next_kickoff
from odds_lambda.storage.readers import OddsReader

logger = structlog.get_logger()

# A session factory yields an async session usable as ``async with factory()``.
SessionFactory = Callable[[], AbstractAsyncContextManager[AsyncSession]]

# (overnight_start_utc, overnight_resume_utc)
OvernightWindow = tuple[int, int]

# Per-sport overnight suppression windows (start_utc, resume_utc) — the hours
# during which no forward job should fire because no relevant games are live.
# EPL: last KO ~20:00 UTC → suppress 22:00-06:00.
# MLB: last first-pitch ~02:00 UTC, games run to ~05:00 UTC → suppress 05:00-14:00.
OVERNIGHT_WINDOWS: dict[str, OvernightWindow] = {
    "soccer_epl": (22, 6),
    "baseball_mlb": (5, 14),
}

# A collector that discovers its own events goes dormant when its sport has no
# fixture inside this horizon. It is longer than any gap between two Premier
# League fixtures (about 20 days) so dormancy only ever means an off-season.
DORMANCY_HORIZON_DAYS = 28


class KickoffLookup(Protocol):
    async def __call__(self, sport_key: str, *, now: datetime) -> datetime | None: ...


# A close run due within this gap would rerun the job straight away; the run
# in progress already sits at the close.
MIN_CLOSE_RUN_GAP = timedelta(seconds=60)


@dataclass(slots=True, frozen=True)
class ScheduleDecision:
    """Decision about whether to execute now and when to run next (immutable).

    Attributes:
        should_execute: Whether the job should run now.
        reason: Human-readable explanation for the decision.
        next_execution: When to schedule the next run (None = no more runs).
        tier: Which fetch tier applies (None when not proximity-driven).
    """

    should_execute: bool
    reason: str
    next_execution: datetime | None
    tier: FetchTier | None


@dataclass(slots=True, frozen=True)
class CadenceConfig:
    """Per-job polling cadence (hours) keyed by canonical ``FetchTier``.

    Tier boundaries are always the canonical :class:`FetchTier` boundaries so
    every job agrees on what "closing" / "pregame" / etc. mean; only the
    interval values differ per job (e.g. a cheap API polls more aggressively
    than a browser-driven scrape).

    ``no_game`` is the check-in interval used when there is no upcoming game
    (off-season / fixtures not yet posted); the check-in does no overnight skip
    and is brought forward to the moment the next fixture enters the lookahead.
    ``db_fallback`` is the interval a caller should use when the kickoff query
    itself fails (DB unreachable), so scheduling never blocks on DB
    availability. ``close_lead``, when set,
    pulls ``next_execution`` forward to ``close_lead`` before the next kickoff
    the cadence would otherwise step over, so a run always lands just before
    each kickoff.
    """

    closing: float
    pregame: float
    sharp: float
    early: float
    opening: float
    no_game: float
    db_fallback: float = 1.0
    close_lead: timedelta | None = None

    def interval_for(self, tier: FetchTier) -> float:
        """Return the polling interval (hours) for a tier.

        ``IN_PLAY`` shares the ``closing`` cadence (incidental collection while
        a game is live).
        """
        mapping: dict[FetchTier, float] = {
            FetchTier.IN_PLAY: self.closing,
            FetchTier.CLOSING: self.closing,
            FetchTier.PREGAME: self.pregame,
            FetchTier.SHARP: self.sharp,
            FetchTier.EARLY: self.early,
            FetchTier.OPENING: self.opening,
        }
        return mapping[tier]


def _next_time(
    now: datetime,
    interval_hours: float,
    overnight: OvernightWindow | None,
) -> datetime:
    """Add ``interval_hours`` to ``now`` and apply the overnight skip if set."""
    candidate = now + timedelta(hours=interval_hours)
    if overnight is None:
        return candidate
    start_utc, resume_utc = overnight
    return apply_overnight_skip(
        candidate,
        overnight_start_utc=start_utc,
        overnight_resume_utc=resume_utc,
    )


def _idle_next_execution(
    now: datetime,
    cadence: CadenceConfig,
    next_kickoff: datetime | None,
    *,
    lookahead_days: int,
) -> datetime:
    """Check-in time while no fixture is within the lookahead.

    The check-in is brought forward to the moment a known fixture enters the
    lookahead, so the job is awake exactly when its fixtures are.
    """
    check_in = now + timedelta(hours=cadence.no_game)
    if next_kickoff is None:
        return check_in
    return min(check_in, next_kickoff - timedelta(days=lookahead_days))


def _decide_from_kickoff(
    sport_key: str,
    cadence: CadenceConfig,
    next_kickoff: datetime | None,
    *,
    now: datetime,
    overnight: OvernightWindow | None,
    lookahead_days: int,
    close_kickoff: datetime | None,
) -> ScheduleDecision:
    """Classify an already-resolved kickoff into a decision (no DB access).

    Shared core of :func:`decide_forward` and :func:`decide_forward_resilient`:
    gates ``should_execute=False`` when ``next_kickoff`` is absent or beyond
    ``lookahead_days``; otherwise maps proximity to a ``FetchTier`` cadence.
    """
    if next_kickoff is None or next_kickoff > now + timedelta(days=lookahead_days):
        return ScheduleDecision(
            should_execute=False,
            reason=f"No upcoming {sport_key} game within {lookahead_days}d",
            next_execution=_idle_next_execution(
                now, cadence, next_kickoff, lookahead_days=lookahead_days
            ),
            tier=None,
        )

    hours_until = (next_kickoff - now).total_seconds() / 3600
    tier = classify_tier(hours_until)
    interval = cadence.interval_for(tier)
    next_execution = _next_time(now, interval, overnight)
    reason = f"Next {sport_key} game in {hours_until:.1f}h ({tier.value} tier)"

    if cadence.close_lead is not None and close_kickoff is not None:
        close_run = close_kickoff - cadence.close_lead
        if close_run - now >= MIN_CLOSE_RUN_GAP and close_run < next_execution:
            next_execution = close_run
            reason += f", close run {cadence.close_lead} before {close_kickoff:%H:%M} UTC"

    return ScheduleDecision(
        should_execute=True,
        reason=reason,
        next_execution=next_execution,
        tier=tier,
    )


async def _get_close_kickoff(
    sport_key: str,
    cadence: CadenceConfig,
    *,
    now: datetime,
    lookup: KickoffLookup | None = None,
) -> datetime | None:
    """Earliest kickoff whose close run (``close_lead`` before it) is still ahead of ``now``."""
    if cadence.close_lead is None:
        return None
    return await (lookup or get_next_kickoff)(sport_key, now=now + cadence.close_lead)


async def decide_forward(
    sport_key: str,
    cadence: CadenceConfig,
    *,
    overnight: OvernightWindow | None = None,
    lookahead_days: int = 7,
    now: datetime | None = None,
    next_kickoff: datetime | None = None,
) -> ScheduleDecision:
    """Forward-looking decision keyed off the next upcoming game for a sport.

    Gates ``should_execute=False`` when no upcoming game exists within
    ``lookahead_days``; otherwise classifies proximity into a ``FetchTier`` and
    returns the matching cadence interval.

    Args:
        sport_key: Sport to scope the next-game query to.
        cadence: Per-job intervals keyed by ``FetchTier``.
        overnight: Optional (start_utc, resume_utc) suppression window applied
            to ``next_execution``.
        lookahead_days: How far ahead a game must be to count as "upcoming".
        now: Injectable clock (defaults to ``datetime.now(UTC)``).
        next_kickoff: Pre-fetched kickoff to avoid a redundant query (used by
            callers that re-query after doing work). When omitted the kickoff is
            queried via :func:`get_next_kickoff`.
    """
    now = now or datetime.now(UTC)

    if next_kickoff is None:
        next_kickoff = await get_next_kickoff(sport_key, now=now)

    return _decide_from_kickoff(
        sport_key,
        cadence,
        next_kickoff,
        now=now,
        overnight=overnight,
        lookahead_days=lookahead_days,
        close_kickoff=await _get_close_kickoff(sport_key, cadence, now=now),
    )


async def decide_forward_resilient(
    sport_keys: list[str],
    cadence: CadenceConfig,
    *,
    overnight: OvernightWindow | None = None,
    lookahead_days: int = 7,
    now: datetime | None = None,
    next_kickoff_lookup: KickoffLookup | None = None,
    close_kickoff_lookup: KickoffLookup | None = None,
) -> ScheduleDecision:
    """``decide_forward`` over one-or-more sports, resilient to query failure.

    The proximity gate keys off the *soonest* kickoff across ``sport_keys`` (a
    single sport for per-sport jobs, several for combined runs), and the close
    run off ``close_kickoff_lookup`` (stored events by default). If the kickoff
    query itself fails, falls back to ``cadence.db_fallback`` so scheduling
    never blocks on DB availability — preserving the self-scheduling chain.
    """
    now = now or datetime.now(UTC)
    lookup = next_kickoff_lookup or get_next_kickoff
    try:
        soonest: datetime | None = None
        soonest_sport = sport_keys[0]
        close_kickoff: datetime | None = None
        for sk in sport_keys:
            ko = await lookup(sk, now=now)
            if ko is not None and (soonest is None or ko < soonest):
                soonest, soonest_sport = ko, sk
            close_ko = await _get_close_kickoff(sk, cadence, now=now, lookup=close_kickoff_lookup)
            if close_ko is not None and (close_kickoff is None or close_ko < close_kickoff):
                close_kickoff = close_ko
        return _decide_from_kickoff(
            soonest_sport,
            cadence,
            soonest,
            now=now,
            overnight=overnight,
            lookahead_days=lookahead_days,
            close_kickoff=close_kickoff,
        )
    except Exception as e:
        logger.warning("next_kickoff_query_failed", error=str(e), exc_info=True)
        return ScheduleDecision(
            should_execute=True,
            reason="kickoff query failed — db_fallback cadence",
            next_execution=_next_time(now, cadence.db_fallback, overnight),
            tier=None,
        )


async def decide_backward(
    sport_key: str,
    *,
    window: timedelta,
    active_interval: float,
    idle_interval: float,
    statuses_needing_update: set[EventStatus] | None = None,
    overnight: OvernightWindow | None = None,
    now: datetime | None = None,
    session_factory: SessionFactory | None = None,
) -> ScheduleDecision:
    """Backward-looking decision keyed off recent games for a sport.

    Executes (active cadence) while at least one game that started within
    ``window`` still lacks a final status; otherwise idles. This preserves the
    semantics of the previous ``should_execute_scores`` (3-day window) and
    ``should_execute_status_update`` (4-hour window), now sport-scoped.

    Args:
        sport_key: Sport to scope the recent-games query to.
        window: How far back to look for games needing an update.
        active_interval: Cadence (hours) while games still need updating.
        idle_interval: Cadence (hours) when nothing needs updating.
        statuses_needing_update: When provided, a game counts as needing an
            update only if its status is in this set (status job: SCHEDULED
            games that may have started). When omitted, any non-FINAL game
            counts (scores job).
        overnight: Optional (start_utc, resume_utc) suppression window.
        now: Injectable clock (defaults to ``datetime.now(UTC)``).
        session_factory: Injectable session factory for testing.
    """
    now = now or datetime.now(UTC)
    factory = session_factory or async_session_maker

    async with factory() as session:
        reader = OddsReader(session)
        events = await reader.get_events_by_date_range(
            start_date=now - window,
            end_date=now,
            sport_key=sport_key,
        )

    if statuses_needing_update is not None:
        needs_update = sum(1 for e in events if e.status in statuses_needing_update)
    else:
        needs_update = sum(1 for e in events if e.status != EventStatus.FINAL)

    if needs_update > 0:
        return ScheduleDecision(
            should_execute=True,
            reason=f"{needs_update} {sport_key} game(s) need updates",
            next_execution=_next_time(now, active_interval, overnight),
            tier=None,
        )

    return ScheduleDecision(
        should_execute=False,
        reason=f"No {sport_key} games need updates",
        next_execution=_next_time(now, idle_interval, overnight),
        tier=None,
    )


async def decide_backward_resilient(
    sport_keys: list[str],
    *,
    window: timedelta,
    active_interval: float,
    idle_interval: float,
    db_fallback: float = 1.0,
    statuses_needing_update: set[EventStatus] | None = None,
    overnight: OvernightWindow | None = None,
    now: datetime | None = None,
    session_factory: SessionFactory | None = None,
) -> ScheduleDecision:
    """``decide_backward`` over one-or-more sports, resilient to query failure.

    Executes (returns the first ``should_execute=True``) if *any* sport has a
    recent game needing an update; otherwise idles. If the recent-games query
    fails (DB unreachable), falls back to ``should_execute=True`` at the
    ``db_fallback`` cadence so the self-scheduling chain survives a DB outage —
    the same crash-survival guarantee :func:`decide_forward_resilient` gives the
    forward jobs.
    """
    now = now or datetime.now(UTC)
    try:
        decision: ScheduleDecision | None = None
        for sk in sport_keys:
            decision = await decide_backward(
                sk,
                window=window,
                active_interval=active_interval,
                idle_interval=idle_interval,
                statuses_needing_update=statuses_needing_update,
                overnight=overnight,
                now=now,
                session_factory=session_factory,
            )
            if decision.should_execute:
                return decision
        assert decision is not None  # noqa: S101 — sport_keys non-empty by caller contract
        return decision
    except Exception as e:
        logger.warning("recent_games_query_failed", error=str(e), exc_info=True)
        return ScheduleDecision(
            should_execute=True,
            reason="recent-games query failed — db_fallback cadence",
            next_execution=_next_time(now, db_fallback, overnight),
            tier=None,
        )


def classify_tier(hours_until: float) -> FetchTier:
    """Classify hours-until-kickoff into a canonical ``FetchTier``.

    Reads the boundaries straight off ``FetchTier.max_hours`` so the canonical
    thresholds live in one place, and applies the strict ``<`` comparison its
    contract documents (``hours_before < tier.max_hours``). A game exactly on a
    boundary (e.g. 3.0h) therefore falls into the *next* tier up — not the tier
    whose ``max_hours`` it equals. This differs from ``tier_utils.calculate_tier``,
    which uses ``<=`` and so disagrees at exact boundaries.
    """
    if hours_until < 0:
        return FetchTier.IN_PLAY
    for tier in (
        FetchTier.CLOSING,
        FetchTier.PREGAME,
        FetchTier.SHARP,
        FetchTier.EARLY,
    ):
        if hours_until < tier.max_hours:
            return tier
    return FetchTier.OPENING
