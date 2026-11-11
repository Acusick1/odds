"""What the stored events say about the fixtures around a match at a decision time: each team's
previous result and the kick-offs near the match's own. Read-only input to a rule, loaded by
the one query pair the runner, the replay backtest and the parity check share."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from odds_core.as_of import RESULT_LAG
from odds_core.market_prices import Outcome
from odds_core.models import Event, EventStatus
from odds_core.time import season_start
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

# Kick-offs at most this far from a match's own are concurrent with it, the boundary included.
CONCURRENT_WINDOW = timedelta(minutes=10)
# A result stored later than this after kick-off is overdue: the daily results jobs (08:00 UTC,
# heartbeat expected within 26 hours) have had a run to store it, with margin (inferred).
RESULT_STORE_LAG = timedelta(hours=36)
# Events that are not a match to be played, or that was played, as far as their status says.
NOT_A_MATCH: tuple[EventStatus, ...] = (
    EventStatus.CANCELLED,
    EventStatus.POSTPONED,
    EventStatus.UNMATCHED,
)


class FixtureInput(StrEnum):
    """A fixture input a rule reads."""

    PREVIOUS_RESULT = "previous_result"
    CONCURRENT_KICKOFFS = "concurrent_kickoffs"


class TeamResult(StrEnum):
    WIN = "win"
    DRAW = "draw"
    LOSS = "loss"


@dataclass(frozen=True)
class PreviousResult:
    """A team's previous league match, from the team's side."""

    event_id: int
    kickoff: datetime
    goals_for: int
    goals_against: int

    @property
    def goal_difference(self) -> int:
        return self.goals_for - self.goals_against

    @property
    def result(self) -> TeamResult:
        if self.goal_difference > 0:
            return TeamResult.WIN
        return TeamResult.LOSS if self.goal_difference < 0 else TeamResult.DRAW


@dataclass(frozen=True)
class FixtureContext:
    """The fixture inputs a rule declared; what it did not declare stays empty or ``None``.

    ``previous`` holds, for the home and the away team (``Outcome.HOME``, ``Outcome.AWAY``),
    the previous league match of the match's season, or ``None`` for a team with no earlier
    match that season. A team whose previous match exists but has no result known at the
    decision time is in ``pending_results`` when the match kicked off less than
    ``RESULT_STORE_LAG`` before it, and in ``overdue_results`` otherwise; a decision with
    either has no inputs. ``concurrent_kickoffs`` counts the match itself and every other
    kick-off within ``CONCURRENT_WINDOW`` of its own. ``event_ids`` are the events read, the
    match's own excluded, in id order.
    """

    previous: Mapping[Outcome, PreviousResult | None] = field(
        default_factory=lambda: MappingProxyType({})
    )
    pending_results: tuple[Outcome, ...] = ()
    overdue_results: tuple[Outcome, ...] = ()
    concurrent_kickoffs: int | None = None
    event_ids: tuple[int, ...] = ()

    @property
    def unknown_results(self) -> tuple[Outcome, ...]:
        return self.pending_results + self.overdue_results

    def to_record(self) -> dict[str, Any] | None:
        """The values a decision consumed, as JSON; ``None`` when none were declared."""
        record: dict[str, Any] = {}
        if self.previous:
            record["previous"] = {
                side.value: None
                if previous is None
                else {
                    "event_id": previous.event_id,
                    "kickoff": previous.kickoff.isoformat(),
                    "goals_for": previous.goals_for,
                    "goals_against": previous.goals_against,
                }
                for side, previous in self.previous.items()
            }
        if self.concurrent_kickoffs is not None:
            record["concurrent_kickoffs"] = self.concurrent_kickoffs
        return record or None

    @classmethod
    def from_record(
        cls, record: Mapping[str, Any] | None, event_ids: Collection[int] = ()
    ) -> FixtureContext:
        """The context ``to_record`` recorded; an empty one for ``None``."""
        record = record or {}
        previous: dict[Outcome, PreviousResult | None] = {}
        for side, values in record.get("previous", {}).items():
            previous[Outcome(side)] = (
                None
                if values is None
                else PreviousResult(
                    values["event_id"],
                    datetime.fromisoformat(values["kickoff"]),
                    values["goals_for"],
                    values["goals_against"],
                )
            )
        return cls(
            previous=MappingProxyType(previous),
            concurrent_kickoffs=record.get("concurrent_kickoffs"),
            event_ids=tuple(sorted(event_ids)),
        )


def _stored(known_at: datetime | None) -> list[object]:
    """The conditions on which events count: a match, stored by ``known_at``."""
    conditions: list[object] = [col(Event.status).not_in(NOT_A_MATCH)]
    if known_at is not None:
        conditions.append(col(Event.created_at) <= known_at)
    return conditions


def _known_result(
    previous: Event, team: str, decision_time: datetime, known_at: datetime | None
) -> PreviousResult | None:
    """``previous``'s result for ``team`` when it was known at the decision time: the event is
    FINAL with both scores, kicked off at least ``RESULT_LAG`` before it, and was completed
    (``completed_at``) by the decision time and by ``known_at`` when that is given.

    ``completed_at`` is the time of the latest FINAL write, so a corrected score postdates the
    first result: a replay treats a corrected match as unknown before the correction, never
    as known earlier than it was.
    """
    stored_by = decision_time if known_at is None else min(decision_time, known_at)
    if (
        previous.id is None
        or previous.status != EventStatus.FINAL
        or previous.home_score is None
        or previous.away_score is None
        or previous.commence_time + RESULT_LAG > decision_time
        or previous.completed_at is None
        or previous.completed_at > stored_by
    ):
        return None
    home = previous.home_team == team
    return PreviousResult(
        previous.id,
        previous.commence_time,
        previous.home_score if home else previous.away_score,
        previous.away_score if home else previous.home_score,
    )


async def load_fixture_context(
    session: AsyncSession,
    event: Event,
    kickoff: datetime,
    decision_time: datetime,
    inputs: frozenset[FixtureInput],
    leagues: Collection[str],
    known_at: datetime | None = None,
) -> FixtureContext:
    """The fixture inputs ``inputs`` names, for ``event`` kicking off at ``kickoff``, as known
    at ``decision_time``.

    The previous match of a team is its latest earlier match of ``event``'s league in the
    season of ``kickoff``, whatever its status but a match not played (``NOT_A_MATCH``); a
    team with none has ``None``, and one whose match has no result known then (``_known_result``)
    is pending or overdue, never replaced by an older match. The concurrent count is the
    events of ``leagues`` kicking off within ``CONCURRENT_WINDOW`` of ``kickoff`` plus the match
    itself. With ``known_at``, an event stored (``created_at``) after it does not count.
    Kick-offs are the events' stored ones; an event keeps no history of them.
    """
    previous: dict[Outcome, PreviousResult | None] = {}
    pending: list[Outcome] = []
    overdue: list[Outcome] = []
    ids: set[int] = set()
    if FixtureInput.PREVIOUS_RESULT in inputs:
        for side, team in ((Outcome.HOME, event.home_team), (Outcome.AWAY, event.away_team)):
            found = (
                await session.execute(
                    select(Event)
                    .where(
                        col(Event.sport_key) == event.sport_key,
                        col(Event.id) != event.id,
                        or_(col(Event.home_team) == team, col(Event.away_team) == team),
                        col(Event.commence_time) >= season_start(kickoff, event.sport_key),
                        col(Event.commence_time) < kickoff,
                        *_stored(known_at),
                    )
                    .order_by(col(Event.commence_time).desc(), col(Event.id).desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
            if found is None:
                previous[side] = None
                continue
            assert found.id is not None  # noqa: S101
            ids.add(found.id)
            result = _known_result(found, team, decision_time, known_at)
            previous[side] = result
            if result is None:
                recent = found.commence_time + RESULT_STORE_LAG > decision_time
                (pending if recent else overdue).append(side)
    concurrent: int | None = None
    if FixtureInput.CONCURRENT_KICKOFFS in inputs:
        near = (
            (
                await session.execute(
                    select(col(Event.id)).where(
                        col(Event.sport_key).in_(sorted(leagues)),
                        col(Event.id) != event.id,
                        col(Event.commence_time) >= kickoff - CONCURRENT_WINDOW,
                        col(Event.commence_time) <= kickoff + CONCURRENT_WINDOW,
                        *_stored(known_at),
                    )
                )
            )
            .scalars()
            .all()
        )
        ids.update(i for i in near if i is not None)
        concurrent = len(near) + 1
    return FixtureContext(
        previous=MappingProxyType(previous),
        pending_results=tuple(pending),
        overdue_results=tuple(overdue),
        concurrent_kickoffs=concurrent,
        event_ids=tuple(sorted(ids)),
    )
