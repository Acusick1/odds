"""Backfill the estimated EV of candidate bets placed before bets recorded it.

Each decision of a version of a registered rule candidate is replayed from its recorded
inputs under the registry's rule, params and book set, as the parity check replays it (``odds_analytics.candidate_parity.recorded_intents``), in the order
the live runs took it, dropping the selections the match's earlier decisions placed bets on.
A bet takes the estimate of the replayed bet on its selection, line, bookmaker and price; a
recorded bet with no such replayed bet keeps none. The registry holds only a name's current version, and a
name's rule never changes across versions, so an earlier version is replayed with the current
entry: a recorded bet whose replayed bet does not reproduce stays unmatched.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from odds_core.bet_models import Bet
from odds_core.bookmakers import BOOK_SETS
from odds_core.candidates import Candidate, RuleCandidate
from odds_core.market_prices import Outcome
from odds_core.models import Event
from odds_lambda.candidate_rules import BetIntent, ModelRuleSpec, Selection
from odds_lambda.candidate_rules.registry import resolve_rule
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from odds_analytics.candidate_parity import (
    load_decisions,
    load_recorded_game,
    recorded_intents,
    recorded_quotes,
)

# A bet's price matches a replayed bet's when they are equal to this many decimal places, the
# tolerance the parity check compares prices at (``odds_analytics.candidate_parity.bet_key``).
PRICE_DECIMALS = 9


@dataclass(frozen=True)
class BetMatch:
    """What a bet and a replayed bet must share to be the same bet."""

    selection: str
    line: float | None
    bookmaker: str
    price: float

    @classmethod
    def of(cls, selection: str, line: float | None, bookmaker: str, price: float) -> BetMatch:
        return cls(selection, line, bookmaker, round(price, PRICE_DECIMALS))


@dataclass(frozen=True)
class BetEstimate:
    """A decision's bet and the estimate of its replayed bet; ``matched`` is False when the
    replay placed no bet on its selection, line, bookmaker and price."""

    bet_id: int
    decision_id: int
    event_id: int
    key: BetMatch
    matched: bool
    estimated_ev: float | None
    stored: float | None


@dataclass(frozen=True)
class VersionBackfill:
    name: str
    version: int
    bets: tuple[BetEstimate, ...]

    @property
    def to_write(self) -> list[BetEstimate]:
        """Matched bets with an estimate and none stored; a stored estimate is never
        overwritten."""
        return [
            t for t in self.bets if t.matched and t.estimated_ev is not None and t.stored is None
        ]

    @property
    def mismatched(self) -> list[BetEstimate]:
        """Matched bets whose stored estimate differs from the replay's."""
        return [
            t
            for t in self.bets
            if t.matched and t.stored is not None and t.stored != t.estimated_ev
        ]

    @property
    def unmatched(self) -> list[BetEstimate]:
        return [t for t in self.bets if not t.matched]


@dataclass(frozen=True)
class EstimateBackfill:
    """The backfill of each version of a registered rule candidate with decision bets,
    and the bets of versions it cannot replay, by (name, version): a name the registry does
    not hold, or a model rule, which places no live bets."""

    versions: tuple[VersionBackfill, ...]
    skipped: dict[tuple[str, int], int]


async def _version_backfill(
    session: AsyncSession, candidate: RuleCandidate, version: int
) -> VersionBackfill:
    spec, params = resolve_rule(candidate)
    assert not isinstance(spec, ModelRuleSpec)  # noqa: S101
    decisions = await load_decisions(session, candidate.name, version)
    games = await load_recorded_game(session, decisions) if spec.game_inputs else {}
    bets: dict[int, list[Bet]] = defaultdict(list)
    if decisions:
        rows = await session.scalars(
            select(Bet)
            .where(col(Bet.decision_id).in_([d.id for d in decisions]))
            .order_by(col(Bet.id))
        )
        for bet in rows:
            assert bet.decision_id is not None  # noqa: S101
            bets[bet.decision_id].append(bet)
    live_selections: dict[int, set[Selection]] = defaultdict(set)
    estimates: list[BetEstimate] = []
    for decision in decisions:
        assert decision.id is not None  # noqa: S101
        placed = bets[decision.id]
        if not placed:
            continue
        event = await session.get(Event, decision.event_id)
        assert event is not None  # noqa: S101
        quoted = await recorded_quotes(session, spec, params, event, candidate.market, decision)
        intents: list[BetIntent] = recorded_intents(
            spec,
            params,
            event,
            candidate.market,
            decision,
            quoted,
            BOOK_SETS[candidate.books],
            games.get(decision.id, ([], None)),
            live_selections[decision.event_id],
        )
        replayed = {
            BetMatch.of(i.outcome.value, i.line, i.bookmaker, i.price_decimal): i for i in intents
        }
        for bet in placed:
            assert bet.id is not None  # noqa: S101
            key = BetMatch.of(bet.selection, bet.line, bet.bookmaker, bet.price_decimal)
            intent = replayed.get(key)
            estimates.append(
                BetEstimate(
                    bet_id=bet.id,
                    decision_id=decision.id,
                    event_id=decision.event_id,
                    key=key,
                    matched=intent is not None,
                    estimated_ev=None if intent is None else intent.estimated_ev,
                    stored=bet.estimated_ev,
                )
            )
        live_selections[decision.event_id].update((Outcome(t.selection), t.line) for t in placed)
    return VersionBackfill(candidate.name, version, tuple(estimates))


async def plan_backfill(session: AsyncSession, registry: Sequence[Candidate]) -> EstimateBackfill:
    """The estimate of every decision bet of each version of a registered rule
    candidate, and the count of decision bets of the versions it cannot replay."""
    replayable = {
        c.name: c
        for c in registry
        if isinstance(c, RuleCandidate) and not isinstance(resolve_rule(c)[0], ModelRuleSpec)
    }
    name, version = col(Bet.candidate_name), col(Bet.candidate_version)
    rows = await session.execute(
        select(name, version, func.count())
        .where(col(Bet.decision_id).is_not(None))
        .group_by(name, version)
        .order_by(name, version)
    )
    versions: list[VersionBackfill] = []
    skipped: dict[tuple[str, int], int] = {}
    for key_name, key_version, count in rows.tuples():
        candidate = replayable.get(key_name)
        if candidate is None:
            skipped[(key_name, key_version)] = count
        else:
            versions.append(await _version_backfill(session, candidate, key_version))
    return EstimateBackfill(tuple(versions), skipped)


async def apply_backfill(session: AsyncSession, backfill: EstimateBackfill) -> int:
    """Write the estimate of each bet in ``VersionBackfill.to_write`` and commit; the number of
    bets written."""
    written = 0
    for version in backfill.versions:
        for bet in version.to_write:
            await session.execute(
                update(Bet).where(col(Bet.id) == bet.bet_id).values(estimated_ev=bet.estimated_ev)
            )
            written += 1
    await session.commit()
    return written
