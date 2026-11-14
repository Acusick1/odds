"""Backfill the estimated EV of candidate trades placed before trades recorded it.

Each decision of a version of a registered mechanical candidate is replayed from its recorded
inputs under the registry's rule, params and book set, as the parity check replays it (``odds_analytics.candidate_parity.recorded_intents``), in the order
the live runs took it, dropping the selections the match's earlier decisions placed trades on.
A trade takes the estimate of the replayed bet on its selection, line, bookmaker and price; a
trade with no such bet keeps none. The registry holds only a name's current version, and a
name's rule never changes across versions, so an earlier version is replayed with the current
entry: a trade whose bet that does not reproduce stays unmatched.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from odds_core.bookmakers import BOOK_SETS
from odds_core.candidates import Candidate, MechanicalCandidate
from odds_core.market_prices import Outcome
from odds_core.models import Event
from odds_core.paper_trade_models import PaperTrade
from odds_lambda.candidate_rules import BetIntent, ModelRuleSpec, Selection
from odds_lambda.candidate_rules.registry import resolve_rule
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from odds_analytics.candidate_parity import (
    load_decisions,
    load_recorded_fixture,
    recorded_intents,
    recorded_quotes,
)

MatchKey = tuple[str, float | None, str, float]


def _key(selection: str, line: float | None, bookmaker: str, price: float) -> MatchKey:
    return (selection, line, bookmaker, round(price, 9))


@dataclass(frozen=True)
class TradeEstimate:
    """A decision's trade and the estimate of its replayed bet; ``matched`` is False when the
    replay placed no bet on its selection, line, bookmaker and price."""

    trade_id: int
    decision_id: int
    event_id: int
    key: MatchKey
    matched: bool
    estimated_ev: float | None
    stored: float | None


@dataclass(frozen=True)
class VersionBackfill:
    name: str
    version: int
    trades: tuple[TradeEstimate, ...]

    @property
    def to_write(self) -> list[TradeEstimate]:
        """Matched trades with an estimate that differs from the one stored."""
        return [
            t
            for t in self.trades
            if t.matched and t.estimated_ev is not None and t.stored != t.estimated_ev
        ]

    @property
    def unmatched(self) -> list[TradeEstimate]:
        return [t for t in self.trades if not t.matched]


@dataclass(frozen=True)
class EstimateBackfill:
    """The backfill of each version of a registered mechanical candidate with decision trades,
    and the trades of versions it cannot replay, by (name, version): a name the registry does
    not hold, or a model rule, which places no live trades."""

    versions: tuple[VersionBackfill, ...]
    skipped: dict[tuple[str, int], int]


async def _version_backfill(
    session: AsyncSession, candidate: MechanicalCandidate, version: int
) -> VersionBackfill:
    spec, params = resolve_rule(candidate)
    assert not isinstance(spec, ModelRuleSpec)  # noqa: S101
    decisions = await load_decisions(session, candidate.name, version)
    fixtures = await load_recorded_fixture(session, decisions) if spec.fixture_inputs else {}
    trades: dict[int, list[PaperTrade]] = defaultdict(list)
    if decisions:
        rows = await session.scalars(
            select(PaperTrade)
            .where(col(PaperTrade.decision_id).in_([d.id for d in decisions]))
            .order_by(col(PaperTrade.id))
        )
        for trade in rows:
            assert trade.decision_id is not None  # noqa: S101
            trades[trade.decision_id].append(trade)
    live_selections: dict[int, set[Selection]] = defaultdict(set)
    estimates: list[TradeEstimate] = []
    for decision in decisions:
        assert decision.id is not None  # noqa: S101
        placed = trades[decision.id]
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
            fixtures.get(decision.id, ([], None)),
            live_selections[decision.event_id],
        )
        replayed = {_key(i.outcome.value, i.line, i.bookmaker, i.price_decimal): i for i in intents}
        for trade in placed:
            assert trade.id is not None  # noqa: S101
            key = _key(trade.selection, trade.line, trade.bookmaker, trade.price_decimal)
            intent = replayed.get(key)
            estimates.append(
                TradeEstimate(
                    trade_id=trade.id,
                    decision_id=decision.id,
                    event_id=decision.event_id,
                    key=key,
                    matched=intent is not None,
                    estimated_ev=None if intent is None else intent.estimated_ev,
                    stored=trade.estimated_ev,
                )
            )
        live_selections[decision.event_id].update((Outcome(t.selection), t.line) for t in placed)
    return VersionBackfill(candidate.name, version, tuple(estimates))


async def plan_backfill(session: AsyncSession, registry: Sequence[Candidate]) -> EstimateBackfill:
    """The estimate of every decision trade of each version of a registered mechanical
    candidate, and the count of decision trades of the versions it cannot replay."""
    replayable = {
        c.name: c
        for c in registry
        if isinstance(c, MechanicalCandidate) and not isinstance(resolve_rule(c)[0], ModelRuleSpec)
    }
    name, version = col(PaperTrade.candidate_name), col(PaperTrade.candidate_version)
    rows = await session.execute(
        select(name, version, func.count())
        .where(col(PaperTrade.decision_id).is_not(None))
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
    """Write each matched trade's estimate and commit; the number of trades written."""
    written = 0
    for version in backfill.versions:
        for trade in version.to_write:
            await session.execute(
                update(PaperTrade)
                .where(col(PaperTrade.id) == trade.trade_id)
                .values(estimated_ev=trade.estimated_ev)
            )
            written += 1
    await session.commit()
    return written
