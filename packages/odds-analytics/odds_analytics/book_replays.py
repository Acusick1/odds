"""Counterfactual book replays of a candidate version's recorded decisions, for information.

Each non-missed decision is rebuilt from its recorded input snapshots (read once) under every
named book set (``odds_core.bookmakers.BookSet``) and under the candidate's own set with each of
its books left out in turn, each variant chaining "once per selection" across a match's decisions
as the backtest and the parity check do. The bets are scored against the close of the candidate
report and settled on the event's result. The candidate's own set reproduces the live bets (the
parity check), so its row is the live series; the other rows say what another set would have
done on the same decisions. Nothing in a checkpoint, verdict or family count reads any of it.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import structlog
from odds_core.bookmakers import BOOK_SETS, BookSet
from odds_core.candidates import Candidate, MechanicalCandidate
from odds_core.close import close_quotes
from odds_core.models import Event
from odds_core.paper_trade_models import CandidateDecision
from odds_lambda.candidate_rules import BetIntent, ModelRuleSpec, Selection
from odds_lambda.candidate_rules.registry import resolve_rule
from odds_lambda.paper_trading import VOIDING_STATUSES
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

from odds_analytics.candidate_parity import (
    load_decisions,
    load_recorded_fixture,
    recorded_intents,
    recorded_quotes,
)
from odds_analytics.candidate_report import (
    ScoredBet,
    running_score,
    scoreable_prefix,
    scored_intent,
    selection_close_probability,
    stake_weighted_mean_ev,
)

logger = structlog.get_logger()


@dataclass(frozen=True)
class BookScore:
    """A series of bets: all of them, those settled, the bets with a close in the scoreable
    prefix, their stake-weighted mean EV (None without one) and the P&L of the settled bets."""

    bets: int
    settled_bets: int
    bets_with_close: int
    mean_ev: float | None
    pnl: float


@dataclass(frozen=True)
class BookSetReplay:
    """The version's decisions replayed under one named book set. ``registered`` marks the set
    the version declares. ``running_mean_ev`` is the running mean EV after each bet with a close
    (``RunningScore.mean_ev``)."""

    books: BookSet
    registered: bool
    score: BookScore
    running_mean_ev: tuple[float, ...]


@dataclass(frozen=True)
class BookRow:
    """One book of the version's set: the paper trades placed at it, and the version's decisions
    replayed without it."""

    book: str
    placed: BookScore
    without: BookScore


@dataclass(frozen=True)
class BookReplay:
    """The counterfactual replays of one candidate version over its ``decisions``. ``books``
    lists the books of the declared set that a trade was placed at or whose removal changes the
    replay's bets or EV, in key order."""

    name: str
    version: int
    decisions: int
    sets: tuple[BookSetReplay, ...]
    books: tuple[BookRow, ...]


def score_bets(bets: Sequence[ScoredBet], with_close: Sequence[ScoredBet]) -> BookScore:
    """``bets`` summarised, ``with_close`` being those of them with a close in the scoreable
    prefix."""
    settled = [b.pnl for b in bets if b.pnl is not None]
    return BookScore(
        bets=len(bets),
        settled_bets=len(settled),
        bets_with_close=len(with_close),
        mean_ev=stake_weighted_mean_ev(with_close),
        pnl=sum(settled),
    )


@dataclass(frozen=True)
class ReplayedBets:
    """The scored bets of each variant of ``decisions`` replayed decisions, in the order the
    live runs took them: under each named book set, and under the version's own set without
    each of its books."""

    decisions: int
    by_set: dict[BookSet, list[ScoredBet]]
    without: dict[str, list[ScoredBet]]


async def replay_variants(
    session: AsyncSession, candidate: MechanicalCandidate
) -> ReplayedBets | None:
    """The version's decisions replayed under every variant; None for a model rule, which has
    none. Each decision's recorded quotes are read once and every variant's state is built from
    them in memory."""
    spec, params = resolve_rule(candidate)
    if isinstance(spec, ModelRuleSpec):
        return None
    decisions = await load_decisions(session, candidate.name, candidate.version)
    fixtures = await load_recorded_fixture(session, decisions) if spec.fixture_inputs else {}
    events = {
        e.id: e
        for e in (
            await session.scalars(
                select(Event).where(col(Event.id).in_({d.event_id for d in decisions}))
            )
        ).all()
    }
    own = BOOK_SETS[candidate.books]
    named = list(BookSet)
    left_out = sorted(own)
    variants = [BOOK_SETS[s] for s in named] + [own - {book} for book in left_out]
    chains: list[dict[int, set[Selection]]] = [defaultdict(set) for _ in variants]
    placed: list[list[tuple[CandidateDecision, list[BetIntent]]]] = [[] for _ in variants]
    for decision in decisions:
        assert decision.id is not None  # noqa: S101
        event = events[decision.event_id]
        quoted = await recorded_quotes(session, spec, params, event, candidate.market, decision)
        for i, books in enumerate(variants):
            intents = recorded_intents(
                spec,
                params,
                event,
                candidate.market,
                decision,
                quoted,
                books,
                fixtures.get(decision.id, ([], None)),
                chains[i][decision.event_id],
            )
            chains[i][decision.event_id].update(intent.selection for intent in intents)
            placed[i].append((decision, intents))
    betting = [events[i] for i in {d.event_id for v in placed for d, b in v if b}]
    closes = await close_quotes(session, betting, candidate.market)
    scored = [
        [
            scored_intent(
                candidate.name,
                candidate.version,
                candidate.market,
                events[decision.event_id],
                intent,
                decision.known_at,
                selection_close_probability(
                    candidate.market,
                    intent.outcome.value,
                    intent.line,
                    events[decision.event_id],
                    closes.get(decision.event_id, []),
                ),
            )
            for decision, intents in per_decision
            if events[decision.event_id].status not in VOIDING_STATUSES
            for intent in intents
        ]
        for per_decision in placed
    ]
    return ReplayedBets(
        len(decisions),
        dict(zip(named, scored[: len(named)], strict=True)),
        dict(zip(left_out, scored[len(named) :], strict=True)),
    )


def build_replay(
    candidate: MechanicalCandidate,
    replayed: ReplayedBets,
    trades: Sequence[ScoredBet],
    now: datetime,
) -> BookReplay:
    """The replay of ``candidate`` from its variants' bets and its paper ``trades``, both in
    placement order."""
    prefixes = {s: scoreable_prefix(bets, now)[0] for s, bets in replayed.by_set.items()}
    sets = tuple(
        BookSetReplay(
            books=s,
            registered=s is candidate.books,
            score=score_bets(replayed.by_set[s], prefixes[s]),
            running_mean_ev=running_score(prefixes[s]).mean_ev,
        )
        for s in BookSet
    )
    baseline = next(s.score for s in sets if s.registered)
    live_with_close = scoreable_prefix(trades, now)[0]
    rows: list[BookRow] = []
    for book, bets in replayed.without.items():
        placed = score_bets(
            [t for t in trades if t.bookmaker == book],
            [t for t in live_with_close if t.bookmaker == book],
        )
        without = score_bets(bets, scoreable_prefix(bets, now)[0])
        if placed.bets or (without.bets, without.mean_ev) != (baseline.bets, baseline.mean_ev):
            rows.append(BookRow(book, placed, without))
    return BookReplay(candidate.name, candidate.version, replayed.decisions, sets, tuple(rows))


async def book_replays(
    session: AsyncSession,
    registry: Sequence[Candidate | None],
    trades: Sequence[ScoredBet],
    now: datetime,
) -> dict[tuple[str, int], BookReplay]:
    """The replay of each mechanical entry of ``registry`` (the registered versions the report
    holds) by name and version, from ``trades`` in placement order. A version whose replay
    fails is logged and left out: nothing else reads these."""
    by_version: dict[tuple[str, int], list[ScoredBet]] = defaultdict(list)
    for trade in trades:
        by_version[(trade.candidate_name, trade.candidate_version)].append(trade)
    found: dict[tuple[str, int], BookReplay] = {}
    for entry in registry:
        if not isinstance(entry, MechanicalCandidate):
            continue
        key = (entry.name, entry.version)
        try:
            replayed = await replay_variants(session, entry)
        except Exception:
            logger.error(
                "book_replay_failed", candidate=entry.name, version=entry.version, exc_info=True
            )
            continue
        if replayed is not None:
            found[key] = build_replay(entry, replayed, by_version.get(key, []), now)
    return found
