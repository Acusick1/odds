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
from odds_core.close import close_grace, close_quotes
from odds_core.models import Event, EventStatus
from odds_core.odds_math import calculate_profit_from_decimal, selection_won
from odds_core.paper_trade_models import CandidateDecision
from odds_core.quotes import Quote
from odds_lambda.candidate_rules import BetIntent, ModelRuleSpec, Selection
from odds_lambda.candidate_rules.registry import resolve_rule
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
    RunningScore,
    ScoredBet,
    running_score,
    scoreable_prefix,
    selection_close_probability,
)

logger = structlog.get_logger()

# A variant is a named book set or the candidate's own set without one book.
_SET = "set"
_WITHOUT = "without"
_Variant = tuple[str, str]


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
    stake = sum(b.stake for b in with_close)
    settled = [b.pnl for b in bets if b.pnl is not None]
    return BookScore(
        bets=len(bets),
        settled_bets=len(settled),
        bets_with_close=len(with_close),
        mean_ev=sum(b.stake * b.ev for b in with_close if b.ev is not None) / stake
        if stake
        else None,
        pnl=sum(settled),
    )


def _variants(candidate: MechanicalCandidate) -> dict[_Variant, frozenset[str]]:
    own = BOOK_SETS[candidate.books]
    return {
        **{(_SET, s.value): BOOK_SETS[s] for s in BookSet},
        **{(_WITHOUT, book): own - {book} for book in sorted(own)},
    }


def _scored(
    candidate: MechanicalCandidate,
    decision_time: datetime,
    event: Event,
    intents: Sequence[BetIntent],
    closes: dict[int, list[Quote]],
) -> list[ScoredBet]:
    settleable = (
        event.status == EventStatus.FINAL
        and event.home_score is not None
        and event.away_score is not None
    )
    bets: list[ScoredBet] = []
    for intent in intents:
        p_close = selection_close_probability(
            candidate.market, intent.outcome.value, intent.line, event, closes.get(event.id, [])
        )
        pnl = None
        if settleable:
            assert event.home_score is not None and event.away_score is not None  # noqa: S101
            pnl = calculate_profit_from_decimal(
                intent.stake,
                intent.price_decimal,
                won=selection_won(
                    intent.outcome.value, intent.line, event.home_score, event.away_score
                ),
            )
        bets.append(
            ScoredBet(
                candidate_name=candidate.name,
                candidate_version=candidate.version,
                event_id=event.id,
                commence_time=event.commence_time,
                stake=intent.stake,
                pnl=pnl,
                price_decimal=intent.price_decimal,
                p_close=None if p_close is None else p_close.proportional,
                placed_at=decision_time,
                close_grace=close_grace(event.sport_key, candidate.market),
                sport_key=event.sport_key,
                selection=intent.outcome.value,
                line=intent.line,
                bookmaker=intent.bookmaker,
                p_close_power=None if p_close is None else p_close.power,
                close_source=None if p_close is None else p_close.source,
            )
        )
    return bets


async def replay_variants(
    session: AsyncSession, candidate: MechanicalCandidate
) -> tuple[int, dict[_Variant, list[ScoredBet]]] | None:
    """The number of decisions replayed and each variant's scored bets in the order the live
    runs took the decisions; None for a model rule, which has none. Each decision's recorded
    quotes are read once and every variant's state is built from them in memory."""
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
    variants = _variants(candidate)
    chains: dict[_Variant, dict[int, set[Selection]]] = {v: defaultdict(set) for v in variants}
    placed: dict[_Variant, list[tuple[CandidateDecision, list[BetIntent]]]] = {
        v: [] for v in variants
    }
    for decision in decisions:
        assert decision.id is not None  # noqa: S101
        event = events[decision.event_id]
        quoted = await recorded_quotes(session, spec, params, event, candidate.market, decision)
        for variant, books in variants.items():
            intents = recorded_intents(
                spec,
                params,
                event,
                candidate.market,
                decision,
                quoted,
                books,
                fixtures.get(decision.id, ([], None)),
                chains[variant][decision.event_id],
            )
            chains[variant][decision.event_id].update(i.selection for i in intents)
            placed[variant].append((decision, intents))
    betting = [events[i] for i in {d.event_id for v in placed.values() for d, b in v if b}]
    closes = await close_quotes(session, betting, candidate.market)
    return len(decisions), {
        variant: [
            bet
            for decision, intents in per_decision
            for bet in _scored(
                candidate, decision.known_at, events[decision.event_id], intents, closes
            )
        ]
        for variant, per_decision in placed.items()
    }


def build_replay(
    candidate: MechanicalCandidate,
    decisions: int,
    variants: dict[_Variant, list[ScoredBet]],
    trades: Sequence[ScoredBet],
    now: datetime,
) -> BookReplay:
    """The replay of ``candidate`` from its variants' bets and its paper ``trades``, both in
    placement order."""
    own = BOOK_SETS[candidate.books]
    prefixes = {v: scoreable_prefix(bets, now)[0] for v, bets in variants.items()}
    scores = {v: score_bets(bets, prefixes[v]) for v, bets in variants.items()}
    running: dict[_Variant, RunningScore] = {v: running_score(p) for v, p in prefixes.items()}
    sets = tuple(
        BookSetReplay(
            books=s,
            registered=s is candidate.books,
            score=scores[(_SET, s.value)],
            running_mean_ev=running[(_SET, s.value)].mean_ev,
        )
        for s in BookSet
    )
    live_with_close = scoreable_prefix(trades, now)[0]
    baseline = scores[(_SET, candidate.books.value)]
    rows: list[BookRow] = []
    for book in sorted(own):
        at_book = [t for t in trades if t.bookmaker == book]
        placed = score_bets(at_book, [t for t in live_with_close if t.bookmaker == book])
        without = scores[(_WITHOUT, book)]
        if placed.bets or (without.bets, without.mean_ev) != (baseline.bets, baseline.mean_ev):
            rows.append(BookRow(book, placed, without))
    return BookReplay(candidate.name, candidate.version, decisions, sets, tuple(rows))


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
            found[key] = build_replay(entry, *replayed, by_version.get(key, []), now)
    return found
