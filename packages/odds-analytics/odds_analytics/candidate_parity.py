"""Check that replaying a mechanical candidate reproduces the bets it placed live.

Two comparisons per recorded decision. The parity comparison rebuilds the market state from
exactly the snapshots the decision recorded (``input_snapshot_ids``), with its decision time
and kick-off, runs the rule and compares the bets with the decision's trades; any difference
is a mismatch. Decisions are replayed in the order the live runs took them, and each outcome
of a match is bet once: the parity comparison drops the outcomes the match's earlier
decisions placed trades on, as the live run did. A rule that reads fixture inputs gets them
from the events the decision recorded (``input_event_ids``), as known at its ``known_at``. The
data comparison replays from the database as it is now, with
``known_at`` the decision's, dropping the outcomes its own replays of the earlier decisions
bet, as the backtest does, and reports where that differs from the recorded-input replay (a
later twin merge, commit lag): informational, never a failure.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum
from typing import Any

from odds_core.bookmakers import BOOK_SETS
from odds_core.candidates import MechanicalCandidate
from odds_core.market_prices import Outcome
from odds_core.models import Event, OddsSnapshot
from odds_core.paper_trade_models import CandidateDecision, PaperTrade
from odds_core.quotes import Quote
from odds_core.vocabulary import Market
from odds_lambda.candidate_rules import (
    BetIntent,
    FixtureContext,
    ModelRuleSpec,
    RuleParams,
    RuleSpec,
    Selection,
    apply_rule,
    build_market_state,
    candidate_leagues,
    input_window,
    lookback_points,
    max_input_age,
    run_rule,
)
from odds_lambda.candidate_rules.inputs import rule_quotes
from odds_lambda.candidate_rules.registry import resolve_rule
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import defer
from sqlmodel import col

BetKey = tuple[str, float | None, str, float, float]


class ParityStatus(StrEnum):
    MATCH = "match"
    MISMATCH = "mismatch"


@dataclass(frozen=True)
class DecisionParity:
    decision_id: int
    event_id: int
    decision_time: datetime
    status: ParityStatus
    live: tuple[BetKey, ...]
    # The bets from the recorded input snapshots.
    replayed: tuple[BetKey, ...]
    # The bets from the database as it is now, as of the decision's ``known_at``.
    db_replayed: tuple[BetKey, ...]
    # Recorded input ids absent from the database; the recorded-input replay cannot be exact.
    missing_snapshot_ids: tuple[int, ...] = ()
    missing_event_ids: tuple[int, ...] = ()

    @property
    def data_difference(self) -> bool:
        return self.db_replayed != self.replayed


@dataclass(frozen=True)
class ParityReport:
    name: str
    version: int
    decisions: list[DecisionParity]

    def of(self, status: ParityStatus) -> list[DecisionParity]:
        return [d for d in self.decisions if d.status == status]

    @property
    def data_differences(self) -> list[DecisionParity]:
        return [d for d in self.decisions if d.data_difference]

    @property
    def checked(self) -> int:
        return len(self.decisions)

    @property
    def ok(self) -> bool:
        return not self.of(ParityStatus.MISMATCH)


def bet_key(outcome: str, line: float | None, bookmaker: str, price: float, stake: float) -> BetKey:
    return (outcome, line, bookmaker, round(price, 9), round(stake, 9))


def intent_keys(intents: list[BetIntent]) -> tuple[BetKey, ...]:
    return tuple(
        sorted(
            bet_key(i.outcome.value, i.line, i.bookmaker, i.price_decimal, i.stake) for i in intents
        )
    )


async def _absent_events(session: AsyncSession, ids: list[int]) -> tuple[int, ...]:
    """The recorded event ids absent from the database."""
    if not ids:
        return ()
    rows = await session.execute(select(col(Event.id)).where(col(Event.id).in_(ids)))
    return tuple(sorted(set(ids) - set(rows.scalars())))


async def recorded_quotes(
    session: AsyncSession,
    spec: RuleSpec,
    params: RuleParams,
    event: Event,
    market: Market,
    decision: CandidateDecision,
) -> list[Quote]:
    """The quotes of the decision's recorded input snapshots within the rule's input window
    before its decision time: what ``load_market_state`` reads for a decision."""
    return await rule_quotes(
        session,
        event.id,
        market,
        spec.sharp_inputs,
        since=decision.decision_time - input_window(max_input_age(params), lookback_points(params)),
        until=decision.decision_time,
        snapshot_ids=list(decision.input_snapshot_ids),
        also_markets=spec.also_markets,
    )


def recorded_intents(
    spec: RuleSpec,
    params: RuleParams,
    event: Event,
    market: Market,
    decision: CandidateDecision,
    quoted: Sequence[Quote],
    books: frozenset[str],
    fixture: tuple[list[int], dict[str, Any] | None],
    bet_selections: Collection[Selection],
) -> list[BetIntent]:
    """The bets on the state built from the decision's recorded ``quoted`` under ``books``, with
    its recorded fixture values and kick-off, dropping the selections ``bet_selections``. A
    rule's fixture context is the recorded one, not read again: events' scores and kick-offs
    change, and a merge changes their ids."""
    event_ids, values = fixture
    state = build_market_state(
        event,
        market,
        quoted,
        decision.decision_time,
        max_input_age(params),
        spec.sharp_inputs,
        books,
        spec.also_markets,
        lookback_points(params),
    )
    if spec.fixture_inputs and state.has_input:
        state = replace(state, fixture=FixtureContext.from_record(values, event_ids))
    return apply_rule(spec, params, state, decision.kickoff_at_decision, bet_selections)[1]


async def _recorded_inputs(
    session: AsyncSession,
    candidate: MechanicalCandidate,
    spec: RuleSpec,
    params: RuleParams,
    event: Event,
    market: Market,
    decision: CandidateDecision,
    fixture: tuple[list[int], dict[str, Any] | None],
    bet_selections: set[Selection],
) -> tuple[list[BetIntent], tuple[int, ...], tuple[int, ...]]:
    """The bets from the decision's recorded snapshots and fixture values, and the recorded
    snapshot ids and event ids not found."""
    ids = list(decision.input_snapshot_ids)
    found: set[int] = set()
    if ids:
        rows = await session.execute(
            select(col(OddsSnapshot.id)).where(col(OddsSnapshot.id).in_(ids))
        )
        found = {snapshot_id for snapshot_id in rows.scalars() if snapshot_id is not None}
    missing = tuple(sorted(set(ids) - found))
    quoted = await recorded_quotes(session, spec, params, event, market, decision)
    intents = recorded_intents(
        spec,
        params,
        event,
        market,
        decision,
        quoted,
        BOOK_SETS[candidate.books],
        fixture,
        bet_selections,
    )
    return intents, missing, await _absent_events(session, fixture[0])


async def _db_replayed(
    session: AsyncSession,
    candidate: MechanicalCandidate,
    spec: RuleSpec,
    params: RuleParams,
    event: Event,
    market: Market,
    decision: CandidateDecision,
    bet_selections: set[Selection],
) -> list[BetIntent]:
    _, intents = await run_rule(
        session,
        spec,
        params,
        event,
        market,
        decision.decision_time,
        decision.known_at,
        BOOK_SETS[candidate.books],
        kickoff=decision.kickoff_at_decision,
        bet_selections=bet_selections,
        fixture_leagues=candidate_leagues(candidate),
    )
    return intents


async def load_decisions(session: AsyncSession, name: str, version: int) -> list[CandidateDecision]:
    """The non-missed decisions of a candidate version in the order the live runs took them."""
    rows = await session.execute(
        # The recorded fixture is read only for a rule with fixture inputs
        # (``load_recorded_fixture``), so a database not yet migrated to hold it can still be
        # checked for other rules. The league set is never read: parity replays the candidate
        # file's rule.
        select(CandidateDecision)
        .options(
            defer(CandidateDecision.input_event_ids),  # type: ignore[arg-type]
            defer(CandidateDecision.input_fixture),  # type: ignore[arg-type]
            defer(CandidateDecision.sports),  # type: ignore[arg-type]
        )
        .where(
            CandidateDecision.candidate_name == name,
            CandidateDecision.candidate_version == version,
            CandidateDecision.missed.is_(False),  # type: ignore[attr-defined]
        )
        # The order the live runs committed them: a kick-off moved earlier can put a later
        # decision's time before an earlier one's.
        .order_by(CandidateDecision.known_at, CandidateDecision.id)  # type: ignore[arg-type]
    )
    return list(rows.scalars().all())


async def load_recorded_fixture(
    session: AsyncSession, decisions: Sequence[CandidateDecision]
) -> dict[int, tuple[list[int], dict[str, Any] | None]]:
    """Each decision's recorded fixture event ids and values, by decision id."""
    if not decisions:
        return {}
    rows = await session.execute(
        select(
            CandidateDecision.id,
            CandidateDecision.input_event_ids,
            CandidateDecision.input_fixture,
        ).where(
            CandidateDecision.id.in_([d.id for d in decisions])  # type: ignore[union-attr]
        )
    )
    return {i: (list(ids), values) for i, ids, values in rows.tuples()}


async def check_parity(
    session: AsyncSession, candidate: MechanicalCandidate, version: int | None = None
) -> ParityReport:
    """Replay every non-missed decision of ``candidate``'s ``version`` (default its own), in
    the order the live runs took them.

    The rule and params are the candidate's, so ``version`` must be the one the file holds;
    ``ValueError`` otherwise, and for a rule that needs a fitted model, which has no live
    decisions to replay.
    """
    version = candidate.version if version is None else version
    if version != candidate.version:
        raise ValueError(
            f"candidate {candidate.name!r} is at version {candidate.version}; the rule and "
            f"params of version {version} are only in git history"
        )
    spec, params = resolve_rule(candidate)
    if isinstance(spec, ModelRuleSpec):
        raise ValueError(
            f"candidate {candidate.name!r}: rule {candidate.rule!r} needs a fitted model; the "
            "live runner does not run it, so it has no decisions to check"
        )

    decisions = await load_decisions(session, candidate.name, version)
    recorded_fixture = (
        await load_recorded_fixture(session, decisions) if spec.fixture_inputs else {}
    )
    trades: dict[int, list[PaperTrade]] = defaultdict(list)
    if decisions:
        rows = await session.execute(
            select(PaperTrade).where(
                PaperTrade.decision_id.in_([d.id for d in decisions])  # type: ignore[union-attr]
            )
        )
        for trade in rows.scalars():
            assert trade.decision_id is not None  # noqa: S101
            trades[trade.decision_id].append(trade)

    # Selections bet so far per event, live and by the database replay; decisions come in
    # live order.
    live_selections: dict[int, set[Selection]] = defaultdict(set)
    db_replayed_selections: dict[int, set[Selection]] = defaultdict(set)
    results: list[DecisionParity] = []
    for decision in decisions:
        assert decision.id is not None  # noqa: S101
        event = await session.get(Event, decision.event_id)
        assert event is not None  # noqa: S101
        live = tuple(
            sorted(
                bet_key(t.selection, t.line, t.bookmaker, t.price_decimal, t.stake)
                for t in trades[decision.id]
            )
        )
        replayed_bets, missing, missing_events = await _recorded_inputs(
            session,
            candidate,
            spec,
            params,
            event,
            candidate.market,
            decision,
            recorded_fixture.get(decision.id, ([], None)),
            live_selections[decision.event_id],
        )
        db_replayed_bets = await _db_replayed(
            session,
            candidate,
            spec,
            params,
            event,
            candidate.market,
            decision,
            db_replayed_selections[decision.event_id],
        )
        live_selections[decision.event_id].update(
            (Outcome(t.selection), t.line) for t in trades[decision.id]
        )
        db_replayed_selections[decision.event_id].update(b.selection for b in db_replayed_bets)
        replayed, db_replayed = intent_keys(replayed_bets), intent_keys(db_replayed_bets)
        results.append(
            DecisionParity(
                decision_id=decision.id,
                event_id=decision.event_id,
                decision_time=decision.decision_time,
                status=(
                    ParityStatus.MATCH
                    if replayed == live and not missing and not missing_events
                    else ParityStatus.MISMATCH
                ),
                live=live,
                replayed=replayed,
                db_replayed=db_replayed,
                missing_snapshot_ids=missing,
                missing_event_ids=missing_events,
            )
        )
    return ParityReport(candidate.name, version, results)
