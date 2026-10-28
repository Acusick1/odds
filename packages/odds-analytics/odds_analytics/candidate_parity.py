"""Check that replaying a mechanical candidate reproduces the bets it placed live.

Two comparisons per recorded decision. The parity comparison rebuilds the market state from
exactly the snapshots the decision recorded (``input_snapshot_ids``), with its decision time
and kick-off, runs the rule and compares the bets with the decision's trades; any difference
is a mismatch. Decisions are replayed in the order the live runs took them, and each outcome
of a match is bet once: the parity comparison drops the outcomes the match's earlier
decisions placed trades on, as the live run did. The data comparison replays from the database as it is now, with
``known_at`` the decision's, dropping the outcomes its own replays of the earlier decisions
bet, as the backtest does, and reports where that differs from the recorded-input replay (a
later twin merge, commit lag): informational, never a failure.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from odds_core.candidates import MechanicalCandidate
from odds_core.market_prices import Outcome
from odds_core.models import Event, OddsSnapshot
from odds_core.paper_trade_models import CandidateDecision, PaperTrade
from odds_core.vocabulary import Market
from odds_lambda.candidate_rules import (
    BetIntent,
    ModelRuleSpec,
    RuleParams,
    RuleSpec,
    Selection,
    apply_rule,
    load_market_state,
    max_input_age,
    resolve_rule,
    run_rule,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

BetKey = tuple[str, str, float, float]


class ParityStatus(StrEnum):
    MATCH = "match"
    MISMATCH = "mismatch"


@dataclass(frozen=True)
class DecisionParity:
    decision_id: int
    event_id: str
    decision_time: datetime
    status: ParityStatus
    live: tuple[BetKey, ...]
    # The bets from the recorded input snapshots.
    replayed: tuple[BetKey, ...]
    # The bets from the database as it is now, as of the decision's ``known_at``.
    db_replayed: tuple[BetKey, ...]
    # Recorded input ids absent from the database; the recorded-input replay cannot be exact.
    missing_snapshot_ids: tuple[int, ...] = ()

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


def bet_key(outcome: str, bookmaker: str, price: float, stake: float) -> BetKey:
    return (outcome, bookmaker, round(price, 9), round(stake, 9))


def intent_keys(intents: list[BetIntent]) -> tuple[BetKey, ...]:
    return tuple(
        sorted(bet_key(i.outcome.value, i.bookmaker, i.price_decimal, i.stake) for i in intents)
    )


async def _recorded_inputs(
    session: AsyncSession,
    spec: RuleSpec,
    params: RuleParams,
    event: Event,
    market: Market,
    decision: CandidateDecision,
    bet_selections: set[Selection],
) -> tuple[list[BetIntent], tuple[int, ...]]:
    """The bets from the decision's recorded snapshots, and the recorded ids not found."""
    ids = list(decision.input_snapshot_ids)
    found: set[int] = set()
    if ids:
        rows = await session.execute(
            select(col(OddsSnapshot.id)).where(col(OddsSnapshot.id).in_(ids))
        )
        found = {snapshot_id for snapshot_id in rows.scalars() if snapshot_id is not None}
    missing = tuple(sorted(set(ids) - found))
    state = await load_market_state(
        session,
        event,
        market,
        decision.decision_time,
        max_input_age(params),
        spec.sharp_inputs,
        snapshot_ids=ids,
    )
    _, intents = apply_rule(spec, params, state, decision.kickoff_at_decision, bet_selections)
    return intents, missing


async def _db_replayed(
    session: AsyncSession,
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
        kickoff=decision.kickoff_at_decision,
        bet_selections=bet_selections,
    )
    return intents


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

    decisions = (
        (
            await session.execute(
                select(CandidateDecision)
                .where(
                    CandidateDecision.candidate_name == candidate.name,
                    CandidateDecision.candidate_version == version,
                    CandidateDecision.missed.is_(False),  # type: ignore[attr-defined]
                )
                # The order the live runs committed them: a kick-off moved earlier can put a
                # later decision's time before an earlier one's.
                .order_by(CandidateDecision.known_at, CandidateDecision.id)  # type: ignore[arg-type]
            )
        )
        .scalars()
        .all()
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
    # live order. A paper trade records no line.
    live_selections: dict[str, set[Selection]] = defaultdict(set)
    db_replayed_selections: dict[str, set[Selection]] = defaultdict(set)
    results: list[DecisionParity] = []
    for decision in decisions:
        assert decision.id is not None  # noqa: S101
        event = await session.get(Event, decision.event_id)
        assert event is not None  # noqa: S101
        live = tuple(
            sorted(
                bet_key(t.selection, t.bookmaker, t.price_decimal, t.stake)
                for t in trades[decision.id]
            )
        )
        replayed_bets, missing = await _recorded_inputs(
            session,
            spec,
            params,
            event,
            candidate.market,
            decision,
            live_selections[decision.event_id],
        )
        db_replayed_bets = await _db_replayed(
            session,
            spec,
            params,
            event,
            candidate.market,
            decision,
            db_replayed_selections[decision.event_id],
        )
        live_selections[decision.event_id].update(
            (Outcome(t.selection), None) for t in trades[decision.id]
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
                    if replayed == live and not missing
                    else ParityStatus.MISMATCH
                ),
                live=live,
                replayed=replayed,
                db_replayed=db_replayed,
                missing_snapshot_ids=missing,
            )
        )
    return ParityReport(candidate.name, version, results)
