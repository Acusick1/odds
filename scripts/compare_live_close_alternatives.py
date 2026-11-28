"""Compare the other live Betfair Exchange prices before kick-off with the Betfair job's live close.

Read-only. For each FINAL EPL match kicked off from ``--from`` to ``--to`` with a live Betfair
close (``odds_core.close.event_close`` over the Betfair job's snapshots), takes each alternative
source's Betfair Exchange back prices in its latest live (``observed``) snapshot at or before
kick-off, optionally within ``--max-age-minutes``: OddsPortal's on ``1x2`` and totals 2.5, the
Odds API's on ``1x2``. A price counts only when it is complete, not in-play, and its back
overround is at most ``MAX_EXCHANGE_OVERROUND``. Both are devigged proportionally, and
``dp = p_alternative - p_live`` is reported per outcome and per band of the live close
probability against the thresholds pre-set on issue 730. Spends no Odds API units.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

from odds_analytics.close_agreement import (
    THRESHOLDS,
    Delta,
    agreement,
    freshness,
    render_agreement,
    render_freshness,
)
from odds_analytics.holdout import require_clear
from odds_core.as_of import latest_snapshot
from odds_core.close import event_close, live_close_quotes
from odds_core.database import async_session_maker
from odds_core.exchange_price import (
    EXCHANGE_BOOKMAKER,
    MAX_EXCHANGE_OVERROUND,
    ExchangePrice,
    exchange_price,
)
from odds_core.market_prices import market_outcomes
from odds_core.models import Event, EventStatus
from odds_core.quotes import Quote, quotes
from odds_core.vocabulary import Market, Side, Source, TimeBasis
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

SPORT = "soccer_epl"
# The totals line OddsPortal's live snapshot and the Betfair job both carry.
TOTALS_LINE = 2.5


@dataclass(frozen=True)
class Pairing:
    """An alternative source on one market and line, compared with the live close there."""

    source: Source
    market: Market
    line: float | None

    @property
    def name(self) -> str:
        line = "" if self.line is None else f" {self.line:g}"
        return f"{self.source.value} {self.market.value}{line}"


# The Odds API has no Betfair Exchange totals price (#707).
PAIRINGS: tuple[Pairing, ...] = (
    Pairing(Source.ODDSPORTAL, Market.ONE_X_TWO, None),
    Pairing(Source.ODDSPORTAL, Market.TOTALS, TOTALS_LINE),
    Pairing(Source.ODDS_API, Market.ONE_X_TWO, None),
)


class Status(StrEnum):
    """How far a match got: it is compared, or the first thing it lacks."""

    NO_LIVE_CLOSE = "no_live_close"
    NO_SNAPSHOT = "no_alternative_snapshot"
    SNAPSHOT_UNUSABLE = "alternative_unusable"
    COMPARED = "compared"


@dataclass(frozen=True)
class EventComparison:
    """One match's live close and alternative price, or what it lacks. ``age_minutes`` is how
    long before kick-off the alternative price was observed."""

    event_id: int
    status: Status
    live: ExchangePrice | None = None
    alternative: ExchangePrice | None = None
    age_minutes: float | None = None


def alternative_price(
    quoted: Iterable[Quote],
    kickoff: datetime,
    pairing: Pairing,
    max_age: timedelta | None = None,
    max_overround: float = MAX_EXCHANGE_OVERROUND,
) -> ExchangePrice | None:
    """The pairing's Betfair Exchange back price in the latest observed snapshot of its source
    at or before ``kickoff``, and at most ``max_age`` before it, that is complete, not in-play
    and within ``max_overround``; None when there is none. Devigged."""
    candidates = [
        q
        for q in quoted
        if q.source is pairing.source
        and q.time_basis is TimeBasis.OBSERVED
        and (max_age is None or q.snapshot_time >= kickoff - max_age)
    ]

    def usable(snapshot: Sequence[Quote]) -> bool:
        price = exchange_price(snapshot, pairing.market, pairing.line)
        return price is not None and not price.in_play and price.overround <= max_overround

    snapshot = latest_snapshot(candidates, kickoff, usable)
    return None if snapshot is None else exchange_price(snapshot, pairing.market, pairing.line)


def compare_event(
    event_id: int,
    kickoff: datetime,
    pairing: Pairing,
    live_quotes: Sequence[Quote],
    alternative_quotes: Sequence[Quote],
    max_age: timedelta | None = None,
) -> EventComparison:
    """The live close of an event (``odds_core.close.event_close`` over its Betfair job quotes)
    and the alternative price ``alternative_price`` takes within ``max_age``."""
    live = event_close(live_quotes, kickoff, SPORT, pairing.market, line=pairing.line)
    if live is None:
        return EventComparison(event_id, Status.NO_LIVE_CLOSE)
    if not any(
        q.snapshot_time <= kickoff and (max_age is None or q.snapshot_time >= kickoff - max_age)
        for q in alternative_quotes
    ):
        return EventComparison(event_id, Status.NO_SNAPSHOT, live=live)
    price = alternative_price(alternative_quotes, kickoff, pairing, max_age)
    if price is None:
        return EventComparison(event_id, Status.SNAPSHOT_UNUSABLE, live=live)
    age = (kickoff - price.time).total_seconds() / 60
    return EventComparison(event_id, Status.COMPARED, live, price, age)


def deltas(comparisons: Iterable[EventComparison], market: Market) -> list[Delta]:
    """One delta per outcome of each compared match, against the live close."""
    return [
        Delta(c.event_id, outcome, c.live.fair[outcome], c.alternative.fair[outcome])
        for c in comparisons
        if c.live is not None and c.alternative is not None
        for outcome in market_outcomes(market)
    ]


def build_section(comparisons: Sequence[EventComparison], pairing: Pairing) -> dict[str, Any]:
    """One pairing's tables: the statistics and verdicts, coverage and the price ages."""
    with_live = [c for c in comparisons if c.live is not None]
    compared = [c for c in comparisons if c.status is Status.COMPARED]
    ages = [c.age_minutes for c in compared if c.age_minutes is not None]
    return {
        "source": pairing.source.value,
        "market": pairing.market.value,
        "line": pairing.line,
        **agreement(deltas(comparisons, pairing.market), market_outcomes(pairing.market)),
        "matches": {
            "final_events": len(comparisons),
            "with_live_close": len(with_live),
            "compared": len(compared),
            "coverage": len(compared) / len(with_live) if with_live else None,
            "missing": {
                s.value: sum(c.status is s for c in comparisons)
                for s in Status
                if s is not Status.COMPARED
            },
        },
        **freshness(ages, len(with_live)),
    }


def build_report(
    comparisons: dict[Pairing, Sequence[EventComparison]], parameters: dict[str, Any]
) -> dict[str, Any]:
    """Every pairing's section, as the JSON written and the text printed."""
    return {
        "parameters": parameters,
        "thresholds": {**THRESHOLDS, "max_exchange_overround": MAX_EXCHANGE_OVERROUND},
        "pairings": {p.name: build_section(found, p) for p, found in comparisons.items()},
    }


def render_section(name: str, section: dict[str, Any]) -> list[str]:
    m = section["matches"]
    coverage = "-" if m["coverage"] is None else f"{m['coverage']:.1%}"
    return [
        f"== {name}: {section['verdict'].upper()}",
        "",
        *render_agreement(section, "Live close probability band"),
        f"Final events {m['final_events']}; with a live close {m['with_live_close']}; "
        f"compared {m['compared']} ({coverage})",
        "Missing: " + ", ".join(f"{k} {v}" for k, v in m["missing"].items()),
        *render_freshness(section, "a live close", "alternative"),
        "",
    ]


def render(report: dict[str, Any]) -> str:
    return "\n".join(
        line for name, s in report["pairings"].items() for line in render_section(name, s)
    )


def _by_event(found: Iterable[Quote]) -> dict[int, list[Quote]]:
    grouped: dict[int, list[Quote]] = defaultdict(list)
    for quote in found:
        grouped[quote.event_id].append(quote)
    return grouped


async def load_comparisons(
    session: AsyncSession,
    start: date,
    end: date,
    max_age: timedelta | None,
    pairings: Sequence[Pairing] = PAIRINGS,
) -> dict[Pairing, list[EventComparison]]:
    """Compare every FINAL EPL event kicked off on ``start`` to ``end`` (UTC days), per pairing.

    Raises ``HoldoutRefused`` before any read when the range touches the sealed holdout."""
    require_clear(SPORT, start, end)
    events = (
        (
            await session.execute(
                select(Event)
                .where(
                    col(Event.sport_key) == SPORT,
                    col(Event.status) == EventStatus.FINAL,
                    col(Event.commence_time) >= datetime.combine(start, time.min, tzinfo=UTC),
                    col(Event.commence_time)
                    < datetime.combine(end + timedelta(days=1), time.min, tzinfo=UTC),
                )
                .order_by(col(Event.commence_time), col(Event.id))
            )
        )
        .scalars()
        .all()
    )
    ids = [e.id for e in events if e.id is not None]
    live: dict[Market, dict[int, list[Quote]]] = {}
    result: dict[Pairing, list[EventComparison]] = {}
    for pairing in pairings:
        if pairing.market not in live:
            live[pairing.market] = _by_event(await live_close_quotes(session, ids, pairing.market))
        alternative = _by_event(
            await quotes(
                session,
                ids,
                pairing.market,
                sources=[pairing.source],
                time_bases=[TimeBasis.OBSERVED],
                bookmakers=[EXCHANGE_BOOKMAKER],
                sides=[Side.BACK],
                before_kickoff=max_age,
            )
        )
        result[pairing] = [
            compare_event(
                event.id,
                event.commence_time,
                pairing,
                live[pairing.market][event.id],
                alternative[event.id],
                max_age,
            )
            for event in events
            if event.id is not None
        ]
    return result


async def run(args: argparse.Namespace) -> dict[str, Any]:
    max_age = None if args.max_age_minutes is None else timedelta(minutes=args.max_age_minutes)
    async with async_session_maker() as session:
        await session.execute(text("SET TRANSACTION READ ONLY"))
        comparisons = await load_comparisons(session, args.start, args.end, max_age)
    return build_report(
        comparisons,
        {
            "sport": SPORT,
            "from": args.start.isoformat(),
            "to": args.end.isoformat(),
            "max_age_minutes": args.max_age_minutes,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # The Betfair job's first EPL live close is on 2026-05-01.
    parser.add_argument(
        "--from", dest="start", type=date.fromisoformat, default=date(2026, 5, 1), metavar="DATE"
    )
    parser.add_argument("--to", dest="end", type=date.fromisoformat, default=None, metavar="DATE")
    parser.add_argument(
        "--max-age-minutes",
        type=float,
        default=None,
        help="only alternative snapshots observed at most this long before kick-off",
    )
    parser.add_argument("--json", type=Path, default=None, help="write the report here")
    args = parser.parse_args()
    if args.end is None:
        args.end = datetime.now(UTC).date()
    report = asyncio.run(run(args))
    print(render(report))
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
