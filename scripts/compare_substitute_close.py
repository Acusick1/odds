"""Compare the Odds API's Betfair Exchange price near kick-off with FDUK's Betfair Exchange close.

Read-only. For each FINAL match of ``--sport`` kicked off from ``--from`` to ``--to`` with an
FDUK Betfair Exchange closing price, takes the Odds API's ``betfair_exchange`` back prices in the
latest historical (``source_stamped``) snapshot at or before kick-off, optionally within
``--max-age-minutes``. A price counts only when it is complete and its back overround is at most
``MAX_EXCHANGE_OVERROUND``, for both sources. Both are devigged proportionally, and
``dp = p_oddsapi - p_fduk`` is reported per outcome and per band of the FDUK probability against
the thresholds pre-set on issue 730. Spends no Odds API units.
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
from odds_core.close import FDUK_CLOSE, FDUK_CLOSED_LEAGUES, event_close, substitute_close
from odds_core.database import async_session_maker
from odds_core.exchange_price import EXCHANGE_BOOKMAKER, MAX_EXCHANGE_OVERROUND, ExchangePrice
from odds_core.market_prices import market_outcomes
from odds_core.models import Event, EventStatus
from odds_core.quotes import Quote, quotes
from odds_core.vocabulary import Market, Side, Source, TimeBasis
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

MARKET = Market.ONE_X_TWO


class Status(StrEnum):
    """How far a match got: it is compared, or the first thing it lacks."""

    NO_FDUK_CLOSE = "no_fduk_close"
    FDUK_UNUSABLE = "fduk_unusable"
    NO_SNAPSHOT = "no_oddsapi_snapshot"
    SNAPSHOT_UNUSABLE = "oddsapi_unusable"
    COMPARED = "compared"


@dataclass(frozen=True)
class EventComparison:
    """One match's two prices, or what it lacks. ``age_minutes`` is how long before kick-off the
    Odds API price was stamped."""

    event_id: int
    status: Status
    fduk: ExchangePrice | None = None
    oddsapi: ExchangePrice | None = None
    age_minutes: float | None = None


def compare_event(
    event_id: int,
    kickoff: datetime,
    sport: str,
    fduk_quotes: Sequence[Quote],
    oddsapi_quotes: Sequence[Quote],
    max_age: timedelta | None = None,
    max_overround: float = MAX_EXCHANGE_OVERROUND,
) -> EventComparison:
    """The FDUK close of an event (``odds_core.close.event_close`` over its FDUK quotes) and the
    Odds API price ``odds_core.close.substitute_close`` takes within ``max_age``."""
    if not fduk_quotes:
        return EventComparison(event_id, Status.NO_FDUK_CLOSE)
    close = event_close(fduk_quotes, kickoff, sport, MARKET, max_overround)
    if close is None:
        return EventComparison(event_id, Status.FDUK_UNUSABLE)

    if not any(
        q.snapshot_time <= kickoff and (max_age is None or q.snapshot_time >= kickoff - max_age)
        for q in oddsapi_quotes
    ):
        return EventComparison(event_id, Status.NO_SNAPSHOT, fduk=close)
    price = substitute_close(oddsapi_quotes, kickoff, MARKET, max_age, max_overround)
    if price is None:
        return EventComparison(event_id, Status.SNAPSHOT_UNUSABLE, fduk=close)
    age = (kickoff - price.time).total_seconds() / 60
    return EventComparison(event_id, Status.COMPARED, close, price, age)


def deltas(comparisons: Iterable[EventComparison]) -> list[Delta]:
    """One delta per outcome of each compared match, against the FDUK close."""
    return [
        Delta(c.event_id, outcome, c.fduk.fair[outcome], c.oddsapi.fair[outcome])
        for c in comparisons
        if c.fduk is not None and c.oddsapi is not None
        for outcome in market_outcomes(MARKET)
    ]


def build_report(
    comparisons: Sequence[EventComparison],
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """The tables of the comparison, as the JSON written and the text printed."""
    counts = {s: sum(c.status is s for c in comparisons) for s in Status}
    with_fduk = [c for c in comparisons if c.fduk is not None]
    compared = [c for c in comparisons if c.status is Status.COMPARED]
    ages = [c.age_minutes for c in compared if c.age_minutes is not None]
    return {
        "parameters": parameters,
        "thresholds": {**THRESHOLDS, "max_exchange_overround": MAX_EXCHANGE_OVERROUND},
        **agreement(deltas(comparisons), market_outcomes(MARKET)),
        "matches": {
            "final_events": len(comparisons),
            "with_fduk_close": len(with_fduk),
            "compared": len(compared),
            "missing": {s.value: counts[s] for s in Status if s is not Status.COMPARED},
        },
        **freshness(ages, len(with_fduk)),
    }


def render(report: dict[str, Any]) -> str:
    m = report["matches"]
    lines = [
        f"Verdict: {report['verdict'].upper()}",
        "",
        *render_agreement(report, "FDUK probability band"),
        f"Final events {m['final_events']}; with a usable FDUK close {m['with_fduk_close']}; "
        f"compared {m['compared']}",
        "Missing: " + ", ".join(f"{k} {v}" for k, v in m["missing"].items()),
        "",
        *render_freshness(report, "an FDUK close", "Odds API"),
    ]
    return "\n".join(lines)


async def load_comparisons(
    session: AsyncSession,
    sport: str,
    start: date,
    end: date,
    max_age: timedelta | None,
) -> list[EventComparison]:
    """Compare every FINAL event of ``sport`` kicked off on ``start`` to ``end`` (UTC days).

    Raises ``HoldoutRefused`` before any read when the range touches the sealed holdout."""
    require_clear(sport, start, end)
    events = (
        (
            await session.execute(
                select(Event)
                .where(
                    col(Event.sport_key) == sport,
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
    fduk = await quotes(
        session,
        ids,
        MARKET,
        sources=[FDUK_CLOSE.source],
        time_bases=[FDUK_CLOSE.time_basis],
        bookmakers=[FDUK_CLOSE.bookmaker],
        sides=FDUK_CLOSE.sides,
    )
    oddsapi = await quotes(
        session,
        ids,
        MARKET,
        sources=[Source.ODDS_API],
        time_bases=[TimeBasis.SOURCE_STAMPED],
        bookmakers=[EXCHANGE_BOOKMAKER],
        sides=[Side.BACK],
        before_kickoff=max_age,
    )
    by_event: tuple[dict[int, list[Quote]], dict[int, list[Quote]]] = (
        defaultdict(list),
        defaultdict(list),
    )
    for grouped, found in zip(by_event, (fduk, oddsapi), strict=True):
        for quote in found:
            grouped[quote.event_id].append(quote)
    return [
        compare_event(
            event.id,
            event.commence_time,
            sport,
            by_event[0][event.id],
            by_event[1][event.id],
            max_age,
        )
        for event in events
        if event.id is not None
    ]


async def run(args: argparse.Namespace) -> dict[str, Any]:
    max_age = None if args.max_age_minutes is None else timedelta(minutes=args.max_age_minutes)
    async with async_session_maker() as session:
        await session.execute(text("SET TRANSACTION READ ONLY"))
        comparisons = await load_comparisons(session, args.sport, args.start, args.end, max_age)
    return build_report(
        comparisons,
        {
            "sport": args.sport,
            "market": MARKET.value,
            "from": args.start.isoformat(),
            "to": args.end.isoformat(),
            "max_age_minutes": args.max_age_minutes,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sport", choices=FDUK_CLOSED_LEAGUES, default="soccer_epl")
    parser.add_argument(
        "--from", dest="start", type=date.fromisoformat, default=date(2024, 8, 1), metavar="DATE"
    )
    parser.add_argument(
        "--to", dest="end", type=date.fromisoformat, default=date(2026, 7, 31), metavar="DATE"
    )
    parser.add_argument(
        "--max-age-minutes",
        type=float,
        default=None,
        help="only Odds API snapshots stamped at most this long before kick-off",
    )
    parser.add_argument("--json", type=Path, default=None, help="write the report here")
    args = parser.parse_args()
    report = asyncio.run(run(args))
    print(render(report))
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
