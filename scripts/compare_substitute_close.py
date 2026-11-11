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
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
from odds_core.as_of import latest_snapshot
from odds_core.close import FDUK_CLOSE, FDUK_CLOSED_LEAGUES, event_close
from odds_core.database import async_session_maker
from odds_core.exchange_price import (
    EXCHANGE_BOOKMAKER,
    MAX_EXCHANGE_OVERROUND,
    ExchangePrice,
    exchange_price,
)
from odds_core.market_prices import Outcome, market_outcomes
from odds_core.models import Event, EventStatus
from odds_core.quotes import Quote, quotes
from odds_core.vocabulary import Market, Side, Source, TimeBasis
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import col

MARKET = Market.ONE_X_TWO

# Pre-set on issue 730, in percentage points.
OUTCOME_MAX_ABS_MEAN_PP = 0.3
OUTCOME_MAX_SD_PP = 1.0
BAND_MAX_ABS_MEAN_PP = 0.5
# A band with fewer rows is inconclusive.
MIN_BAND_ROWS = 50
# Bands of the FDUK close probability: below the first edge, up to the second, and from it.
BAND_EDGES: tuple[float, ...] = (0.25, 0.5)
BAND_LABELS: tuple[str, ...] = ("<0.25", "0.25-0.5", ">=0.5")
# Minutes before kick-off that the Odds API price must lie within, for the share reported.
FRESHNESS_MINUTES: tuple[int, ...] = (5, 10, 30, 60)
AGE_PERCENTILES: tuple[int, ...] = (25, 50, 75, 90)


class Verdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


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


@dataclass(frozen=True)
class Delta:
    event_id: int
    outcome: Outcome
    p_fduk: float
    p_oddsapi: float

    @property
    def dp_pp(self) -> float:
        return (self.p_oddsapi - self.p_fduk) * 100


@dataclass(frozen=True)
class Stat:
    n: int
    mean_pp: float | None
    sd_pp: float | None


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
    latest Odds API exchange snapshot at or before ``kickoff`` that is complete and narrow."""
    if not fduk_quotes:
        return EventComparison(event_id, Status.NO_FDUK_CLOSE)
    close = event_close(fduk_quotes, kickoff, sport, MARKET, max_overround)
    if close is None:
        return EventComparison(event_id, Status.FDUK_UNUSABLE)

    def usable(snapshot: Sequence[Quote]) -> bool:
        price = exchange_price(snapshot, MARKET)
        return price is not None and price.overround <= max_overround

    window = [
        q
        for q in oddsapi_quotes
        if q.snapshot_time <= kickoff and (max_age is None or q.snapshot_time >= kickoff - max_age)
    ]
    if not window:
        return EventComparison(event_id, Status.NO_SNAPSHOT, fduk=close)
    snapshot = latest_snapshot(window, kickoff, usable)
    price = None if snapshot is None else exchange_price(snapshot, MARKET)
    if price is None:
        return EventComparison(event_id, Status.SNAPSHOT_UNUSABLE, fduk=close)
    age = (kickoff - price.time).total_seconds() / 60
    return EventComparison(event_id, Status.COMPARED, close, price, age)


def deltas(comparisons: Iterable[EventComparison]) -> list[Delta]:
    """One delta per outcome of each compared match."""
    return [
        Delta(c.event_id, outcome, c.fduk.fair[outcome], c.oddsapi.fair[outcome])
        for c in comparisons
        if c.fduk is not None and c.oddsapi is not None
        for outcome in market_outcomes(MARKET)
    ]


def stat(values_pp: Sequence[float]) -> Stat:
    n = len(values_pp)
    if n == 0:
        return Stat(0, None, None)
    return Stat(n, float(np.mean(values_pp)), float(np.std(values_pp, ddof=1)) if n > 1 else None)


def outcome_verdict(s: Stat) -> Verdict:
    if s.mean_pp is None or s.sd_pp is None:
        return Verdict.INCONCLUSIVE
    ok = abs(s.mean_pp) <= OUTCOME_MAX_ABS_MEAN_PP and s.sd_pp <= OUTCOME_MAX_SD_PP
    return Verdict.PASS if ok else Verdict.FAIL


def band_verdict(s: Stat) -> Verdict:
    if s.mean_pp is None or s.n < MIN_BAND_ROWS:
        return Verdict.INCONCLUSIVE
    return Verdict.PASS if abs(s.mean_pp) <= BAND_MAX_ABS_MEAN_PP else Verdict.FAIL


def overall_verdict(verdicts: Iterable[Verdict]) -> Verdict:
    """Fail if any slice fails; pass only when every slice passes; otherwise inconclusive."""
    seen = set(verdicts)
    if Verdict.FAIL in seen:
        return Verdict.FAIL
    if Verdict.INCONCLUSIVE in seen or not seen:
        return Verdict.INCONCLUSIVE
    return Verdict.PASS


def band_of(p_fduk: float) -> str:
    return BAND_LABELS[bisect_right(BAND_EDGES, p_fduk)]


def _slice(rows: Sequence[Delta]) -> Stat:
    return stat([d.dp_pp for d in rows])


def build_report(
    comparisons: Sequence[EventComparison],
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """The tables of the comparison, as the JSON written and the text printed."""
    rows = deltas(comparisons)
    outcomes = {o: _slice([d for d in rows if d.outcome is o]) for o in market_outcomes(MARKET)}
    bands = {b: _slice([d for d in rows if band_of(d.p_fduk) == b]) for b in BAND_LABELS}
    outcome_verdicts = {o: outcome_verdict(s) for o, s in outcomes.items()}
    band_verdicts = {b: band_verdict(s) for b, s in bands.items()}

    counts = {s: sum(c.status is s for c in comparisons) for s in Status}
    with_fduk = [c for c in comparisons if c.fduk is not None]
    compared = [c for c in comparisons if c.status is Status.COMPARED]
    ages = [c.age_minutes for c in compared if c.age_minutes is not None]
    within = {
        str(m): {
            "matches": sum(a <= m for a in ages),
            "share": sum(a <= m for a in ages) / len(with_fduk) if with_fduk else None,
        }
        for m in FRESHNESS_MINUTES
    }
    age_distribution = (
        {
            "n": len(ages),
            "min": float(np.min(ages)),
            **{f"p{p}": float(np.percentile(ages, p)) for p in AGE_PERCENTILES},
            "max": float(np.max(ages)),
            "mean": float(np.mean(ages)),
        }
        if ages
        else {"n": 0}
    )

    def table(stats: dict[Any, Stat], verdicts: dict[Any, Verdict]) -> dict[str, Any]:
        return {
            str(k): {
                "n": s.n,
                "mean_pp": s.mean_pp,
                "sd_pp": s.sd_pp,
                "verdict": verdicts[k],
            }
            for k, s in stats.items()
        }

    return {
        "parameters": parameters,
        "thresholds": {
            "outcome_max_abs_mean_pp": OUTCOME_MAX_ABS_MEAN_PP,
            "outcome_max_sd_pp": OUTCOME_MAX_SD_PP,
            "band_max_abs_mean_pp": BAND_MAX_ABS_MEAN_PP,
            "min_band_rows": MIN_BAND_ROWS,
            "max_exchange_overround": MAX_EXCHANGE_OVERROUND,
        },
        "verdict": overall_verdict([*outcome_verdicts.values(), *band_verdicts.values()]),
        "outcomes": table(outcomes, outcome_verdicts),
        "bands": table(bands, band_verdicts),
        "matches": {
            "final_events": len(comparisons),
            "with_fduk_close": len(with_fduk),
            "compared": len(compared),
            "missing": {s.value: counts[s] for s in Status if s is not Status.COMPARED},
        },
        "within_minutes_of_kickoff": within,
        "age_minutes": age_distribution,
    }


def _fmt(value: float | None, spec: str = "+.2f") -> str:
    return "-" if value is None else format(value, spec)


def render(report: dict[str, Any]) -> str:
    lines = [f"Verdict: {report['verdict'].upper()}", ""]
    for title, key in (("Outcome", "outcomes"), ("FDUK probability band", "bands")):
        lines.append(f"{title:<24}{'n':>7}{'mean pp':>10}{'sd pp':>8}  verdict")
        for name, row in report[key].items():
            lines.append(
                f"{name:<24}{row['n']:>7}{_fmt(row['mean_pp']):>10}"
                f"{_fmt(row['sd_pp'], '.2f'):>8}  {row['verdict']}"
            )
        lines.append("")
    m = report["matches"]
    lines.append(
        f"Final events {m['final_events']}; with a usable FDUK close {m['with_fduk_close']}; "
        f"compared {m['compared']}"
    )
    lines.append("Missing: " + ", ".join(f"{k} {v}" for k, v in m["missing"].items()))
    lines.append("")
    lines.append("Share of matches with an FDUK close whose Odds API price is within")
    for minutes, row in report["within_minutes_of_kickoff"].items():
        share = "-" if row["share"] is None else f"{row['share']:.1%}"
        lines.append(f"  {minutes:>3} min before kick-off: {share} ({row['matches']})")
    age = report["age_minutes"]
    lines.append("")
    lines.append(
        "Age of the Odds API prices used (minutes before kick-off): "
        + (", ".join(f"{k} {v:.1f}" if k != "n" else f"n {v}" for k, v in age.items()))
    )
    return "\n".join(lines)


async def load_comparisons(
    session: AsyncSession,
    sport: str,
    start: date,
    end: date,
    max_age: timedelta | None,
) -> list[EventComparison]:
    """Compare every FINAL event of ``sport`` kicked off on ``start`` to ``end`` (UTC days)."""
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
