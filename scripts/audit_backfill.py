"""Audit what an Odds API acquisition manifest stored, by season, against what it planned.

Read-only. For each season of ``--manifest`` it reads the season's FINAL events, the historical
(``source_stamped``) Odds API snapshots stored on them and the recorded provider gaps, and prints
Markdown tables of: fixtures with and without a stored snapshot, and listings in stored responses
that attached to no event; each book's fixture coverage; the share of each coverage band's
whole-minute decisions that find a snapshot within the band's max input age at the snapshots'
actual times; the latest Betfair Exchange price at or before kick-off and its age; provider gaps
and off-phase snapshots; and the planned calls against the stored snapshot times. Prices are
counted, never read as values. Spends no Odds API units.
"""

from __future__ import annotations

import argparse
import asyncio
from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from odds_analytics.backfill_checks import TIMESTAMP_TOLERANCE, lag_ok, stored_market
from odds_analytics.backfill_plan import Band, Call, Fixture, Manifest, Season, plan
from odds_core.as_of import latest_snapshot
from odds_core.bookmakers import (
    BOOKMAKER_ALIASES_BY_SOURCE,
    EXCHANGE_BOOKMAKERS,
    UK_RETAIL_BOOKMAKERS,
)
from odds_core.database import async_session_maker
from odds_core.exchange_price import EXCHANGE_BOOKMAKER, MAX_EXCHANGE_OVERROUND, exchange_price
from odds_core.quotes import Quote, quotes
from odds_core.vocabulary import Market, Side, Source, TimeBasis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

# Ages within which a near-kick-off exchange price is counted, for the shares reported.
CLOSE_WINDOWS_MINUTES: tuple[int, ...] = (5, 10)
AGE_PERCENTILES: tuple[int, ...] = (50, 75, 90, 99)
# How far before kick-off the exchange's latest price is searched for.
CLOSE_LOOKBACK = timedelta(hours=2)
MINUTE_SECONDS = 60


@dataclass(frozen=True)
class StoredSnapshot:
    time: datetime
    books: frozenset[str]


@dataclass(frozen=True)
class AuditFixture:
    event_id: int
    home: str
    away: str
    kickoff: datetime
    # The requested market's snapshots, in time order.
    snapshots: tuple[StoredSnapshot, ...]

    def before_kickoff(self) -> list[StoredSnapshot]:
        return [s for s in self.snapshots if s.time <= self.kickoff]


@dataclass(frozen=True)
class Gap:
    planned: datetime
    returned: datetime
    next: datetime | None


@dataclass(frozen=True)
class BandCoverage:
    """A fixture band's whole-minute decisions: those served, all of them, and of the unserved,
    those before the fixture's first stored snapshot (not yet listed) and, of the rest, those a
    provider gap's planned snapshot would have served."""

    served: int
    total: int
    before_listing: int
    in_gaps: int


def _seconds(times: Iterable[datetime]) -> np.ndarray:
    return np.array([t.timestamp() for t in times], dtype=np.int64)


def band_coverage(
    kickoff: datetime, snapshot_times: Sequence[datetime], band: Band, gaps: Sequence[Gap] = ()
) -> BandCoverage:
    """Count ``band``'s whole-minute decisions before ``kickoff`` that find a snapshot in
    ``snapshot_times`` (sorted) at most the band's max age old and no later than the decision.
    The fixture counts as listed from the first of ``snapshot_times``."""
    first, last = band.decision_span(kickoff)
    decisions = np.arange(
        int(first.timestamp()), int(last.timestamp()) + 1, MINUTE_SECONDS, dtype=np.int64
    )
    max_age = int(band.max_age.total_seconds())
    snaps = _seconds(snapshot_times)
    if len(snaps):
        idx = np.searchsorted(snaps, decisions, side="right") - 1
        served = (idx >= 0) & (decisions - snaps[np.clip(idx, 0, None)] <= max_age)
        unlisted = decisions < snaps[0]
    else:
        served = np.zeros(len(decisions), dtype=bool)
        unlisted = np.ones(len(decisions), dtype=bool)
    in_gap = np.zeros(len(decisions), dtype=bool)
    for gap in gaps:
        planned = int(gap.planned.timestamp())
        in_gap |= (decisions >= planned) & (decisions - max_age <= planned)
    return BandCoverage(
        served=int(served.sum()),
        total=len(decisions),
        before_listing=int((~served & unlisted).sum()),
        in_gaps=int((~served & ~unlisted & in_gap).sum()),
    )


def off_phase(manifest: Manifest, snapshot: datetime) -> bool:
    """``snapshot`` is not an input to the same whole-minute decisions as its nearest grid
    time (``lag_ok``), so the provider stamped it off the grid's phase."""
    try:
        step = manifest.grid_step_at(snapshot)
        nearest = manifest.floor_to_grid(snapshot + step / 2)
    except ValueError:
        return True
    return not lag_ok(nearest, snapshot)


@dataclass(frozen=True)
class CallReconciliation:
    planned: int
    stored: int
    gap: int
    missing: tuple[datetime, ...]
    # Stored snapshot times no planned call accounts for: gap returns, probes, off-phase stamps.
    unplanned: tuple[datetime, ...]


def gap_covers(manifest: Manifest, gap: Gap, call: datetime) -> bool:
    """A recorded gap accounts for the call at ``call``: recorded for it, or inside its hole."""
    if gap.planned == call:
        return True
    return (
        gap.next is not None
        and gap.returned < call - TIMESTAMP_TOLERANCE
        and gap.next > manifest.request_time(call)
    )


def reconcile_calls(
    manifest: Manifest,
    planned: Sequence[datetime],
    stored: Sequence[datetime],
    gaps: Sequence[Gap],
) -> CallReconciliation:
    """Each planned call is stored when a stored snapshot lies in ``(request - step, request]``,
    where its response falls, else a gap when a recorded gap covers it, else missing."""
    stored_sorted = sorted(set(stored))
    used: set[datetime] = set()
    n_stored = n_gap = 0
    missing: list[datetime] = []
    for call in sorted(planned):
        request = manifest.request_time(call)
        lo = request - manifest.grid_step_at(call)
        i = bisect_right(stored_sorted, request) - 1
        if i >= 0 and stored_sorted[i] > lo:
            n_stored += 1
            used.add(stored_sorted[i])
        elif any(gap_covers(manifest, gap, call) for gap in gaps):
            n_gap += 1
        else:
            missing.append(call)
    return CallReconciliation(
        planned=len(planned),
        stored=n_stored,
        gap=n_gap,
        missing=tuple(missing),
        unplanned=tuple(t for t in stored_sorted if t not in used),
    )


def calls_before_listing(calls: Sequence[Call], listed: dict[int, datetime]) -> list[datetime]:
    """Planned calls none of whose served fixtures was listed by then, judged by ``listed``,
    each fixture's first stored snapshot (a fixture never stored counts as never listed)."""
    return [
        call.time
        for call in calls
        if not any(
            (first := listed.get(t.fixture.event_id)) is not None
            and first <= call.time + TIMESTAMP_TOLERANCE
            for t in call.serves
        )
    ]


@dataclass(frozen=True)
class ClosePoint:
    """A fixture's latest Betfair Exchange snapshot at or before kick-off within
    ``CLOSE_LOOKBACK``: its age, whether it has a back price on every outcome, and whether that
    price is usable (back overround at most ``MAX_EXCHANGE_OVERROUND``)."""

    age: timedelta | None
    complete: bool
    usable_age: timedelta | None


def close_point(kickoff: datetime, exchange_quotes: Sequence[Quote], market: Market) -> ClosePoint:
    any_quote = latest_snapshot(exchange_quotes, kickoff)
    usable = latest_snapshot(
        exchange_quotes,
        kickoff,
        lambda group: (
            (price := exchange_price(group, market)) is not None
            and price.overround <= MAX_EXCHANGE_OVERROUND
        ),
    )
    return ClosePoint(
        age=kickoff - any_quote[0].snapshot_time if any_quote else None,
        complete=any_quote is not None and exchange_price(any_quote, market) is not None,
        usable_age=kickoff - usable[0].snapshot_time if usable else None,
    )


def share_within(ages: Sequence[timedelta | None], minutes: int) -> float:
    if not ages:
        return 0.0
    limit = timedelta(minutes=minutes)
    return sum(1 for a in ages if a is not None and a <= limit) / len(ages)


def age_percentiles(ages: Sequence[timedelta | None]) -> dict[int, float]:
    """Percentiles of the present ages, in minutes."""
    present = [a.total_seconds() / 60 for a in ages if a is not None]
    if not present:
        return {}
    return {p: float(np.percentile(present, p)) for p in AGE_PERCENTILES}


def book_role(book: str) -> str:
    if book in UK_RETAIL_BOOKMAKERS:
        return "UK retail"
    if book in EXCHANGE_BOOKMAKERS:
        return "exchange"
    return "neither"


def book_aliases(book: str) -> list[str]:
    """The Odds API keys stored under ``book``, other than itself."""
    return sorted(k for k, v in BOOKMAKER_ALIASES_BY_SOURCE[Source.ODDS_API].items() if v == book)


# --- Database reads -----------------------------------------------------------------------


async def load_audit_fixtures(
    session: AsyncSession, sport: str, season: Season, market: Market
) -> list[AuditFixture]:
    events = (
        await session.execute(
            text(
                "SELECT id, home_team, away_team, commence_time FROM events "
                "WHERE sport_key = :sport AND status = 'FINAL' "
                "AND commence_time >= :start AND commence_time < :end ORDER BY commence_time, id"
            ),
            {"sport": sport, "start": season.start, "end": season.end},
        )
    ).all()
    rows = (
        await session.execute(
            text(
                "SELECT s.event_id, s.snapshot_time, array_agg(DISTINCT p.bookmaker) "
                "FROM odds_snapshots s JOIN prices p ON p.snapshot_id = s.id "
                "WHERE s.event_id = ANY(:ids) AND s.source = :source "
                "AND s.time_basis = :basis AND s.market = :market "
                "GROUP BY s.id ORDER BY s.snapshot_time"
            ),
            {
                "ids": [e.id for e in events],
                "source": Source.ODDS_API.value,
                "basis": TimeBasis.SOURCE_STAMPED.value,
                "market": market.value,
            },
        )
    ).all()
    by_event: dict[int, list[StoredSnapshot]] = defaultdict(list)
    for event_id, time, books in rows:
        by_event[event_id].append(StoredSnapshot(time, frozenset(books)))
    return [
        AuditFixture(e.id, e.home_team, e.away_team, e.commence_time, tuple(by_event[e.id]))
        for e in events
    ]


async def load_other_markets(
    session: AsyncSession, event_ids: Sequence[int], market: Market
) -> list[tuple[str | None, bool, int, int]]:
    """Odds API historical snapshots of the events in any other market (or none): the market,
    whether stamped before kick-off, and the snapshots and events."""
    rows = await session.execute(
        text(
            "SELECT s.market, s.snapshot_time <= e.commence_time, count(*), "
            "count(DISTINCT s.event_id) FROM odds_snapshots s JOIN events e ON e.id = s.event_id "
            "WHERE s.event_id = ANY(:ids) AND s.source = :source AND s.time_basis = :basis "
            "AND s.market IS DISTINCT FROM :market GROUP BY 1, 2 ORDER BY 1, 2"
        ),
        {
            "ids": list(event_ids),
            "source": Source.ODDS_API.value,
            "basis": TimeBasis.SOURCE_STAMPED.value,
            "market": market.value,
        },
    )
    return [(m, bool(pre), int(n), int(ev)) for m, pre, n, ev in rows.all()]


async def load_unmatched_listings(
    session: AsyncSession, event_ids: Sequence[int]
) -> list[tuple[str, str, str, int]]:
    """Games listed in the stored responses of the events that attached to no event in that
    response: home, away, listed kick-off and the number of responses listing it. A response
    that attached nothing is not stored, so it is not seen here."""
    rows = await session.execute(
        text(
            "WITH p AS (SELECT DISTINCT payload_id FROM odds_snapshots "
            "WHERE event_id = ANY(:ids) AND source = :source AND time_basis = :basis "
            "AND payload_id IS NOT NULL) "
            "SELECT x->>'home_team', x->>'away_team', x->>'commence_time', count(*) "
            "FROM p JOIN snapshot_payloads sp ON sp.id = p.payload_id "
            "CROSS JOIN LATERAL jsonb_array_elements(sp.payload) x "
            "WHERE NOT EXISTS (SELECT 1 FROM odds_snapshots s2 JOIN event_external_ids ee "
            "ON ee.event_id = s2.event_id AND ee.source = :source "
            "WHERE s2.payload_id = p.payload_id AND ee.external_id = x->>'id') "
            "GROUP BY 1, 2, 3 ORDER BY 3"
        ),
        {
            "ids": list(event_ids),
            "source": Source.ODDS_API.value,
            "basis": TimeBasis.SOURCE_STAMPED.value,
        },
    )
    return [(h, a, k, int(n)) for h, a, k, n in rows.all()]


async def load_stored_times(
    session: AsyncSession, event_ids: Sequence[int], market: Market | None = None
) -> list[datetime]:
    """Every distinct Odds API historical snapshot time on the events, in ``market`` or, when
    None, in any market or none."""
    rows = await session.execute(
        text(
            "SELECT DISTINCT snapshot_time FROM odds_snapshots WHERE event_id = ANY(:ids) "
            "AND source = :source AND time_basis = :basis "
            "AND (CAST(:market AS text) IS NULL OR market = :market) ORDER BY 1"
        ),
        {
            "ids": list(event_ids),
            "source": Source.ODDS_API.value,
            "basis": TimeBasis.SOURCE_STAMPED.value,
            "market": market.value if market else None,
        },
    )
    return [t for (t,) in rows.all()]


async def load_gaps(session: AsyncSession, sport: str, market: Market) -> list[Gap]:
    rows = await session.execute(
        text(
            "SELECT planned_snapshot, returned_snapshot, next_snapshot FROM odds_api_snapshot_gaps "
            "WHERE sport_key = :sport AND market = :market ORDER BY planned_snapshot"
        ),
        {"sport": sport, "market": market.value},
    )
    return [Gap(p, r, n) for p, r, n in rows.all()]


# --- Report ---------------------------------------------------------------------------------


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _table(header: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _iso(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _band_label(band: Band) -> str:
    return (
        f"KO−{band.min_lead_minutes}m..KO−{band.max_lead_minutes}m, "
        f"max age {band.max_input_age_minutes}m"
    )


def farthest_lead(manifest: Manifest) -> int:
    """Minutes before kick-off of the earliest snapshot any band can read."""
    return max(b.max_lead_minutes + b.max_input_age_minutes for b in manifest.coverage)


def _season_gaps(
    gaps: Sequence[Gap], fixtures: Sequence[AuditFixture], manifest: Manifest
) -> list[Gap]:
    if not fixtures:
        return []
    lo = min(f.kickoff for f in fixtures) - timedelta(minutes=farthest_lead(manifest))
    hi = max(f.kickoff for f in fixtures)
    return [g for g in gaps if lo <= g.planned <= hi]


async def audit_season(
    session: AsyncSession, manifest: Manifest, name: str, season: Season, gaps: Sequence[Gap]
) -> str:
    market = stored_market(manifest, manifest.markets[0])
    fixtures = await load_audit_fixtures(session, manifest.sport, season, market)
    ids = [f.event_id for f in fixtures]
    season_gaps = _season_gaps(gaps, fixtures, manifest)
    out: list[str] = [f"## {name}"]

    # Fixtures.
    with_snaps = [f for f in fixtures if f.before_kickoff()]
    missed = [f for f in fixtures if not f.before_kickoff()]
    unmatched = await load_unmatched_listings(session, ids)
    out.append(
        f"### Fixtures, {name}\n\n{len(fixtures)} FINAL fixtures; {len(with_snaps)} with a stored "
        f"`{market}` snapshot at or before kick-off; {len(missed)} without."
    )
    if missed:
        out.append(
            _table(
                ["Fixture", "Kick-off"], ((f"{f.home} v {f.away}", _iso(f.kickoff)) for f in missed)
            )
        )
    out.append(f"Listings in stored responses that attached to no event: {len(unmatched)}.")
    if unmatched:
        out.append(_table(["Home", "Away", "Listed kick-off", "Responses"], unmatched))
    others = await load_other_markets(session, ids, market)
    if others:
        out.append(f"Odds API historical snapshots of these fixtures outside `{market}`:")
        out.append(
            _table(
                ["Market", "Stamped", "Snapshots", "Fixtures"],
                (
                    (m or "none", "before KO" if pre else "after KO", n, ev)
                    for m, pre, n, ev in others
                ),
            )
        )

    # Bookmakers.
    fixture_books: Counter[str] = Counter()
    snapshot_books: Counter[str] = Counter()
    n_pre = 0
    for f in fixtures:
        pre = f.before_kickoff()
        n_pre += len(pre)
        fixture_books.update(set().union(*(s.books for s in pre)) if pre else set())
        for s in pre:
            snapshot_books.update(s.books)
    out.append(
        f"### Bookmakers, {name}\n\nShare of the {len(fixtures)} fixtures with the book in any "
        "snapshot "
        f"at or before kick-off, and of the {n_pre:,} such snapshots."
    )
    out.append(
        _table(
            ["Book", "Fixtures", "Snapshots", "Role", "Odds API keys aliased to it"],
            (
                (
                    f"`{b}`",
                    _pct(n / len(fixtures)),
                    _pct(snapshot_books[b] / n_pre) if n_pre else "-",
                    book_role(b),
                    ", ".join(f"`{k}`" for k in book_aliases(b)) or "none",
                )
                for b, n in sorted(fixture_books.items(), key=lambda kv: (-kv[1], kv[0]))
            ),
        )
    )

    # Leads.
    rows = []
    for band in sorted(manifest.coverage, key=lambda b: b.min_lead_minutes):
        any_cov = [
            band_coverage(f.kickoff, [s.time for s in f.snapshots], band, season_gaps)
            for f in fixtures
        ]
        pair_cov = [
            band_coverage(
                f.kickoff,
                [
                    s.time
                    for s in f.snapshots
                    if EXCHANGE_BOOKMAKER in s.books and s.books & UK_RETAIL_BOOKMAKERS
                ],
                band,
            )
            for f in fixtures
        ]
        served = sum(c.served for c in any_cov)
        total = sum(c.total for c in any_cov)
        unlisted = sum(c.before_listing for c in any_cov)
        in_gaps = sum(c.in_gaps for c in any_cov)
        pair_served = sum(c.served for c in pair_cov)
        rows.append(
            (
                _band_label(band),
                f"{total:,}",
                _pct(served / total) if total else "-",
                f"{unlisted:,}",
                f"{in_gaps:,}",
                f"{total - served - unlisted - in_gaps:,}",
                f"{sum(1 for c in any_cov if c.served == c.total)}/{len(fixtures)}",
                _pct(pair_served / total) if total else "-",
            )
        )
    out.append(
        f"### Leads, {name}\n\nWhole-minute decisions per fixture and band, served when a stored "
        "snapshot lies in `[t − max age, t]` at its actual time. Unserved decisions are split "
        "into those before the fixture's first stored snapshot (not yet listed), those a "
        "recorded gap's planned snapshot would have served, and the rest. The last column "
        "needs `betfair_exchange` and a UK retail book in the same snapshot."
    )
    out.append(
        _table(
            [
                "Band",
                "Decisions",
                "Served",
                "Unserved: not listed",
                "Unserved: gap",
                "Unserved: other",
                "Fixtures fully served",
                "Served, exchange + retail",
            ],
            rows,
        )
    )
    horizon = timedelta(minutes=farthest_lead(manifest))
    late = [
        (f, f.snapshots[0].time)
        for f in fixtures
        if f.snapshots and f.snapshots[0].time > f.kickoff - horizon
    ]
    if late:
        by_lead = Counter(f"{(f.kickoff - first).total_seconds() / 3600:.0f}" for f, first in late)
        out.append(
            f"Fixtures first stored less than {horizon.total_seconds() / 3600:.0f} h before "
            "kick-off: "
            f"{len(late)}, first stored at leads (h: fixtures) "
            + ", ".join(
                f"{h}: {n}" for h, n in sorted(by_lead.items(), key=lambda kv: float(kv[0]))
            )
            + "."
        )

    # Near kick-off exchange.
    exchange = await quotes(
        session,
        ids,
        market,
        sources=[Source.ODDS_API],
        time_bases=[TimeBasis.SOURCE_STAMPED],
        bookmakers=[EXCHANGE_BOOKMAKER],
        sides=[Side.BACK],
        before_kickoff=CLOSE_LOOKBACK,
    )
    by_event: dict[int, list[Quote]] = defaultdict(list)
    for q in exchange:
        by_event[q.event_id].append(q)
    points = [close_point(f.kickoff, by_event[f.event_id], market) for f in fixtures]
    any_ages = [p.age for p in points]
    usable_ages = [p.usable_age for p in points]
    pct_any = age_percentiles(any_ages)
    pct_usable = age_percentiles(usable_ages)
    out.append(
        f"### Near kick-off `{EXCHANGE_BOOKMAKER}`, {name}\n\nPer fixture, the latest exchange "
        "snapshot at or before kick-off within "
        f"{CLOSE_LOOKBACK.total_seconds() / 60:.0f} min; usable means a back price on every "
        f"outcome with back overround at most {MAX_EXCHANGE_OVERROUND:.0%}. Research inputs, not "
        "validated closing labels."
    )
    out.append(
        _table(
            ["Price", "Fixtures with one"]
            + [f"Within {m} min" for m in CLOSE_WINDOWS_MINUTES]
            + [f"p{p} age (min)" for p in AGE_PERCENTILES],
            [
                (
                    label,
                    f"{sum(1 for a in ages if a is not None)}/{len(fixtures)}",
                    *(_pct(share_within(ages, m)) for m in CLOSE_WINDOWS_MINUTES),
                    *(f"{pct[p]:.1f}" if pct else "-" for p in AGE_PERCENTILES),
                )
                for label, ages, pct in (
                    ("Any exchange quote", any_ages, pct_any),
                    ("Usable", usable_ages, pct_usable),
                )
            ],
        )
    )
    incomplete = sum(1 for p in points if p.age is not None and not p.complete)
    out.append(f"Latest exchange snapshot missing an outcome's back price: {incomplete} fixtures.")

    # Gaps and off-phase snapshots.
    stored = await load_stored_times(session, ids)
    off = [t for t in stored if off_phase(manifest, t)]
    out.append(
        f"### Provider gaps and off-phase snapshots, {name}\n\nRecorded gaps: {len(season_gaps)}."
    )
    if season_gaps:
        out.append(
            _table(
                ["Planned", "Returned", "Next"],
                (
                    (_iso(g.planned), _iso(g.returned), _iso(g.next) if g.next else "-")
                    for g in season_gaps
                ),
            )
        )
    out.append(
        "Stored snapshot times off the grid's phase (outside their nearest grid time's "
        f"whole minute): "
        f"{len(off)}" + (f" ({', '.join(_iso(t) for t in off)})." if off else ".")
    )

    # Calls.
    calls = plan(manifest, [Fixture(f.event_id, f.kickoff) for f in fixtures])
    # A call counts as stored as the executor's skip judges it: by a requested-market snapshot.
    rec = reconcile_calls(
        manifest,
        [c.time for c in calls],
        await load_stored_times(session, ids, market),
        season_gaps,
    )
    unlisted_calls = calls_before_listing(
        calls, {f.event_id: f.snapshots[0].time for f in fixtures if f.snapshots}
    )
    out.append(
        f"### Calls, {name}\n\nPlanned {rec.planned:,}; stored {rec.stored:,}; covered by a "
        "recorded gap "
        f"{rec.gap:,}; missing {len(rec.missing):,}. Distinct stored snapshot times, any "
        f"market: {len(stored):,}, × {manifest.units_per_call} units = "
        f"{len(stored) * manifest.units_per_call:,} units. `{market}` snapshot times matching no "
        f"planned call (gap returns, probes): {len(rec.unplanned):,}. Planned calls that serve "
        "only "
        f"fixtures not yet stored at that time (not listed): {len(unlisted_calls):,}."
    )
    if rec.missing:
        out.append("Missing calls: " + ", ".join(_iso(t) for t in rec.missing[:20]))
    return "\n\n".join(out)


async def audit(manifest: Manifest, seasons: Sequence[str]) -> str:
    async with async_session_maker() as session:
        await session.execute(text("SET TRANSACTION READ ONLY"))
        gaps = await load_gaps(
            session, manifest.sport, stored_market(manifest, manifest.markets[0])
        )
        parts = [
            await audit_season(session, manifest, name, manifest.seasons[name], gaps)
            for name in seasons
        ]
    return "\n\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--season", action="append", help="default: every season")
    args = parser.parse_args()
    manifest = Manifest.load(args.manifest)
    if len(manifest.markets) != 1:
        parser.error("the audit covers single-market manifests")
    seasons = args.season or list(manifest.seasons)
    print(asyncio.run(audit(manifest, seasons)))


if __name__ == "__main__":
    main()
