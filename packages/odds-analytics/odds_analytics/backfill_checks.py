"""A historical response judged against the acquisition manifest that requested it.

The probe reports every check; a run stops on a unit mismatch and on a streak of sparse
responses.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from odds_core.api_models import HistoricalOddsResponse
from odds_core.bookmakers import ODDS_API_BOOKMAKER_KEY_MAP
from odds_core.time import parse_api_datetime
from odds_core.utils import is_three_way_sport
from odds_core.vocabulary import Market

from odds_analytics.backfill_plan import Manifest

# The provider stamps snapshots to the second, so a neighbour is one grid step away to within
# this, and a returned snapshot is the requested time to within it.
TIMESTAMP_TOLERANCE = timedelta(seconds=60)
# A call's required books are judged on fixtures at most this far from kick-off, so a call that
# lists only games days ahead, as across a break, is not judged on books that are not posted
# that early (inferred). The probe checks that each required book appears this far ahead.
BOOK_JUDGED_LEAD = timedelta(hours=24)


def stored_market(manifest: Manifest, market: Market) -> Market:
    """The market key a response and the database carry for a requested market: soccer's
    three-way h2h is stored as 1x2 (``odds_core.utils.relabel_three_way_markets``)."""
    if market == Market.H2H and is_three_way_sport(manifest.sport):
        return Market.ONE_X_TWO
    return Market(market)


@dataclass(frozen=True)
class ResponseCheck:
    """What a response returned, against what the manifest expects of it.

    ``book_events`` counts, for each required (market, bookmaker), the events judged that carry
    it; it is empty when no event was judged. ``book_leads`` is the farthest lead, kick-off less
    snapshot, of an event carrying each. ``served`` fixtures were planned for the call and
    ``served_stored`` of them were stored.
    """

    manifest: Manifest
    lag: timedelta
    snapshot_time: datetime
    units: int | None
    events: int
    previous_gap: timedelta | None
    next_gap: timedelta | None
    farthest_lead: timedelta | None
    book_events: dict[tuple[Market, str], int]
    book_leads: dict[tuple[Market, str], timedelta]
    served: int = 0
    served_stored: int = 0

    @property
    def units_ok(self) -> bool:
        """The bill is the manifest's. An empty response may also bill nothing, which the API
        does not count against the quota."""
        if self.events == 0:
            return self.units in (None, 0, self.manifest.probe.units_per_call)
        return self.units == self.manifest.probe.units_per_call

    @property
    def lag_ok(self) -> bool:
        """The snapshot is the requested grid time, which the planner's coverage and
        skip-existing assume."""
        return self.lag <= TIMESTAMP_TOLERANCE

    @property
    def grid_ok(self) -> bool:
        """Both neighbouring snapshots are one grid step away. The step before the snapshot is
        the one in force at that earlier snapshot."""
        if self.previous_gap is None or self.next_gap is None:
            return False
        before = self.manifest.grid_step_at(self.snapshot_time - self.previous_gap)
        after = self.manifest.grid_step_at(self.snapshot_time)
        return (
            abs(self.previous_gap - before) <= TIMESTAMP_TOLERANCE
            and abs(self.next_gap - after) <= TIMESTAMP_TOLERANCE
        )

    @property
    def lead_ok(self) -> bool:
        return self.farthest_lead is not None and self.farthest_lead >= timedelta(
            hours=self.manifest.probe.min_fixture_lead_hours
        )

    @property
    def missing_books(self) -> list[tuple[Market, str]]:
        return [key for key, count in self.book_events.items() if count == 0]

    @property
    def books_ok(self) -> bool:
        return not self.missing_books

    @property
    def horizon_ok(self) -> bool:
        """Each required book is listed at least ``BOOK_JUDGED_LEAD`` ahead of kick-off, so a
        run's book judging, which stops there, never rests on a book not yet posted. A book with
        no events at all fails ``books_ok`` instead."""
        return all(lead >= BOOK_JUDGED_LEAD for lead in self.book_leads.values())

    @property
    def sparse_reason(self) -> str | None:
        """Why the response is too thin to be a run's data, None when it is not."""
        if not self.events:
            return "no events"
        if self.served and not self.served_stored:
            return f"stored none of the {self.served} fixtures it serves"
        if self.missing_books:
            return "lacks " + ", ".join(
                f"{book} on {market}" for market, book in self.missing_books
            )
        return None

    @property
    def ok(self) -> bool:
        return (
            self.units_ok
            and self.lag_ok
            and self.grid_ok
            and self.lead_ok
            and self.books_ok
            and self.horizon_ok
        )


def check_response(
    manifest: Manifest,
    response: HistoricalOddsResponse,
    requested: datetime,
    *,
    judged_events: Sequence[Mapping[str, Any]] | None,
    served: int = 0,
    served_stored: int = 0,
) -> ResponseCheck:
    """``response`` to a request for ``requested``, with its required books counted over
    ``judged_events`` (None: no event is fit to judge them on)."""
    snapshot = response.snapshot_time
    required: dict[tuple[Market, str], int] = {}
    if judged_events is not None:
        required = {
            (stored_market(manifest, market), ODDS_API_BOOKMAKER_KEY_MAP.get(book, book)): 0
            for market, books in manifest.probe.required_bookmakers.items()
            for book in books
        }
    leads: dict[tuple[Market, str], timedelta] = {}
    for raw in judged_events or ():
        lead = parse_api_datetime(raw["commence_time"]) - snapshot
        carried = {
            (market["key"], bookmaker["key"])
            for bookmaker in raw.get("bookmakers", [])
            for market in bookmaker.get("markets", [])
        }
        for key in required.keys() & carried:
            required[key] += 1
            leads[key] = max(lead, leads.get(key, lead))
    return ResponseCheck(
        manifest=manifest,
        lag=requested - snapshot,
        snapshot_time=snapshot,
        units=response.units_used,
        events=len(response.events),
        previous_gap=(
            snapshot - response.previous_snapshot_time if response.previous_snapshot_time else None
        ),
        next_gap=(response.next_snapshot_time - snapshot if response.next_snapshot_time else None),
        farthest_lead=(
            max(event.commence_time for event in response.events) - snapshot
            if response.events
            else None
        ),
        book_events=required,
        book_leads=leads,
        served=served,
        served_stored=served_stored,
    )
