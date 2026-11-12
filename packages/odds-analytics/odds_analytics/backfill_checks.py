"""A historical response judged against the acquisition manifest that requested it.

The probe reports every check; a run stops on a unit mismatch, on a snapshot later than the
call requested or stamped for the next grid time, and on a streak of sparse responses or of gaps.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

from odds_core.api_models import HistoricalOddsResponse
from odds_core.bookmakers import ODDS_API_BOOKMAKER_KEY_MAP
from odds_core.time import parse_api_datetime
from odds_core.utils import is_three_way_sport
from odds_core.vocabulary import Market

from odds_analytics.backfill_plan import Manifest, ceil_minute, floor_minute

# The provider stamps snapshots to the second, so a neighbour is one grid step away to within
# this.
TIMESTAMP_TOLERANCE = timedelta(seconds=60)
# A call's required books are judged on fixtures at most this far from kick-off, so a call that
# lists only games days ahead, as across a break, is not judged on books that are not posted
# that early (inferred).
MAX_BOOK_JUDGED_LEAD = timedelta(hours=24)


def lag_ok(modelled: datetime, snapshot: datetime) -> bool:
    """``snapshot`` is an input to the same whole-minute decisions, at any whole-minute max age,
    as the ``modelled`` one the planner's coverage assumes: it falls within the modelled time's
    whole minute, or exactly on the next minute."""
    return ceil_minute(snapshot) <= ceil_minute(modelled) and floor_minute(
        snapshot
    ) >= floor_minute(modelled)


class ReturnedSnapshot(StrEnum):
    """Where the snapshot a call returned lies against the grid snapshot it requested. The API
    returns the latest snapshot at or before the request time, so any snapshot at or before it
    is the provider's own, stored at its own time."""

    # An input to the same whole-minute decisions as the planned snapshot (``lag_ok``).
    PLANNED = "planned"
    # The provider's snapshot for the planned grid step, stamped off the grid's phase: from
    # ``TIMESTAMP_TOLERANCE`` before the planned snapshot to the request time.
    OFF_PHASE = "off_phase"
    # The provider has nothing from an earlier snapshot to the request time.
    PROVIDER_GAP = "provider_gap"
    # Later than the request time, which the API's contract rules out, or stamped for the next
    # grid time, so the request or the manifest's grid is wrong.
    LATER = "later"


def next_grid_time(manifest: Manifest, modelled: datetime) -> datetime:
    """The grid time after the one the snapshot at ``modelled`` is stamped for, to within
    ``TIMESTAMP_TOLERANCE``: a step starting within that tolerance after ``modelled``, or the
    one after the grid time at ``modelled`` plus the tolerance."""
    tolerance = modelled + TIMESTAMP_TOLERANCE
    starts = [s.start for s in manifest.snapshot_grid if modelled < s.start <= tolerance]
    return min(starts, default=manifest.next_grid_time(manifest.floor_to_grid(tolerance)))


def classify_snapshot(
    manifest: Manifest, modelled: datetime, snapshot: datetime
) -> ReturnedSnapshot:
    """Where ``snapshot``, returned for the grid snapshot at ``modelled``, lies."""
    if (
        snapshot > manifest.request_time(modelled)
        or snapshot >= next_grid_time(manifest, modelled) - TIMESTAMP_TOLERANCE
    ):
        return ReturnedSnapshot.LATER
    if lag_ok(modelled, snapshot):
        return ReturnedSnapshot.PLANNED
    if snapshot >= modelled - TIMESTAMP_TOLERANCE:
        return ReturnedSnapshot.OFF_PHASE
    return ReturnedSnapshot.PROVIDER_GAP


def gap_covers(
    manifest: Manifest,
    modelled: datetime,
    planned: datetime,
    returned: datetime,
    next_snapshot: datetime | None,
) -> bool:
    """Whether a call for ``modelled`` would return the snapshot a recorded provider gap
    returned, as a provider gap: the gap was recorded for ``modelled``, or the gap's
    ``returned`` snapshot is a provider gap for ``modelled`` and its ``next_snapshot`` is
    later than the call's request time. The API returns the latest snapshot at or before the
    request time, so the call would get ``returned``. ``next_snapshot`` is trusted only when
    later than the gap call's own request time, as that call's response implies."""
    if modelled == planned:
        return True
    if next_snapshot is None or next_snapshot <= manifest.request_time(planned):
        return False
    return (
        next_snapshot > manifest.request_time(modelled)
        and classify_snapshot(manifest, modelled, returned) is ReturnedSnapshot.PROVIDER_GAP
    )


def book_judged_lead(manifest: Manifest) -> timedelta:
    """How far from kick-off a call's required books are judged: at most
    ``MAX_BOOK_JUDGED_LEAD``, and no farther than the manifest needs fixtures listed, for a
    sport whose fixtures are listed only hours ahead. The probe checks that each required book
    appears this far ahead."""
    return min(MAX_BOOK_JUDGED_LEAD, timedelta(hours=manifest.probe.min_fixture_lead_hours))


def stored_market(manifest: Manifest, market: Market) -> Market:
    """The market key a response and the database carry for a requested market: soccer's
    three-way h2h is stored as 1x2 (``odds_core.utils.relabel_three_way_markets``)."""
    if market == Market.H2H and is_three_way_sport(manifest.sport):
        return Market.ONE_X_TWO
    return Market(market)


@dataclass(frozen=True)
class ResponseCheck:
    """What a response returned, against what the manifest expects of it: ``modelled`` is the
    grid snapshot the call requested and ``snapshot_time`` the one the provider returned.

    ``book_events`` counts, for each required (market, bookmaker), the events judged that carry
    it; it is empty when no event was judged. ``book_leads`` is the farthest lead, kick-off less
    snapshot, of an event carrying each. ``served`` fixtures were planned for the call and
    ``served_stored`` of them were stored.
    """

    manifest: Manifest
    modelled: datetime
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
            return self.units in (None, 0, self.manifest.units_per_call)
        return self.units == self.manifest.units_per_call

    @property
    def lag(self) -> timedelta:
        return self.modelled - self.snapshot_time

    @property
    def lag_ok(self) -> bool:
        """The snapshot is an input to the same whole-minute decisions, at any whole-minute max
        age, as the modelled one the planner's coverage assumes: it falls within the modelled
        time's whole minute, or exactly on the next minute."""
        return lag_ok(self.modelled, self.snapshot_time)

    @property
    def returned(self) -> ReturnedSnapshot:
        return classify_snapshot(self.manifest, self.modelled, self.snapshot_time)

    @property
    def provider_gap(self) -> bool:
        return self.returned is ReturnedSnapshot.PROVIDER_GAP

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
        """Each required book is listed at least ``book_judged_lead`` ahead of kick-off, so a
        run's book judging, which stops there, never rests on a book not yet posted. A book with
        no events at all fails ``books_ok`` instead."""
        horizon = book_judged_lead(self.manifest)
        return all(lead >= horizon for lead in self.book_leads.values())

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
    modelled: datetime,
    *,
    judged_events: Sequence[Mapping[str, Any]] | None,
    served: int = 0,
    served_stored: int = 0,
) -> ResponseCheck:
    """``response`` to a request for the grid snapshot at ``modelled``, with its required books
    counted over ``judged_events`` (None: no event is fit to judge them on)."""
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
        modelled=modelled,
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
