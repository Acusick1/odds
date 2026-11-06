"""A historical response judged against the acquisition manifest that requested it.

The probe reports every check; a run stops on a unit mismatch and on a streak of sparse
responses.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from odds_core.api_models import HistoricalOddsResponse
from odds_core.bookmakers import canonical_bookmaker
from odds_core.utils import is_three_way_sport
from odds_core.vocabulary import Market, Source

from odds_analytics.backfill_plan import Manifest


def stored_market(manifest: Manifest, market: Market) -> Market:
    """The market key a response and the database carry for a requested market: soccer's
    three-way h2h is stored as 1x2 (``odds_core.utils.relabel_three_way_markets``)."""
    if market == Market.H2H and is_three_way_sport(manifest.sport):
        return Market.ONE_X_TWO
    return Market(market)


@dataclass(frozen=True)
class ResponseCheck:
    """What a response returned, against what the manifest expects of it.

    ``book_events`` counts, for each required (market, bookmaker), the events carrying it.
    """

    manifest: Manifest
    snapshot_time: datetime
    units: int | None
    events: int
    previous_gap: timedelta | None
    next_gap: timedelta | None
    farthest_lead: timedelta | None
    book_events: dict[tuple[Market, str], int]

    @property
    def units_ok(self) -> bool:
        return self.units == self.manifest.probe.units_per_call

    @property
    def grid_ok(self) -> bool:
        """Both neighbouring snapshots are one grid step away. The step before the snapshot is
        the one in force at that earlier snapshot."""
        if self.previous_gap is None or self.next_gap is None:
            return False
        before = self.manifest.grid_step_at(self.snapshot_time - self.previous_gap)
        after = self.manifest.grid_step_at(self.snapshot_time)
        return self.previous_gap == before and self.next_gap == after

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
    def sparse_reason(self) -> str | None:
        """Why the response is too thin to be a run's data, None when it is not."""
        if not self.events:
            return "no events"
        if self.missing_books:
            return "lacks " + ", ".join(
                f"{book} on {market}" for market, book in self.missing_books
            )
        return None

    @property
    def ok(self) -> bool:
        return self.units_ok and self.grid_ok and self.lead_ok and self.books_ok


def check_response(manifest: Manifest, response: HistoricalOddsResponse) -> ResponseCheck:
    snapshot = response.snapshot_time
    required = {
        (stored_market(manifest, market), canonical_bookmaker(Source.ODDS_API, book)): 0
        for market, books in manifest.probe.required_bookmakers.items()
        for book in books
    }
    for raw in response.raw_events_data:
        carried = {
            (market["key"], bookmaker["key"])
            for bookmaker in raw.get("bookmakers", [])
            for market in bookmaker.get("markets", [])
        }
        for key in required.keys() & carried:
            required[key] += 1
    return ResponseCheck(
        manifest=manifest,
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
    )
