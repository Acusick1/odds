"""The backfill audit's coverage, call reconciliation and near-kick-off arithmetic."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from odds_analytics.backfill_plan import Band, Call, Fixture, Manifest, Target
from odds_core.quotes import Quote
from odds_core.vocabulary import Market, PriceOutcome, Side, Source, TimeBasis

from scripts import audit_backfill as audit

MANIFEST = Manifest.load(
    Path(__file__).parents[2] / "backfills" / "odds-api-bundesliga-2024-26.yaml"
)
KICKOFF = datetime(2025, 3, 1, 14, 30, tzinfo=UTC)
# A grid time of the manifest's 5-minute grid, 38 s past the mark.
GRID = datetime(2025, 3, 1, 14, 25, 38, tzinfo=UTC)
NEAR = Band(min_lead_minutes=1, max_lead_minutes=120, max_input_age_minutes=10)


def every(start: datetime, end: datetime, step: timedelta) -> list[datetime]:
    out = []
    t = start
    while t <= end:
        out.append(t)
        t += step
    return out


class TestBandCoverage:
    def test_snapshots_every_max_age_serve_every_decision(self) -> None:
        snaps = every(KICKOFF - timedelta(minutes=130, seconds=22), KICKOFF, timedelta(minutes=10))
        cov = audit.band_coverage(KICKOFF, snaps, NEAR)
        assert (cov.served, cov.total) == (120, 120)

    def test_decisions_before_the_first_snapshot_are_not_listed(self) -> None:
        # Listed 60 min before kick-off, then every 5 min: decisions KO-120..KO-61 are unlisted.
        snaps = every(KICKOFF - timedelta(minutes=60), KICKOFF, timedelta(minutes=5))
        cov = audit.band_coverage(KICKOFF, snaps, NEAR)
        assert cov.served == 60
        assert cov.before_listing == 60
        assert cov.in_gaps == 0

    def test_a_missing_snapshot_leaves_decisions_unserved_after_its_max_age(self) -> None:
        start = KICKOFF - timedelta(minutes=125)
        snaps = [
            t
            for t in every(start, KICKOFF, timedelta(minutes=5))
            if t != KICKOFF - timedelta(minutes=60)
        ]
        cov = audit.band_coverage(KICKOFF, snaps, NEAR)
        # With a 5-minute grid and a 10-minute max age, one missing snapshot loses nothing.
        assert cov.served == 120

    def test_unserved_decisions_a_gap_would_have_served_count_as_gap(self) -> None:
        start = KICKOFF - timedelta(minutes=130)
        hole = (KICKOFF - timedelta(minutes=70), KICKOFF - timedelta(minutes=40))
        snaps = [
            t for t in every(start, KICKOFF, timedelta(minutes=10)) if not hole[0] < t < hole[1]
        ]
        gap = audit.Gap(planned=KICKOFF - timedelta(minutes=60), returned=hole[0], next=hole[1])
        cov = audit.band_coverage(KICKOFF, snaps, NEAR, [gap])
        # The 70-minute snapshot serves to KO-60; KO-59..KO-41 are unserved, KO-59..KO-50 within
        # the planned snapshot's max age.
        assert cov.total - cov.served == 19
        assert cov.in_gaps == 10
        assert cov.before_listing == 0

    def test_no_snapshots_leaves_every_decision_unlisted(self) -> None:
        cov = audit.band_coverage(KICKOFF, [], NEAR)
        assert (cov.served, cov.before_listing) == (0, 120)


class TestOffPhase:
    @pytest.mark.parametrize(
        ("offset", "expected"),
        [
            (timedelta(0), False),
            (timedelta(seconds=-38), False),
            (timedelta(seconds=22), False),
            # Stamped across the minute boundary, as 04:26:09 for 04:25:38.
            (timedelta(seconds=31), True),
            (timedelta(seconds=-46), True),
            (timedelta(minutes=2, seconds=19), True),
        ],
    )
    def test_off_phase_is_outside_the_nearest_grid_time_minute(
        self, offset: timedelta, expected: bool
    ) -> None:
        assert audit.off_phase(MANIFEST, GRID + offset) is expected


class TestReconcileCalls:
    def test_a_stored_snapshot_in_the_response_window_stores_the_call(self) -> None:
        rec = audit.reconcile_calls(MANIFEST, [GRID], [GRID + timedelta(seconds=2)], [])
        assert (rec.stored, rec.gap, rec.missing, rec.unplanned) == (1, 0, (), ())

    def test_an_off_phase_snapshot_still_stores_its_call(self) -> None:
        rec = audit.reconcile_calls(MANIFEST, [GRID], [GRID + timedelta(seconds=31)], [])
        assert rec.stored == 1

    def test_a_gap_covers_its_call_and_calls_inside_its_hole(self) -> None:
        returned = GRID - timedelta(hours=3)
        gap = audit.Gap(planned=GRID, returned=returned, next=GRID + timedelta(minutes=12))
        later = GRID + timedelta(minutes=5)
        rec = audit.reconcile_calls(MANIFEST, [GRID, later], [returned], [gap])
        assert (rec.stored, rec.gap, rec.missing) == (0, 2, ())
        assert rec.unplanned == (returned,)

    def test_a_call_with_nothing_stored_is_missing(self) -> None:
        rec = audit.reconcile_calls(MANIFEST, [GRID], [GRID - timedelta(minutes=5)], [])
        assert rec.missing == (GRID,)


class TestCallsBeforeListing:
    def call(self, time: datetime, *event_ids: int) -> Call:
        return Call(
            time,
            tuple(Target(Fixture(e, KICKOFF), NEAR, time, time) for e in event_ids),
        )

    def test_a_call_serving_only_unlisted_fixtures_is_reported(self) -> None:
        listed = {1: GRID + timedelta(hours=1), 2: GRID + timedelta(hours=2)}
        assert audit.calls_before_listing([self.call(GRID, 1, 2)], listed) == [GRID]

    def test_one_listed_fixture_is_enough(self) -> None:
        listed = {1: GRID + timedelta(seconds=3), 2: GRID + timedelta(hours=2)}
        assert audit.calls_before_listing([self.call(GRID, 1, 2)], listed) == []

    def test_a_fixture_never_stored_counts_as_unlisted(self) -> None:
        assert audit.calls_before_listing([self.call(GRID, 1)], {}) == [GRID]


def exchange_quotes(
    snapshot_id: int, time: datetime, prices: dict[PriceOutcome, float]
) -> list[Quote]:
    return [
        Quote(
            snapshot_id=snapshot_id,
            event_id=1,
            snapshot_time=time,
            created_at=time,
            source=Source.ODDS_API,
            market=Market.ONE_X_TWO,
            time_basis=TimeBasis.SOURCE_STAMPED,
            in_play=None,
            market_status=None,
            total_matched=None,
            market_start_time=None,
            bookmaker="betfair_exchange",
            outcome=outcome,
            line=None,
            side=Side.BACK,
            price=price,
            size=None,
            matched=None,
            changed_at=None,
        )
        for outcome, price in prices.items()
    ]


TIGHT = {PriceOutcome.HOME: 2.0, PriceOutcome.DRAW: 3.6, PriceOutcome.AWAY: 4.2}
WIDE = {PriceOutcome.HOME: 1.8, PriceOutcome.DRAW: 3.0, PriceOutcome.AWAY: 3.5}
NO_DRAW = {PriceOutcome.HOME: 2.0, PriceOutcome.AWAY: 4.2}


class TestClosePoint:
    def test_latest_usable_price_at_or_before_kickoff(self) -> None:
        quoted = exchange_quotes(1, KICKOFF - timedelta(minutes=9), TIGHT) + exchange_quotes(
            2, KICKOFF - timedelta(minutes=4), TIGHT
        )
        point = audit.close_point(KICKOFF, quoted, Market.ONE_X_TWO)
        assert point.age == point.usable_age == timedelta(minutes=4)
        assert point.complete

    def test_snapshots_after_kickoff_are_ignored(self) -> None:
        quoted = exchange_quotes(1, KICKOFF - timedelta(minutes=9), TIGHT) + exchange_quotes(
            2, KICKOFF + timedelta(minutes=1), TIGHT
        )
        assert audit.close_point(KICKOFF, quoted, Market.ONE_X_TWO).age == timedelta(minutes=9)

    def test_a_wide_book_is_not_usable(self) -> None:
        quoted = exchange_quotes(1, KICKOFF - timedelta(minutes=9), TIGHT) + exchange_quotes(
            2, KICKOFF - timedelta(minutes=4), WIDE
        )
        point = audit.close_point(KICKOFF, quoted, Market.ONE_X_TWO)
        assert point.age == timedelta(minutes=4)
        assert point.usable_age == timedelta(minutes=9)

    def test_a_missing_outcome_is_incomplete(self) -> None:
        quoted = exchange_quotes(1, KICKOFF - timedelta(minutes=4), NO_DRAW)
        point = audit.close_point(KICKOFF, quoted, Market.ONE_X_TWO)
        assert not point.complete
        assert point.usable_age is None

    def test_no_quotes(self) -> None:
        point = audit.close_point(KICKOFF, [], Market.ONE_X_TWO)
        assert (point.age, point.complete, point.usable_age) == (None, False, None)


class TestSummaries:
    def test_share_within_counts_missing_ages_in_the_denominator(self) -> None:
        ages = [timedelta(minutes=4), timedelta(minutes=7), None, timedelta(minutes=12)]
        assert audit.share_within(ages, 5) == 0.25
        assert audit.share_within(ages, 10) == 0.5

    def test_book_role_and_aliases(self) -> None:
        assert audit.book_role("ladbrokes") == "UK retail"
        assert audit.book_role("betfair_exchange") == "exchange"
        assert audit.book_role("casumo") == "neither"
        assert "betfair_ex_uk" in audit.book_aliases("betfair_exchange")
        assert audit.book_aliases("casumo") == []
