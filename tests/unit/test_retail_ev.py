"""Unit tests for the ADR-0012 retail-EV decision rows."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from odds_analytics.retail_ev import (
    Outcome,
    RetailEvDataset,
    build_event_rows,
    market_outcomes,
    retail_ev_feature_names,
    retail_ev_features,
)
from odds_core.bookmakers import canonical_bookmaker_key
from odds_core.models import Event, OddsSnapshot
from odds_core.odds_math import devig_probabilities
from odds_core.snapshot_sources import FdukTimeBasis, SnapshotKind, snapshot_kind

from tests.test_helpers import make_epl_event, make_tagged_snapshot, three_way_bookmaker

KICKOFF = datetime(2025, 10, 4, 14, 0, tzinfo=UTC)
EVENT_ID = "fduk_ARS_CHE_2025-10-04"
OUTCOMES = market_outcomes("1x2")


def _betfair(prices: tuple[float, float, float], inplay: bool = False) -> dict:
    return three_way_bookmaker(
        "betfair_exchange", prices, betfair_meta={"inplay": inplay, "market_status": "OPEN"}
    )


def _fduk(hours_before: float, basis: FdukTimeBasis, *books: dict) -> OddsSnapshot:
    return make_tagged_snapshot(
        EVENT_ID, KICKOFF - timedelta(hours=hours_before), list(books), "football_data_uk", basis
    )


def _live(source: str, minutes_before: float, *books: dict) -> OddsSnapshot:
    return make_tagged_snapshot(
        EVENT_ID, KICKOFF - timedelta(minutes=minutes_before), list(books), source
    )


def _fair(prices: tuple[float, float, float]) -> dict[Outcome, float]:
    probs = devig_probabilities(*(1 / p for p in prices))
    return dict(zip(OUTCOMES, probs, strict=True))


FDUK_OPEN_BFE = (2.1, 3.6, 4.2)
LIVE_BFE = (2.2, 3.5, 3.8)
LIVE_CLOSE = (2.3, 3.4, 3.6)
FDUK_CLOSE = (2.25, 3.45, 3.7)
OP_BET365 = (2.05, 3.4, 3.9)


@pytest.fixture
def event() -> Event:
    return make_epl_event(EVENT_ID, KICKOFF)


@pytest.fixture
def snapshots() -> list[OddsSnapshot]:
    return [
        _live("oddsportal_live", 8 * 24 * 60, three_way_bookmaker("bet365", (2.0, 3.3, 4.0))),
        _fduk(
            48,
            FdukTimeBasis.COLLECTION_DEADLINE,
            three_way_bookmaker("bet365", (2.0, 3.5, 4.0)),
            three_way_bookmaker("interwetten", (2.0, 3.4, 4.0)),
            three_way_bookmaker("market_avg", (2.0, 3.4, 4.0)),
            three_way_bookmaker("pinnacle", (2.1, 3.5, 4.1)),
            three_way_bookmaker("betfair_exchange", FDUK_OPEN_BFE),
        ),
        _live("oddsportal_live", 10 * 60, three_way_bookmaker("bet365", (2.0, 3.4, 4.0))),
        _live("betfair_api", 4 * 60, _betfair(LIVE_BFE)),
        _live(
            "oddsportal_live",
            3 * 60,
            three_way_bookmaker("bet365", OP_BET365),
            three_way_bookmaker("betfair", (2.0, 3.3, 3.8)),
            three_way_bookmaker("stakecom", (2.1, 3.5, 4.0)),
            three_way_bookmaker("betfair_exchange", (9.0, 9.0, 9.0)),
        ),
        _live("betfair_api", 10, _betfair(LIVE_CLOSE)),
        _live("oddsportal_live", 2, three_way_bookmaker("bet365", (2.1, 3.3, 3.7))),
        _live("betfair_api", 1, _betfair((5.0, 5.0, 1.5), inplay=True)),
        _fduk(
            0,
            FdukTimeBasis.KICKOFF,
            three_way_bookmaker("bet365", (2.2, 3.4, 3.7)),
            three_way_bookmaker("betfair_exchange", FDUK_CLOSE),
        ),
    ]


class TestSnapshotKind:
    @pytest.mark.parametrize(
        ("raw_data", "kind"),
        [
            ({"bookmakers": []}, SnapshotKind.ODDS_API),
            ({"source": "betfair_api"}, SnapshotKind.BETFAIR_LIVE),
            ({"source": "oddsportal_live"}, SnapshotKind.ODDSPORTAL_LIVE),
            (
                {"source": "football_data_uk", "timing": {"basis": "fduk_collection_deadline"}},
                SnapshotKind.FDUK_OPENING,
            ),
            (
                {"source": "football_data_uk", "timing": {"basis": "kickoff"}},
                SnapshotKind.FDUK_CLOSING,
            ),
        ],
    )
    def test_classifies_by_tags(self, raw_data: dict, kind: SnapshotKind) -> None:
        assert snapshot_kind(raw_data) is kind

    def test_unknown_source_raises(self) -> None:
        with pytest.raises(ValueError):
            snapshot_kind({"source": "somewhere_new"})


class TestCanonicalBookmakerKey:
    def test_aliases_map_to_one_key(self) -> None:
        assert canonical_bookmaker_key("betfred_uk", SnapshotKind.ODDS_API) == "betfred"
        assert canonical_bookmaker_key("betfair", SnapshotKind.ODDSPORTAL_LIVE) == (
            "betfair_sportsbook"
        )

    def test_odds_api_betmgm_is_the_us_book(self) -> None:
        assert canonical_bookmaker_key("betmgm", SnapshotKind.ODDS_API) == "betmgm_us"
        assert canonical_bookmaker_key("betmgm", SnapshotKind.ODDSPORTAL) == "betmgm"


class TestBuildEventRows:
    def test_row_set(self, event: Event, snapshots: list[OddsSnapshot]) -> None:
        rows, drops = build_event_rows(event, snapshots, "1x2")

        keys = {(r.retail_kind, r.book, r.outcome) for r in rows}
        assert keys == {
            (kind, book, outcome)
            for kind, book in [
                (SnapshotKind.FDUK_OPENING, "bet365"),
                (SnapshotKind.ODDSPORTAL_LIVE, "bet365"),
                (SnapshotKind.ODDSPORTAL_LIVE, "betfair_sportsbook"),
            ]
            for outcome in OUTCOMES
        }
        assert len(rows) == 9
        assert drops.rows_stale_exchange == 3
        assert drops.rows_without_exchange == 0

    def test_decision_window_bounds(self, event: Event, snapshots: list[OddsSnapshot]) -> None:
        rows, _ = build_event_rows(event, snapshots, "1x2")
        assert all(
            timedelta(minutes=5) <= KICKOFF - r.decision_time <= timedelta(days=7) for r in rows
        )
        assert not any(r.retail_kind is SnapshotKind.FDUK_CLOSING for r in rows)

    def test_live_close_wins_and_in_play_is_ignored(
        self, event: Event, snapshots: list[OddsSnapshot]
    ) -> None:
        rows, _ = build_event_rows(event, snapshots, "1x2")
        assert {r.close.kind for r in rows} == {SnapshotKind.BETFAIR_LIVE}
        assert rows[0].close.fair == pytest.approx(_fair(LIVE_CLOSE))

    def test_fduk_close_without_live_close_in_window(
        self, event: Event, snapshots: list[OddsSnapshot]
    ) -> None:
        early_close = _live("betfair_api", 40, _betfair(LIVE_CLOSE))
        kept = [s for s in snapshots if s.snapshot_time != KICKOFF - timedelta(minutes=10)]
        rows, _ = build_event_rows(event, [*kept, early_close], "1x2")
        assert {r.close.kind for r in rows} == {SnapshotKind.FDUK_CLOSING}
        assert rows[0].close.fair == pytest.approx(_fair(FDUK_CLOSE))
        assert rows[0].close.time == KICKOFF

    def test_event_without_close_is_dropped(
        self, event: Event, snapshots: list[OddsSnapshot]
    ) -> None:
        no_close = [
            s
            for s in snapshots
            if s.snapshot_time < KICKOFF - timedelta(minutes=30)
            or snapshot_kind(s.raw_data) is SnapshotKind.ODDSPORTAL_LIVE
        ]
        rows, drops = build_event_rows(event, no_close, "1x2")
        assert rows == []
        assert drops.events_without_close == 1

    def test_target_and_baseline(self, event: Event, snapshots: list[OddsSnapshot]) -> None:
        rows, _ = build_event_rows(event, snapshots, "1x2")
        row = next(
            r
            for r in rows
            if r.retail_kind is SnapshotKind.ODDSPORTAL_LIVE
            and r.book == "bet365"
            and r.outcome is Outcome.HOME
        )
        assert row.exchange.kind is SnapshotKind.BETFAIR_LIVE
        assert row.exchange_age_hours == pytest.approx(1.0)
        assert row.target == pytest.approx(_fair(LIVE_CLOSE)[Outcome.HOME] * 2.05 - 1)
        assert row.baseline == pytest.approx(_fair(LIVE_BFE)[Outcome.HOME] * 2.05 - 1)

    def test_fduk_opening_is_its_own_exchange_price(
        self, event: Event, snapshots: list[OddsSnapshot]
    ) -> None:
        rows, _ = build_event_rows(event, snapshots, "1x2")
        fduk = [r for r in rows if r.retail_kind is SnapshotKind.FDUK_OPENING]
        assert {r.exchange.kind for r in fduk} == {SnapshotKind.FDUK_OPENING}
        assert fduk[0].exchange.fair == pytest.approx(_fair(FDUK_OPEN_BFE))

    def test_stale_exchange_price_is_dropped(self, event: Event) -> None:
        snapshots = [
            _live("betfair_api", 3 * 60, _betfair(LIVE_BFE)),
            _live("oddsportal_live", 60 - 1, three_way_bookmaker("bet365", OP_BET365)),
            _live("betfair_api", 10, _betfair(LIVE_CLOSE)),
        ]
        rows, drops = build_event_rows(event, snapshots, "1x2")
        assert rows == []
        assert drops.rows_stale_exchange == 3

    def test_wide_exchange_book_is_not_a_price(self, event: Event) -> None:
        snapshots = [
            _live("betfair_api", 3 * 60, _betfair(LIVE_BFE)),
            _live("betfair_api", 2 * 60, _betfair((1.01, 1.5, 2.0))),
            _live("oddsportal_live", 2 * 60 - 1, three_way_bookmaker("bet365", OP_BET365)),
            _live("betfair_api", 10, _betfair(LIVE_CLOSE)),
        ]
        rows, drops = build_event_rows(event, snapshots, "1x2")
        assert drops.wide_exchange_snapshots == 1
        assert {r.exchange.fair[Outcome.HOME] for r in rows} == {_fair(LIVE_BFE)[Outcome.HOME]}


class TestNoLookAhead:
    def test_rows_depend_only_on_data_at_or_before_t(
        self, event: Event, snapshots: list[OddsSnapshot]
    ) -> None:
        rows, _ = build_event_rows(event, snapshots, "1x2")
        assert rows
        for row in rows:
            closes = [
                s
                for s in snapshots
                if s.snapshot_time > row.decision_time
                and snapshot_kind(s.raw_data)
                in {SnapshotKind.BETFAIR_LIVE, SnapshotKind.FDUK_CLOSING}
                and s.snapshot_time >= KICKOFF - timedelta(minutes=30)
            ]
            past = [s for s in snapshots if s.snapshot_time <= row.decision_time]
            truncated, _ = build_event_rows(event, past + closes, "1x2")
            match = next(
                r
                for r in truncated
                if (r.decision_time, r.retail_kind, r.book, r.outcome)
                == (row.decision_time, row.retail_kind, row.book, row.outcome)
            )
            np.testing.assert_array_equal(
                retail_ev_features(row, OUTCOMES), retail_ev_features(match, OUTCOMES)
            )
            assert row.exchange.time <= row.decision_time

    def test_future_exchange_move_does_not_reach_earlier_rows(
        self, event: Event, snapshots: list[OddsSnapshot]
    ) -> None:
        rows, _ = build_event_rows(event, snapshots, "1x2")
        shock = _live("betfair_api", 3 * 60 - 1, _betfair((1.2, 8.0, 15.0)))
        shocked, _ = build_event_rows(event, [*snapshots, shock], "1x2")
        early = [r for r in rows if r.decision_time <= KICKOFF - timedelta(hours=3)]
        early_shocked = [r for r in shocked if r.decision_time <= KICKOFF - timedelta(hours=3)]
        for a, b in zip(early, early_shocked, strict=True):
            np.testing.assert_array_equal(
                retail_ev_features(a, OUTCOMES), retail_ev_features(b, OUTCOMES)
            )


class TestFeaturesAndWeights:
    def test_feature_vector_matches_names(
        self, event: Event, snapshots: list[OddsSnapshot]
    ) -> None:
        rows, _ = build_event_rows(event, snapshots, "1x2")
        names = retail_ev_feature_names(OUTCOMES)
        row = rows[-1]
        values = dict(zip(names, retail_ev_features(row, OUTCOMES), strict=True))
        assert values["hours_to_kickoff"] == pytest.approx(3.0)
        assert values[f"outcome_{row.outcome.value}"] == 1.0
        assert values[f"book_{row.book}"] == 1.0
        assert values["baseline_ev"] == pytest.approx(row.baseline)
        assert values["retail_implied_prob"] == pytest.approx(1 / row.retail_decimal)

    def test_loss_weight_is_inverse_odds_squared_over_event_rows(
        self, event: Event, snapshots: list[OddsSnapshot]
    ) -> None:
        rows, _ = build_event_rows(event, snapshots, "1x2")
        dataset = RetailEvDataset(rows=rows)
        odds = np.array([r.retail_decimal for r in rows])
        np.testing.assert_allclose(dataset.event_weights, 1 / len(rows))
        np.testing.assert_allclose(dataset.loss_weights, 1 / odds**2 / len(rows))
