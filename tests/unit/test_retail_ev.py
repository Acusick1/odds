"""Unit tests for the ADR-0012 retail-EV decision rows."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import get_args

import numpy as np
import pandas as pd
import pytest
from odds_analytics.as_of_team_features import TeamHistory, results_known_before
from odds_analytics.retail_ev import (
    _FEATURE_GROUPS,
    LOSS_WEIGHT_SCALE,
    RETAIL_KINDS,
    Outcome,
    RetailEvDataset,
    RetailEvFeatures,
    RetailEvRow,
    build_event_rows,
    market_outcomes,
)
from odds_analytics.standings_features import epl_season_key
from odds_analytics.training.config import RetailEvFeatureGroupName
from odds_core.bookmakers import canonical_bookmaker_key
from odds_core.models import Event, EventStatus, OddsSnapshot
from odds_core.odds_math import devig_probabilities
from odds_core.snapshot_sources import FdukTimeBasis, SnapshotKind, snapshot_kind

from tests.test_helpers import make_epl_event, make_tagged_snapshot, three_way_bookmaker

KICKOFF = datetime(2025, 10, 4, 14, 0, tzinfo=UTC)
EVENT_ID = "fduk_ARS_CHE_2025-10-04"
OUTCOMES = market_outcomes("1x2")
PRICE_GROUPS = RetailEvFeatures(OUTCOMES, ("polymarket_at_t", "dynamics"))


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


PM_EARLY = (2.0, 3.5, 4.5)
PM_LATE = (2.2, 3.6, 3.9)
BFE_400 = (2.0, 3.6, 4.4)
BFE_300 = (2.1, 3.6, 4.0)
OP_250_BOOKS = (
    three_way_bookmaker("bet365", (2.0, 3.4, 4.1)),
    three_way_bookmaker("williamhill", (2.1, 3.3, 4.0)),
)


def _polymarket(
    minutes_before: float, prices: tuple[float, float, float], **meta: dict
) -> OddsSnapshot:
    probs = dict(zip(("home", "draw", "away"), (1 / p for p in prices), strict=True))
    return _live(
        "polymarket",
        minutes_before,
        three_way_bookmaker(
            "polymarket", prices, polymarket_meta={"yes_probabilities": probs, **meta}
        ),
    )


@pytest.fixture
def rich_snapshots(snapshots: list[OddsSnapshot]) -> list[OddsSnapshot]:
    """The base snapshots plus Polymarket prices and denser exchange and retail history."""
    return [
        *snapshots,
        _live("betfair_api", 400, _betfair(BFE_400)),
        _live("betfair_api", 300, _betfair(BFE_300)),
        _live("oddsportal_live", 250, *OP_250_BOOKS),
        _polymarket(260, PM_EARLY),
        _polymarket(245, (2.4, 3.3, 3.2)),
        _live("oddsportal_live", 230, three_way_bookmaker("bet365", (1.9, 3.6, 4.3))),
        _live("oddsportal", 245, three_way_bookmaker("bet365", (1.8, 3.7, 4.6))),
        _polymarket(
            190,
            PM_LATE,
            best_bid={"home": 0.44, "draw": 0.27, "away": 0.25},
            best_ask={"home": 0.46, "draw": 0.28, "away": 0.27},
        ),
        _polymarket(-1, (1.5, 4.0, 6.0)),
    ]


def _row(
    rows: list[RetailEvRow], minutes_before: float, book: str, outcome: Outcome
) -> RetailEvRow:
    return next(
        r
        for r in rows
        if r.decision_time == KICKOFF - timedelta(minutes=minutes_before)
        and r.book == book
        and r.outcome is outcome
    )


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
        assert drops.prices_outside_roster == {
            "interwetten": 3,
            "market_avg": 3,
            "pinnacle": 3,
            "betfair_exchange": 6,
            "stakecom": 3,
        }

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
        self, event: Event, rich_snapshots: list[OddsSnapshot]
    ) -> None:
        snapshots = rich_snapshots
        rows, _ = build_event_rows(event, snapshots, "1x2")
        assert rows
        closes = [
            s
            for s in snapshots
            if snapshot_kind(s.raw_data) in {SnapshotKind.BETFAIR_LIVE, SnapshotKind.FDUK_CLOSING}
            and s.snapshot_time >= KICKOFF - timedelta(minutes=30)
        ]

        def at(built: list, t: datetime) -> dict:
            return {
                (r.retail_kind, r.book, r.outcome): PRICE_GROUPS.row(r)
                for r in built
                if r.decision_time == t
            }

        retail_times = {
            s.snapshot_time
            for s in snapshots
            if snapshot_kind(s.raw_data) in RETAIL_KINDS
            and timedelta(minutes=5) <= KICKOFF - s.snapshot_time <= timedelta(days=7)
        }
        assert {r.decision_time for r in rows} < retail_times
        for t in sorted(retail_times):
            past = [s for s in snapshots if s.snapshot_time <= t]
            truncated, _ = build_event_rows(
                event, past + [c for c in closes if c.snapshot_time > t], "1x2"
            )
            full_at_t, truncated_at_t = at(rows, t), at(truncated, t)
            assert full_at_t.keys() == truncated_at_t.keys()
            for key, features in full_at_t.items():
                np.testing.assert_array_equal(features, truncated_at_t[key])
        assert all(r.exchange.time <= r.decision_time for r in rows)
        assert all(r.polymarket.time <= r.decision_time for r in rows if r.polymarket)
        assert any(r.polymarket for r in rows)
        assert any(m is not None for r in rows for m in r.dynamics.exchange_moves)
        assert any(m is not None for r in rows for m in r.dynamics.consensus_moves)

    def test_future_exchange_move_does_not_reach_earlier_rows(
        self, event: Event, snapshots: list[OddsSnapshot]
    ) -> None:
        rows, _ = build_event_rows(event, snapshots, "1x2")
        shock = _live("betfair_api", 3 * 60 - 1, _betfair((1.2, 8.0, 15.0)))
        shocked, _ = build_event_rows(event, [*snapshots, shock], "1x2")
        early = [r for r in rows if r.decision_time <= KICKOFF - timedelta(hours=3)]
        early_shocked = [r for r in shocked if r.decision_time <= KICKOFF - timedelta(hours=3)]
        for a, b in zip(early, early_shocked, strict=True):
            np.testing.assert_array_equal(PRICE_GROUPS.row(a), PRICE_GROUPS.row(b))


class TestFeaturesAndWeights:
    def test_feature_vector_matches_names(
        self, event: Event, snapshots: list[OddsSnapshot]
    ) -> None:
        rows, _ = build_event_rows(event, snapshots, "1x2")
        features = RetailEvFeatures(OUTCOMES)
        row = rows[-1]
        values = dict(zip(features.names, features.row(row), strict=True))
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
        np.testing.assert_allclose(dataset.loss_weights, LOSS_WEIGHT_SCALE / odds**2 / len(rows))

    def test_loss_weights_do_not_depend_on_other_events(
        self, event: Event, snapshots: list[OddsSnapshot]
    ) -> None:
        rows, _ = build_event_rows(event, snapshots, "1x2")
        dense_rows = [replace(r, event_id="dense") for r in rows for _ in range(10)]
        alone = RetailEvDataset(rows=rows).loss_weights
        with_dense = RetailEvDataset(rows=rows + dense_rows).loss_weights
        np.testing.assert_array_equal(with_dense[: len(rows)], alone)


class TestRowLabels:
    @pytest.mark.parametrize(
        ("hours", "bucket"),
        [
            (0.1, "0-1h"),
            (1.0, "1-6h"),
            (23.9, "6-24h"),
            (24.0, "1-3d"),
            (72.0, "3-7d"),
            (168.0, "3-7d"),
        ],
    )
    def test_horizon_bucket_by_hours_to_kickoff(
        self, event: Event, snapshots: list[OddsSnapshot], hours: float, bucket: str
    ) -> None:
        rows, _ = build_event_rows(event, snapshots, "1x2")
        row = replace(rows[0], decision_time=rows[0].commence_time - timedelta(hours=hours))
        assert row.horizon_bucket == bucket

    @pytest.mark.parametrize(
        ("kickoff", "season"),
        [
            (datetime(2025, 8, 15, 19, tzinfo=UTC), "2025-26"),
            (datetime(2026, 5, 24, 15, tzinfo=UTC), "2025-26"),
            (datetime(2026, 7, 1, 12, tzinfo=UTC), "2026-27"),
        ],
    )
    def test_season_starts_in_july(
        self, event: Event, snapshots: list[OddsSnapshot], kickoff: datetime, season: str
    ) -> None:
        rows, _ = build_event_rows(event, snapshots, "1x2")
        assert replace(rows[0], commence_time=kickoff).season == season

    @pytest.mark.parametrize("hours", [-0.1, 168.1])
    def test_horizon_bucket_rejects_hours_outside_decision_window(
        self, event: Event, snapshots: list[OddsSnapshot], hours: float
    ) -> None:
        rows, _ = build_event_rows(event, snapshots, "1x2")
        row = replace(rows[0], decision_time=rows[0].commence_time - timedelta(hours=hours))
        with pytest.raises(ValueError, match="outside the decision window"):
            _ = row.horizon_bucket


class TestPolymarketAtT:
    def test_latest_price_within_age_cap(self, event: Event, rich_snapshots: list) -> None:
        rows, _ = build_event_rows(event, rich_snapshots, "1x2")
        early = _row(rows, 250, "bet365", Outcome.HOME)
        assert early.polymarket is not None
        assert early.polymarket.fair == pytest.approx(_fair(PM_EARLY))
        assert early.polymarket.spread == {}
        late = _row(rows, 180, "bet365", Outcome.AWAY)
        assert late.polymarket is not None
        assert late.polymarket.fair == pytest.approx(_fair(PM_LATE))
        assert late.polymarket.spread[Outcome.AWAY] == pytest.approx(0.02)

    def test_no_price_older_than_cap(self, event: Event, rich_snapshots: list) -> None:
        rows, _ = build_event_rows(event, rich_snapshots, "1x2")
        assert _row(rows, 48 * 60, "bet365", Outcome.HOME).polymarket is None
        late_only = {KICKOFF - timedelta(minutes=m) for m in (190, 245, 260)}
        kept = [s for s in rich_snapshots if s.snapshot_time not in late_only]
        kept.append(_polymarket(330, PM_EARLY))
        rows, _ = build_event_rows(event, kept, "1x2")
        assert _row(rows, 250, "bet365", Outcome.HOME).polymarket is not None
        assert _row(rows, 180, "bet365", Outcome.HOME).polymarket is None

    def test_features(self, event: Event, rich_snapshots: list) -> None:
        rows, _ = build_event_rows(event, rich_snapshots, "1x2")
        row = _row(rows, 180, "bet365", Outcome.AWAY)
        features = RetailEvFeatures(OUTCOMES, ("polymarket_at_t",))
        values = dict(zip(features.names, features.row(row), strict=True))
        pm = _fair(PM_LATE)[Outcome.AWAY]
        assert values["pm_prob"] == pytest.approx(pm)
        assert values["pm_exchange_diff"] == pytest.approx(pm - row.exchange.fair[Outcome.AWAY])
        assert values["pm_consensus_diff"] == pytest.approx(pm - row.consensus_prob)
        assert values["pm_baseline_ev"] == pytest.approx(pm * row.retail_decimal - 1)
        assert values["pm_spread"] == pytest.approx(0.02)
        assert values["pm_age_hours"] == pytest.approx(10 / 60)
        missing = _row(rows, 48 * 60, "bet365", Outcome.AWAY)
        assert np.isnan(features.row(missing)[-6:]).all()


class TestDynamics:
    def test_moves_take_the_latest_price_at_or_before_t_minus_window(
        self, event: Event, rich_snapshots: list
    ) -> None:
        rows, _ = build_event_rows(event, rich_snapshots, "1x2")
        row = _row(rows, 180, "bet365", Outcome.HOME)
        now = _fair(LIVE_BFE)[Outcome.HOME]
        one_h, three_h, six_h = row.dynamics.exchange_moves
        assert one_h is None, "the latest exchange price at t - 1h is the price at t itself"
        assert three_h == pytest.approx(now - _fair(BFE_400)[Outcome.HOME])
        assert six_h is None, "only the FDUK opening price lies within the cap of t - 6h"

    def test_consensus_moves_compare_books_of_one_source(
        self, event: Event, rich_snapshots: list
    ) -> None:
        rows, _ = build_event_rows(event, rich_snapshots, "1x2")
        row = _row(rows, 180, "bet365", Outcome.HOME)
        bet365_move = 1 / 2.05 - 1 / 2.0
        assert row.dynamics.consensus_moves[0] == pytest.approx(bet365_move)
        assert row.dynamics.consensus_moves[1] is None
        assert row.dynamics.consensus_moves[2] == pytest.approx(bet365_move)

    def test_exchange_move_never_crosses_source(self, event: Event, rich_snapshots: list) -> None:
        live_near_fduk = [
            _live("betfair_api", 43 * 60, _betfair(LIVE_BFE)),
            _live("oddsportal_live", 42 * 60, three_way_bookmaker("bet365", OP_BET365)),
        ]
        rows, _ = build_event_rows(event, [*rich_snapshots, *live_near_fduk], "1x2")
        row = _row(rows, 42 * 60, "bet365", Outcome.HOME)
        assert row.exchange.kind is SnapshotKind.BETFAIR_LIVE
        assert row.dynamics.exchange_moves == (None, None, None), (
            "the FDUK opening price lies within the cap of t - 6h but is another source"
        )

    def test_fduk_opening_has_no_moves(self, event: Event, rich_snapshots: list) -> None:
        rows, _ = build_event_rows(event, rich_snapshots, "1x2")
        row = _row(rows, 48 * 60, "bet365", Outcome.HOME)
        assert row.dynamics.exchange_moves == (None, None, None)
        assert row.dynamics.consensus_moves == (None, None, None)

    def test_dispersion_across_uk_books(self, event: Event, rich_snapshots: list) -> None:
        rows, _ = build_event_rows(event, rich_snapshots, "1x2")
        row = _row(rows, 180, "bet365", Outcome.HOME)
        assert row.dynamics.retail_dispersion == pytest.approx(np.std([1 / 2.05, 1 / 2.0]))
        single = _row(rows, 48 * 60, "bet365", Outcome.HOME)
        assert single.dynamics.retail_dispersion is None

    def test_features_match_names(self, event: Event, rich_snapshots: list) -> None:
        rows, _ = build_event_rows(event, rich_snapshots, "1x2")
        features = RetailEvFeatures(OUTCOMES, ("dynamics",))
        values = dict(zip(features.names, features.row(rows[0]), strict=True))
        assert set(values) >= {
            "exchange_move_1h",
            "exchange_move_3h",
            "exchange_move_6h",
            "consensus_move_1h",
            "consensus_move_3h",
            "consensus_move_6h",
            "retail_dispersion",
        }


MATCH_LENGTH = timedelta(hours=2)
TEAM_T = (
    KICKOFF - timedelta(days=7),
    KICKOFF - timedelta(days=4),
    datetime(2025, 10, 1, 20, 30, tzinfo=UTC),
    datetime(2025, 10, 1, 21, 0, tzinfo=UTC),
    datetime(2025, 10, 1, 21, 30, tzinfo=UTC),
    datetime(2025, 10, 1, 23, 30, tzinfo=UTC),
    KICKOFF - timedelta(days=1),
    KICKOFF - timedelta(minutes=5),
)


def _final(event_id: str, kickoff: datetime, home: str, away: str, score: tuple[int, int]) -> Event:
    return make_epl_event(
        event_id,
        kickoff,
        home,
        away,
        status=EventStatus.FINAL,
        home_score=score[0],
        away_score=score[1],
    )


@pytest.fixture
def season(event: Event) -> list[Event]:
    """Finished matches around the event, including a midweek match between t and kick-off."""
    played = make_epl_event(
        event.id,
        KICKOFF,
        event.home_team,
        event.away_team,
        status=EventStatus.FINAL,
        home_score=2,
        away_score=0,
    )
    return [
        _final("m1", datetime(2025, 9, 20, 14, 0, tzinfo=UTC), "Arsenal", "Everton", (1, 0)),
        _final("m2", datetime(2025, 9, 21, 16, 30, tzinfo=UTC), "Chelsea", "Fulham", (0, 0)),
        _final("m3", datetime(2025, 9, 27, 14, 0, tzinfo=UTC), "Fulham", "Arsenal", (2, 1)),
        _final("m4", datetime(2025, 9, 28, 14, 0, tzinfo=UTC), "Everton", "Chelsea", (1, 3)),
        _final("m5", datetime(2025, 10, 1, 19, 45, tzinfo=UTC), "Arsenal", "Fulham", (4, 0)),
        played,
        _final("m6", datetime(2025, 10, 18, 14, 0, tzinfo=UTC), "Chelsea", "Arsenal", (1, 1)),
    ]


@pytest.fixture
def fixtures() -> pd.DataFrame:
    rows = [
        ("2025-09-27T14:00:00Z", "Arsenal", "Fulham", "Premier League", "Full Time"),
        ("2025-09-28T14:00:00Z", "Chelsea", "Everton", "Premier League", "Full Time"),
        ("2025-09-30T19:00:00Z", "Chelsea", "Benfica", "Champions League", "Full Time"),
        ("2025-10-01T19:30:00Z", "Chelsea", "Brighton", "League Cup", "Postponed"),
        ("2025-10-01T19:45:00Z", "Arsenal", "Fulham", "Premier League", "Full Time"),
        ("2025-10-04T14:00:00Z", "Arsenal", "Chelsea", "Premier League", "Full Time"),
        ("2025-10-07T19:00:00Z", "Chelsea", "Ajax", "Champions League", "Scheduled"),
    ]
    df = pd.DataFrame(rows, columns=["date", "team", "opponent", "competition", "status"])
    df["date"] = pd.to_datetime(df["date"], utc=True)
    return df


def _history(events: list[Event], fixtures: pd.DataFrame) -> TeamHistory:
    cache: dict[str, list[Event]] = {}
    for e in sorted(events, key=lambda e: e.commence_time):
        cache.setdefault(epl_season_key(e.commence_time), []).append(e)
    return TeamHistory(season_results=cache, fixtures=fixtures, form_window=5)


def _known_at(
    t: datetime, season: list[Event], fixtures: pd.DataFrame
) -> tuple[list[Event], pd.DataFrame]:
    """Only matches that have finished by t, as a database written at t would hold."""
    finished = [e for e in season if e.commence_time + MATCH_LENGTH <= t]
    return finished, fixtures.loc[fixtures["date"] + MATCH_LENGTH <= t]


class TestTeamFeaturesAsOfT:
    def test_rebuilt_from_matches_finished_by_t(
        self, event: Event, season: list[Event], fixtures: pd.DataFrame
    ) -> None:
        full = _history(season, fixtures)
        for t in TEAM_T:
            truncated = _history(*_known_at(t, season, fixtures))
            np.testing.assert_array_equal(
                full.schedule_at(event, t).to_array(), truncated.schedule_at(event, t).to_array()
            )
            np.testing.assert_array_equal(
                full.standings_at(event, t).to_array(), truncated.standings_at(event, t).to_array()
            )

    def test_midweek_match_counts_only_once_its_result_is_known(
        self, event: Event, season: list[Event], fixtures: pd.DataFrame
    ) -> None:
        history = _history(season, fixtures)
        in_play, after = TEAM_T[2], TEAM_T[5]
        assert history.schedule_at(event, in_play).home_rest_days == pytest.approx(7.0)
        assert history.schedule_at(event, after).home_rest_days == pytest.approx(
            (KICKOFF - datetime(2025, 10, 1, 19, 45, tzinfo=UTC)).total_seconds() / 86400
        )
        assert history.standings_at(event, in_play).home_goal_difference == 0.0
        assert history.standings_at(event, after).home_goal_difference == 4.0
        assert history.schedule_at(event, TEAM_T[1]).away_european_last_7d == 0.0
        assert history.schedule_at(event, TEAM_T[-1]).away_european_last_7d == 1.0

    def test_result_known_three_hours_after_kick_off(self) -> None:
        assert results_known_before(KICKOFF) == KICKOFF - timedelta(hours=3)

    def test_unplayed_fixture_rows_do_not_count(
        self, event: Event, season: list[Event], fixtures: pd.DataFrame
    ) -> None:
        features = _history(season, fixtures).schedule_at(event, TEAM_T[-1])
        assert features.away_matches_last_14d == 2.0

    def test_team_without_played_fixtures_has_no_congestion(
        self, season: list[Event], fixtures: pd.DataFrame
    ) -> None:
        newcomer = make_epl_event("new", KICKOFF, "Brentford", "Chelsea")
        features = _history(season, fixtures).schedule_at(newcomer, TEAM_T[-1])
        assert features.home_matches_last_14d is None
        assert features.home_european_last_7d is None
        assert features.away_matches_last_14d == 2.0

    def test_feature_vector_includes_team_groups(
        self,
        event: Event,
        snapshots: list[OddsSnapshot],
        season: list[Event],
        fixtures: pd.DataFrame,
    ) -> None:
        rows, _ = build_event_rows(event, snapshots, "1x2")
        features = RetailEvFeatures(
            OUTCOMES,
            ("standings_as_of_t", "schedule_as_of_t"),
            teams=_history(season, fixtures),
            events={event.id: event},
        )
        values = dict(zip(features.names, features.row(rows[-1]), strict=True))
        assert values["eplsched_home_rest_days"] == pytest.approx(
            (KICKOFF - datetime(2025, 10, 1, 19, 45, tzinfo=UTC)).total_seconds() / 86400
        )
        assert values["stnd_home_goal_difference"] == 4.0
        assert features.names.index("eplsched_home_rest_days") < features.names.index(
            "stnd_home_goal_difference"
        )

    def test_team_groups_need_history(self) -> None:
        with pytest.raises(ValueError, match="teams"):
            RetailEvFeatures(OUTCOMES, ("schedule_as_of_t",))


class TestFeatureGroupTable:
    def test_every_config_group_has_an_entry(self) -> None:
        assert set(_FEATURE_GROUPS) == set(get_args(RetailEvFeatureGroupName))


class TestUpperBoundDecisionTime:
    def test_fduk_opening_row_takes_no_polymarket_price(
        self, event: Event, rich_snapshots: list
    ) -> None:
        pm_before_fduk = _polymarket(48 * 60 + 30, PM_EARLY)
        rows, _ = build_event_rows(event, [*rich_snapshots, pm_before_fduk], "1x2")
        fduk = _row(rows, 48 * 60, "bet365", Outcome.HOME)
        live = _row(rows, 180, "bet365", Outcome.HOME)
        assert not fduk.decision_time_is_exact
        assert fduk.polymarket is None
        assert live.polymarket is not None

    def test_fduk_opening_row_pairs_only_with_its_own_exchange_price(self, event: Event) -> None:
        snapshots = [
            _live("betfair_api", 49 * 60, _betfair(LIVE_BFE)),
            _fduk(48, FdukTimeBasis.COLLECTION_DEADLINE, three_way_bookmaker("bet365", OP_BET365)),
            _live("betfair_api", 10, _betfair(LIVE_CLOSE)),
        ]
        rows, drops = build_event_rows(event, snapshots, "1x2")
        assert rows == []
        assert drops.rows_without_own_exchange == 3
        assert drops.rows_stale_exchange == 0

    def test_fduk_opening_row_has_no_team_features(
        self,
        event: Event,
        snapshots: list[OddsSnapshot],
        season: list[Event],
        fixtures: pd.DataFrame,
    ) -> None:
        rows, _ = build_event_rows(event, snapshots, "1x2")
        fduk = _row(rows, 48 * 60, "bet365", Outcome.HOME)
        live = _row(rows, 180, "bet365", Outcome.HOME)
        features = RetailEvFeatures(
            OUTCOMES,
            ("schedule_as_of_t", "standings_as_of_t"),
            teams=_history(season, fixtures),
            events={event.id: event},
        )
        team_columns = slice(len(features.names) - 19, None)
        assert np.isnan(features.row(fduk)[team_columns]).all()
        assert not np.isnan(features.row(live)[team_columns][:4]).any()
