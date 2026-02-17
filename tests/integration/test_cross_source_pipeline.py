"""Integration tests for the cross-source Polymarket feature pipeline.

Verifies that the 5-layer adapter pipeline correctly:
- NaN-fills Polymarket features for events without PM data (keeps row)
- Extracts PM features for events with complete PM + SB data
- Integrates with prepare_training_data() via EventDataBundle
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from odds_analytics.feature_groups import (
    XGBoostAdapter,
    collect_event_data,
    prepare_training_data,
)
from odds_analytics.training import FeatureConfig, SamplingConfig
from odds_core.models import Event, EventStatus, Odds, OddsSnapshot
from odds_core.polymarket_models import (
    PolymarketEvent,
    PolymarketMarket,
    PolymarketMarketType,
    PolymarketPriceSnapshot,
)
from odds_lambda.fetch_tier import FetchTier

pytestmark = pytest.mark.integration


# =============================================================================
# Test data helpers
# =============================================================================


def make_sb_event(
    idx: int,
    commence_time: datetime,
    home_team: str,
    away_team: str,
    final: bool = True,
) -> Event:
    return Event(
        id=f"test_event_{idx}",
        sport_key="basketball_nba",
        sport_title="NBA",
        commence_time=commence_time,
        home_team=home_team,
        away_team=away_team,
        status=EventStatus.FINAL if final else EventStatus.SCHEDULED,
        home_score=110 if final else None,
        away_score=105 if final else None,
        completed_at=commence_time + timedelta(hours=3) if final else None,
    )


def make_odds_snapshots(
    event_id: str,
    commence_time: datetime,
    home_team: str,
    away_team: str,
) -> list[OddsSnapshot]:
    """Create SB snapshots at early, pregame, and closing tiers."""
    snapshots = []
    bookmakers = ["pinnacle", "fanduel", "draftkings"]
    for tier_name, hours_before in [("early", 48.0), ("pregame", 7.5), ("closing", 0.5)]:
        t = commence_time - timedelta(hours=hours_before)
        raw_data = {
            "bookmakers": [
                {
                    "key": bk,
                    "title": bk.title(),
                    "last_update": t.isoformat(),
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": home_team, "price": -130},
                                {"name": away_team, "price": 110},
                            ],
                        }
                    ],
                }
                for bk in bookmakers
            ]
        }
        snapshots.append(
            OddsSnapshot(
                event_id=event_id,
                snapshot_time=t,
                raw_data=raw_data,
                bookmaker_count=len(bookmakers),
                fetch_tier=tier_name,
                hours_until_commence=hours_before,
            )
        )
    return snapshots


def make_odds_records(
    event_id: str,
    commence_time: datetime,
    home_team: str,
    away_team: str,
) -> list[Odds]:
    """Create Odds records aligned with snapshot times."""
    records = []
    bookmakers = ["pinnacle", "fanduel", "draftkings"]
    for hours_before in [48.0, 7.5, 0.5]:
        t = commence_time - timedelta(hours=hours_before)
        for bk in bookmakers:
            records.extend(
                [
                    Odds(
                        event_id=event_id,
                        bookmaker_key=bk,
                        bookmaker_title=bk.title(),
                        market_key="h2h",
                        outcome_name=home_team,
                        price=-130,
                        point=None,
                        odds_timestamp=t,
                        last_update=t,
                    ),
                    Odds(
                        event_id=event_id,
                        bookmaker_key=bk,
                        bookmaker_title=bk.title(),
                        market_key="h2h",
                        outcome_name=away_team,
                        price=110,
                        point=None,
                        odds_timestamp=t,
                        last_update=t,
                    ),
                ]
            )
    return records


def make_pm_event(event_id: str, commence_time: datetime) -> PolymarketEvent:
    return PolymarketEvent(
        pm_event_id=f"pm_{event_id}",
        ticker=f"nba-test-{event_id}",
        slug=f"test-{event_id}",
        title=f"Test event {event_id}",
        event_id=event_id,
        start_date=commence_time - timedelta(days=7),
        end_date=commence_time + timedelta(hours=3),
        active=False,
        closed=True,
        volume=5000.0,
        liquidity=10000.0,
        markets_count=1,
    )


def make_pm_market(pm_event_id: int, home_team: str, away_team: str) -> PolymarketMarket:
    home_alias = home_team.split()[-1]
    away_alias = away_team.split()[-1]
    return PolymarketMarket(
        polymarket_event_id=pm_event_id,
        pm_market_id=f"market_{pm_event_id}",
        condition_id=f"cond_{pm_event_id}",
        question=f"Will {home_alias} win?",
        clob_token_ids=[f"tok_{pm_event_id}_0", f"tok_{pm_event_id}_1"],
        outcomes=[home_alias, away_alias],
        market_type=PolymarketMarketType.MONEYLINE,
    )


def make_pm_price_snapshots(
    market_id: int,
    commence_time: datetime,
    home_prob: float = 0.62,
) -> list[PolymarketPriceSnapshot]:
    """Create PM price snapshots in velocity window + at decision time (~7.5h before game)."""
    snapshots = []
    base_prob = home_prob - 0.04
    decision_time = commence_time - timedelta(hours=7.5)
    for i in range(4):
        t = decision_time - timedelta(minutes=90 - i * 30)
        prob = round(base_prob + i * 0.01, 4)
        snapshots.append(
            PolymarketPriceSnapshot(
                polymarket_market_id=market_id,
                snapshot_time=t,
                outcome_0_price=prob,
                outcome_1_price=round(1.0 - prob, 4),
                best_bid=round(prob - 0.01, 4),
                best_ask=round(prob + 0.01, 4),
                spread=0.02,
                midpoint=prob,
                volume=1000.0,
                liquidity=5000.0,
                fetch_tier="pregame",
                hours_until_commence=(decision_time - t).total_seconds() / 3600 + 7.5,
            )
        )
    snapshots.append(
        PolymarketPriceSnapshot(
            polymarket_market_id=market_id,
            snapshot_time=decision_time,
            outcome_0_price=home_prob,
            outcome_1_price=round(1.0 - home_prob, 4),
            best_bid=round(home_prob - 0.01, 4),
            best_ask=round(home_prob + 0.01, 4),
            spread=0.02,
            midpoint=home_prob,
            volume=2000.0,
            liquidity=8000.0,
            fetch_tier="pregame",
            hours_until_commence=7.5,
        )
    )
    return snapshots


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
async def cross_source_test_data(pglite_async_session):
    """
    Seed 3 events:
    - event_0: has SB + PM data (Lakers vs Celtics)
    - event_1: has SB + PM data (Warriors vs Heat)
    - event_2: has SB data only (no linked PM event)
    """
    base_time = datetime(2026, 1, 20, 19, 0, tzinfo=UTC)
    games = [
        ("Los Angeles Lakers", "Boston Celtics"),
        ("Golden State Warriors", "Miami Heat"),
        ("Denver Nuggets", "Dallas Mavericks"),  # SB only
    ]

    events = []
    pm_events = []
    pm_markets = []

    for i, (home, away) in enumerate(games):
        commence = base_time + timedelta(days=i)
        sb_event = make_sb_event(i, commence, home, away)
        pglite_async_session.add(sb_event)
        events.append(sb_event)

        for snap in make_odds_snapshots(sb_event.id, commence, home, away):
            pglite_async_session.add(snap)
        for odds in make_odds_records(sb_event.id, commence, home, away):
            pglite_async_session.add(odds)

    await pglite_async_session.flush()

    for i in range(2):
        home, away = games[i]
        commence = base_time + timedelta(days=i)
        pm_ev = make_pm_event(events[i].id, commence)
        pglite_async_session.add(pm_ev)
        pm_events.append(pm_ev)

    await pglite_async_session.flush()

    for i, pm_ev in enumerate(pm_events):
        home, away = games[i]
        pm_mkt = make_pm_market(pm_ev.id, home, away)
        pglite_async_session.add(pm_mkt)
        pm_markets.append(pm_mkt)

    await pglite_async_session.flush()

    for i, pm_mkt in enumerate(pm_markets):
        home_prob = 0.62 if i == 0 else 0.45
        commence = base_time + timedelta(days=i)
        for snap in make_pm_price_snapshots(pm_mkt.id, commence, home_prob):
            pglite_async_session.add(snap)

    await pglite_async_session.commit()

    for e in events:
        await pglite_async_session.refresh(e)

    return {
        "events": events,
        "pm_events": pm_events,
        "pm_markets": pm_markets,
    }


# =============================================================================
# Tests
# =============================================================================


class TestCollectEventData:
    """Integration tests for collect_event_data (Layer 1)."""

    def _make_config(self, with_pm: bool = True) -> FeatureConfig:
        groups = ("tabular", "polymarket") if with_pm else ("tabular",)
        return FeatureConfig(
            sharp_bookmakers=["pinnacle"],
            retail_bookmakers=["fanduel", "draftkings"],
            markets=["h2h"],
            outcome="home",
            closing_tier=FetchTier.CLOSING,
            feature_groups=groups,
            pm_velocity_window_hours=2.0,
            pm_price_tolerance_minutes=60,
        )

    async def test_pm_context_none_for_sb_only_event(
        self, pglite_async_session, cross_source_test_data
    ):
        """SB-only event (event_2) should have pm_context=None in bundle."""
        sb_only_event = cross_source_test_data["events"][2]
        config = self._make_config(with_pm=True)
        bundle = await collect_event_data(sb_only_event, pglite_async_session, config)
        assert bundle.pm_context is None

    async def test_pm_context_populated_for_pm_event(
        self, pglite_async_session, cross_source_test_data
    ):
        """PM-linked event (event_0) should have pm_context with valid market/home_idx."""
        pm_event = cross_source_test_data["events"][0]
        config = self._make_config(with_pm=True)
        bundle = await collect_event_data(pm_event, pglite_async_session, config)
        assert bundle.pm_context is not None
        assert bundle.pm_context.home_idx in (0, 1)
        assert len(bundle.pm_prices) > 0

    async def test_no_pm_context_when_group_not_requested(
        self, pglite_async_session, cross_source_test_data
    ):
        """When polymarket not in feature_groups, pm_context should be None."""
        pm_event = cross_source_test_data["events"][0]
        config = self._make_config(with_pm=False)
        bundle = await collect_event_data(pm_event, pglite_async_session, config)
        assert bundle.pm_context is None
        assert bundle.pm_prices == []


class TestXGBoostAdapterFeatureCount:
    """Tests for XGBoostAdapter feature counts."""

    def test_tabular_only_feature_names(self):
        from odds_analytics.feature_extraction import TabularFeatures

        config = FeatureConfig(feature_groups=["tabular"])
        adapter = XGBoostAdapter()
        names = adapter.feature_names(config)
        expected = len(TabularFeatures.get_feature_names()) + 1  # +1 for hours_until_event
        assert len(names) == expected
        assert "hours_until_event" in names

    def test_tabular_polymarket_feature_names(self):
        from odds_analytics.feature_extraction import TabularFeatures
        from odds_analytics.polymarket_features import (
            CrossSourceFeatures,
            PolymarketTabularFeatures,
        )

        config = FeatureConfig(feature_groups=["tabular", "polymarket"])
        adapter = XGBoostAdapter()
        names = adapter.feature_names(config)
        expected = (
            len(TabularFeatures.get_feature_names())
            + len(PolymarketTabularFeatures.get_feature_names())
            + len(CrossSourceFeatures.get_feature_names())
            + 1  # hours_until_event
        )
        assert len(names) == expected


class TestPrepareTrainingDataWithPolymarket:
    """Tests for prepare_training_data with polymarket feature group."""

    def _make_config(self) -> FeatureConfig:
        return FeatureConfig(
            sharp_bookmakers=["pinnacle"],
            retail_bookmakers=["fanduel", "draftkings"],
            markets=["h2h"],
            outcome="home",
            closing_tier=FetchTier.CLOSING,
            feature_groups=("tabular", "polymarket"),
            pm_velocity_window_hours=2.0,
            pm_price_tolerance_minutes=60,
            target_type="raw",
            sampling=SamplingConfig(
                strategy="time_range",
                min_hours=3.0,
                max_hours=12.0,
                max_samples_per_event=5,
            ),
        )

    async def test_all_events_produce_rows(self, pglite_async_session, cross_source_test_data):
        """PM NaN-fill means all events produce rows (not just PM-linked ones)."""
        events = cross_source_test_data["events"]  # 3 events (2 with PM, 1 without)
        config = self._make_config()
        result = await prepare_training_data(events, pglite_async_session, config)
        # The pregame snapshot (7.5h before) falls in [3h, 12h] range → 1 row each
        assert result.num_samples == 3

    async def test_output_shape_correct(self, pglite_async_session, cross_source_test_data):
        from odds_analytics.feature_extraction import TabularFeatures
        from odds_analytics.polymarket_features import (
            CrossSourceFeatures,
            PolymarketTabularFeatures,
        )

        events = cross_source_test_data["events"]
        config = self._make_config()
        result = await prepare_training_data(events, pglite_async_session, config)

        max_features = (
            len(TabularFeatures.get_feature_names())
            + len(PolymarketTabularFeatures.get_feature_names())
            + len(CrossSourceFeatures.get_feature_names())
            + 1  # hours_until_event
        )
        assert result.X.shape[1] == len(result.feature_names)
        assert result.X.shape[1] <= max_features
        assert result.y.shape == (result.num_samples,)

    async def test_sb_only_event_has_zero_pm_features(
        self, pglite_async_session, cross_source_test_data
    ):
        """The SB-only event row should have zeros (NaN→0) for PM feature columns."""
        events = cross_source_test_data["events"]
        config = self._make_config()
        result = await prepare_training_data(events, pglite_async_session, config)

        # Find the row for the SB-only event (event_2)
        sb_only_id = cross_source_test_data["events"][2].id
        sb_only_rows = result.X[result.event_ids == sb_only_id]
        assert len(sb_only_rows) > 0

        # PM/cross-source features should be 0 (NaN-filled and then nan_to_num)
        pm_indices = [
            i
            for i, name in enumerate(result.feature_names)
            if name.startswith("pm_") or name.startswith("xsrc_")
        ]
        if pm_indices:
            pm_block = sb_only_rows[0, pm_indices]
            assert np.all(pm_block == 0.0)
