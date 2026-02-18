"""Unit tests for multi-horizon data preparation and group-aware CV."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from odds_analytics.feature_groups import (
    AdapterOutput,
    EventDataBundle,
    LSTMAdapter,
    TierSampler,
    TimeRangeSampler,
    XGBoostAdapter,
    prepare_training_data,
)
from odds_analytics.training.config import FeatureConfig, MLTrainingConfig, SamplingConfig
from odds_analytics.training.cross_validation import CVResult, run_cv
from odds_core.models import Event, EventStatus, OddsSnapshot


@pytest.fixture
def sample_events():
    """Three events spread over a week, sorted chronologically."""
    return [
        Event(
            id=f"event_{i}",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime(2024, 11, 1 + i, 19, 0, 0, tzinfo=UTC),
            home_team="Los Angeles Lakers",
            away_team="Boston Celtics",
            status=EventStatus.FINAL,
            home_score=110,
            away_score=105,
        )
        for i in range(3)
    ]


def _make_snapshot(
    idx: int,
    event_id: str,
    snapshot_time: datetime,
    home_team: str = "Los Angeles Lakers",
    away_team: str = "Boston Celtics",
    home_price: int = -150,
    away_price: int = 130,
) -> OddsSnapshot:
    """Create an OddsSnapshot with Pinnacle h2h odds in raw_data."""
    return OddsSnapshot(
        id=idx,
        event_id=event_id,
        snapshot_time=snapshot_time,
        bookmaker_count=1,
        raw_data={
            "id": event_id,
            "sport_key": "basketball_nba",
            "commence_time": "2024-11-01T19:00:00Z",
            "home_team": home_team,
            "away_team": away_team,
            "bookmakers": [
                {
                    "key": "pinnacle",
                    "title": "Pinnacle",
                    "last_update": snapshot_time.isoformat(),
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": home_team, "price": home_price},
                                {"name": away_team, "price": away_price},
                            ],
                        }
                    ],
                }
            ],
        },
    )


class TestGroupTimeseriesCV:
    """Tests for group_timeseries cross-validation method."""

    def _make_config(
        self, n_folds: int = 3, cv_method: str = "group_timeseries"
    ) -> MLTrainingConfig:
        """Create a minimal MLTrainingConfig for CV testing."""
        return MLTrainingConfig.model_validate(
            {
                "experiment": {"name": "test_cv", "description": "test"},
                "training": {
                    "strategy_type": "xgboost_line_movement",
                    "features": {
                        "feature_groups": ["tabular"],
                        "markets": ["h2h"],
                        "outcome": "home",
                        "closing_tier": "closing",
                    },
                    "data": {
                        "start_date": "2024-10-01",
                        "end_date": "2024-12-31",
                        "cv_method": cv_method,
                        "n_folds": n_folds,
                        "random_seed": 42,
                    },
                    "model": {
                        "n_estimators": 10,
                        "max_depth": 3,
                        "learning_rate": 0.1,
                    },
                },
            }
        )

    def _make_mock_strategy(self):
        """Create a mock strategy that returns plausible metrics."""
        strategy = MagicMock()

        def fake_train(config, X_train, y_train, feature_names, X_val, y_val):
            return {
                "train_mse": 0.01,
                "train_mae": 0.08,
                "train_r2": 0.90,
                "val_mse": 0.02,
                "val_mae": 0.10,
                "val_r2": 0.80,
            }

        strategy.train_from_config = fake_train
        return strategy

    def test_events_dont_span_folds(self):
        """No event should appear in both train and val within a fold."""
        n_events = 5
        rows_per_event = 3
        n_rows = n_events * rows_per_event
        n_features = 5

        X = np.random.randn(n_rows, n_features).astype(np.float32)
        y = np.random.randn(n_rows).astype(np.float32)
        feature_names = [f"f_{i}" for i in range(n_features)]

        event_ids = np.array([f"evt_{i}" for i in range(n_events) for _ in range(rows_per_event)])

        config = self._make_config(n_folds=3)
        strategy = self._make_mock_strategy()

        result = run_cv(strategy, config, X, y, feature_names, event_ids=event_ids)

        assert isinstance(result, CVResult)
        assert result.n_folds == 3
        assert result.cv_method == "group_timeseries"

    def test_temporal_ordering_preserved(self):
        """Train events should always be earlier than val events."""
        n_events = 6
        rows_per_event = 2
        n_rows = n_events * rows_per_event
        n_features = 3

        X = np.random.randn(n_rows, n_features).astype(np.float32)
        y = np.random.randn(n_rows).astype(np.float32)
        feature_names = [f"f_{i}" for i in range(n_features)]

        event_ids = np.array([f"evt_{i}" for i in range(n_events) for _ in range(rows_per_event)])

        config = self._make_config(n_folds=3)

        fold_splits = []

        def capture_train(config, X_train, y_train, feature_names, X_val, y_val):
            fold_splits.append((len(X_train), len(X_val)))
            return {
                "train_mse": 0.01,
                "train_mae": 0.08,
                "train_r2": 0.90,
                "val_mse": 0.02,
                "val_mae": 0.10,
                "val_r2": 0.80,
            }

        strategy = MagicMock()
        strategy.train_from_config = capture_train

        run_cv(strategy, config, X, y, feature_names, event_ids=event_ids)

        for i in range(len(fold_splits) - 1):
            assert (
                fold_splits[i][0] <= fold_splits[i + 1][0]
            ), f"Fold {i} train size ({fold_splits[i][0]}) > fold {i + 1} ({fold_splits[i + 1][0]})"

    def test_group_timeseries_with_no_event_ids_falls_back_to_timeseries(self):
        """group_timeseries without event_ids warns and falls back to timeseries CV."""
        n_rows = 20
        n_features = 3
        X = np.random.randn(n_rows, n_features).astype(np.float32)
        y = np.random.randn(n_rows).astype(np.float32)
        feature_names = [f"f_{i}" for i in range(n_features)]

        config = self._make_config(n_folds=3)
        strategy = self._make_mock_strategy()

        result = run_cv(strategy, config, X, y, feature_names, event_ids=None)
        assert isinstance(result, CVResult)
        assert result.n_folds == 3
        assert result.cv_method == "timeseries"

    def test_fold_results_populated(self):
        """Each fold should produce valid metrics."""
        n_events = 4
        rows_per_event = 3
        n_rows = n_events * rows_per_event
        n_features = 5

        X = np.random.randn(n_rows, n_features).astype(np.float32)
        y = np.random.randn(n_rows).astype(np.float32)
        feature_names = [f"f_{i}" for i in range(n_features)]
        event_ids = np.array([f"evt_{i}" for i in range(n_events) for _ in range(rows_per_event)])

        config = self._make_config(n_folds=3)
        strategy = self._make_mock_strategy()

        result = run_cv(strategy, config, X, y, feature_names, event_ids=event_ids)

        assert len(result.fold_results) == 3
        for fold in result.fold_results:
            assert fold.n_train > 0
            assert fold.n_val > 0
            assert fold.val_mse == 0.02
            assert fold.val_r2 == 0.80


class TestFeatureConfigValidation:
    """Tests for new FeatureConfig and SamplingConfig validation."""

    def test_default_sampling_config(self):
        config = FeatureConfig(
            feature_groups=["tabular"],
            markets=["h2h"],
            outcome="home",
            closing_tier="closing",
        )
        assert config.sampling.strategy == "time_range"
        assert config.sampling.min_hours == 3.0
        assert config.sampling.max_hours == 12.0
        assert config.sampling.max_samples_per_event == 5

    def test_sampling_config_time_range(self):
        config = FeatureConfig(
            feature_groups=["tabular"],
            markets=["h2h"],
            outcome="home",
            closing_tier="closing",
            sampling=SamplingConfig(
                strategy="time_range",
                min_hours=3.0,
                max_hours=12.0,
                max_samples_per_event=5,
            ),
        )
        assert config.sampling.strategy == "time_range"
        assert config.sampling.min_hours == 3.0
        assert config.sampling.max_hours == 12.0

    def test_sampling_config_tier(self):
        from odds_lambda.fetch_tier import FetchTier

        config = FeatureConfig(
            feature_groups=["tabular"],
            markets=["h2h"],
            outcome="home",
            closing_tier="closing",
            sampling=SamplingConfig(
                strategy="tier",
                decision_tier=FetchTier.PREGAME,
            ),
        )
        assert config.sampling.strategy == "tier"
        assert config.sampling.decision_tier == FetchTier.PREGAME

    def test_invalid_hours_range_reversed(self):
        """min >= max should raise validation error."""
        with pytest.raises(ValueError, match="must be less than"):
            SamplingConfig(strategy="time_range", min_hours=12.0, max_hours=3.0)

    def test_default_target_type_is_raw(self):
        config = FeatureConfig(
            feature_groups=["tabular"],
            markets=["h2h"],
            outcome="home",
            closing_tier="closing",
        )
        assert config.target_type == "raw"

    def test_devigged_pinnacle_target_type(self):
        config = FeatureConfig(
            feature_groups=["tabular"],
            markets=["h2h"],
            outcome="home",
            closing_tier="closing",
            target_type="devigged_pinnacle",
        )
        assert config.target_type == "devigged_pinnacle"

    def test_adapter_default_is_xgboost(self):
        config = FeatureConfig()
        assert config.adapter == "xgboost"

    def test_adapter_lstm(self):
        config = FeatureConfig(adapter="lstm")
        assert config.adapter == "lstm"


class TestTierSampler:
    """Tests for TierSampler snapshot selection."""

    @pytest.fixture
    def event(self):
        return Event(
            id="tier_test",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime(2024, 11, 1, 19, 0, 0, tzinfo=UTC),
            home_team="Los Angeles Lakers",
            away_team="Boston Celtics",
            status=EventStatus.FINAL,
            home_score=110,
            away_score=105,
        )

    def _make_bundle(self, event: Event, snapshots: list[OddsSnapshot]) -> EventDataBundle:
        return EventDataBundle(
            event=event,
            snapshots=snapshots,
            closing_snapshot=None,
            pm_context=None,
        )

    def test_returns_latest_snapshot_in_decision_tier(self, event):
        """Should pick the most recent pregame snapshot."""
        commence = event.commence_time
        snaps = [
            OddsSnapshot(
                id=1,
                event_id=event.id,
                snapshot_time=commence - timedelta(hours=10),
                raw_data={},
                bookmaker_count=1,
                fetch_tier="pregame",
            ),
            OddsSnapshot(
                id=2,
                event_id=event.id,
                snapshot_time=commence - timedelta(hours=5),
                raw_data={},
                bookmaker_count=1,
                fetch_tier="pregame",
            ),
            OddsSnapshot(
                id=3,
                event_id=event.id,
                snapshot_time=commence - timedelta(hours=1),
                raw_data={},
                bookmaker_count=1,
                fetch_tier="closing",
            ),
        ]
        sampler = TierSampler("pregame")
        result = sampler.sample(self._make_bundle(event, snaps))

        assert len(result) == 1
        assert result[0].id == 2  # Latest pregame, not the closing one

    def test_includes_earlier_tiers(self, event):
        """When no snapshot in decision tier, falls back to earlier tiers."""
        commence = event.commence_time
        snaps = [
            OddsSnapshot(
                id=1,
                event_id=event.id,
                snapshot_time=commence - timedelta(hours=50),
                raw_data={},
                bookmaker_count=1,
                fetch_tier="early",
            ),
            OddsSnapshot(
                id=2,
                event_id=event.id,
                snapshot_time=commence - timedelta(hours=1),
                raw_data={},
                bookmaker_count=1,
                fetch_tier="closing",
            ),
        ]
        sampler = TierSampler("pregame")
        result = sampler.sample(self._make_bundle(event, snaps))

        assert len(result) == 1
        assert result[0].fetch_tier == "early"

    def test_excludes_closer_tiers(self, event):
        """Closing-tier snapshot should not be selected when decision_tier=pregame."""
        commence = event.commence_time
        snaps = [
            OddsSnapshot(
                id=1,
                event_id=event.id,
                snapshot_time=commence - timedelta(hours=1),
                raw_data={},
                bookmaker_count=1,
                fetch_tier="closing",
            ),
        ]
        sampler = TierSampler("pregame")
        result = sampler.sample(self._make_bundle(event, snaps))

        assert result == []

    def test_excludes_in_play_snapshots(self, event):
        """IN_PLAY snapshots must not be selected for pre-game decision tiers."""
        commence = event.commence_time
        snaps = [
            OddsSnapshot(
                id=1,
                event_id=event.id,
                snapshot_time=commence - timedelta(hours=5),
                raw_data={},
                bookmaker_count=1,
                fetch_tier="pregame",
            ),
            OddsSnapshot(
                id=2,
                event_id=event.id,
                snapshot_time=commence + timedelta(hours=1),
                raw_data={},
                bookmaker_count=1,
                fetch_tier="in_play",
            ),
        ]
        sampler = TierSampler("pregame")
        result = sampler.sample(self._make_bundle(event, snaps))

        assert len(result) == 1
        assert result[0].id == 1  # Pregame, not in_play

    def test_in_play_tier_can_select_in_play(self, event):
        """IN_PLAY decision tier should select in-play snapshots."""
        commence = event.commence_time
        snaps = [
            OddsSnapshot(
                id=1,
                event_id=event.id,
                snapshot_time=commence - timedelta(hours=5),
                raw_data={},
                bookmaker_count=1,
                fetch_tier="pregame",
            ),
            OddsSnapshot(
                id=2,
                event_id=event.id,
                snapshot_time=commence + timedelta(hours=1),
                raw_data={},
                bookmaker_count=1,
                fetch_tier="in_play",
            ),
        ]
        sampler = TierSampler("in_play")
        result = sampler.sample(self._make_bundle(event, snaps))

        assert len(result) == 1
        assert result[0].id == 2  # in_play is the latest

    def test_empty_snapshots(self, event):
        sampler = TierSampler("pregame")
        assert sampler.sample(self._make_bundle(event, [])) == []


class TestTimeRangeSampler:
    """Tests for TimeRangeSampler stratified snapshot selection."""

    @pytest.fixture
    def event(self):
        return Event(
            id="range_test",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime(2024, 11, 1, 19, 0, 0, tzinfo=UTC),
            home_team="Los Angeles Lakers",
            away_team="Boston Celtics",
            status=EventStatus.FINAL,
            home_score=110,
            away_score=105,
        )

    def _make_bundle(self, event: Event, snapshots: list[OddsSnapshot]) -> EventDataBundle:
        return EventDataBundle(
            event=event,
            snapshots=snapshots,
            closing_snapshot=None,
            pm_context=None,
        )

    def test_returns_snapshots_in_range(self, event):
        """Only snapshots within [min_hours, max_hours] before game are included."""
        commence = event.commence_time
        snaps = [
            OddsSnapshot(
                id=1,
                event_id=event.id,
                snapshot_time=commence - timedelta(hours=24),
                raw_data={},
                bookmaker_count=1,
            ),
            OddsSnapshot(
                id=2,
                event_id=event.id,
                snapshot_time=commence - timedelta(hours=8),
                raw_data={},
                bookmaker_count=1,
            ),
            OddsSnapshot(
                id=3,
                event_id=event.id,
                snapshot_time=commence - timedelta(hours=6),
                raw_data={},
                bookmaker_count=1,
            ),
            OddsSnapshot(
                id=4,
                event_id=event.id,
                snapshot_time=commence - timedelta(hours=4),
                raw_data={},
                bookmaker_count=1,
            ),
            OddsSnapshot(
                id=5,
                event_id=event.id,
                snapshot_time=commence - timedelta(hours=1),
                raw_data={},
                bookmaker_count=1,
            ),
        ]
        sampler = TimeRangeSampler(min_hours=3.0, max_hours=12.0, max_samples=10)
        result = sampler.sample(self._make_bundle(event, snaps))

        assert len(result) == 3
        result_ids = {s.id for s in result}
        assert result_ids == {2, 3, 4}

    def test_stratified_sampling_caps_at_max(self, event):
        """When more snapshots than max_samples, picks one per bin."""
        commence = event.commence_time
        snaps = [
            OddsSnapshot(
                id=i,
                event_id=event.id,
                snapshot_time=commence - timedelta(hours=3 + i * 0.9),
                raw_data={},
                bookmaker_count=1,
            )
            for i in range(10)
        ]
        sampler = TimeRangeSampler(min_hours=3.0, max_hours=12.0, max_samples=3)
        result = sampler.sample(self._make_bundle(event, snaps))

        assert len(result) <= 3
        # Results should be chronologically sorted
        for i in range(len(result) - 1):
            assert result[i].snapshot_time < result[i + 1].snapshot_time

    def test_no_duplicates_in_stratified_sample(self, event):
        """Same snapshot shouldn't appear twice even if it's nearest to multiple bins."""
        commence = event.commence_time
        # Only 2 snapshots, ask for 5 bins — should get at most 2, no dupes
        snaps = [
            OddsSnapshot(
                id=1,
                event_id=event.id,
                snapshot_time=commence - timedelta(hours=6),
                raw_data={},
                bookmaker_count=1,
            ),
            OddsSnapshot(
                id=2,
                event_id=event.id,
                snapshot_time=commence - timedelta(hours=4),
                raw_data={},
                bookmaker_count=1,
            ),
        ]
        sampler = TimeRangeSampler(min_hours=3.0, max_hours=12.0, max_samples=5)
        result = sampler.sample(self._make_bundle(event, snaps))

        assert len(result) == 2
        assert result[0].id != result[1].id

    def test_empty_range(self, event):
        """No snapshots in range returns empty list."""
        commence = event.commence_time
        snaps = [
            OddsSnapshot(
                id=1,
                event_id=event.id,
                snapshot_time=commence - timedelta(hours=24),
                raw_data={},
                bookmaker_count=1,
            ),
        ]
        sampler = TimeRangeSampler(min_hours=3.0, max_hours=12.0, max_samples=5)
        assert sampler.sample(self._make_bundle(event, snaps)) == []

    def test_empty_snapshots(self, event):
        sampler = TimeRangeSampler(min_hours=3.0, max_hours=12.0, max_samples=5)
        assert sampler.sample(self._make_bundle(event, [])) == []


class TestPrepareTrainingData:
    """Tests for the new unified prepare_training_data function."""

    def _make_bundle(
        self,
        event: Event,
        closing_snap: OddsSnapshot,
        all_snaps: list[OddsSnapshot],
    ) -> EventDataBundle:
        return EventDataBundle(
            event=event,
            snapshots=all_snaps,
            closing_snapshot=closing_snap,
            pm_context=None,
            pm_prices=[],
            pm_orderbooks=[],
            sequences=[],
        )

    @pytest.mark.asyncio
    async def test_basic_multi_horizon(self, sample_events):
        """Time-range sampling produces multiple rows per event."""
        event = sample_events[0]
        commence = event.commence_time

        closing_snap = _make_snapshot(
            100,
            event.id,
            commence - timedelta(hours=0.5),
            home_price=-200,
            away_price=170,
        )
        decision_snaps = [
            _make_snapshot(
                i,
                event.id,
                commence - timedelta(hours=h),
                home_price=-150 - i * 5,
                away_price=130 + i * 5,
            )
            for i, h in enumerate([4, 6, 8, 10])
        ]
        all_snaps = decision_snaps + [closing_snap]

        config = FeatureConfig(
            feature_groups=["tabular"],
            markets=["h2h"],
            outcome="home",
            closing_tier="closing",
            target_type="devigged_pinnacle",
            sampling=SamplingConfig(
                strategy="time_range",
                min_hours=3.0,
                max_hours=12.0,
                max_samples_per_event=5,
            ),
        )

        bundle = self._make_bundle(event, closing_snap, all_snaps)
        mock_session = AsyncMock()

        with patch(
            "odds_analytics.feature_groups.collect_event_data",
            new=AsyncMock(return_value=bundle),
        ):
            result = await prepare_training_data(
                events=[event],
                session=mock_session,
                config=config,
            )

        assert result.num_samples >= 1
        assert result.num_samples <= 4
        assert result.event_ids is not None
        assert all(eid == event.id for eid in result.event_ids)
        assert "hours_until_event" in result.feature_names
        assert not all(y == 0 for y in result.y)

    @pytest.mark.asyncio
    async def test_drops_event_without_pinnacle_closing(self, sample_events):
        """Events without Pinnacle closing data are skipped with devigged_pinnacle target."""
        event = sample_events[0]
        commence = event.commence_time

        # Closing snapshot with NO Pinnacle (only fanduel)
        closing_snap = OddsSnapshot(
            id=100,
            event_id=event.id,
            snapshot_time=commence - timedelta(hours=0.5),
            bookmaker_count=1,
            raw_data={
                "id": event.id,
                "sport_key": "basketball_nba",
                "commence_time": commence.isoformat(),
                "home_team": event.home_team,
                "away_team": event.away_team,
                "bookmakers": [
                    {
                        "key": "fanduel",
                        "title": "FanDuel",
                        "last_update": (commence - timedelta(hours=0.5)).isoformat(),
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": event.home_team, "price": -200},
                                    {"name": event.away_team, "price": 170},
                                ],
                            }
                        ],
                    }
                ],
            },
        )

        config = FeatureConfig(
            feature_groups=["tabular"],
            markets=["h2h"],
            outcome="home",
            closing_tier="closing",
            target_type="devigged_pinnacle",
            sampling=SamplingConfig(
                strategy="time_range",
                min_hours=3.0,
                max_hours=12.0,
                max_samples_per_event=5,
            ),
        )

        bundle = self._make_bundle(event, closing_snap, [])
        mock_session = AsyncMock()

        with patch(
            "odds_analytics.feature_groups.collect_event_data",
            new=AsyncMock(return_value=bundle),
        ):
            with pytest.raises(ValueError, match="No valid training data"):
                await prepare_training_data(
                    events=[event],
                    session=mock_session,
                    config=config,
                )

    @pytest.mark.asyncio
    async def test_hours_until_event_correct(self, sample_events):
        """hours_until_event feature matches actual time difference."""
        event = sample_events[0]
        commence = event.commence_time

        closing_snap = _make_snapshot(
            100,
            event.id,
            commence - timedelta(hours=0.5),
            home_price=-200,
            away_price=170,
        )
        decision_snap = _make_snapshot(
            1,
            event.id,
            commence - timedelta(hours=6),
            home_price=-150,
            away_price=130,
        )
        all_snaps = [decision_snap, closing_snap]

        config = FeatureConfig(
            feature_groups=["tabular"],
            markets=["h2h"],
            outcome="home",
            closing_tier="closing",
            target_type="devigged_pinnacle",
            sampling=SamplingConfig(
                strategy="time_range",
                min_hours=3.0,
                max_hours=12.0,
                max_samples_per_event=5,
            ),
        )

        bundle = self._make_bundle(event, closing_snap, all_snaps)
        mock_session = AsyncMock()

        with patch(
            "odds_analytics.feature_groups.collect_event_data",
            new=AsyncMock(return_value=bundle),
        ):
            result = await prepare_training_data(
                events=[event],
                session=mock_session,
                config=config,
            )

        assert result.num_samples == 1
        hours_idx = result.feature_names.index("hours_until_event")
        assert result.X[0, hours_idx] == pytest.approx(6.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_max_samples_per_event_respected(self, sample_events):
        """Should not produce more rows than max_samples_per_event."""
        event = sample_events[0]
        commence = event.commence_time

        closing_snap = _make_snapshot(
            100,
            event.id,
            commence - timedelta(hours=0.5),
            home_price=-200,
            away_price=170,
        )
        decision_snaps = [
            _make_snapshot(
                i,
                event.id,
                commence - timedelta(hours=3.0 + i * 0.9),
                home_price=-150 - i,
                away_price=130 + i,
            )
            for i in range(10)
        ]
        all_snaps = decision_snaps + [closing_snap]

        config = FeatureConfig(
            feature_groups=["tabular"],
            markets=["h2h"],
            outcome="home",
            closing_tier="closing",
            target_type="devigged_pinnacle",
            sampling=SamplingConfig(
                strategy="time_range",
                min_hours=3.0,
                max_hours=12.0,
                max_samples_per_event=3,
            ),
        )

        bundle = self._make_bundle(event, closing_snap, all_snaps)
        mock_session = AsyncMock()

        with patch(
            "odds_analytics.feature_groups.collect_event_data",
            new=AsyncMock(return_value=bundle),
        ):
            result = await prepare_training_data(
                events=[event],
                session=mock_session,
                config=config,
            )

        assert result.num_samples <= 3


class TestAdapterOutput:
    """Tests for the AdapterOutput dataclass."""

    def test_xgboost_adapter_output_has_no_mask(self):
        features = np.array([1.0, 2.0, 3.0])
        output = AdapterOutput(features=features)
        assert output.mask is None
        np.testing.assert_array_equal(output.features, features)

    def test_lstm_adapter_output_has_mask(self):
        timesteps, n_features = 8, 4
        features = np.zeros((timesteps, n_features))
        mask = np.ones(timesteps, dtype=bool)
        output = AdapterOutput(features=features, mask=mask)
        assert output.mask is not None
        assert output.features.shape == (timesteps, n_features)
        assert output.mask.shape == (timesteps,)


class TestLSTMAdapter:
    """Tests for LSTMAdapter feature extraction."""

    @pytest.fixture
    def event(self):
        return Event(
            id="lstm_test",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime(2024, 11, 1, 19, 0, 0, tzinfo=UTC),
            home_team="Los Angeles Lakers",
            away_team="Boston Celtics",
            status=EventStatus.FINAL,
            home_score=110,
            away_score=105,
        )

    @pytest.fixture
    def config(self):
        return FeatureConfig(
            adapter="lstm",
            feature_groups=["tabular"],
            markets=["h2h"],
            outcome="home",
            closing_tier="closing",
            lookback_hours=72,
            timesteps=8,
        )

    def _make_bundle(self, event: Event, sequences=None) -> EventDataBundle:
        return EventDataBundle(
            event=event,
            snapshots=[],
            closing_snapshot=None,
            pm_context=None,
            sequences=sequences or [],
        )

    def _make_snapshot(self, event_id: str, snapshot_time: datetime) -> OddsSnapshot:
        return OddsSnapshot(
            id=1,
            event_id=event_id,
            snapshot_time=snapshot_time,
            bookmaker_count=1,
            raw_data={},
        )

    def test_feature_names_returns_sequence_feature_names(self, config):
        from odds_analytics.feature_extraction import SequenceFeatures

        adapter = LSTMAdapter()
        names = adapter.feature_names(config)
        assert names == SequenceFeatures.get_feature_names()
        assert len(names) > 0

    def test_transform_returns_adapter_output_with_mask(self, event, config):
        adapter = LSTMAdapter()
        snapshot_time = event.commence_time - timedelta(hours=6)
        snapshot = self._make_snapshot(event.id, snapshot_time)
        bundle = self._make_bundle(event)

        mock_sequence = np.zeros((8, 10))
        mock_mask = np.ones(8, dtype=bool)

        with patch(
            "odds_analytics.feature_extraction.SequenceFeatureExtractor.extract_features",
            return_value={"sequence": mock_sequence, "mask": mock_mask},
        ):
            output = adapter.transform(bundle, snapshot, config)

        assert output is not None
        assert isinstance(output, AdapterOutput)
        assert output.features.shape == (8, 10)
        assert output.mask is not None
        assert output.mask.shape == (8,)

    def test_transform_filters_sequences_by_snapshot_time(self, event, config):
        """Only sequences with first-entry timestamp <= snapshot_time are passed."""
        from odds_core.models import Odds

        snapshot_time = event.commence_time - timedelta(hours=6)
        snapshot = self._make_snapshot(event.id, snapshot_time)

        # Two sequences: one before snapshot, one after
        early_odds = Odds(
            event_id=event.id,
            bookmaker_key="pinnacle",
            bookmaker_title="Pinnacle",
            market_key="h2h",
            outcome_name=event.home_team,
            price=-150,
            point=None,
            odds_timestamp=snapshot_time - timedelta(hours=1),
            last_update=snapshot_time - timedelta(hours=1),
        )
        late_odds = Odds(
            event_id=event.id,
            bookmaker_key="pinnacle",
            bookmaker_title="Pinnacle",
            market_key="h2h",
            outcome_name=event.home_team,
            price=-140,
            point=None,
            odds_timestamp=snapshot_time + timedelta(hours=1),
            last_update=snapshot_time + timedelta(hours=1),
        )

        sequences = [[early_odds], [late_odds]]
        bundle = self._make_bundle(event, sequences=sequences)

        captured_odds_data = []

        def fake_extract(self_extractor, event, odds_data, outcome=None, market="h2h", **kwargs):
            captured_odds_data.append(odds_data)
            n = config.timesteps
            return {"sequence": np.zeros((n, 10)), "mask": np.zeros(n, dtype=bool)}

        adapter = LSTMAdapter()

        with patch(
            "odds_analytics.feature_extraction.SequenceFeatureExtractor.extract_features",
            new=fake_extract,
        ):
            adapter.transform(bundle, snapshot, config)

        assert len(captured_odds_data) == 1
        passed_seqs = captured_odds_data[0]
        # Only the early sequence (timestamp <= snapshot_time) should be passed
        assert len(passed_seqs) == 1
        assert passed_seqs[0][0].odds_timestamp <= snapshot_time

    def test_transform_returns_none_on_extraction_failure(self, event, config):
        adapter = LSTMAdapter()
        snapshot = self._make_snapshot(event.id, event.commence_time - timedelta(hours=6))
        bundle = self._make_bundle(event)

        with patch(
            "odds_analytics.feature_extraction.SequenceFeatureExtractor.extract_features",
            side_effect=ValueError("extraction failed"),
        ):
            output = adapter.transform(bundle, snapshot, config)

        assert output is None


class TestPrepareTrainingDataMaskCollection:
    """Tests that prepare_training_data correctly collects masks for LSTM adapter."""

    @pytest.fixture
    def event(self):
        return Event(
            id="mask_test",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime(2024, 11, 1, 19, 0, 0, tzinfo=UTC),
            home_team="Los Angeles Lakers",
            away_team="Boston Celtics",
            status=EventStatus.FINAL,
            home_score=110,
            away_score=105,
        )

    def _make_bundle(self, event: Event, closing_snap: OddsSnapshot) -> EventDataBundle:
        return EventDataBundle(
            event=event,
            snapshots=[closing_snap],
            closing_snapshot=closing_snap,
            pm_context=None,
            pm_prices=[],
            pm_orderbooks=[],
            sequences=[],
        )

    @pytest.mark.asyncio
    async def test_xgboost_adapter_produces_no_masks(self, event):
        """XGBoostAdapter.mask=None → result.masks is None."""
        commence = event.commence_time
        closing_snap = _make_snapshot(100, event.id, commence - timedelta(hours=0.5))
        decision_snap = _make_snapshot(1, event.id, commence - timedelta(hours=6))
        decision_snap.fetch_tier = "pregame"
        closing_snap.fetch_tier = "closing"
        all_snaps = [decision_snap, closing_snap]
        bundle = EventDataBundle(
            event=event,
            snapshots=all_snaps,
            closing_snapshot=closing_snap,
            pm_context=None,
            pm_prices=[],
            pm_orderbooks=[],
            sequences=[],
        )

        config = FeatureConfig(
            adapter="xgboost",
            feature_groups=["tabular"],
            markets=["h2h"],
            outcome="home",
            closing_tier="closing",
            target_type="devigged_pinnacle",
            sampling=SamplingConfig(strategy="tier", decision_tier="pregame"),
        )

        mock_features = np.array([0.5, 0.4, 6.0])
        mock_output = AdapterOutput(features=mock_features, mask=None)

        with (
            patch(
                "odds_analytics.feature_groups.collect_event_data",
                new=AsyncMock(return_value=bundle),
            ),
            patch.object(XGBoostAdapter, "transform", return_value=mock_output),
            patch.object(XGBoostAdapter, "feature_names", return_value=["f1", "f2", "f3"]),
            patch(
                "odds_analytics.feature_groups._compute_target",
                return_value=0.05,
            ),
        ):
            result = await prepare_training_data(
                events=[event],
                session=AsyncMock(),
                config=config,
            )

        assert result.masks is None
        assert result.X.shape == (1, 3)

    @pytest.mark.asyncio
    async def test_lstm_adapter_produces_masks(self, event):
        """LSTMAdapter.mask is not None → result.masks stacked into 2D array."""
        commence = event.commence_time
        closing_snap = _make_snapshot(100, event.id, commence - timedelta(hours=0.5))
        decision_snap = _make_snapshot(1, event.id, commence - timedelta(hours=6))
        decision_snap.fetch_tier = "pregame"
        closing_snap.fetch_tier = "closing"
        all_snaps = [decision_snap, closing_snap]
        bundle = EventDataBundle(
            event=event,
            snapshots=all_snaps,
            closing_snapshot=closing_snap,
            pm_context=None,
            pm_prices=[],
            pm_orderbooks=[],
            sequences=[],
        )

        timesteps, n_features = 8, 4
        mock_sequence = np.zeros((timesteps, n_features))
        mock_mask = np.ones(timesteps, dtype=bool)
        mock_output = AdapterOutput(features=mock_sequence, mask=mock_mask)

        config = FeatureConfig(
            adapter="lstm",
            feature_groups=["tabular"],
            markets=["h2h"],
            outcome="home",
            closing_tier="closing",
            target_type="devigged_pinnacle",
            lookback_hours=72,
            timesteps=timesteps,
            sampling=SamplingConfig(strategy="tier", decision_tier="pregame"),
        )

        with (
            patch(
                "odds_analytics.feature_groups.collect_event_data",
                new=AsyncMock(return_value=bundle),
            ),
            patch.object(LSTMAdapter, "transform", return_value=mock_output),
            patch.object(
                LSTMAdapter, "feature_names", return_value=[f"f{i}" for i in range(n_features)]
            ),
            patch(
                "odds_analytics.feature_groups._compute_target",
                return_value=0.05,
            ),
        ):
            result = await prepare_training_data(
                events=[event],
                session=AsyncMock(),
                config=config,
            )

        assert result.masks is not None
        assert result.masks.shape == (1, timesteps)
        assert result.X.shape == (1, timesteps, n_features)
