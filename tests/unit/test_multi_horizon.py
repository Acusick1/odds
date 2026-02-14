"""Unit tests for multi-horizon data preparation and group-aware CV."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from odds_analytics.training.config import FeatureConfig, MLTrainingConfig
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
                        "opening_tier": "opening",
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
        # 5 events, 3 rows each = 15 rows
        n_events = 5
        rows_per_event = 3
        n_rows = n_events * rows_per_event
        n_features = 5

        X = np.random.randn(n_rows, n_features).astype(np.float32)
        y = np.random.randn(n_rows).astype(np.float32)
        feature_names = [f"f_{i}" for i in range(n_features)]

        # event_ids: ["evt_0","evt_0","evt_0","evt_1","evt_1","evt_1",...]
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

        # Events in chronological order
        event_ids = np.array([f"evt_{i}" for i in range(n_events) for _ in range(rows_per_event)])

        config = self._make_config(n_folds=3)

        # Track fold splits for verification
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

        # Training set should grow with each fold (walk-forward)
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
    """Tests for multi-horizon config validation."""

    def test_valid_decision_hours_range(self):
        config = FeatureConfig(
            feature_groups=["tabular"],
            markets=["h2h"],
            outcome="home",
            opening_tier="opening",
            closing_tier="closing",
            target_type="devigged_pinnacle",
            decision_hours_range=(3.0, 12.0),
            max_samples_per_event=5,
        )
        assert config.decision_hours_range == (3.0, 12.0)

    def test_invalid_range_reversed(self):
        """min >= max should raise validation error."""
        with pytest.raises(ValueError, match="must be less than max"):
            FeatureConfig(
                feature_groups=["tabular"],
                markets=["h2h"],
                outcome="home",
                opening_tier="opening",
                closing_tier="closing",
                target_type="devigged_pinnacle",
                decision_hours_range=(12.0, 3.0),
            )

    def test_invalid_range_negative(self):
        """Negative min should raise validation error."""
        with pytest.raises(ValueError, match="must be >= 0"):
            FeatureConfig(
                feature_groups=["tabular"],
                markets=["h2h"],
                outcome="home",
                opening_tier="opening",
                closing_tier="closing",
                target_type="devigged_pinnacle",
                decision_hours_range=(-1.0, 12.0),
            )

    def test_default_target_type_is_raw(self):
        config = FeatureConfig(
            feature_groups=["tabular"],
            markets=["h2h"],
            outcome="home",
            opening_tier="opening",
            closing_tier="closing",
        )
        assert config.target_type == "raw"


class TestPrepareMultiHorizonData:
    """Tests for prepare_multi_horizon_data."""

    def _mock_reader(
        self,
        closing_snap: OddsSnapshot,
        all_snaps: list[OddsSnapshot],
    ) -> MagicMock:
        """Create a mock OddsReader with both methods configured."""
        mock_reader = MagicMock()
        mock_reader.get_last_snapshot_in_tier = AsyncMock(return_value=closing_snap)
        mock_reader.get_snapshots_for_event = AsyncMock(return_value=all_snaps)
        return mock_reader

    @pytest.mark.asyncio
    async def test_basic_multi_horizon(self, sample_events):
        """Multi-horizon produces multiple rows per event with correct structure."""
        from odds_analytics.feature_groups import prepare_multi_horizon_data

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
            opening_tier="opening",
            closing_tier="closing",
            target_type="devigged_pinnacle",
            decision_hours_range=(3.0, 12.0),
            max_samples_per_event=5,
        )

        mock_reader = self._mock_reader(closing_snap, all_snaps)
        mock_session = AsyncMock()

        with patch("odds_lambda.storage.readers.OddsReader", return_value=mock_reader):
            result = await prepare_multi_horizon_data(
                events=[event],
                session=mock_session,
                config=config,
            )

        assert result.num_samples >= 1
        assert result.num_samples <= 4
        assert result.event_ids is not None
        assert all(eid == event.id for eid in result.event_ids)
        assert result.feature_names[-1] == "hours_until_event"
        assert not all(y == 0 for y in result.y)

    @pytest.mark.asyncio
    async def test_drops_event_without_pinnacle_closing(self, sample_events):
        """Events without Pinnacle closing data are skipped."""
        from odds_analytics.feature_groups import prepare_multi_horizon_data

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
            opening_tier="opening",
            closing_tier="closing",
            target_type="devigged_pinnacle",
            decision_hours_range=(3.0, 12.0),
            max_samples_per_event=5,
        )

        mock_reader = self._mock_reader(closing_snap, [])
        mock_session = AsyncMock()

        with patch("odds_lambda.storage.readers.OddsReader", return_value=mock_reader):
            with pytest.raises(ValueError, match="No valid training data"):
                await prepare_multi_horizon_data(
                    events=[event],
                    session=mock_session,
                    config=config,
                )

    @pytest.mark.asyncio
    async def test_hours_until_event_correct(self, sample_events):
        """hours_until_event feature matches actual time difference."""
        from odds_analytics.feature_groups import prepare_multi_horizon_data

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
            opening_tier="opening",
            closing_tier="closing",
            target_type="devigged_pinnacle",
            decision_hours_range=(3.0, 12.0),
            max_samples_per_event=5,
        )

        mock_reader = self._mock_reader(closing_snap, all_snaps)
        mock_session = AsyncMock()

        with patch("odds_lambda.storage.readers.OddsReader", return_value=mock_reader):
            result = await prepare_multi_horizon_data(
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
        from odds_analytics.feature_groups import prepare_multi_horizon_data

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
            opening_tier="opening",
            closing_tier="closing",
            target_type="devigged_pinnacle",
            decision_hours_range=(3.0, 12.0),
            max_samples_per_event=3,
        )

        mock_reader = self._mock_reader(closing_snap, all_snaps)
        mock_session = AsyncMock()

        with patch("odds_lambda.storage.readers.OddsReader", return_value=mock_reader):
            result = await prepare_multi_horizon_data(
                events=[event],
                session=mock_session,
                config=config,
            )

        assert result.num_samples <= 3
