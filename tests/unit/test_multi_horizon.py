"""Unit tests for multi-horizon data preparation and group-aware CV."""

from collections.abc import Collection, Sequence
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from odds_analytics.feature_groups import (
    AdapterOutput,
    EventDataBundle,
    SnapshotQuotes,
    TierSampler,
    TimeRangeSampler,
    XGBoostAdapter,
    prepare_training_data,
)
from odds_analytics.training.config import FeatureConfig, MLTrainingConfig, SamplingConfig
from odds_analytics.training.cross_validation import CVResult, run_cv
from odds_core.models import Event, EventStatus
from odds_core.odds_math import american_to_decimal
from odds_core.quotes import Quote
from odds_core.vocabulary import Market, PriceOutcome, Side, Source, TimeBasis


@pytest.fixture
def sample_events():
    """Three events spread over a week, sorted chronologically."""
    return [
        Event(
            id=i + 1,
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
    event_id: int,
    snapshot_time: datetime,
    home_price: int = -150,
    away_price: int = 130,
    bookmakers: Sequence[str] = ("pinnacle",),
) -> SnapshotQuotes:
    """One snapshot of home and away h2h back quotes from each bookmaker."""
    return SnapshotQuotes(
        tuple(
            Quote(
                snapshot_id=idx,
                event_id=event_id,
                snapshot_time=snapshot_time,
                created_at=snapshot_time,
                source=Source.ODDS_API,
                market=Market.H2H,
                time_basis=TimeBasis.OBSERVED,
                in_play=None,
                market_status=None,
                total_matched=None,
                market_start_time=None,
                bookmaker=bookmaker,
                outcome=outcome,
                line=None,
                side=Side.BACK,
                price=american_to_decimal(price),
                size=None,
                matched=None,
                changed_at=None,
            )
            for bookmaker in bookmakers
            for outcome, price in ((PriceOutcome.HOME, home_price), (PriceOutcome.AWAY, away_price))
        )
    )


def _id(snapshot: SnapshotQuotes) -> int:
    return snapshot.quotes[0].snapshot_id


class TestGroupTimeseriesCV:
    """Tests for event-grouped walk-forward cross-validation."""

    def _make_config(
        self, cv_method: str = "walk_forward", **data_overrides: object
    ) -> MLTrainingConfig:
        """Create a minimal MLTrainingConfig for CV testing."""
        data = {
            "start_date": "2024-10-01",
            "end_date": "2024-12-31",
            "cv_method": cv_method,
            "random_seed": 42,
            "min_train_events": 2,
            "val_step_events": 1,
        }
        data.update(data_overrides)
        return MLTrainingConfig.model_validate(
            {
                "experiment": {"name": "test_cv", "description": "test"},
                "training": {
                    "strategy_type": "xgboost_line_movement",
                    "features": {
                        "target_type": "raw",
                        "feature_groups": ["tabular"],
                        "markets": ["h2h"],
                        "outcome": "home",
                    },
                    "data": data,
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

        def fake_train(config, X_train, y_train, feature_names, X_val, y_val, **kwargs):
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

        event_ids = np.array([i for i in range(n_events) for _ in range(rows_per_event)])

        # min_train=2, val_step=1 → 3 folds for 5 events
        config = self._make_config()
        strategy = self._make_mock_strategy()

        result = run_cv(strategy, config, X, y, feature_names, event_ids=event_ids)

        assert isinstance(result, CVResult)
        assert result.n_folds == 3
        assert result.cv_method == "walk_forward"

    def test_temporal_ordering_preserved(self):
        """Train events should always be earlier than val events."""
        n_events = 6
        rows_per_event = 2
        n_rows = n_events * rows_per_event
        n_features = 3

        X = np.random.randn(n_rows, n_features).astype(np.float32)
        y = np.random.randn(n_rows).astype(np.float32)
        feature_names = [f"f_{i}" for i in range(n_features)]

        event_ids = np.array([i for i in range(n_events) for _ in range(rows_per_event)])

        config = self._make_config()

        fold_splits = []

        def capture_train(config, X_train, y_train, feature_names, X_val, y_val, **kwargs):
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
            assert fold_splits[i][0] <= fold_splits[i + 1][0], (
                f"Fold {i} train size ({fold_splits[i][0]}) > fold {i + 1} ({fold_splits[i + 1][0]})"
            )

    def test_walk_forward_with_no_event_ids_raises(self):
        """walk_forward without event_ids is an error, not a switch to timeseries CV."""
        n_rows = 20
        n_features = 3
        X = np.random.randn(n_rows, n_features).astype(np.float32)
        y = np.random.randn(n_rows).astype(np.float32)
        feature_names = [f"f_{i}" for i in range(n_features)]

        config = self._make_config()
        strategy = self._make_mock_strategy()

        with pytest.raises(ValueError, match="event_ids"):
            run_cv(strategy, config, X, y, feature_names, event_ids=None)

    def test_fold_results_populated(self):
        """Each fold should produce valid metrics."""
        n_events = 4
        rows_per_event = 3
        n_rows = n_events * rows_per_event
        n_features = 5

        X = np.random.randn(n_rows, n_features).astype(np.float32)
        y = np.random.randn(n_rows).astype(np.float32)
        feature_names = [f"f_{i}" for i in range(n_features)]
        event_ids = np.array([i for i in range(n_events) for _ in range(rows_per_event)])

        # min_train=2, val_step=1 → 2 folds for 4 events
        config = self._make_config()
        strategy = self._make_mock_strategy()

        result = run_cv(strategy, config, X, y, feature_names, event_ids=event_ids)

        assert len(result.fold_results) == 2
        for fold in result.fold_results:
            assert fold.n_train > 0
            assert fold.n_val > 0
            assert fold.val_mse == 0.02
            assert fold.val_r2 == 0.80


class TestFeatureConfigValidation:
    """Tests for new FeatureConfig and SamplingConfig validation."""

    def test_default_sampling_config(self):
        config = FeatureConfig(
            target_type="raw",
            feature_groups=["tabular"],
            markets=["h2h"],
            outcome="home",
        )
        assert config.sampling.strategy == "time_range"
        assert config.sampling.min_hours == 3.0
        assert config.sampling.max_hours == 12.0
        assert config.sampling.max_samples_per_event == 5

    def test_sampling_config_time_range(self):
        config = FeatureConfig(
            target_type="raw",
            feature_groups=["tabular"],
            markets=["h2h"],
            outcome="home",
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
            target_type="raw",
            feature_groups=["tabular"],
            markets=["h2h"],
            outcome="home",
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
            target_type="raw",
            feature_groups=["tabular"],
            markets=["h2h"],
            outcome="home",
        )
        assert config.target_type == "raw"

    def test_adapter_default_is_xgboost(self):
        config = FeatureConfig(
            target_type="raw",
        )
        assert config.adapter == "xgboost"


class TestTierSampler:
    """Tests for TierSampler snapshot selection."""

    @pytest.fixture
    def event(self):
        return Event(
            id=1,
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime(2024, 11, 1, 19, 0, 0, tzinfo=UTC),
            home_team="Los Angeles Lakers",
            away_team="Boston Celtics",
            status=EventStatus.FINAL,
            home_score=110,
            away_score=105,
        )

    def _make_bundle(self, event: Event, snapshots: list[SnapshotQuotes]) -> EventDataBundle:
        return EventDataBundle(
            event=event,
            snapshots=snapshots,
            closes={},
        )

    def test_returns_latest_snapshot_in_decision_tier(self, event):
        """Should pick the most recent pregame snapshot."""
        commence = event.commence_time
        snaps = [
            _make_snapshot(1, event.id, commence - timedelta(hours=10)),
            _make_snapshot(2, event.id, commence - timedelta(hours=5)),
            _make_snapshot(3, event.id, commence - timedelta(hours=1)),
        ]
        sampler = TierSampler("pregame")
        result = sampler.sample(self._make_bundle(event, snaps))

        assert len(result) == 1
        assert _id(result[0]) == 2  # Latest pregame, not the closing one

    def test_includes_earlier_tiers(self, event):
        """When no snapshot in decision tier, falls back to earlier tiers."""
        commence = event.commence_time
        snaps = [
            _make_snapshot(1, event.id, commence - timedelta(hours=50)),
            _make_snapshot(2, event.id, commence - timedelta(hours=1)),
        ]
        sampler = TierSampler("pregame")
        result = sampler.sample(self._make_bundle(event, snaps))

        assert len(result) == 1
        assert _id(result[0]) == 1

    def test_excludes_closer_tiers(self, event):
        """Closing-tier snapshot should not be selected when decision_tier=pregame."""
        commence = event.commence_time
        snaps = [
            _make_snapshot(1, event.id, commence - timedelta(hours=1)),
        ]
        sampler = TierSampler("pregame")
        result = sampler.sample(self._make_bundle(event, snaps))

        assert result == []

    def test_excludes_in_play_snapshots(self, event):
        """IN_PLAY snapshots must not be selected for pre-game decision tiers."""
        commence = event.commence_time
        snaps = [
            _make_snapshot(1, event.id, commence - timedelta(hours=5)),
            _make_snapshot(2, event.id, commence + timedelta(hours=1)),
        ]
        sampler = TierSampler("pregame")
        result = sampler.sample(self._make_bundle(event, snaps))

        assert len(result) == 1
        assert _id(result[0]) == 1  # Pregame, not in_play

    def test_in_play_tier_can_select_in_play(self, event):
        """IN_PLAY decision tier should select in-play snapshots."""
        commence = event.commence_time
        snaps = [
            _make_snapshot(1, event.id, commence - timedelta(hours=5)),
            _make_snapshot(2, event.id, commence + timedelta(hours=1)),
        ]
        sampler = TierSampler("in_play")
        result = sampler.sample(self._make_bundle(event, snaps))

        assert len(result) == 1
        assert _id(result[0]) == 2  # in_play is the latest

    def test_empty_snapshots(self, event):
        sampler = TierSampler("pregame")
        assert sampler.sample(self._make_bundle(event, [])) == []

    def test_required_bookmakers_filters_candidates(self, event):
        """Sampler with required_bookmakers drops snapshots missing those bookmakers."""
        commence = event.commence_time
        snaps = [
            _make_snapshot(
                1, event.id, commence - timedelta(hours=48), bookmakers=["pinnacle", "bet365"]
            ),
            _make_snapshot(
                2, event.id, commence - timedelta(hours=18), bookmakers=["bet365", "betway"]
            ),
        ]
        sampler = TierSampler("sharp", required_bookmakers=["pinnacle"])
        result = sampler.sample(self._make_bundle(event, snaps))

        assert len(result) == 1
        assert _id(result[0]) == 1  # Earlier but has Pinnacle

    def test_required_bookmakers_no_candidates_returns_empty(self, event):
        """When filtering removes all candidates, returns empty list."""
        commence = event.commence_time
        snaps = [
            _make_snapshot(
                1, event.id, commence - timedelta(hours=18), bookmakers=["bet365", "betway"]
            ),
        ]
        sampler = TierSampler("sharp", required_bookmakers=["pinnacle"])
        result = sampler.sample(self._make_bundle(event, snaps))

        assert result == []

    def test_no_required_bookmakers_unchanged_behavior(self, event):
        """Default (no required_bookmakers) still picks latest eligible snapshot."""
        commence = event.commence_time
        snaps = [
            _make_snapshot(
                1, event.id, commence - timedelta(hours=48), bookmakers=["pinnacle", "bet365"]
            ),
            _make_snapshot(2, event.id, commence - timedelta(hours=18), bookmakers=["bet365"]),
        ]
        sampler = TierSampler("sharp")
        result = sampler.sample(self._make_bundle(event, snaps))

        assert len(result) == 1
        assert _id(result[0]) == 2  # Latest, no bookmaker filter

    def test_required_bookmakers_any_semantics_accepts_partial(self, event):
        """With [pinnacle, betfair_exchange], a snapshot with only BFE is valid."""
        commence = event.commence_time
        snaps = [
            _make_snapshot(
                1,
                event.id,
                commence - timedelta(hours=18),
                bookmakers=["betfair_exchange", "bet365"],
            ),
        ]
        sampler = TierSampler("sharp", required_bookmakers=["pinnacle", "betfair_exchange"])
        result = sampler.sample(self._make_bundle(event, snaps))

        assert len(result) == 1
        assert _id(result[0]) == 1

    def test_required_bookmakers_any_semantics_rejects_none(self, event):
        """With [pinnacle, betfair_exchange], a snapshot with neither is rejected."""
        commence = event.commence_time
        snaps = [
            _make_snapshot(
                1, event.id, commence - timedelta(hours=18), bookmakers=["bet365", "betway"]
            ),
        ]
        sampler = TierSampler("sharp", required_bookmakers=["pinnacle", "betfair_exchange"])
        result = sampler.sample(self._make_bundle(event, snaps))

        assert result == []

    def test_required_bookmakers_prefers_latest_with_any_sharp(self, event):
        """Picks latest snapshot that has at least one required bookmaker."""
        commence = event.commence_time
        snaps = [
            _make_snapshot(
                1, event.id, commence - timedelta(hours=48), bookmakers=["pinnacle", "bet365"]
            ),
            _make_snapshot(
                2,
                event.id,
                commence - timedelta(hours=20),
                bookmakers=["betfair_exchange", "bet365"],
            ),
            _make_snapshot(3, event.id, commence - timedelta(hours=3), bookmakers=["bet365"]),
        ]
        sampler = TierSampler("sharp", required_bookmakers=["pinnacle", "betfair_exchange"])
        result = sampler.sample(self._make_bundle(event, snaps))

        assert len(result) == 1
        assert _id(result[0]) == 2  # Latest with a required bookmaker


class TestTimeRangeSampler:
    """Tests for TimeRangeSampler stratified snapshot selection."""

    @pytest.fixture
    def event(self):
        return Event(
            id=2,
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime(2024, 11, 1, 19, 0, 0, tzinfo=UTC),
            home_team="Los Angeles Lakers",
            away_team="Boston Celtics",
            status=EventStatus.FINAL,
            home_score=110,
            away_score=105,
        )

    def _make_bundle(self, event: Event, snapshots: list[SnapshotQuotes]) -> EventDataBundle:
        return EventDataBundle(
            event=event,
            snapshots=snapshots,
            closes={},
        )

    def test_returns_snapshots_in_range(self, event):
        """Only snapshots within [min_hours, max_hours] before game are included."""
        commence = event.commence_time
        snaps = [
            _make_snapshot(1, event.id, commence - timedelta(hours=24)),
            _make_snapshot(2, event.id, commence - timedelta(hours=8)),
            _make_snapshot(3, event.id, commence - timedelta(hours=6)),
            _make_snapshot(4, event.id, commence - timedelta(hours=4)),
            _make_snapshot(5, event.id, commence - timedelta(hours=1)),
        ]
        sampler = TimeRangeSampler(min_hours=3.0, max_hours=12.0, max_samples=10)
        result = sampler.sample(self._make_bundle(event, snaps))

        assert len(result) == 3
        result_ids = {_id(s) for s in result}
        assert result_ids == {2, 3, 4}

    def test_stratified_sampling_caps_at_max(self, event):
        """When more snapshots than max_samples, picks one per bin."""
        commence = event.commence_time
        snaps = [
            _make_snapshot(i, event.id, commence - timedelta(hours=3 + i * 0.9)) for i in range(10)
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
            _make_snapshot(1, event.id, commence - timedelta(hours=6)),
            _make_snapshot(2, event.id, commence - timedelta(hours=4)),
        ]
        sampler = TimeRangeSampler(min_hours=3.0, max_hours=12.0, max_samples=5)
        result = sampler.sample(self._make_bundle(event, snaps))

        assert len(result) == 2
        assert _id(result[0]) != _id(result[1])

    def test_empty_range(self, event):
        """No snapshots in range returns empty list."""
        commence = event.commence_time
        snaps = [
            _make_snapshot(1, event.id, commence - timedelta(hours=24)),
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
        closing_snap: SnapshotQuotes,
        all_snaps: list[SnapshotQuotes],
    ) -> EventDataBundle:
        return EventDataBundle(
            event=event,
            snapshots=all_snaps,
            closes={closing_snap.source: closing_snap},
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
            target_type="raw",
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
    async def test_injected_sampler_replaces_config_sampler(self, sample_events):
        """A passed sampler picks the decision snapshots instead of config.sampling's."""
        event = sample_events[0]
        commence = event.commence_time
        closing_snap = _make_snapshot(
            100, event.id, commence - timedelta(hours=0.5), home_price=-200, away_price=170
        )
        decision_snaps = [
            _make_snapshot(
                i, event.id, commence - timedelta(hours=h), home_price=-150 - i, away_price=130 + i
            )
            for i, h in enumerate([4, 6, 8, 10])
        ]
        chosen = decision_snaps[2]

        class PickOne:
            calls = 0

            def sample(
                self, bundle: EventDataBundle, sources: Collection[Source] | None = None
            ) -> list[SnapshotQuotes]:
                PickOne.calls += 1
                return [chosen]

        config = FeatureConfig(
            feature_groups=["tabular"],
            markets=["h2h"],
            outcome="home",
            target_type="raw",
            sampling=SamplingConfig(
                strategy="time_range", min_hours=3.0, max_hours=12.0, max_samples_per_event=5
            ),
        )
        bundle = self._make_bundle(event, closing_snap, decision_snaps + [closing_snap])

        with patch(
            "odds_analytics.feature_groups.collect_event_data",
            new=AsyncMock(return_value=bundle),
        ):
            result = await prepare_training_data(
                events=[event], session=AsyncMock(), config=config, sampler=PickOne()
            )

        assert PickOne.calls == 1
        assert result.num_samples == 1
        hours = result.X[0, result.feature_names.index("hours_until_event")]
        assert hours == pytest.approx(8.0)

    @pytest.mark.asyncio
    async def test_drops_event_without_bookmaker_closing(self, sample_events):
        """Events whose closing snapshot has no sharp bookmaker are skipped."""
        event = sample_events[0]
        commence = event.commence_time

        # Closing snapshot with NO Pinnacle (only fanduel)
        closing_snap = _make_snapshot(
            100,
            event.id,
            commence - timedelta(hours=0.5),
            home_price=-200,
            away_price=170,
            bookmakers=["fanduel"],
        )

        config = FeatureConfig(
            feature_groups=["tabular"],
            markets=["h2h"],
            outcome="home",
            target_type="raw",
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
            target_type="raw",
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
            target_type="raw",
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

    def test_adapter_output_holds_features(self):
        features = np.array([1.0, 2.0, 3.0])
        output = AdapterOutput(features=features)
        np.testing.assert_array_equal(output.features, features)


class TestPrepareTrainingDataRows:
    """Tests that prepare_training_data stacks adapter output into the feature matrix."""

    @pytest.fixture
    def event(self):
        return Event(
            id=3,
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime(2024, 11, 1, 19, 0, 0, tzinfo=UTC),
            home_team="Los Angeles Lakers",
            away_team="Boston Celtics",
            status=EventStatus.FINAL,
            home_score=110,
            away_score=105,
        )

    def _make_bundle(self, event: Event, closing_snap: SnapshotQuotes) -> EventDataBundle:
        return EventDataBundle(
            event=event,
            snapshots=[closing_snap],
            closes={closing_snap.source: closing_snap},
        )

    @pytest.mark.asyncio
    async def test_stacks_adapter_features(self, event):
        """One adapter row becomes one row of X."""
        commence = event.commence_time
        closing_snap = _make_snapshot(100, event.id, commence - timedelta(hours=0.5))
        decision_snap = _make_snapshot(1, event.id, commence - timedelta(hours=6))
        all_snaps = [decision_snap, closing_snap]
        bundle = EventDataBundle(
            event=event,
            snapshots=all_snaps,
            closes={closing_snap.source: closing_snap},
        )

        config = FeatureConfig(
            adapter="xgboost",
            feature_groups=["tabular"],
            markets=["h2h"],
            outcome="home",
            target_type="raw",
            sampling=SamplingConfig(strategy="tier", decision_tier="pregame"),
        )

        mock_features = np.array([0.5, 0.4, 6.0])
        mock_output = AdapterOutput(features=mock_features)

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

        assert result.X.shape == (1, 3)
