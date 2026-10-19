"""prepare_training_data_from_config with target_type retail_ev against a database."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest
from odds_analytics.feature_groups import TierSampler, prepare_training_data
from odds_analytics.training import (
    DataConfig,
    ExperimentConfig,
    FeatureConfig,
    MLTrainingConfig,
    TrainingConfig,
    XGBoostConfig,
    prepare_training_data_from_config,
)
from odds_core.models import EventStatus
from odds_lambda.fetch_tier import FetchTier

from tests.test_helpers import make_epl_event, make_tagged_snapshot, three_way_bookmaker

pytestmark = pytest.mark.integration

FIRST_KICKOFF = datetime(2025, 9, 13, 14, 0, tzinfo=UTC)


@pytest.fixture
async def retail_ev_events(pglite_async_session):
    session = pglite_async_session
    for i in range(3):
        kickoff = FIRST_KICKOFF + timedelta(days=7 * i)
        event_id = f"fduk_retail_ev_{i}"
        session.add(
            make_epl_event(event_id, kickoff, status=EventStatus.FINAL, home_score=1, away_score=0)
        )
        session.add(
            make_tagged_snapshot(
                event_id,
                kickoff - timedelta(hours=40),
                [
                    three_way_bookmaker("bet365", (2.0, 3.5, 4.0)),
                    three_way_bookmaker("williamhill", (1.95, 3.6, 4.1)),
                    three_way_bookmaker("betfair_exchange", (2.1, 3.7, 4.3)),
                ],
                "football_data_uk",
                "fduk_collection_deadline",
            )
        )
        session.add(
            make_tagged_snapshot(
                event_id,
                kickoff,
                [three_way_bookmaker("betfair_exchange", (2.2, 3.6, 4.0))],
                "football_data_uk",
                "kickoff",
            )
        )
    await session.commit()
    return session


def _config() -> MLTrainingConfig:
    return MLTrainingConfig(
        experiment=ExperimentConfig(name="retail_ev_integration"),
        training=TrainingConfig(
            strategy_type="xgboost_line_movement",
            data=DataConfig(
                start_date=date(2025, 9, 1),
                end_date=date(2025, 10, 31),
                test_split=0.0,
                validation_split=0.0,
                cv_method="walk_forward",
                min_train_events=1,
                val_step_events=1,
                sport_key="soccer_epl",
            ),
            model=XGBoostConfig(n_estimators=5),
            features=FeatureConfig(target_type="retail_ev", markets=["1x2"], feature_groups=()),
        ),
    )


class TestRetailEvTrainingData:
    async def test_rows_targets_and_weights(self, retail_ev_events) -> None:
        result = await prepare_training_data_from_config(_config(), retail_ev_events)

        assert result.num_train_samples == 3 * 6
        assert result.sample_weights_train is not None
        names = result.feature_names
        implied = result.X_train[:, names.index("retail_implied_prob")].astype(np.float64)
        expected = implied**2 / 6
        np.testing.assert_allclose(
            result.sample_weights_train, expected / expected.mean(), rtol=1e-5
        )
        assert result.event_ids_train is not None
        assert len(set(result.event_ids_train.tolist())) == 3
        assert result.commence_times_train is not None
        assert len(set(result.commence_times_train.tolist())) == 3

    async def test_rejects_a_sampler(self, retail_ev_events) -> None:
        config = _config().training.features
        with pytest.raises(ValueError, match="sampler"):
            await prepare_training_data(
                [], retail_ev_events, config, sampler=TierSampler(FetchTier.SHARP)
            )
