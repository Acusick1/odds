"""Tests for the score_predictions job."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from odds_analytics.feature_groups import XGBoostAdapter, collect_event_data
from odds_analytics.training.config import FeatureConfig
from odds_core.models import Event, EventStatus, OddsSnapshot
from odds_lambda.jobs.score_predictions import _preload_caches, score_events

_TEST_FEATURE_CONFIG = FeatureConfig(
    adapter="xgboost",
    sharp_bookmakers=["bet365"],
    retail_bookmakers=["betway", "betfred", "bwin"],
    markets=["h2h"],
    outcome="home",
    feature_groups=("tabular",),
    target_type="devigged_bookmaker",
    target_bookmaker="bet365",
    sport_key="soccer_epl",
)

_EXPECTED_FEATURE_NAMES = XGBoostAdapter().feature_names(_TEST_FEATURE_CONFIG)

_TABULAR_STANDINGS_CONFIG = FeatureConfig(
    adapter="xgboost",
    sharp_bookmakers=["bet365"],
    retail_bookmakers=["betway", "betfred", "bwin"],
    markets=["h2h"],
    outcome="home",
    feature_groups=("tabular", "standings"),
    target_type="devigged_bookmaker",
    target_bookmaker="bet365",
    sport_key="soccer_epl",
)

_TABULAR_STANDINGS_FEATURE_NAMES = XGBoostAdapter().feature_names(_TABULAR_STANDINGS_CONFIG)


def _make_event(
    event_id: str = "test_event_1",
    home_team: str = "Arsenal",
    away_team: str = "Chelsea",
    hours_from_now: float = 48.0,
) -> Event:
    return Event(
        id=event_id,
        sport_key="soccer_epl",
        sport_title="EPL",
        commence_time=datetime.now(UTC) + timedelta(hours=hours_from_now),
        home_team=home_team,
        away_team=away_team,
        status=EventStatus.SCHEDULED,
    )


def _make_snapshot(
    event: Event,
    snapshot_id: int = 1,
    hours_before: float = 24.0,
) -> OddsSnapshot:
    raw_data = {
        "bookmakers": [
            {
                "key": "bet365",
                "title": "Bet365",
                "last_update": "2026-03-07T12:00:00Z",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": event.home_team, "price": -150},
                            {"name": "Draw", "price": 250},
                            {"name": event.away_team, "price": 200},
                        ],
                    }
                ],
            },
            {
                "key": "betway",
                "title": "Betway",
                "last_update": "2026-03-07T12:00:00Z",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": event.home_team, "price": -140},
                            {"name": "Draw", "price": 240},
                            {"name": event.away_team, "price": 190},
                        ],
                    }
                ],
            },
        ]
    }
    return OddsSnapshot(
        id=snapshot_id,
        event_id=event.id,
        snapshot_time=event.commence_time - timedelta(hours=hours_before),
        raw_data=raw_data,
        bookmaker_count=2,
        fetch_tier="pregame",
        hours_until_commence=hours_before,
    )


class TestAdapterTabular:
    @pytest.mark.asyncio
    async def test_tabular_only_vector_matches_legacy_extraction(self) -> None:
        """Adapter output for tabular-only must equal the prior hand-rolled vector.

        Replicates the pre-#354 extraction (TabularFeatureExtractor +
        hours_until_event concatenation, modulo np.nan_to_num) and asserts
        byte equality with the new XGBoostAdapter path.
        """
        from odds_analytics.backtesting import BacktestEvent
        from odds_analytics.feature_extraction import TabularFeatureExtractor
        from odds_analytics.feature_groups import resolve_outcome_name
        from odds_core.snapshot_utils import extract_odds_from_snapshot

        event = _make_event()
        snapshot = _make_snapshot(event)

        market = _TEST_FEATURE_CONFIG.primary_market
        outcome_name = resolve_outcome_name(_TEST_FEATURE_CONFIG, event)
        odds = extract_odds_from_snapshot(snapshot, event.id, market=market)
        backtest_event = BacktestEvent(
            id=event.id,
            commence_time=event.commence_time,
            home_team=event.home_team,
            away_team=event.away_team,
            home_score=0,
            away_score=0,
            status=event.status,
        )
        extractor = TabularFeatureExtractor.from_config(_TEST_FEATURE_CONFIG)
        tab_array = extractor.extract_features(
            event=backtest_event,
            odds_data=odds,
            outcome=outcome_name,
            market=market,
        ).to_array()
        hours_until = (event.commence_time - snapshot.snapshot_time).total_seconds() / 3600
        legacy_vector = np.concatenate([tab_array, np.array([hours_until])]).astype(np.float32)

        session = AsyncMock()
        snapshots_result = MagicMock()
        snapshots_result.scalars.return_value.all.return_value = [snapshot]
        session.execute = AsyncMock(return_value=snapshots_result)

        bundle = await collect_event_data(event, session, _TEST_FEATURE_CONFIG)
        adapter = XGBoostAdapter()
        output = adapter.transform(bundle, snapshot, _TEST_FEATURE_CONFIG)

        assert output is not None
        adapter_vector = output.features.astype(np.float32)
        assert adapter_vector.shape == legacy_vector.shape
        assert np.array_equal(adapter_vector, legacy_vector, equal_nan=True)

    @pytest.mark.asyncio
    async def test_returns_none_for_empty_snapshot(self) -> None:
        event = _make_event()
        snapshot = OddsSnapshot(
            id=1,
            event_id=event.id,
            snapshot_time=event.commence_time - timedelta(hours=24),
            raw_data={"bookmakers": []},
            bookmaker_count=0,
        )

        session = AsyncMock()
        snapshots_result = MagicMock()
        snapshots_result.scalars.return_value.all.return_value = [snapshot]
        session.execute = AsyncMock(return_value=snapshots_result)

        bundle = await collect_event_data(event, session, _TEST_FEATURE_CONFIG)
        adapter = XGBoostAdapter()
        output = adapter.transform(bundle, snapshot, _TEST_FEATURE_CONFIG)

        assert output is None


class TestScoreEvents:
    @pytest.mark.asyncio
    @patch("odds_lambda.jobs.score_predictions.collect_event_data")
    @patch("odds_lambda.jobs.score_predictions.load_model")
    @patch("odds_lambda.jobs.score_predictions.get_cached_version")
    @patch("odds_lambda.jobs.score_predictions.async_session_maker")
    async def test_scores_unscored_snapshots(
        self,
        mock_session_maker: MagicMock,
        mock_version: MagicMock,
        mock_load: MagicMock,
        mock_collect: MagicMock,
    ) -> None:
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.015])
        mock_load.return_value = {
            "model": mock_model,
            "feature_names": _EXPECTED_FEATURE_NAMES,
            "params": {},
            "feature_config": _TEST_FEATURE_CONFIG,
        }
        mock_version.return_value = '"etag123"'

        event = _make_event()
        snapshot = _make_snapshot(event)

        from odds_analytics.feature_groups import EventDataBundle

        mock_collect.return_value = EventDataBundle(
            event=event,
            snapshots=[snapshot],
            closing_snapshot=snapshot,
            pm_context=None,
        )

        mock_session = AsyncMock()
        events_result = MagicMock()
        events_result.scalars.return_value.all.return_value = [event]
        snapshots_result = MagicMock()
        snapshots_result.scalars.return_value.all.return_value = [snapshot]
        insert_result = MagicMock()
        insert_result.rowcount = 1
        mock_session.execute = AsyncMock(
            side_effect=[events_result, snapshots_result, insert_result]
        )

        mock_session_maker.return_value.__aenter__.return_value = mock_session

        stats = await score_events(model_name="epl-clv-home", bucket="test-bucket")

        assert stats["events_checked"] == 1
        assert stats["snapshots_scored"] == 1
        assert stats["errors"] == 0
        assert mock_session.execute.call_count == 3
        mock_model.predict.assert_called_once()

    @pytest.mark.asyncio
    @patch("odds_lambda.jobs.score_predictions._preload_caches")
    @patch("odds_lambda.jobs.score_predictions.collect_event_data")
    @patch("odds_lambda.jobs.score_predictions.load_model")
    @patch("odds_lambda.jobs.score_predictions.get_cached_version")
    @patch("odds_lambda.jobs.score_predictions.async_session_maker")
    async def test_scores_multi_group_path(
        self,
        mock_session_maker: MagicMock,
        mock_version: MagicMock,
        mock_load: MagicMock,
        mock_collect: MagicMock,
        mock_preload: AsyncMock,
    ) -> None:
        """Tabular + standings model scores end-to-end with caches provided."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.02])
        mock_load.return_value = {
            "model": mock_model,
            "feature_names": _TABULAR_STANDINGS_FEATURE_NAMES,
            "params": {},
            "feature_config": _TABULAR_STANDINGS_CONFIG,
        }
        mock_version.return_value = '"v2"'

        event = _make_event()
        snapshot = _make_snapshot(event)
        sentinel_standings: dict[str, list[Event]] = {"2025-26": []}
        mock_preload.return_value = (sentinel_standings, None, None, None)

        from odds_analytics.feature_groups import EventDataBundle

        mock_collect.return_value = EventDataBundle(
            event=event,
            snapshots=[snapshot],
            closing_snapshot=snapshot,
            pm_context=None,
        )

        mock_session = AsyncMock()
        events_result = MagicMock()
        events_result.scalars.return_value.all.return_value = [event]
        snapshots_result = MagicMock()
        snapshots_result.scalars.return_value.all.return_value = [snapshot]
        insert_result = MagicMock()
        insert_result.rowcount = 1
        mock_session.execute = AsyncMock(
            side_effect=[events_result, snapshots_result, insert_result]
        )
        mock_session_maker.return_value.__aenter__.return_value = mock_session

        stats = await score_events(model_name="epl-clv-multi", bucket="test-bucket")

        assert stats["events_checked"] == 1
        assert stats["snapshots_scored"] == 1
        assert stats["errors"] == 0

        mock_preload.assert_awaited_once()
        collect_kwargs = mock_collect.await_args.kwargs
        assert collect_kwargs["standings_cache"] is sentinel_standings
        assert collect_kwargs["match_stats_cache"] is None
        assert collect_kwargs["fixtures_df"] is None
        assert collect_kwargs["lineup_cache"] is None

        predict_input = mock_model.predict.call_args.args[0]
        assert predict_input.shape == (1, len(_TABULAR_STANDINGS_FEATURE_NAMES))

    @pytest.mark.asyncio
    @patch("odds_lambda.jobs.score_predictions.async_session_maker")
    @patch("odds_lambda.jobs.score_predictions.get_cached_version")
    @patch("odds_lambda.jobs.score_predictions.load_model")
    async def test_skips_on_feature_name_mismatch(
        self,
        mock_load: MagicMock,
        mock_version: MagicMock,
        mock_session_maker: MagicMock,
    ) -> None:
        """When the model's bundled feature names diverge from the adapter's,
        scoring logs and returns without invoking the model or DB."""
        mock_model = MagicMock()
        mock_load.return_value = {
            "model": mock_model,
            "feature_names": ["wrong_feature_1", "wrong_feature_2"],
            "params": {},
            "feature_config": _TEST_FEATURE_CONFIG,
        }
        mock_version.return_value = '"etag123"'

        stats = await score_events(model_name="bad-model", bucket="test-bucket")

        assert stats["events_checked"] == 0
        assert stats["snapshots_scored"] == 0
        mock_model.predict.assert_not_called()
        mock_session_maker.assert_not_called()

    @pytest.mark.asyncio
    @patch("odds_lambda.jobs.score_predictions.load_model")
    async def test_returns_empty_stats_on_model_load_failure(self, mock_load: MagicMock) -> None:
        mock_load.side_effect = FileNotFoundError("Model not found")

        stats = await score_events(model_name="missing", bucket="test-bucket")

        assert stats["events_checked"] == 0
        assert stats["snapshots_scored"] == 0

    @pytest.mark.asyncio
    @patch("odds_lambda.jobs.score_predictions.load_model")
    @patch("odds_lambda.jobs.score_predictions.get_cached_version")
    @patch("odds_lambda.jobs.score_predictions.async_session_maker")
    async def test_no_upcoming_events(
        self,
        mock_session_maker: MagicMock,
        mock_version: MagicMock,
        mock_load: MagicMock,
    ) -> None:
        mock_load.return_value = {
            "model": MagicMock(),
            "feature_names": _EXPECTED_FEATURE_NAMES,
            "params": {},
            "feature_config": _TEST_FEATURE_CONFIG,
        }
        mock_version.return_value = '"v1"'

        mock_session = AsyncMock()
        events_result = MagicMock()
        events_result.scalars.return_value.all.return_value = []
        mock_session.execute = AsyncMock(return_value=events_result)
        mock_session_maker.return_value.__aenter__.return_value = mock_session

        stats = await score_events(model_name="test", bucket="test-bucket")

        assert stats["events_checked"] == 0
        assert stats["snapshots_scored"] == 0

    @pytest.mark.asyncio
    @patch("odds_lambda.jobs.score_predictions.load_model")
    async def test_returns_empty_stats_when_no_config(self, mock_load: MagicMock) -> None:
        mock_load.return_value = {
            "model": MagicMock(),
            "feature_names": [],
            "params": {},
            "feature_config": None,
        }

        stats = await score_events(model_name="test", bucket="test-bucket")

        assert stats["events_checked"] == 0
        assert stats["snapshots_scored"] == 0

    @pytest.mark.asyncio
    @patch("odds_lambda.jobs.score_predictions.load_model")
    async def test_returns_empty_stats_when_no_sport_key(self, mock_load: MagicMock) -> None:
        config_no_sport = FeatureConfig(
            adapter="xgboost",
            sharp_bookmakers=["bet365"],
            retail_bookmakers=["betway"],
            markets=["h2h"],
        )
        mock_load.return_value = {
            "model": MagicMock(),
            "feature_names": [],
            "params": {},
            "feature_config": config_no_sport,
        }

        stats = await score_events(model_name="test", bucket="test-bucket")

        assert stats["events_checked"] == 0
        assert stats["snapshots_scored"] == 0


def _config_with_groups(*groups: str) -> FeatureConfig:
    return FeatureConfig(
        adapter="xgboost",
        sharp_bookmakers=["bet365"],
        retail_bookmakers=["betway", "betfred", "bwin"],
        markets=["h2h"],
        outcome="home",
        feature_groups=tuple(groups),
        target_type="devigged_bookmaker",
        target_bookmaker="bet365",
        sport_key="soccer_epl",
    )


class TestPreloadCaches:
    """Direct unit tests for _preload_caches gating."""

    @pytest.mark.asyncio
    @patch("odds_lambda.jobs.score_predictions._load_lineup_cache", new_callable=AsyncMock)
    @patch("odds_lambda.jobs.score_predictions._load_fixtures_df", new_callable=AsyncMock)
    @patch("odds_analytics.match_stats_features.load_match_stats_cache", new_callable=AsyncMock)
    @patch("odds_analytics.standings_features.load_season_events_cache", new_callable=AsyncMock)
    async def test_tabular_only_loads_no_caches(
        self,
        mock_standings: AsyncMock,
        mock_match_stats: AsyncMock,
        mock_fixtures: AsyncMock,
        mock_lineup: AsyncMock,
    ) -> None:
        config = _config_with_groups("tabular")
        session = AsyncMock()

        result = await _preload_caches(session, config, "soccer_epl")

        assert result == (None, None, None, None)
        mock_standings.assert_not_called()
        mock_match_stats.assert_not_called()
        mock_fixtures.assert_not_called()
        mock_lineup.assert_not_called()

    @pytest.mark.asyncio
    @patch("odds_lambda.jobs.score_predictions._load_lineup_cache", new_callable=AsyncMock)
    @patch("odds_lambda.jobs.score_predictions._load_fixtures_df", new_callable=AsyncMock)
    @patch("odds_analytics.match_stats_features.load_match_stats_cache", new_callable=AsyncMock)
    @patch("odds_analytics.standings_features.load_season_events_cache", new_callable=AsyncMock)
    async def test_standings_only_loads_standings_cache(
        self,
        mock_standings: AsyncMock,
        mock_match_stats: AsyncMock,
        mock_fixtures: AsyncMock,
        mock_lineup: AsyncMock,
    ) -> None:
        sentinel: dict[str, list[Event]] = {"2025-26": []}
        mock_standings.return_value = sentinel
        config = _config_with_groups("tabular", "standings")
        session = AsyncMock()

        standings_cache, match_stats_cache, fixtures_df, lineup_cache = await _preload_caches(
            session, config, "soccer_epl"
        )

        assert standings_cache is sentinel
        assert match_stats_cache is None
        assert fixtures_df is None
        assert lineup_cache is None
        mock_standings.assert_awaited_once_with(session, "soccer_epl")
        mock_match_stats.assert_not_called()
        mock_fixtures.assert_not_called()
        mock_lineup.assert_not_called()

    @pytest.mark.asyncio
    @patch("odds_lambda.jobs.score_predictions._load_lineup_cache", new_callable=AsyncMock)
    @patch("odds_lambda.jobs.score_predictions._load_fixtures_df", new_callable=AsyncMock)
    @patch("odds_analytics.match_stats_features.load_match_stats_cache", new_callable=AsyncMock)
    @patch("odds_analytics.standings_features.load_season_events_cache", new_callable=AsyncMock)
    async def test_match_stats_only_loads_match_stats_cache(
        self,
        mock_standings: AsyncMock,
        mock_match_stats: AsyncMock,
        mock_fixtures: AsyncMock,
        mock_lineup: AsyncMock,
    ) -> None:
        sentinel = MagicMock(name="match_stats_cache")
        mock_match_stats.return_value = sentinel
        config = _config_with_groups("tabular", "match_stats")
        session = AsyncMock()

        standings_cache, match_stats_cache, fixtures_df, lineup_cache = await _preload_caches(
            session, config, "soccer_epl"
        )

        assert standings_cache is None
        assert match_stats_cache is sentinel
        assert fixtures_df is None
        assert lineup_cache is None
        mock_standings.assert_not_called()
        mock_match_stats.assert_awaited_once_with(session, "soccer_epl")
        mock_fixtures.assert_not_called()
        mock_lineup.assert_not_called()

    @pytest.mark.asyncio
    @patch("odds_lambda.jobs.score_predictions._load_lineup_cache", new_callable=AsyncMock)
    @patch("odds_lambda.jobs.score_predictions._load_fixtures_df", new_callable=AsyncMock)
    @patch("odds_analytics.match_stats_features.load_match_stats_cache", new_callable=AsyncMock)
    @patch("odds_analytics.standings_features.load_season_events_cache", new_callable=AsyncMock)
    async def test_epl_schedule_loads_fixtures_and_standings(
        self,
        mock_standings: AsyncMock,
        mock_match_stats: AsyncMock,
        mock_fixtures: AsyncMock,
        mock_lineup: AsyncMock,
    ) -> None:
        """epl_schedule pulls in standings_cache too (shared by season-rest features)."""
        standings_sentinel: dict[str, list[Event]] = {"2025-26": []}
        fixtures_sentinel = MagicMock(name="fixtures_df")
        mock_standings.return_value = standings_sentinel
        mock_fixtures.return_value = fixtures_sentinel
        config = _config_with_groups("tabular", "epl_schedule")
        session = AsyncMock()

        standings_cache, match_stats_cache, fixtures_df, lineup_cache = await _preload_caches(
            session, config, "soccer_epl"
        )

        assert standings_cache is standings_sentinel
        assert match_stats_cache is None
        assert fixtures_df is fixtures_sentinel
        assert lineup_cache is None
        mock_standings.assert_awaited_once_with(session, "soccer_epl")
        mock_fixtures.assert_awaited_once_with(session)
        mock_match_stats.assert_not_called()
        mock_lineup.assert_not_called()

    @pytest.mark.asyncio
    @patch("odds_lambda.jobs.score_predictions._load_lineup_cache", new_callable=AsyncMock)
    @patch("odds_lambda.jobs.score_predictions._load_fixtures_df", new_callable=AsyncMock)
    @patch("odds_analytics.match_stats_features.load_match_stats_cache", new_callable=AsyncMock)
    @patch("odds_analytics.standings_features.load_season_events_cache", new_callable=AsyncMock)
    async def test_epl_lineup_loads_lineup_cache(
        self,
        mock_standings: AsyncMock,
        mock_match_stats: AsyncMock,
        mock_fixtures: AsyncMock,
        mock_lineup: AsyncMock,
    ) -> None:
        sentinel = MagicMock(name="lineup_cache")
        mock_lineup.return_value = sentinel
        config = _config_with_groups("tabular", "epl_lineup")
        session = AsyncMock()

        standings_cache, match_stats_cache, fixtures_df, lineup_cache = await _preload_caches(
            session, config, "soccer_epl"
        )

        assert standings_cache is None
        assert match_stats_cache is None
        assert fixtures_df is None
        assert lineup_cache is sentinel
        mock_standings.assert_not_called()
        mock_match_stats.assert_not_called()
        mock_fixtures.assert_not_called()
        mock_lineup.assert_awaited_once_with(session)

    @pytest.mark.asyncio
    @patch("odds_lambda.jobs.score_predictions._load_lineup_cache", new_callable=AsyncMock)
    @patch("odds_lambda.jobs.score_predictions._load_fixtures_df", new_callable=AsyncMock)
    @patch("odds_analytics.match_stats_features.load_match_stats_cache", new_callable=AsyncMock)
    @patch("odds_analytics.standings_features.load_season_events_cache", new_callable=AsyncMock)
    async def test_all_groups_load_all_caches(
        self,
        mock_standings: AsyncMock,
        mock_match_stats: AsyncMock,
        mock_fixtures: AsyncMock,
        mock_lineup: AsyncMock,
    ) -> None:
        standings_sentinel: dict[str, list[Event]] = {"2025-26": []}
        match_stats_sentinel = MagicMock(name="match_stats_cache")
        fixtures_sentinel = MagicMock(name="fixtures_df")
        lineup_sentinel = MagicMock(name="lineup_cache")
        mock_standings.return_value = standings_sentinel
        mock_match_stats.return_value = match_stats_sentinel
        mock_fixtures.return_value = fixtures_sentinel
        mock_lineup.return_value = lineup_sentinel

        config = _config_with_groups(
            "tabular",
            "standings",
            "match_stats",
            "epl_schedule",
            "epl_lineup",
        )
        session = AsyncMock()

        standings_cache, match_stats_cache, fixtures_df, lineup_cache = await _preload_caches(
            session, config, "soccer_epl"
        )

        assert standings_cache is standings_sentinel
        assert match_stats_cache is match_stats_sentinel
        assert fixtures_df is fixtures_sentinel
        assert lineup_cache is lineup_sentinel
        mock_standings.assert_awaited_once_with(session, "soccer_epl")
        mock_match_stats.assert_awaited_once_with(session, "soccer_epl")
        mock_fixtures.assert_awaited_once_with(session)
        mock_lineup.assert_awaited_once_with(session)
