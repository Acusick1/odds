"""Unit tests for feature_groups helpers."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from odds_analytics.feature_groups import (
    _should_filter_missing_sharp,
    preload_feature_group_caches,
    snapshot_has_bookmaker,
)
from odds_analytics.training.config import FeatureConfig
from odds_core.models import Event, EventStatus, OddsSnapshot


def _make_event() -> Event:
    return Event(
        id="evt_1",
        sport_key="soccer_epl",
        sport_title="EPL",
        commence_time=datetime(2025, 12, 1, 15, 0, tzinfo=UTC),
        home_team="Arsenal",
        away_team="Chelsea",
        status=EventStatus.FINAL,
        home_score=2,
        away_score=1,
    )


def _make_snapshot(
    bookmakers: list[dict[str, list[str]]],
    *,
    event_id: str = "evt_1",
    hours_before: float = 0.5,
) -> OddsSnapshot:
    """Build snapshot with explicit bookmaker -> markets mapping.

    Args:
        bookmakers: List of dicts like ``{"key": "pinnacle", "markets": ["h2h"]}``.
    """
    commence = datetime(2025, 12, 1, 15, 0, tzinfo=UTC)
    snapshot_time = commence - timedelta(hours=hours_before)
    raw_bookmakers = []
    for bm in bookmakers:
        bm_key = bm["key"]
        markets_raw = []
        for mkt_key in bm.get("markets", []):
            markets_raw.append(
                {
                    "key": mkt_key,
                    "outcomes": [
                        {"name": "Arsenal", "price": 1.80},
                        {"name": "Draw", "price": 3.40},
                        {"name": "Chelsea", "price": 4.50},
                    ],
                }
            )
        raw_bookmakers.append(
            {
                "key": bm_key,
                "title": bm_key.title(),
                "last_update": snapshot_time.isoformat(),
                "markets": markets_raw,
            }
        )
    return OddsSnapshot(
        event_id=event_id,
        snapshot_time=snapshot_time,
        raw_data={"bookmakers": raw_bookmakers},
        bookmaker_count=len(raw_bookmakers),
        fetch_tier="closing",
        hours_until_commence=hours_before,
        api_request_id="test",
    )


class TestSnapshotHasBookmaker:
    def test_bookmaker_present_with_market(self) -> None:
        snap = _make_snapshot([{"key": "pinnacle", "markets": ["h2h"]}])
        assert snapshot_has_bookmaker(snap, "pinnacle", "h2h") is True

    def test_bookmaker_present_wrong_market(self) -> None:
        snap = _make_snapshot([{"key": "pinnacle", "markets": ["totals"]}])
        assert snapshot_has_bookmaker(snap, "pinnacle", "h2h") is False

    def test_bookmaker_absent(self) -> None:
        snap = _make_snapshot([{"key": "bet365", "markets": ["h2h"]}])
        assert snapshot_has_bookmaker(snap, "pinnacle", "h2h") is False

    def test_no_raw_data(self) -> None:
        snap = OddsSnapshot(
            event_id="evt_1",
            snapshot_time=datetime(2025, 12, 1, 14, 0, tzinfo=UTC),
            raw_data=None,
            bookmaker_count=0,
            fetch_tier="closing",
            hours_until_commence=1.0,
            api_request_id="test",
        )
        assert snapshot_has_bookmaker(snap, "pinnacle", "h2h") is False

    def test_empty_bookmakers_list(self) -> None:
        snap = OddsSnapshot(
            event_id="evt_1",
            snapshot_time=datetime(2025, 12, 1, 14, 0, tzinfo=UTC),
            raw_data={"bookmakers": []},
            bookmaker_count=0,
            fetch_tier="closing",
            hours_until_commence=1.0,
            api_request_id="test",
        )
        assert snapshot_has_bookmaker(snap, "pinnacle", "h2h") is False

    def test_raw_data_missing_bookmakers_key(self) -> None:
        snap = OddsSnapshot(
            event_id="evt_1",
            snapshot_time=datetime(2025, 12, 1, 14, 0, tzinfo=UTC),
            raw_data={"other_field": 123},
            bookmaker_count=0,
            fetch_tier="closing",
            hours_until_commence=1.0,
            api_request_id="test",
        )
        assert snapshot_has_bookmaker(snap, "pinnacle", "h2h") is False

    def test_multiple_bookmakers(self) -> None:
        snap = _make_snapshot(
            [
                {"key": "bet365", "markets": ["h2h"]},
                {"key": "pinnacle", "markets": ["h2h", "totals"]},
            ]
        )
        assert snapshot_has_bookmaker(snap, "pinnacle", "h2h") is True
        assert snapshot_has_bookmaker(snap, "pinnacle", "totals") is True
        assert snapshot_has_bookmaker(snap, "bet365", "h2h") is True
        assert snapshot_has_bookmaker(snap, "bet365", "totals") is False

    def test_bookmaker_with_no_markets(self) -> None:
        snap = _make_snapshot([{"key": "pinnacle", "markets": []}])
        assert snapshot_has_bookmaker(snap, "pinnacle", "h2h") is False


class TestShouldFilterMissingSharp:
    def _make_config(self, sharp_bookmakers: list[str], target_bookmaker: str) -> FeatureConfig:
        return FeatureConfig(
            sharp_bookmakers=sharp_bookmakers,
            target_bookmaker=target_bookmaker,
            markets=["h2h"],
        )

    def test_sharp_equals_target_no_filter(self) -> None:
        config = self._make_config(["bet365"], "bet365")
        assert _should_filter_missing_sharp(config) is False

    def test_sharp_differs_from_target_filter(self) -> None:
        config = self._make_config(["pinnacle"], "bet365")
        assert _should_filter_missing_sharp(config) is True

    def test_multiple_sharp_includes_target_still_filters(self) -> None:
        config = self._make_config(["pinnacle", "bet365"], "bet365")
        assert _should_filter_missing_sharp(config) is True

    def test_multiple_sharp_none_is_target(self) -> None:
        config = self._make_config(["pinnacle", "circa"], "bet365")
        assert _should_filter_missing_sharp(config) is True


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


class TestPreloadFeatureGroupCaches:
    """Gating behaviour of the shared preload helper."""

    @pytest.mark.asyncio
    @patch("odds_analytics.feature_groups.load_lineup_cache", new_callable=AsyncMock)
    @patch("odds_analytics.feature_groups.load_fixtures_df", new_callable=AsyncMock)
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

        caches = await preload_feature_group_caches(session, config, "soccer_epl")

        assert caches.standings is None
        assert caches.match_stats is None
        assert caches.fixtures_df is None
        assert caches.lineup_cache is None
        mock_standings.assert_not_called()
        mock_match_stats.assert_not_called()
        mock_fixtures.assert_not_called()
        mock_lineup.assert_not_called()

    @pytest.mark.asyncio
    @patch("odds_analytics.feature_groups.load_lineup_cache", new_callable=AsyncMock)
    @patch("odds_analytics.feature_groups.load_fixtures_df", new_callable=AsyncMock)
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

        caches = await preload_feature_group_caches(session, config, "soccer_epl")

        assert caches.standings is sentinel
        assert caches.match_stats is None
        assert caches.fixtures_df is None
        assert caches.lineup_cache is None
        mock_standings.assert_awaited_once_with(session, "soccer_epl")
        mock_match_stats.assert_not_called()
        mock_fixtures.assert_not_called()
        mock_lineup.assert_not_called()

    @pytest.mark.asyncio
    @patch("odds_analytics.feature_groups.load_lineup_cache", new_callable=AsyncMock)
    @patch("odds_analytics.feature_groups.load_fixtures_df", new_callable=AsyncMock)
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

        caches = await preload_feature_group_caches(session, config, "soccer_epl")

        assert caches.standings is None
        assert caches.match_stats is sentinel
        assert caches.fixtures_df is None
        assert caches.lineup_cache is None
        mock_standings.assert_not_called()
        mock_match_stats.assert_awaited_once_with(session, "soccer_epl")
        mock_fixtures.assert_not_called()
        mock_lineup.assert_not_called()

    @pytest.mark.asyncio
    @patch("odds_analytics.feature_groups.load_lineup_cache", new_callable=AsyncMock)
    @patch("odds_analytics.feature_groups.load_fixtures_df", new_callable=AsyncMock)
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

        caches = await preload_feature_group_caches(session, config, "soccer_epl")

        assert caches.standings is standings_sentinel
        assert caches.match_stats is None
        assert caches.fixtures_df is fixtures_sentinel
        assert caches.lineup_cache is None
        mock_standings.assert_awaited_once_with(session, "soccer_epl")
        mock_fixtures.assert_awaited_once_with(session)
        mock_match_stats.assert_not_called()
        mock_lineup.assert_not_called()

    @pytest.mark.asyncio
    @patch("odds_analytics.feature_groups.load_lineup_cache", new_callable=AsyncMock)
    @patch("odds_analytics.feature_groups.load_fixtures_df", new_callable=AsyncMock)
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

        caches = await preload_feature_group_caches(session, config, "soccer_epl")

        assert caches.standings is None
        assert caches.match_stats is None
        assert caches.fixtures_df is None
        assert caches.lineup_cache is sentinel
        mock_standings.assert_not_called()
        mock_match_stats.assert_not_called()
        mock_fixtures.assert_not_called()
        mock_lineup.assert_awaited_once_with(session)

    @pytest.mark.asyncio
    @patch("odds_analytics.feature_groups.load_lineup_cache", new_callable=AsyncMock)
    @patch("odds_analytics.feature_groups.load_fixtures_df", new_callable=AsyncMock)
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

        caches = await preload_feature_group_caches(session, config, "soccer_epl")

        assert caches.standings is standings_sentinel
        assert caches.match_stats is match_stats_sentinel
        assert caches.fixtures_df is fixtures_sentinel
        assert caches.lineup_cache is lineup_sentinel
        mock_standings.assert_awaited_once_with(session, "soccer_epl")
        mock_match_stats.assert_awaited_once_with(session, "soccer_epl")
        mock_fixtures.assert_awaited_once_with(session)
        mock_lineup.assert_awaited_once_with(session)

    @pytest.mark.asyncio
    @patch("odds_analytics.feature_groups.load_lineup_cache", new_callable=AsyncMock)
    @patch("odds_analytics.feature_groups.load_fixtures_df", new_callable=AsyncMock)
    @patch("odds_analytics.match_stats_features.load_match_stats_cache", new_callable=AsyncMock)
    @patch("odds_analytics.standings_features.load_season_events_cache", new_callable=AsyncMock)
    async def test_sport_scoped_caches_skipped_when_sport_key_missing(
        self,
        mock_standings: AsyncMock,
        mock_match_stats: AsyncMock,
        mock_fixtures: AsyncMock,
        mock_lineup: AsyncMock,
    ) -> None:
        """Standings and match_stats skip when sport_key is None; sport-agnostic caches still load."""
        fixtures_sentinel = MagicMock(name="fixtures_df")
        mock_fixtures.return_value = fixtures_sentinel
        config = _config_with_groups("standings", "match_stats", "epl_schedule")
        session = AsyncMock()

        caches = await preload_feature_group_caches(session, config, None)

        assert caches.standings is None
        assert caches.match_stats is None
        assert caches.fixtures_df is fixtures_sentinel
        assert caches.lineup_cache is None
        mock_standings.assert_not_called()
        mock_match_stats.assert_not_called()
        mock_fixtures.assert_awaited_once_with(session)
        mock_lineup.assert_not_called()
