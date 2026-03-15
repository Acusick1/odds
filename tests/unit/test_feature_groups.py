"""Unit tests for feature_groups helpers."""

from datetime import UTC, datetime, timedelta

from odds_analytics.feature_groups import (
    _should_filter_missing_sharp,
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
