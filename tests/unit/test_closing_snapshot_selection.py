"""Unit tests for closing snapshot selection logic."""

from datetime import UTC, datetime, timedelta

from odds_analytics.feature_groups import _select_closing_snapshot
from odds_analytics.training.config import FeatureConfig
from odds_core.models import Event, EventStatus, OddsSnapshot


def _make_event() -> Event:
    return Event(
        id="evt_1",
        sport_key="basketball_nba",
        sport_title="NBA",
        commence_time=datetime(2024, 11, 1, 19, 0, tzinfo=UTC),
        home_team="Los Angeles Lakers",
        away_team="Boston Celtics",
        status=EventStatus.FINAL,
        home_score=110,
        away_score=105,
    )


def _make_snapshot(
    event: Event,
    bookmakers: list[str],
    hours_before: float,
    home_price: int = -120,
    away_price: int = 100,
    api_request_id: str = "test",
) -> OddsSnapshot:
    snapshot_time = event.commence_time - timedelta(hours=hours_before)
    return OddsSnapshot(
        event_id=event.id,
        snapshot_time=snapshot_time,
        raw_data={
            "bookmakers": [
                {
                    "key": bk,
                    "title": bk.title(),
                    "last_update": snapshot_time.isoformat(),
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": event.home_team, "price": home_price},
                                {"name": event.away_team, "price": away_price},
                            ],
                        }
                    ],
                }
                for bk in bookmakers
            ]
        },
        bookmaker_count=len(bookmakers),
        fetch_tier="closing",
        hours_until_commence=hours_before,
        api_request_id=api_request_id,
    )


class TestSelectClosingSnapshot:
    """Tests for _select_closing_snapshot."""

    def test_empty_candidates_returns_none(self) -> None:
        event = _make_event()
        config = FeatureConfig(target_type="devigged_bookmaker", target_bookmaker="pinnacle")
        assert _select_closing_snapshot([], event, config) is None

    def test_single_candidate_returned(self) -> None:
        event = _make_event()
        snap = _make_snapshot(event, ["pinnacle", "fanduel"], hours_before=0.5)
        config = FeatureConfig(target_type="devigged_bookmaker", target_bookmaker="pinnacle")
        assert _select_closing_snapshot([snap], event, config) is snap

    def test_prefers_candidate_with_target_bookmaker(self) -> None:
        """Odds API snapshot (with pinnacle) preferred over OddsPortal (without)."""
        event = _make_event()
        oddsapi_snap = _make_snapshot(event, ["pinnacle", "fanduel"], hours_before=1.9)
        oddsportal_snap = _make_snapshot(event, ["bet365", "betway"], hours_before=0.0)
        # Ordered by time: oddsapi first, oddsportal last
        candidates = [oddsapi_snap, oddsportal_snap]

        config = FeatureConfig(target_type="devigged_bookmaker", target_bookmaker="pinnacle")
        result = _select_closing_snapshot(candidates, event, config)
        assert result is oddsapi_snap

    def test_falls_back_to_last_when_no_candidate_has_target(self) -> None:
        """When no candidate has the target bookmaker, fall back to last by time."""
        event = _make_event()
        snap_a = _make_snapshot(event, ["fanduel"], hours_before=1.9)
        snap_b = _make_snapshot(event, ["bet365"], hours_before=0.0)
        candidates = [snap_a, snap_b]

        config = FeatureConfig(target_type="devigged_bookmaker", target_bookmaker="pinnacle")
        result = _select_closing_snapshot(candidates, event, config)
        assert result is snap_b

    def test_raw_target_type_uses_last_by_time(self) -> None:
        """Non-bookmaker target types always use last-by-time."""
        event = _make_event()
        oddsapi_snap = _make_snapshot(event, ["pinnacle", "fanduel"], hours_before=1.9)
        oddsportal_snap = _make_snapshot(event, ["bet365", "betway"], hours_before=0.0)
        candidates = [oddsapi_snap, oddsportal_snap]

        config = FeatureConfig(target_type="raw")
        result = _select_closing_snapshot(candidates, event, config)
        assert result is oddsportal_snap

    def test_prefers_latest_candidate_with_target_bookmaker(self) -> None:
        """When multiple candidates have the target bookmaker, pick the latest."""
        event = _make_event()
        snap_early = _make_snapshot(event, ["pinnacle"], hours_before=2.0)
        snap_late = _make_snapshot(event, ["pinnacle"], hours_before=0.5)
        candidates = [snap_early, snap_late]

        config = FeatureConfig(target_type="devigged_bookmaker", target_bookmaker="pinnacle")
        result = _select_closing_snapshot(candidates, event, config)
        assert result is snap_late

    def test_bet365_target_selects_oddsportal_over_oddsapi(self) -> None:
        """bet365 experiments should select the OddsPortal snapshot (which has bet365)."""
        event = _make_event()
        oddsapi_snap = _make_snapshot(event, ["pinnacle", "fanduel"], hours_before=1.9)
        oddsportal_snap = _make_snapshot(event, ["bet365", "betway"], hours_before=0.0)
        candidates = [oddsapi_snap, oddsportal_snap]

        config = FeatureConfig(target_type="devigged_bookmaker", target_bookmaker="bet365")
        result = _select_closing_snapshot(candidates, event, config)
        assert result is oddsportal_snap


class TestClosingSourcePriority:
    """Tests for closing_source_priority in _select_closing_snapshot."""

    def test_priority_selects_matching_source(self) -> None:
        """When priority matches a source, return that source's snapshot."""
        event = _make_event()
        op_snap = _make_snapshot(
            event,
            ["bet365", "betway"],
            hours_before=0.5,
            api_request_id="oddsportal",
        )
        fduk_snap = _make_snapshot(
            event,
            ["bet365", "pinnacle"],
            hours_before=1.0,
            api_request_id="football_data_uk",
        )
        candidates = [fduk_snap, op_snap]

        config = FeatureConfig(
            target_type="devigged_bookmaker",
            target_bookmaker="bet365",
            closing_source_priority=["football_data_uk"],
        )
        result = _select_closing_snapshot(candidates, event, config)
        assert result is fduk_snap

    def test_priority_falls_back_when_no_match(self) -> None:
        """When priority source not available, fall back to latest with bookmaker."""
        event = _make_event()
        op_snap = _make_snapshot(
            event,
            ["bet365", "betway"],
            hours_before=0.5,
            api_request_id="oddsportal",
        )
        candidates = [op_snap]

        config = FeatureConfig(
            target_type="devigged_bookmaker",
            target_bookmaker="bet365",
            closing_source_priority=["football_data_uk"],
        )
        result = _select_closing_snapshot(candidates, event, config)
        assert result is op_snap

    def test_priority_order_respected(self) -> None:
        """First matching priority source wins over later ones."""
        event = _make_event()
        op_snap = _make_snapshot(
            event,
            ["bet365"],
            hours_before=0.5,
            api_request_id="oddsportal",
        )
        fduk_snap = _make_snapshot(
            event,
            ["bet365"],
            hours_before=1.0,
            api_request_id="football_data_uk",
        )
        candidates = [fduk_snap, op_snap]

        config = FeatureConfig(
            target_type="devigged_bookmaker",
            target_bookmaker="bet365",
            closing_source_priority=["oddsportal", "football_data_uk"],
        )
        result = _select_closing_snapshot(candidates, event, config)
        assert result is op_snap

    def test_null_priority_behaves_as_before(self) -> None:
        """When closing_source_priority is None, latest with bookmaker wins."""
        event = _make_event()
        fduk_snap = _make_snapshot(
            event,
            ["bet365"],
            hours_before=1.0,
            api_request_id="football_data_uk",
        )
        op_snap = _make_snapshot(
            event,
            ["bet365"],
            hours_before=0.5,
            api_request_id="oddsportal",
        )
        candidates = [fduk_snap, op_snap]

        config = FeatureConfig(
            target_type="devigged_bookmaker",
            target_bookmaker="bet365",
        )
        result = _select_closing_snapshot(candidates, event, config)
        assert result is op_snap
