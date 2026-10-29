"""Integration tests for collect_event_data and the XGBoost adapter's feature names."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from odds_analytics.feature_groups import EventDataBundle, XGBoostAdapter, collect_event_data
from odds_analytics.training import FeatureConfig
from odds_core.models import Event, EventStatus
from odds_core.vocabulary import Source

from tests.test_helpers import TaggedSnapshot, make_tagged_snapshot, store_tagged

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
        id=idx,
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
    event_id: int,
    commence_time: datetime,
    home_team: str,
    away_team: str,
) -> list[TaggedSnapshot]:
    """Create SB snapshots at early, pregame, and closing times before kick-off."""
    snapshots = []
    bookmakers = ["pinnacle", "fanduel", "draftkings"]
    for hours_before in (48.0, 7.5, 0.5):
        t = commence_time - timedelta(hours=hours_before)
        snapshots.append(
            make_tagged_snapshot(
                event_id,
                t,
                [
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
                ],
                source=None,
            )
        )
    return snapshots


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
async def cross_source_test_data(pglite_async_session):
    """Seed 3 FINAL NBA events, each with early, pregame and closing Odds API snapshots."""
    base_time = datetime(2026, 1, 20, 19, 0, tzinfo=UTC)
    games = [
        ("Los Angeles Lakers", "Boston Celtics"),
        ("Golden State Warriors", "Miami Heat"),
        ("Denver Nuggets", "Dallas Mavericks"),
    ]

    events = []
    for i, (home, away) in enumerate(games):
        commence = base_time + timedelta(days=i)
        sb_event = make_sb_event(i, commence, home, away)
        pglite_async_session.add(sb_event)
        events.append(sb_event)

        for snap in make_odds_snapshots(sb_event.id, commence, home, away):
            await store_tagged(pglite_async_session, snap)

    await pglite_async_session.commit()

    for e in events:
        await pglite_async_session.refresh(e)

    return {"events": events}


# =============================================================================
# Tests
# =============================================================================


class TestCollectEventData:
    """Integration tests for collect_event_data (Layer 1)."""

    def _make_config(self) -> FeatureConfig:
        return FeatureConfig(
            target_type="raw",
            sharp_bookmakers=["pinnacle"],
            retail_bookmakers=["fanduel", "draftkings"],
            markets=["h2h"],
            outcome="home",
            feature_groups=("tabular",),
        )

    async def test_polymarket_only_snapshots_leave_the_bundle(
        self, pglite_async_session, cross_source_test_data
    ):
        event = cross_source_test_data["events"][2]
        before = await collect_event_data(event, pglite_async_session, self._make_config())
        await store_tagged(
            pglite_async_session,
            make_tagged_snapshot(
                event.id,
                event.commence_time - timedelta(minutes=1),
                [
                    {
                        "key": "polymarket",
                        "title": "Polymarket",
                        "markets": [{"key": "h2h", "outcomes": []}],
                        "polymarket_meta": {
                            "yes_probabilities": {"home": 0.6, "away": 0.4},
                            "best_ask": {"home": 0.61, "away": 0.41},
                            "best_bid": {"home": 0.59, "away": 0.39},
                        },
                    }
                ],
                source="polymarket",
            ),
        )
        await pglite_async_session.commit()

        after = await collect_event_data(event, pglite_async_session, self._make_config())

        def snapshot_ids(bundle: EventDataBundle) -> list[int]:
            return [s.quotes[0].snapshot_id for s in bundle.snapshots]

        assert snapshot_ids(after) == snapshot_ids(before)
        assert after.closes
        assert {s: c.quotes[0].snapshot_id for s, c in after.closes.items()} == {
            s: c.quotes[0].snapshot_id for s, c in before.closes.items()
        }

    async def test_closing_snapshot_is_the_half_hour_snapshot(
        self, pglite_async_session, cross_source_test_data
    ) -> None:
        """Of the early, pregame and closing snapshots, only the last can be the close."""
        event = cross_source_test_data["events"][2]

        bundle = await collect_event_data(event, pglite_async_session, self._make_config())

        assert bundle.closes[Source.ODDS_API].snapshot_time == event.commence_time - timedelta(
            hours=0.5
        )

    async def test_latest_close_candidate_wins_and_in_play_is_never_the_close(
        self, pglite_async_session, cross_source_test_data
    ) -> None:
        event = cross_source_test_data["events"][2]
        later = event.commence_time - timedelta(minutes=10)
        in_play = event.commence_time + timedelta(minutes=30)
        for t in (later, in_play):
            await store_tagged(
                pglite_async_session,
                make_tagged_snapshot(
                    event.id,
                    t,
                    [
                        {
                            "key": "pinnacle",
                            "markets": [
                                {
                                    "key": "h2h",
                                    "outcomes": [
                                        {"name": event.home_team, "price": -140},
                                        {"name": event.away_team, "price": 120},
                                    ],
                                }
                            ],
                        }
                    ],
                    source=None,
                ),
            )
        await pglite_async_session.commit()

        bundle = await collect_event_data(event, pglite_async_session, self._make_config())

        assert bundle.closes[Source.ODDS_API].snapshot_time == later
        assert in_play in [s.snapshot_time for s in bundle.snapshots]


class TestXGBoostAdapterFeatureCount:
    """Tests for XGBoostAdapter feature counts."""

    def test_tabular_only_feature_names(self):
        from odds_analytics.feature_extraction import TabularFeatures

        config = FeatureConfig(target_type="raw", feature_groups=["tabular"])
        adapter = XGBoostAdapter()
        names = adapter.feature_names(config)
        expected = len(TabularFeatures.get_feature_names()) + 1  # +1 for hours_until_event
        assert len(names) == expected
        assert "hours_until_event" in names
