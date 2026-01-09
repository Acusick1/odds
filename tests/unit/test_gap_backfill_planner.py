"""Unit tests for gap backfill planner."""

from datetime import UTC, date, datetime, timedelta

import pytest
from odds_analytics.gap_backfill_planner import GapBackfillPlanner
from odds_core.models import Event, EventStatus, OddsSnapshot
from odds_lambda.fetch_tier import FetchTier


@pytest.fixture
async def sample_events_with_gaps(test_session):
    """Create sample events with tier coverage gaps."""
    # Event 1: Missing CLOSING tier (highest priority)
    event1 = Event(
        id="event_missing_closing",
        sport_key="basketball_nba",
        sport_title="NBA",
        commence_time=datetime(2024, 10, 24, 19, 0, 0, tzinfo=UTC),
        home_team="Lakers",
        away_team="Celtics",
        status=EventStatus.FINAL,
        home_score=110,
        away_score=105,
    )

    # Event 2: Missing OPENING and EARLY tiers (lower priority)
    event2 = Event(
        id="event_missing_opening",
        sport_key="basketball_nba",
        sport_title="NBA",
        commence_time=datetime(2024, 10, 24, 21, 0, 0, tzinfo=UTC),
        home_team="Warriors",
        away_team="Nets",
        status=EventStatus.FINAL,
        home_score=115,
        away_score=108,
    )

    # Event 3: Complete coverage (no gaps)
    event3 = Event(
        id="event_complete",
        sport_key="basketball_nba",
        sport_title="NBA",
        commence_time=datetime(2024, 10, 25, 19, 0, 0, tzinfo=UTC),
        home_team="Bulls",
        away_team="Knicks",
        status=EventStatus.FINAL,
        home_score=98,
        away_score=102,
    )

    test_session.add_all([event1, event2, event3])
    await test_session.commit()

    # Add snapshots for event1 (missing CLOSING)
    for tier in [FetchTier.OPENING, FetchTier.EARLY, FetchTier.SHARP, FetchTier.PREGAME]:
        snapshot = OddsSnapshot(
            event_id=event1.id,
            snapshot_time=event1.commence_time - timedelta(hours=tier.interval_hours),
            raw_data={},
            bookmaker_count=8,
            fetch_tier=tier.value,
            hours_until_commence=tier.interval_hours,
        )
        test_session.add(snapshot)

    # Add snapshots for event2 (missing OPENING and EARLY)
    for tier in [FetchTier.SHARP, FetchTier.PREGAME, FetchTier.CLOSING]:
        snapshot = OddsSnapshot(
            event_id=event2.id,
            snapshot_time=event2.commence_time - timedelta(hours=tier.interval_hours),
            raw_data={},
            bookmaker_count=8,
            fetch_tier=tier.value,
            hours_until_commence=tier.interval_hours,
        )
        test_session.add(snapshot)

    # Add complete snapshots for event3
    for tier in FetchTier.get_priority_order():
        snapshot = OddsSnapshot(
            event_id=event3.id,
            snapshot_time=event3.commence_time - timedelta(hours=tier.interval_hours),
            raw_data={},
            bookmaker_count=8,
            fetch_tier=tier.value,
            hours_until_commence=tier.interval_hours,
        )
        test_session.add(snapshot)

    await test_session.commit()

    return [event1, event2, event3]


class TestGapBackfillPlanner:
    """Test GapBackfillPlanner class."""

    async def test_analyze_gaps_detects_missing_tiers(self, test_session, sample_events_with_gaps):
        """Test that analyze_gaps correctly identifies missing tiers."""
        planner = GapBackfillPlanner(test_session)

        analysis = await planner.analyze_gaps(
            start_date=date(2024, 10, 24),
            end_date=date(2024, 10, 25),
        )

        # Should find 3 total games, 2 with gaps
        assert analysis.total_games == 3
        assert analysis.games_with_gaps == 2

        # Should have games missing CLOSING (highest priority)
        assert len(analysis.games_by_priority[FetchTier.CLOSING]) == 1

        # Should have games missing EARLY (for event2, which is missing OPENING and EARLY)
        # EARLY is the highest priority missing tier for event2
        assert len(analysis.games_by_priority[FetchTier.EARLY]) == 1

    async def test_analyze_gaps_no_gaps(self, test_session):
        """Test analyze_gaps when no gaps exist."""
        # Create event with complete coverage
        event = Event(
            id="complete_event",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime(2024, 10, 24, 19, 0, 0, tzinfo=UTC),
            home_team="Lakers",
            away_team="Celtics",
            status=EventStatus.FINAL,
            home_score=110,
            away_score=105,
        )
        test_session.add(event)

        # Add all tiers
        for tier in FetchTier.get_priority_order():
            snapshot = OddsSnapshot(
                event_id=event.id,
                snapshot_time=event.commence_time - timedelta(hours=tier.interval_hours),
                raw_data={},
                bookmaker_count=8,
                fetch_tier=tier.value,
            )
            test_session.add(snapshot)

        await test_session.commit()

        planner = GapBackfillPlanner(test_session)

        analysis = await planner.analyze_gaps(
            start_date=date(2024, 10, 24),
            end_date=date(2024, 10, 24),
        )

        assert analysis.total_games == 1
        assert analysis.games_with_gaps == 0
        assert analysis.total_missing_snapshots == 0

    async def test_prioritize_games_by_highest_missing_tier(
        self, test_session, sample_events_with_gaps
    ):
        """Test that games are prioritized by highest-priority missing tier."""
        planner = GapBackfillPlanner(test_session)

        analysis = await planner.analyze_gaps(
            start_date=date(2024, 10, 24),
            end_date=date(2024, 10, 25),
        )

        prioritized = planner._prioritize_games(analysis)

        # First game should be the one missing CLOSING (highest priority)
        assert prioritized[0].event_id == "event_missing_closing"
        assert prioritized[0].highest_priority_missing == FetchTier.CLOSING

        # Second game should be the one missing EARLY (next highest priority)
        # Note: event_missing_opening is missing both OPENING and EARLY, but EARLY has higher priority
        assert prioritized[1].event_id == "event_missing_opening"
        assert prioritized[1].highest_priority_missing == FetchTier.EARLY

    async def test_get_highest_priority_missing_tier(self, test_session):
        """Test _get_highest_priority_missing_tier method."""
        planner = GapBackfillPlanner(test_session)

        # CLOSING is highest priority
        missing = frozenset([FetchTier.OPENING, FetchTier.EARLY, FetchTier.CLOSING])
        assert planner._get_highest_priority_missing_tier(missing) == FetchTier.CLOSING

        # PREGAME when CLOSING not present
        missing = frozenset([FetchTier.OPENING, FetchTier.EARLY, FetchTier.PREGAME])
        assert planner._get_highest_priority_missing_tier(missing) == FetchTier.PREGAME

        # OPENING alone (lowest priority)
        missing = frozenset([FetchTier.OPENING])
        assert planner._get_highest_priority_missing_tier(missing) == FetchTier.OPENING

    async def test_generate_plan_respects_quota_limit(self, test_session, sample_events_with_gaps):
        """Test that generate_plan respects max_quota constraint."""
        planner = GapBackfillPlanner(test_session)

        # Set very low quota (only enough for 1 snapshot)
        plan = await planner.generate_plan(
            start_date=date(2024, 10, 24),
            end_date=date(2024, 10, 25),
            max_quota=60,  # Only enough for 2 snapshots (2 * 30)
        )

        # Should include 0 or 1 complete games (all-or-nothing)
        # Since event_missing_closing has 1 missing snapshot and event_missing_opening has 2
        assert plan["total_games"] <= 2
        assert plan["estimated_quota_usage"] <= 60

    async def test_generate_plan_unlimited_quota(self, test_session, sample_events_with_gaps):
        """Test that generate_plan includes all games when no quota limit."""
        planner = GapBackfillPlanner(test_session)

        plan = await planner.generate_plan(
            start_date=date(2024, 10, 24),
            end_date=date(2024, 10, 25),
            max_quota=None,  # Unlimited
        )

        # Should include all games with gaps
        assert plan["total_games"] == 2
        assert plan["total_snapshots"] > 0

    async def test_generate_plan_raises_on_impossible_quota(
        self, test_session, sample_events_with_gaps
    ):
        """Test that generate_plan raises error if quota too low for first game."""
        planner = GapBackfillPlanner(test_session)

        # Set quota too low to fit even the first game
        with pytest.raises(ValueError, match="Max quota .* is too low"):
            await planner.generate_plan(
                start_date=date(2024, 10, 24),
                end_date=date(2024, 10, 25),
                max_quota=10,  # Way too low
            )

    async def test_generate_plan_format_compatible_with_executor(
        self, test_session, sample_events_with_gaps
    ):
        """Test that generated plan has correct format for BackfillExecutor."""
        planner = GapBackfillPlanner(test_session)

        plan = await planner.generate_plan(
            start_date=date(2024, 10, 24),
            end_date=date(2024, 10, 25),
        )

        # Verify plan structure
        assert "total_games" in plan
        assert "total_snapshots" in plan
        assert "estimated_quota_usage" in plan
        assert "games" in plan
        assert "start_date" in plan
        assert "end_date" in plan

        # Verify game structure
        if plan["total_games"] > 0:
            game = plan["games"][0]
            assert "event_id" in game
            assert "home_team" in game
            assert "away_team" in game
            assert "commence_time" in game
            assert "snapshots" in game
            assert "snapshot_count" in game

            # Snapshots should be ISO format strings
            assert isinstance(game["snapshots"], list)
            if len(game["snapshots"]) > 0:
                assert isinstance(game["snapshots"][0], str)

    async def test_generate_plan_empty_date_range(self, test_session):
        """Test generate_plan with date range containing no events."""
        planner = GapBackfillPlanner(test_session)

        plan = await planner.generate_plan(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 1),
        )

        assert plan["total_games"] == 0
        assert plan["total_snapshots"] == 0
        assert plan["estimated_quota_usage"] == 0
        assert plan["games"] == []


class TestFetchTierPriorityOrder:
    """Test FetchTier.get_priority_order() classmethod."""

    def test_priority_order_is_correct(self):
        """Test that priority order is CLOSING to OPENING."""
        order = FetchTier.get_priority_order()

        assert order[0] == FetchTier.CLOSING  # Highest priority
        assert order[1] == FetchTier.PREGAME
        assert order[2] == FetchTier.SHARP
        assert order[3] == FetchTier.EARLY
        assert order[4] == FetchTier.OPENING  # Lowest priority

    def test_priority_order_includes_all_tiers(self):
        """Test that all 5 tiers are in priority order."""
        order = FetchTier.get_priority_order()

        assert len(order) == 5
        assert FetchTier.CLOSING in order
        assert FetchTier.PREGAME in order
        assert FetchTier.SHARP in order
        assert FetchTier.EARLY in order
        assert FetchTier.OPENING in order
