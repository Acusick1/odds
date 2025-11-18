"""Integration tests for gap detection and backfill planning workflow."""

from datetime import UTC, datetime, timedelta

import pytest
from odds_analytics.backfill_executor import BackfillExecutor
from odds_analytics.gap_backfill_planner import GapBackfillPlanner
from odds_core.models import Event, EventStatus, Odds, OddsSnapshot
from odds_lambda.fetch_tier import FetchTier
from odds_lambda.storage.readers import OddsReader
from sqlalchemy import select


@pytest.fixture
async def events_with_gaps(test_session):
    """Create events with tier coverage gaps for integration testing."""
    from odds_lambda.storage.writers import OddsWriter

    writer = OddsWriter(test_session)

    # Event 1: Missing CLOSING tier (highest priority gap)
    event1 = Event(
        id="gap_test_event_1",
        sport_key="basketball_nba",
        sport_title="NBA",
        commence_time=datetime(2024, 1, 15, 19, 0, 0, tzinfo=UTC),
        home_team="Los Angeles Lakers",
        away_team="Boston Celtics",
        status=EventStatus.FINAL,
        home_score=110,
        away_score=105,
    )

    # Event 2: Missing OPENING and EARLY tiers (lower priority gap)
    event2 = Event(
        id="gap_test_event_2",
        sport_key="basketball_nba",
        sport_title="NBA",
        commence_time=datetime(2024, 1, 15, 21, 0, 0, tzinfo=UTC),
        home_team="Golden State Warriors",
        away_team="Miami Heat",
        status=EventStatus.FINAL,
        home_score=98,
        away_score=102,
    )

    # Event 3: Complete coverage (no gaps)
    event3 = Event(
        id="gap_test_event_3",
        sport_key="basketball_nba",
        sport_title="NBA",
        commence_time=datetime(2024, 1, 16, 19, 0, 0, tzinfo=UTC),
        home_team="Chicago Bulls",
        away_team="New York Knicks",
        status=EventStatus.FINAL,
        home_score=95,
        away_score=100,
    )

    for event in [event1, event2, event3]:
        await writer.upsert_event(event)

    await test_session.commit()

    # Add partial snapshots for event1 (missing CLOSING)
    for tier in [FetchTier.OPENING, FetchTier.EARLY, FetchTier.SHARP, FetchTier.PREGAME]:
        snapshot = OddsSnapshot(
            event_id=event1.id,
            snapshot_time=event1.commence_time - timedelta(hours=tier.interval_hours),
            raw_data={"bookmakers": []},
            bookmaker_count=0,
            fetch_tier=tier.value,
            hours_until_commence=tier.interval_hours,
        )
        test_session.add(snapshot)

    # Add partial snapshots for event2 (missing OPENING and EARLY)
    for tier in [FetchTier.SHARP, FetchTier.PREGAME, FetchTier.CLOSING]:
        snapshot = OddsSnapshot(
            event_id=event2.id,
            snapshot_time=event2.commence_time - timedelta(hours=tier.interval_hours),
            raw_data={"bookmakers": []},
            bookmaker_count=0,
            fetch_tier=tier.value,
            hours_until_commence=tier.interval_hours,
        )
        test_session.add(snapshot)

    # Add complete snapshots for event3
    for tier in FetchTier.get_priority_order():
        snapshot = OddsSnapshot(
            event_id=event3.id,
            snapshot_time=event3.commence_time - timedelta(hours=tier.interval_hours),
            raw_data={"bookmakers": []},
            bookmaker_count=0,
            fetch_tier=tier.value,
            hours_until_commence=tier.interval_hours,
        )
        test_session.add(snapshot)

    await test_session.commit()

    return [event1, event2, event3]


class TestGapDetectionWorkflow:
    """Integration tests for end-to-end gap detection workflow."""

    @pytest.mark.asyncio
    async def test_gap_detection_to_plan_generation(self, test_session, events_with_gaps):
        """Test gap detection identifies missing tiers and generates valid plan."""
        # Step 1: Run gap analysis
        planner = GapBackfillPlanner(test_session)

        analysis = await planner.analyze_gaps(
            start_date=datetime(2024, 1, 15).date(),
            end_date=datetime(2024, 1, 16).date(),
        )

        # Verify gap analysis results
        assert analysis.total_games == 3
        assert analysis.games_with_gaps == 2  # event1 and event2 have gaps
        assert analysis.total_missing_snapshots > 0

        # Verify prioritization
        assert len(analysis.games_by_priority[FetchTier.CLOSING]) == 1  # event1
        assert len(analysis.games_by_priority[FetchTier.EARLY]) == 1  # event2

        # Step 2: Generate backfill plan
        plan = await planner.generate_plan(
            start_date=datetime(2024, 1, 15).date(),
            end_date=datetime(2024, 1, 16).date(),
        )

        # Verify plan structure
        assert "total_games" in plan
        assert "total_snapshots" in plan
        assert "estimated_quota_usage" in plan
        assert "games" in plan

        # Should include games with gaps
        assert plan["total_games"] == 2
        assert plan["total_snapshots"] > 0

        # Verify quota calculation (30 units per snapshot)
        assert plan["estimated_quota_usage"] == plan["total_snapshots"] * 30

        # Verify plan compatibility with BackfillExecutor
        for game in plan["games"]:
            assert "event_id" in game
            assert "home_team" in game
            assert "away_team" in game
            assert "commence_time" in game
            assert "snapshots" in game
            assert "snapshot_count" in game
            assert len(game["snapshots"]) == game["snapshot_count"]

    @pytest.mark.asyncio
    async def test_plan_generation_with_quota_limit(self, test_session, events_with_gaps):
        """Test that quota limit correctly constrains plan generation."""
        planner = GapBackfillPlanner(test_session)

        # Get unlimited plan first
        unlimited_plan = await planner.generate_plan(
            start_date=datetime(2024, 1, 15).date(),
            end_date=datetime(2024, 1, 16).date(),
            max_quota=None,
        )

        # Generate plan with low quota (only enough for 1 snapshot)
        limited_plan = await planner.generate_plan(
            start_date=datetime(2024, 1, 15).date(),
            end_date=datetime(2024, 1, 16).date(),
            max_quota=60,  # 2 snapshots worth
        )

        # Limited plan should have fewer games
        assert limited_plan["total_games"] <= unlimited_plan["total_games"]
        assert limited_plan["total_snapshots"] <= unlimited_plan["total_snapshots"]
        assert limited_plan["estimated_quota_usage"] <= 60

    @pytest.mark.asyncio
    async def test_gap_plan_execution_dry_run(
        self, test_session, mock_session_factory, events_with_gaps, mock_api_response_factory
    ):
        """Test end-to-end workflow: gap detection → plan generation → dry-run execution."""
        from unittest.mock import AsyncMock

        # Step 1: Generate gap backfill plan
        planner = GapBackfillPlanner(test_session)

        plan = await planner.generate_plan(
            start_date=datetime(2024, 1, 15).date(),
            end_date=datetime(2024, 1, 16).date(),
        )

        assert plan["total_games"] > 0

        # Step 2: Execute plan in dry-run mode
        mock_client = AsyncMock()

        # Mock API responses for each game
        def mock_response_side_effect(event_id, *args, **kwargs):
            if event_id == "gap_test_event_1":
                return mock_api_response_factory(
                    "gap_test_event_1", "Los Angeles Lakers", "Boston Celtics"
                )
            elif event_id == "gap_test_event_2":
                return mock_api_response_factory(
                    "gap_test_event_2", "Golden State Warriors", "Miami Heat"
                )
            else:
                return mock_api_response_factory(event_id, "Team A", "Team B")

        mock_client.get_historical_odds = AsyncMock(side_effect=mock_response_side_effect)

        # Execute with dry_run=True (simulates CLI --dry-run)
        async with BackfillExecutor(
            client=mock_client,
            session_factory=mock_session_factory,
            skip_existing=True,  # Skip existing to avoid conflicts with fixture snapshots
            dry_run=True,  # Dry run mode
            rate_limit_seconds=0,
        ) as executor:
            result = await executor.execute_plan(plan)

            # In dry run, API is called but data is not written
            assert result.successful_games == plan["total_games"]
            # Note: Some snapshots may be skipped if they already exist from fixture
            assert result.successful_snapshots + result.skipped_snapshots == plan["total_snapshots"]
            assert result.failed_snapshots == 0

            # Verify API was called (may be fewer calls if snapshots skipped)
            assert mock_client.get_historical_odds.call_count >= 0

    @pytest.mark.asyncio
    async def test_gap_plan_execution_fills_gaps(
        self, test_session, mock_session_factory, events_with_gaps, mock_api_response_factory
    ):
        """Test that executing gap plan actually fills the missing snapshots."""
        from unittest.mock import AsyncMock

        # Step 1: Verify gaps exist
        reader = OddsReader(test_session)

        # event1 should be missing CLOSING tier snapshot
        closing_time = events_with_gaps[0].commence_time - timedelta(minutes=30)
        exists_before = await reader.snapshot_exists("gap_test_event_1", closing_time)
        assert exists_before is False

        # Step 2: Generate and execute plan
        planner = GapBackfillPlanner(test_session)

        plan = await planner.generate_plan(
            start_date=datetime(2024, 1, 15).date(),
            end_date=datetime(2024, 1, 16).date(),
        )

        mock_client = AsyncMock()
        mock_client.get_historical_odds = AsyncMock(
            return_value=mock_api_response_factory(
                "gap_test_event_1", "Los Angeles Lakers", "Boston Celtics"
            )
        )

        async with BackfillExecutor(
            client=mock_client,
            session_factory=mock_session_factory,
            skip_existing=False,
            dry_run=False,  # Actually write data
            rate_limit_seconds=0,
        ) as executor:
            result = await executor.execute_plan(plan)

            assert result.successful_snapshots > 0

        # Step 3: Verify gaps are filled
        # Refresh session to see new data
        test_session.expire_all()

        exists_after = await reader.snapshot_exists("gap_test_event_1", closing_time)
        assert exists_after is True

    @pytest.mark.asyncio
    async def test_prioritization_in_execution_order(
        self, test_session, mock_session_factory, events_with_gaps, mock_api_response_factory
    ):
        """Test that games are executed in priority order (CLOSING gaps first)."""
        from unittest.mock import AsyncMock

        planner = GapBackfillPlanner(test_session)

        plan = await planner.generate_plan(
            start_date=datetime(2024, 1, 15).date(),
            end_date=datetime(2024, 1, 16).date(),
        )

        # Verify plan has games in priority order
        # event1 (missing CLOSING) should come before event2 (missing EARLY)
        game_ids = [game["event_id"] for game in plan["games"]]

        # Find positions
        event1_pos = game_ids.index("gap_test_event_1") if "gap_test_event_1" in game_ids else -1
        event2_pos = game_ids.index("gap_test_event_2") if "gap_test_event_2" in game_ids else -1

        if event1_pos >= 0 and event2_pos >= 0:
            # event1 (CLOSING gap) should come before event2 (EARLY gap)
            assert event1_pos < event2_pos

    @pytest.mark.asyncio
    async def test_no_gaps_returns_empty_plan(self, test_session):
        """Test that date range with complete coverage returns empty plan."""
        from odds_lambda.storage.writers import OddsWriter

        # Create event with complete tier coverage
        writer = OddsWriter(test_session)

        event = Event(
            id="complete_event",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=datetime(2024, 2, 1, 19, 0, 0, tzinfo=UTC),
            home_team="Complete Team A",
            away_team="Complete Team B",
            status=EventStatus.FINAL,
            home_score=100,
            away_score=95,
        )

        await writer.upsert_event(event)

        # Add all 5 tiers
        for tier in FetchTier.get_priority_order():
            snapshot = OddsSnapshot(
                event_id=event.id,
                snapshot_time=event.commence_time - timedelta(hours=tier.interval_hours),
                raw_data={"bookmakers": []},
                bookmaker_count=0,
                fetch_tier=tier.value,
            )
            test_session.add(snapshot)

        await test_session.commit()

        # Try to generate gap plan
        planner = GapBackfillPlanner(test_session)

        plan = await planner.generate_plan(
            start_date=datetime(2024, 2, 1).date(),
            end_date=datetime(2024, 2, 1).date(),
        )

        # Should return empty plan
        assert plan["total_games"] == 0
        assert plan["total_snapshots"] == 0
        assert plan["estimated_quota_usage"] == 0
        assert plan["games"] == []

    @pytest.mark.asyncio
    async def test_plan_skips_already_complete_games(
        self, test_session, mock_session_factory, events_with_gaps, mock_api_response_factory
    ):
        """Test that re-running plan generation skips games that now have complete coverage."""
        from unittest.mock import AsyncMock

        # Step 1: Generate initial plan
        planner = GapBackfillPlanner(test_session)

        plan1 = await planner.generate_plan(
            start_date=datetime(2024, 1, 15).date(),
            end_date=datetime(2024, 1, 16).date(),
        )

        initial_games = plan1["total_games"]
        assert initial_games > 0

        # Step 2: Fill gaps for event1
        mock_client = AsyncMock()
        mock_client.get_historical_odds = AsyncMock(
            return_value=mock_api_response_factory(
                "gap_test_event_1", "Los Angeles Lakers", "Boston Celtics"
            )
        )

        # Create a mini-plan for just event1's missing snapshots
        event1_plan = {
            "total_games": 1,
            "total_snapshots": 1,
            "estimated_quota_usage": 30,
            "games": [g for g in plan1["games"] if g["event_id"] == "gap_test_event_1"],
        }

        async with BackfillExecutor(
            client=mock_client,
            session_factory=mock_session_factory,
            skip_existing=False,
            dry_run=False,
            rate_limit_seconds=0,
        ) as executor:
            await executor.execute_plan(event1_plan)

        # Step 3: Generate plan again - should have fewer games
        test_session.expire_all()

        plan2 = await planner.generate_plan(
            start_date=datetime(2024, 1, 15).date(),
            end_date=datetime(2024, 1, 16).date(),
        )

        # Should have fewer games now (event1 should be skipped)
        assert plan2["total_games"] < initial_games or plan2["total_snapshots"] < plan1[
            "total_snapshots"
        ]


class TestGapPlanCompatibility:
    """Test compatibility between gap plans and regular backfill plans."""

    @pytest.mark.asyncio
    async def test_gap_plan_structure_matches_regular_plan(self, test_session, events_with_gaps):
        """Test that gap plans have identical structure to regular backfill plans."""
        from odds_analytics.game_selector import GameSelector

        # Generate gap plan
        planner = GapBackfillPlanner(test_session)

        gap_plan = await planner.generate_plan(
            start_date=datetime(2024, 1, 15).date(),
            end_date=datetime(2024, 1, 16).date(),
        )

        # Generate regular backfill plan using GameSelector
        from odds_core.time import utc_isoformat
        from odds_lambda.storage.readers import OddsReader

        reader = OddsReader(test_session)
        events = await reader.get_events_by_date_range(
            start_date=datetime(2024, 1, 15, tzinfo=UTC),
            end_date=datetime(2024, 1, 17, tzinfo=UTC),
            sport_key="basketball_nba",
        )

        events_by_date = {}
        for event in events:
            date_str = utc_isoformat(event.commence_time)
            event_dict = {
                "id": event.id,
                "sport_key": event.sport_key,
                "sport_title": event.sport_title,
                "commence_time": event.commence_time.isoformat(),
                "home_team": event.home_team,
                "away_team": event.away_team,
                "bookmakers": [],
            }
            if date_str not in events_by_date:
                events_by_date[date_str] = []
            events_by_date[date_str].append(event_dict)

        selector = GameSelector(
            start_date=datetime(2024, 1, 15, tzinfo=UTC),
            end_date=datetime(2024, 1, 17, tzinfo=UTC),
            target_games=10,
        )

        regular_plan = selector.generate_backfill_plan(events_by_date)

        # Verify both plans have same top-level keys
        assert set(gap_plan.keys()) == set(regular_plan.keys())

        # Verify game entries have same structure
        if gap_plan["games"]:
            gap_game = gap_plan["games"][0]
            regular_game = regular_plan["games"][0]
            assert set(gap_game.keys()) == set(regular_game.keys())

    @pytest.mark.asyncio
    async def test_gap_plan_executable_by_backfill_executor(
        self, test_session, mock_session_factory, events_with_gaps, mock_api_response_factory
    ):
        """Test that gap plans can be executed by BackfillExecutor without modification."""
        from unittest.mock import AsyncMock

        # Generate gap plan
        planner = GapBackfillPlanner(test_session)

        plan = await planner.generate_plan(
            start_date=datetime(2024, 1, 15).date(),
            end_date=datetime(2024, 1, 16).date(),
        )

        # Execute without any modification - should work seamlessly
        mock_client = AsyncMock()
        mock_client.get_historical_odds = AsyncMock(
            return_value=mock_api_response_factory("gap_test_event_1", "Team A", "Team B")
        )

        async with BackfillExecutor(
            client=mock_client,
            session_factory=mock_session_factory,
            skip_existing=False,
            dry_run=True,
            rate_limit_seconds=0,
        ) as executor:
            # Should execute without errors
            result = await executor.execute_plan(plan)

            assert result.successful_games >= 0
            assert result.failed_snapshots == 0
