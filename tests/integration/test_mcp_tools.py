"""Integration tests for Phase 2 MCP tools (match briefs + find_retail_edges).

Tests call the actual MCP tool handler functions end-to-end, with
async_session_maker patched to use the PGlite test database.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
from odds_core.match_brief_models import BriefDecision, MatchBrief, SharpPriceResult
from odds_core.models import Event, EventStatus, OddsSnapshot
from odds_lambda.storage.readers import OddsReader
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker


@pytest.fixture
def patch_session_maker(pglite_async_engine):
    """Patch async_session_maker in the server module to use the PGlite engine."""
    factory = async_sessionmaker(pglite_async_engine, expire_on_commit=False)
    with patch("odds_mcp.server.async_session_maker", factory):
        yield factory


@pytest.fixture
async def epl_event_with_odds(pglite_async_session):
    """Create an EPL event with a snapshot containing sharp + retail bookmaker odds."""
    commence_time = datetime(2026, 4, 15, 15, 0, tzinfo=UTC)

    event = Event(
        id="epl_test_001",
        sport_key="soccer_epl",
        sport_title="EPL",
        commence_time=commence_time,
        home_team="Arsenal",
        away_team="Chelsea",
        status=EventStatus.SCHEDULED,
    )
    pglite_async_session.add(event)

    snapshot_time = commence_time - timedelta(hours=6)
    raw_data = {
        "bookmakers": [
            {
                "key": "pinnacle",
                "title": "Pinnacle",
                "last_update": snapshot_time.isoformat(),
                "markets": [
                    {
                        "key": "1x2",
                        "outcomes": [
                            {"name": "Arsenal", "price": -120},
                            {"name": "Draw", "price": 280},
                            {"name": "Chelsea", "price": 310},
                        ],
                    }
                ],
            },
            {
                "key": "bet365",
                "title": "Bet365",
                "last_update": snapshot_time.isoformat(),
                "markets": [
                    {
                        "key": "1x2",
                        "outcomes": [
                            {"name": "Arsenal", "price": -130},
                            {"name": "Draw", "price": 260},
                            {"name": "Chelsea", "price": 290},
                        ],
                    }
                ],
            },
            {
                "key": "betway",
                "title": "Betway",
                "last_update": snapshot_time.isoformat(),
                "markets": [
                    {
                        "key": "1x2",
                        "outcomes": [
                            {"name": "Arsenal", "price": -125},
                            {"name": "Draw", "price": 270},
                            {"name": "Chelsea", "price": 300},
                        ],
                    }
                ],
            },
        ]
    }
    snapshot = OddsSnapshot(
        event_id=event.id,
        snapshot_time=snapshot_time,
        raw_data=raw_data,
        bookmaker_count=3,
        fetch_tier="pregame",
        hours_until_commence=6.0,
    )
    pglite_async_session.add(snapshot)
    await pglite_async_session.commit()
    await pglite_async_session.refresh(event)
    await pglite_async_session.refresh(snapshot)
    return event, snapshot


@pytest.fixture
async def epl_event_no_odds(pglite_async_session):
    """Create an EPL event with no snapshots."""
    event = Event(
        id="epl_test_002",
        sport_key="soccer_epl",
        sport_title="EPL",
        commence_time=datetime(2026, 4, 16, 15, 0, tzinfo=UTC),
        home_team="Liverpool",
        away_team="Man City",
        status=EventStatus.SCHEDULED,
    )
    pglite_async_session.add(event)
    await pglite_async_session.commit()
    await pglite_async_session.refresh(event)
    return event


@pytest.fixture
async def partial_sharp_event(pglite_async_session):
    """Event where pinnacle has only home odds, betfair_exchange has all three."""
    commence_time = datetime(2026, 4, 17, 15, 0, tzinfo=UTC)
    event = Event(
        id="epl_test_003",
        sport_key="soccer_epl",
        sport_title="EPL",
        commence_time=commence_time,
        home_team="Tottenham",
        away_team="Everton",
        status=EventStatus.SCHEDULED,
    )
    pglite_async_session.add(event)

    snapshot_time = commence_time - timedelta(hours=4)
    raw_data = {
        "bookmakers": [
            {
                "key": "pinnacle",
                "title": "Pinnacle",
                "last_update": snapshot_time.isoformat(),
                "markets": [
                    {
                        "key": "1x2",
                        "outcomes": [
                            {"name": "Tottenham", "price": -150},
                        ],
                    }
                ],
            },
            {
                "key": "betfair_exchange",
                "title": "Betfair Exchange",
                "last_update": snapshot_time.isoformat(),
                "markets": [
                    {
                        "key": "1x2",
                        "outcomes": [
                            {"name": "Tottenham", "price": -145},
                            {"name": "Draw", "price": 260},
                            {"name": "Everton", "price": 400},
                        ],
                    }
                ],
            },
            {
                "key": "bet365",
                "title": "Bet365",
                "last_update": snapshot_time.isoformat(),
                "markets": [
                    {
                        "key": "1x2",
                        "outcomes": [
                            {"name": "Tottenham", "price": -160},
                            {"name": "Draw", "price": 250},
                            {"name": "Everton", "price": 380},
                        ],
                    }
                ],
            },
        ]
    }
    snapshot = OddsSnapshot(
        event_id=event.id,
        snapshot_time=snapshot_time,
        raw_data=raw_data,
        bookmaker_count=3,
        fetch_tier="pregame",
        hours_until_commence=4.0,
    )
    pglite_async_session.add(snapshot)
    await pglite_async_session.commit()
    await pglite_async_session.refresh(event)
    await pglite_async_session.refresh(snapshot)
    return event, snapshot


class TestSaveMatchBrief:
    """Tests for the save_match_brief MCP tool."""

    @pytest.mark.asyncio
    async def test_happy_path(self, patch_session_maker, epl_event_with_odds):
        """save_match_brief persists a brief with auto-captured sharp prices."""
        from odds_mcp.server import save_match_brief

        event, _ = epl_event_with_odds

        result = await save_match_brief(
            event_id=event.id,
            market="1x2",
            decision="watching",
            summary="Arsenal strong at home, Chelsea missing players",
            brief_text="Arsenal looking strong at home, Chelsea missing key players.",
        )

        assert "error" not in result
        assert result["id"] is not None
        assert result["event_id"] == event.id
        assert result["decision"] == "watching"
        assert result["summary"] == "Arsenal strong at home, Chelsea missing players"
        assert (
            result["brief_text"] == "Arsenal looking strong at home, Chelsea missing key players."
        )
        assert result["created_at"] is not None
        # Sharp prices auto-captured from snapshot
        sharp = result["sharp_price_at_brief"]
        assert sharp is not None
        for outcome_data in sharp.values():
            assert outcome_data["bookmaker"] == "pinnacle"
            assert "price" in outcome_data
            assert "implied_prob" in outcome_data

    @pytest.mark.asyncio
    async def test_event_not_found(self, patch_session_maker):
        """save_match_brief returns error for nonexistent event."""
        from odds_mcp.server import save_match_brief

        result = await save_match_brief(
            event_id="nonexistent_event",
            market="1x2",
            decision="watching",
            summary="Test",
            brief_text="Should fail",
        )

        assert result == {"error": "Event 'nonexistent_event' not found"}

    @pytest.mark.asyncio
    async def test_no_snapshot_still_saves(self, patch_session_maker, epl_event_no_odds):
        """save_match_brief works with no snapshot (sharp prices will be null)."""
        from odds_mcp.server import save_match_brief

        event = epl_event_no_odds

        result = await save_match_brief(
            event_id=event.id,
            market="1x2",
            decision="watching",
            summary="No odds yet",
            brief_text="No odds available yet for this fixture.",
        )

        assert "error" not in result
        assert result["id"] is not None
        assert result["sharp_price_at_brief"] is None

    @pytest.mark.asyncio
    async def test_multiple_briefs_append_only(
        self, patch_session_maker, epl_event_with_odds, pglite_async_session
    ):
        """Multiple briefs for the same event are allowed (append-only)."""
        from odds_mcp.server import save_match_brief

        event, _ = epl_event_with_odds

        ids = []
        for i in range(3):
            result = await save_match_brief(
                event_id=event.id,
                market="1x2",
                decision="watching",
                summary=f"Revision {i}",
                brief_text=f"Brief revision {i}",
            )
            assert "error" not in result
            ids.append(result["id"])

        # All have distinct IDs
        assert len(set(ids)) == 3

        # Verify via DB
        db_result = await pglite_async_session.execute(
            select(MatchBrief).where(MatchBrief.event_id == event.id)
        )
        assert len(list(db_result.scalars().all())) == 3

    @pytest.mark.asyncio
    async def test_sharp_prices_via_lookback(self, patch_session_maker, sharp_lookback_event):
        """save_match_brief stamps sharp prices found via lookback when the
        latest snapshot lacks sharp bookmaker prices."""
        from odds_mcp.server import save_match_brief

        event, _snap_old, _snap_new = sharp_lookback_event

        result = await save_match_brief(
            event_id=event.id,
            market="1x2",
            decision="watching",
            summary="Brighton favoured at home",
            brief_text="Brighton should dominate at home.",
        )

        assert "error" not in result
        sharp = result["sharp_price_at_brief"]
        assert sharp is not None
        # Pinnacle is only in the older snapshot — lookback should find it
        for outcome in ("Brighton", "Draw", "Wolves"):
            assert sharp[outcome]["bookmaker"] == "pinnacle"
            assert "price" in sharp[outcome]
            assert "implied_prob" in sharp[outcome]


class TestGetMatchBrief:
    """Tests for the get_match_brief MCP tool."""

    @pytest.mark.asyncio
    async def test_retrieve_after_save(self, patch_session_maker, epl_event_with_odds):
        """get_match_brief retrieves briefs saved by save_match_brief."""
        from odds_mcp.server import get_match_brief, save_match_brief

        event, _ = epl_event_with_odds

        await save_match_brief(
            event_id=event.id,
            market="1x2",
            decision="watching",
            summary="Arsenal vs Chelsea context",
            brief_text="Context analysis for Arsenal vs Chelsea.",
        )

        result = await get_match_brief(event_id=event.id)

        assert "error" not in result
        assert result["brief_count"] == 1
        assert result["briefs"][0]["brief_text"] == "Context analysis for Arsenal vs Chelsea."
        assert result["event"]["id"] == event.id

    @pytest.mark.asyncio
    async def test_returns_all_briefs(self, patch_session_maker, epl_event_with_odds):
        """All briefs for an event are returned (append-only model)."""
        from odds_mcp.server import get_match_brief, save_match_brief

        event, _ = epl_event_with_odds

        await save_match_brief(
            event_id=event.id,
            market="1x2",
            decision="watching",
            summary="First",
            brief_text="First analysis",
        )
        await save_match_brief(
            event_id=event.id,
            market="1x2",
            decision="bet",
            summary="Updated",
            brief_text="Updated analysis",
        )

        result = await get_match_brief(event_id=event.id)
        assert result["brief_count"] == 2

    @pytest.mark.asyncio
    async def test_empty_results(self, patch_session_maker, epl_event_no_odds):
        """Querying briefs for an event with none returns empty list."""
        from odds_mcp.server import get_match_brief

        event = epl_event_no_odds

        result = await get_match_brief(event_id=event.id)

        assert "error" not in result
        assert result["brief_count"] == 0
        assert result["briefs"] == []

    @pytest.mark.asyncio
    async def test_newest_first_ordering(self, patch_session_maker, epl_event_with_odds):
        """Briefs are returned newest first."""
        from odds_mcp.server import get_match_brief, save_match_brief

        event, _ = epl_event_with_odds

        for i in range(3):
            await save_match_brief(
                event_id=event.id,
                market="1x2",
                decision="watching",
                summary=f"Summary {i}",
                brief_text=f"Brief {i}",
            )

        result = await get_match_brief(event_id=event.id)

        assert result["brief_count"] == 3
        # Newest first: Brief 2 should be first
        assert result["briefs"][0]["brief_text"] == "Brief 2"
        assert result["briefs"][2]["brief_text"] == "Brief 0"

    @pytest.mark.asyncio
    async def test_limit_parameter(self, patch_session_maker, epl_event_with_odds):
        """Limit parameter restricts number of briefs returned."""
        from odds_mcp.server import get_match_brief, save_match_brief

        event, _ = epl_event_with_odds

        for i in range(4):
            await save_match_brief(
                event_id=event.id,
                market="1x2",
                decision="watching",
                summary=f"Summary {i}",
                brief_text=f"Brief {i}",
            )

        result = await get_match_brief(event_id=event.id, limit=2)

        assert result["brief_count"] == 2
        assert len(result["briefs"]) == 2
        # Newest first
        assert result["briefs"][0]["brief_text"] == "Brief 3"
        assert result["briefs"][1]["brief_text"] == "Brief 2"

    @pytest.mark.asyncio
    async def test_event_not_found(self, patch_session_maker):
        """get_match_brief returns error for nonexistent event."""
        from odds_mcp.server import get_match_brief

        result = await get_match_brief(event_id="nonexistent_event")
        assert result == {"error": "Event 'nonexistent_event' not found"}


class TestGetSlateBriefs:
    """Tests for the get_slate_briefs MCP tool."""

    @pytest.fixture
    async def future_epl_events(self, pglite_async_session):
        """Two future EPL events for slate testing."""
        future = datetime.now(UTC) + timedelta(days=2)
        event_a = Event(
            id="slate_test_001",
            sport_key="soccer_epl",
            sport_title="EPL",
            commence_time=future,
            home_team="Arsenal",
            away_team="Chelsea",
            status=EventStatus.SCHEDULED,
        )
        event_b = Event(
            id="slate_test_002",
            sport_key="soccer_epl",
            sport_title="EPL",
            commence_time=future + timedelta(hours=3),
            home_team="Liverpool",
            away_team="Man City",
            status=EventStatus.SCHEDULED,
        )
        pglite_async_session.add_all([event_a, event_b])
        await pglite_async_session.commit()
        await pglite_async_session.refresh(event_a)
        await pglite_async_session.refresh(event_b)
        return event_a, event_b

    @pytest.mark.asyncio
    async def test_happy_path_with_and_without_briefs(
        self, patch_session_maker, future_epl_events, pglite_async_session
    ):
        """Returns latest brief for events that have one, None for those without."""
        from odds_mcp.server import get_slate_briefs

        event_with, event_without = future_epl_events

        # Insert brief directly (no snapshot needed for this test)
        brief = MatchBrief(
            event_id=event_with.id,
            decision=BriefDecision.WATCHING,
            summary="Looks interesting",
            brief_text="Full analysis here.",
        )
        pglite_async_session.add(brief)
        await pglite_async_session.commit()

        result = await get_slate_briefs(league="soccer_epl", days_ahead=7)

        assert result["fixture_count"] == 2
        by_event = {f["event"]["id"]: f for f in result["fixtures"]}

        assert by_event[event_with.id]["latest_brief"] is not None
        assert by_event[event_with.id]["latest_brief"]["decision"] == "watching"
        assert by_event[event_with.id]["latest_brief"]["summary"] == "Looks interesting"

        assert by_event[event_without.id]["latest_brief"] is None

    @pytest.mark.asyncio
    async def test_empty_slate(self, patch_session_maker):
        """No upcoming fixtures returns empty list."""
        from odds_mcp.server import get_slate_briefs

        result = await get_slate_briefs(league="soccer_epl", days_ahead=1)

        assert result["fixture_count"] == 0
        assert result["fixtures"] == []

    @pytest.mark.asyncio
    async def test_returns_latest_brief(
        self, patch_session_maker, future_epl_events, pglite_async_session
    ):
        """When multiple briefs exist, returns the most recent one."""
        from odds_mcp.server import get_slate_briefs

        event, _ = future_epl_events

        brief_old = MatchBrief(
            event_id=event.id,
            decision=BriefDecision.WATCHING,
            summary="First look",
            brief_text="Initial analysis.",
        )
        pglite_async_session.add(brief_old)
        await pglite_async_session.commit()

        brief_new = MatchBrief(
            event_id=event.id,
            decision=BriefDecision.BET,
            summary="Edge found on Arsenal",
            brief_text="Updated analysis with bet.",
        )
        pglite_async_session.add(brief_new)
        await pglite_async_session.commit()

        result = await get_slate_briefs(league="soccer_epl", days_ahead=7)

        by_event = {f["event"]["id"]: f for f in result["fixtures"]}
        brief = by_event[event.id]["latest_brief"]
        assert brief["decision"] == "bet"
        assert brief["summary"] == "Edge found on Arsenal"

    @pytest.mark.asyncio
    async def test_does_not_include_full_brief_text(
        self, patch_session_maker, future_epl_events, pglite_async_session
    ):
        """Slate briefs should not include full brief_text."""
        from odds_mcp.server import get_slate_briefs

        event, _ = future_epl_events

        brief = MatchBrief(
            event_id=event.id,
            decision=BriefDecision.SKIP,
            summary="No edge",
            brief_text="Detailed analysis that should not appear in slate view.",
        )
        pglite_async_session.add(brief)
        await pglite_async_session.commit()

        result = await get_slate_briefs(league="soccer_epl", days_ahead=7)

        by_event = {f["event"]["id"]: f for f in result["fixtures"]}
        latest = by_event[event.id]["latest_brief"]
        assert "brief_text" not in latest


class TestFindRetailEdges:
    """Integration tests for the find_retail_edges MCP tool."""

    @pytest.mark.asyncio
    async def test_happy_path(self, patch_session_maker, epl_event_with_odds):
        """Response has the documented shape with sharp_bookmakers, per_outcome, retail_edges."""
        from odds_mcp.server import find_retail_edges

        event, _ = epl_event_with_odds

        result = await find_retail_edges(event_id=event.id, market="1x2")

        assert "error" not in result
        assert result["event"]["id"] == event.id
        assert result["snapshot_time"] is not None
        assert result["sharp_bookmakers"] == ["pinnacle", "betfair_exchange"]
        assert isinstance(result["per_outcome"], list)
        assert isinstance(result["retail_edges"], list)

        outcomes = {e["outcome"] for e in result["per_outcome"]}
        assert outcomes == {"Arsenal", "Draw", "Chelsea"}

        arsenal = next(e for e in result["per_outcome"] if e["outcome"] == "Arsenal")
        assert arsenal["sharp_implied_prob"] is not None
        assert arsenal["sharp_snapshot_time"] is not None
        assert arsenal["sharp_age_seconds"] is not None
        assert arsenal["n_books"] == 2  # bet365 + betway
        # Required keys on best/worst
        for side in ("best_retail", "worst_retail"):
            assert set(arsenal[side].keys()) == {
                "book",
                "price",
                "implied_prob",
                "divergence",
                "z_score",
                "market_hold",
            }

    @pytest.mark.asyncio
    async def test_no_snapshot_graceful(self, patch_session_maker, epl_event_no_odds):
        """Event with no snapshots returns empty per_outcome/retail_edges with message."""
        from odds_mcp.server import find_retail_edges

        event = epl_event_no_odds

        result = await find_retail_edges(event_id=event.id, market="1x2")

        assert "error" not in result
        assert result["per_outcome"] == []
        assert result["retail_edges"] == []
        assert "message" in result

    @pytest.mark.asyncio
    async def test_event_not_found(self, patch_session_maker):
        """find_retail_edges returns error for nonexistent event."""
        from odds_mcp.server import find_retail_edges

        result = await find_retail_edges(event_id="nonexistent_event", market="1x2")
        assert result == {"error": "Event 'nonexistent_event' not found"}

    @pytest.mark.asyncio
    async def test_per_outcome_sharp_fallback(self, patch_session_maker, partial_sharp_event):
        """Sharp prices fall through per-outcome via lookback reader."""
        from odds_mcp.server import find_retail_edges

        event, _ = partial_sharp_event

        result = await find_retail_edges(event_id=event.id, market="1x2")

        by_outcome = {e["outcome"]: e for e in result["per_outcome"]}
        # All outcomes should have a sharp reference; bet365 divergence is computable
        for name in ("Tottenham", "Draw", "Everton"):
            assert by_outcome[name]["sharp_implied_prob"] is not None
            assert by_outcome[name]["best_retail"]["divergence"] is not None

    @pytest.mark.asyncio
    async def test_multi_line_totals_produces_entries_per_point(
        self, patch_session_maker, pglite_async_session
    ):
        """Totals with multiple point values produce one per_outcome entry per (outcome, point)."""
        from odds_mcp.server import find_retail_edges

        commence_time = datetime(2026, 4, 25, 15, 0, tzinfo=UTC)
        event = Event(
            id="totals_test_001",
            sport_key="baseball_mlb",
            sport_title="MLB",
            commence_time=commence_time,
            home_team="Yankees",
            away_team="Red Sox",
            status=EventStatus.SCHEDULED,
        )
        pglite_async_session.add(event)

        snapshot_time = commence_time - timedelta(hours=3)
        raw_data = {
            "bookmakers": [
                {
                    "key": "betfair_exchange",
                    "title": "Betfair Exchange",
                    "last_update": snapshot_time.isoformat(),
                    "markets": [
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over", "price": -110, "point": 8.5},
                                {"name": "Under", "price": -110, "point": 8.5},
                                {"name": "Over", "price": +120, "point": 9.5},
                                {"name": "Under", "price": -140, "point": 9.5},
                            ],
                        }
                    ],
                },
                {
                    "key": "bet365",
                    "title": "Bet365",
                    "last_update": snapshot_time.isoformat(),
                    "markets": [
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over", "price": -105, "point": 8.5},
                                {"name": "Under", "price": -115, "point": 8.5},
                                {"name": "Over", "price": +115, "point": 9.5},
                                {"name": "Under", "price": -135, "point": 9.5},
                            ],
                        }
                    ],
                },
                {
                    "key": "betway",
                    "title": "Betway",
                    "last_update": snapshot_time.isoformat(),
                    "markets": [
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over", "price": -115, "point": 8.5},
                                {"name": "Under", "price": -105, "point": 8.5},
                                {"name": "Over", "price": +110, "point": 9.5},
                                {"name": "Under", "price": -130, "point": 9.5},
                            ],
                        }
                    ],
                },
            ]
        }
        snapshot = OddsSnapshot(
            event_id=event.id,
            snapshot_time=snapshot_time,
            raw_data=raw_data,
            bookmaker_count=3,
            fetch_tier="pregame",
            hours_until_commence=3.0,
        )
        pglite_async_session.add(snapshot)
        await pglite_async_session.commit()
        await pglite_async_session.refresh(event)

        result = await find_retail_edges(
            event_id=event.id,
            market="totals",
            sharp_bookmakers=["betfair_exchange"],
        )

        keys = {(e["outcome"], e["point"]) for e in result["per_outcome"]}
        assert keys == {
            ("Over", 8.5),
            ("Under", 8.5),
            ("Over", 9.5),
            ("Under", 9.5),
        }

        # Every per_outcome entry is a totals bucket — market_hold must be null
        for entry in result["per_outcome"]:
            if entry["best_retail"] is not None:
                assert entry["best_retail"]["market_hold"] is None
                assert entry["worst_retail"]["market_hold"] is None

        # retail_edges entries include point
        for edge in result["retail_edges"]:
            assert edge["point"] in {8.5, 9.5}

    @pytest.mark.asyncio
    async def test_cry_whu_shaped_fixture_outlier_ranks_first(
        self, patch_session_maker, pglite_async_session
    ):
        """CRY-WHU-shaped fixture: a high-hold midnite book long on one outcome
        surfaces at rank 1 of retail_edges despite its overall hold.
        """
        from odds_mcp.server import find_retail_edges

        commence_time = datetime(2026, 4, 22, 15, 0, tzinfo=UTC)
        event = Event(
            id="cry_whu_test_001",
            sport_key="soccer_epl",
            sport_title="EPL",
            commence_time=commence_time,
            home_team="Crystal Palace",
            away_team="West Ham",
            status=EventStatus.SCHEDULED,
        )
        pglite_async_session.add(event)

        snapshot_time = commence_time - timedelta(hours=6)

        # Sharp (Betfair Exchange) West Ham +198 (33.6%)
        # Retail books cluster around sharp with small positive divergence (SHORTER).
        # midnite is *longer* on West Ham at +230 (30.3%, divergence ~-3.3%),
        # with an asymmetrically-shorter Crystal Palace and Draw to load hold.
        retail_entries = [
            ("bet365", -115, 275, 175),
            ("betway", -110, 270, 170),
            ("betfred", -112, 272, 172),
            ("betvictor", -114, 273, 173),
            ("bwin", -113, 271, 174),
            ("paddypower", -115, 270, 170),
            ("skybet", -117, 275, 175),
            ("williamhill", -115, 272, 171),
            ("10bet", -110, 265, 165),
            ("betmgm", -115, 273, 172),
            ("888sport", -114, 275, 173),
            ("betuk", -112, 270, 172),
            ("spreadex", -114, 270, 170),
            ("betano", -115, 272, 171),
            ("unibet_uk", -113, 273, 174),
            ("allbritishcasino", -112, 271, 173),
            ("7bet", -115, 270, 170),
        ]
        bookmakers = [
            {
                "key": "betfair_exchange",
                "title": "Betfair Exchange",
                "last_update": snapshot_time.isoformat(),
                "markets": [
                    {
                        "key": "1x2",
                        "outcomes": [
                            {"name": "Crystal Palace", "price": -105},
                            {"name": "Draw", "price": 280},
                            {"name": "West Ham", "price": 198},
                        ],
                    }
                ],
            }
        ]
        for key, cp_price, draw_price, wh_price in retail_entries:
            bookmakers.append(
                {
                    "key": key,
                    "title": key.title(),
                    "last_update": snapshot_time.isoformat(),
                    "markets": [
                        {
                            "key": "1x2",
                            "outcomes": [
                                {"name": "Crystal Palace", "price": cp_price},
                                {"name": "Draw", "price": draw_price},
                                {"name": "West Ham", "price": wh_price},
                            ],
                        }
                    ],
                }
            )

        # midnite: shorter on CP and Draw (loads hold), longer on West Ham at +230.
        bookmakers.append(
            {
                "key": "midnite",
                "title": "Midnite",
                "last_update": snapshot_time.isoformat(),
                "markets": [
                    {
                        "key": "1x2",
                        "outcomes": [
                            {"name": "Crystal Palace", "price": -140},
                            {"name": "Draw", "price": 240},
                            {"name": "West Ham", "price": 230},
                        ],
                    }
                ],
            }
        )

        snapshot = OddsSnapshot(
            event_id=event.id,
            snapshot_time=snapshot_time,
            raw_data={"bookmakers": bookmakers},
            bookmaker_count=len(bookmakers),
            fetch_tier="pregame",
            hours_until_commence=6.0,
        )
        pglite_async_session.add(snapshot)
        await pglite_async_session.commit()
        await pglite_async_session.refresh(event)

        result = await find_retail_edges(event_id=event.id, market="1x2")

        edges = result["retail_edges"]
        assert len(edges) >= 1
        # midnite West Ham is at rank 1
        assert edges[0]["book"] == "midnite"
        assert edges[0]["outcome"] == "West Ham"
        assert edges[0]["divergence"] < 0
        # midnite should have a higher (asymmetric) hold than the retail average
        wh_bucket = next(e for e in result["per_outcome"] if e["outcome"] == "West Ham")
        assert wh_bucket["n_books"] == 18  # 17 retail + midnite
        assert wh_bucket["dispersion_stddev"] is not None
        assert wh_bucket["dispersion_stddev"] > 0
        # z_score on the rank-1 edge should be computed and negative
        assert edges[0]["z_score"] is not None
        assert edges[0]["z_score"] < 0
        # midnite market_hold should be populated (1x2) and visibly larger than
        # the retail pack's typical 4-6% hold — fixture prices put it around 18%.
        assert edges[0]["market_hold"] is not None
        assert edges[0]["market_hold"] > 0.12


def _make_snapshot_raw_data(bookmakers: list[dict[str, object]], snapshot_time: datetime) -> dict:
    """Build raw_data blob for a snapshot from a compact bookmaker spec.

    Each entry in *bookmakers* is ``{"key": str, "outcomes": list[dict]}``.
    """
    return {
        "bookmakers": [
            {
                "key": bm["key"],
                "title": str(bm["key"]).title(),
                "last_update": snapshot_time.isoformat(),
                "markets": [{"key": "1x2", "outcomes": bm["outcomes"]}],
            }
            for bm in bookmakers
        ]
    }


@pytest.fixture
async def sharp_lookback_event(pglite_async_session):
    """Event with two snapshots: newest has retail only, older has sharp + retail.

    Timeline (relative to KO at T):
        T-3h  snapshot_old  — pinnacle + bet365
        T-2h  snapshot_new  — bet365 only (sharp missing)
    """
    commence_time = datetime(2026, 4, 20, 15, 0, tzinfo=UTC)
    event = Event(
        id="epl_lookback_001",
        sport_key="soccer_epl",
        sport_title="EPL",
        commence_time=commence_time,
        home_team="Brighton",
        away_team="Wolves",
        status=EventStatus.SCHEDULED,
    )
    pglite_async_session.add(event)

    # Older snapshot WITH sharp bookmaker
    old_time = commence_time - timedelta(hours=3)
    old_raw = _make_snapshot_raw_data(
        [
            {
                "key": "pinnacle",
                "outcomes": [
                    {"name": "Brighton", "price": -140},
                    {"name": "Draw", "price": 270},
                    {"name": "Wolves", "price": 350},
                ],
            },
            {
                "key": "bet365",
                "outcomes": [
                    {"name": "Brighton", "price": -150},
                    {"name": "Draw", "price": 260},
                    {"name": "Wolves", "price": 330},
                ],
            },
        ],
        old_time,
    )
    snap_old = OddsSnapshot(
        event_id=event.id,
        snapshot_time=old_time,
        raw_data=old_raw,
        bookmaker_count=2,
        fetch_tier="pregame",
        hours_until_commence=3.0,
    )
    pglite_async_session.add(snap_old)

    # Newer snapshot WITHOUT sharp bookmaker
    new_time = commence_time - timedelta(hours=2)
    new_raw = _make_snapshot_raw_data(
        [
            {
                "key": "bet365",
                "outcomes": [
                    {"name": "Brighton", "price": -155},
                    {"name": "Draw", "price": 255},
                    {"name": "Wolves", "price": 325},
                ],
            },
        ],
        new_time,
    )
    snap_new = OddsSnapshot(
        event_id=event.id,
        snapshot_time=new_time,
        raw_data=new_raw,
        bookmaker_count=1,
        fetch_tier="pregame",
        hours_until_commence=2.0,
    )
    pglite_async_session.add(snap_new)

    await pglite_async_session.commit()
    await pglite_async_session.refresh(event)
    await pglite_async_session.refresh(snap_old)
    await pglite_async_session.refresh(snap_new)
    return event, snap_old, snap_new


class TestGetSharpPrices:
    """Tests for OddsReader.get_sharp_prices lookback method."""

    @pytest.mark.asyncio
    async def test_sharp_found_in_older_snapshot(self, pglite_async_session, sharp_lookback_event):
        """Sharp price is resolved from an older snapshot when newest lacks it."""
        event, snap_old, snap_new = sharp_lookback_event
        reader = OddsReader(pglite_async_session)

        # Use now=snap_new.snapshot_time so both snapshots are in the 2h window
        result = await reader.get_sharp_prices(
            event.id,
            market="1x2",
            sharp_bookmakers=["pinnacle", "betfair_exchange"],
            lookback_hours=2.0,
            now=snap_new.snapshot_time,
        )

        assert isinstance(result, SharpPriceResult)
        assert "Brighton" in result.prices
        assert result.prices["Brighton"]["bookmaker"] == "pinnacle"
        assert result.prices["Brighton"]["price"] == -140
        assert result.prices["Draw"]["bookmaker"] == "pinnacle"
        assert result.prices["Wolves"]["bookmaker"] == "pinnacle"

        # Metadata points to the older snapshot
        assert result.meta["Brighton"].snapshot_id == snap_old.id
        assert result.meta["Brighton"].age_seconds > 0

    @pytest.mark.asyncio
    async def test_no_sharp_in_window(self, pglite_async_session, sharp_lookback_event):
        """When no snapshot in the window has a sharp bookmaker, result is empty."""
        event, snap_old, snap_new = sharp_lookback_event
        reader = OddsReader(pglite_async_session)

        # Set window so only the newest (retail-only) snapshot is included
        result = await reader.get_sharp_prices(
            event.id,
            market="1x2",
            sharp_bookmakers=["pinnacle"],
            lookback_hours=0.5,
            now=snap_new.snapshot_time,
        )

        assert result.prices == {}
        assert result.meta == {}

    @pytest.mark.asyncio
    async def test_expired_window(self, pglite_async_session, sharp_lookback_event):
        """When now is far in the future, no snapshots are in the lookback window."""
        event, _snap_old, _snap_new = sharp_lookback_event
        reader = OddsReader(pglite_async_session)

        far_future = datetime(2026, 5, 1, 0, 0, tzinfo=UTC)
        result = await reader.get_sharp_prices(
            event.id,
            market="1x2",
            sharp_bookmakers=["pinnacle"],
            lookback_hours=2.0,
            now=far_future,
        )

        assert result.prices == {}
        assert result.meta == {}

    @pytest.mark.asyncio
    async def test_recency_takes_precedence_over_priority(
        self, pglite_async_session, sharp_lookback_event
    ):
        """Recency wins: once an outcome is resolved from a newer snapshot,
        older snapshots are skipped even if they contain a higher-priority bookmaker."""
        event, snap_old, snap_new = sharp_lookback_event
        reader = OddsReader(pglite_async_session)

        # pinnacle is higher priority but only in the older snapshot;
        # bet365 in the newer snapshot resolves first.
        result = await reader.get_sharp_prices(
            event.id,
            market="1x2",
            sharp_bookmakers=["pinnacle", "bet365"],
            lookback_hours=2.0,
            now=snap_new.snapshot_time,
        )

        assert result.prices["Brighton"]["bookmaker"] == "bet365"
        assert result.prices["Brighton"]["price"] == -155


class TestFindRetailEdgesSharpLookback:
    """Tests for find_retail_edges using sharp lookback resolution."""

    @pytest.mark.asyncio
    async def test_lookback_finds_sharp_from_older_snapshot(
        self, patch_session_maker, sharp_lookback_event
    ):
        """find_retail_edges resolves sharp from older snapshot via lookback."""
        from odds_mcp.server import find_retail_edges

        event, _snap_old, _snap_new = sharp_lookback_event

        result = await find_retail_edges(
            event_id=event.id,
            market="1x2",
            sharp_lookback_hours=4.0,
        )

        assert "error" not in result

        # Sharp price for Brighton should come from the older snapshot (pinnacle)
        brighton = next(e for e in result["per_outcome"] if e["outcome"] == "Brighton")
        assert brighton["sharp_implied_prob"] is not None
        assert brighton["sharp_snapshot_time"] is not None
        # Divergence is computed off the lookback sharp price
        assert brighton["best_retail"]["divergence"] is not None

    @pytest.mark.asyncio
    async def test_no_sharp_returns_null_sharp(self, patch_session_maker, sharp_lookback_event):
        """When lookback is too short to find sharp, sharp fields are null and
        no entries appear in retail_edges for those outcomes."""
        from odds_mcp.server import find_retail_edges

        event, _snap_old, _snap_new = sharp_lookback_event

        # Very short lookback — only latest snapshot in window, no pinnacle
        result = await find_retail_edges(
            event_id=event.id,
            market="1x2",
            sharp_lookback_hours=0.01,
        )

        assert result["retail_edges"] == []
        for entry in result["per_outcome"]:
            assert entry["sharp_implied_prob"] is None
            assert entry["sharp_snapshot_time"] is None
            # best_retail/worst_retail still populated but with null divergence/z_score
            assert entry["best_retail"]["divergence"] is None
            assert entry["best_retail"]["z_score"] is None
