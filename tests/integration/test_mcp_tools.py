"""Integration tests for Phase 2 MCP tools (match briefs + sharp/soft spread).

Tests call the actual MCP tool handler functions end-to-end, with
async_session_maker patched to use the PGlite test database.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
from odds_core.match_brief_models import BriefCheckpoint, MatchBrief
from odds_core.models import Event, EventStatus, OddsSnapshot
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
                        "key": "h2h",
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
                        "key": "h2h",
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
                        "key": "h2h",
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
                        "key": "h2h",
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
                        "key": "h2h",
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
                        "key": "h2h",
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
            brief_text="Arsenal looking strong at home, Chelsea missing key players.",
            checkpoint="context",
        )

        assert "error" not in result
        assert result["id"] is not None
        assert result["event_id"] == event.id
        assert result["checkpoint"] == "context"
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
            brief_text="Should fail",
            checkpoint="context",
        )

        assert result == {"error": "Event 'nonexistent_event' not found"}

    @pytest.mark.asyncio
    async def test_no_snapshot_still_saves(self, patch_session_maker, epl_event_no_odds):
        """save_match_brief works with no snapshot (sharp prices will be null)."""
        from odds_mcp.server import save_match_brief

        event = epl_event_no_odds

        result = await save_match_brief(
            event_id=event.id,
            brief_text="No odds available yet for this fixture.",
            checkpoint="decision",
        )

        assert "error" not in result
        assert result["id"] is not None
        assert result["sharp_price_at_brief"] is None
        assert result["checkpoint"] == "decision"

    @pytest.mark.asyncio
    async def test_multiple_briefs_per_checkpoint(
        self, patch_session_maker, epl_event_with_odds, pglite_async_session
    ):
        """Multiple briefs for the same event+checkpoint are allowed."""
        from odds_mcp.server import save_match_brief

        event, _ = epl_event_with_odds

        ids = []
        for i in range(3):
            result = await save_match_brief(
                event_id=event.id,
                brief_text=f"Decision brief revision {i}",
                checkpoint="decision",
            )
            assert "error" not in result
            ids.append(result["id"])

        # All have distinct IDs
        assert len(set(ids)) == 3

        # Verify via DB
        db_result = await pglite_async_session.execute(
            select(MatchBrief)
            .where(MatchBrief.event_id == event.id)
            .where(MatchBrief.checkpoint == BriefCheckpoint.DECISION)
        )
        assert len(list(db_result.scalars().all())) == 3


class TestGetMatchBrief:
    """Tests for the get_match_brief MCP tool."""

    @pytest.mark.asyncio
    async def test_retrieve_after_save(self, patch_session_maker, epl_event_with_odds):
        """get_match_brief retrieves briefs saved by save_match_brief."""
        from odds_mcp.server import get_match_brief, save_match_brief

        event, _ = epl_event_with_odds

        await save_match_brief(
            event_id=event.id,
            brief_text="Context analysis for Arsenal vs Chelsea.",
            checkpoint="context",
        )

        result = await get_match_brief(event_id=event.id)

        assert "error" not in result
        assert result["brief_count"] == 1
        assert result["briefs"][0]["brief_text"] == "Context analysis for Arsenal vs Chelsea."
        assert result["event"]["id"] == event.id

    @pytest.mark.asyncio
    async def test_checkpoint_filtering(self, patch_session_maker, epl_event_with_odds):
        """Filtering by checkpoint returns only matching briefs."""
        from odds_mcp.server import get_match_brief, save_match_brief

        event, _ = epl_event_with_odds

        await save_match_brief(
            event_id=event.id, brief_text="Context analysis", checkpoint="context"
        )
        await save_match_brief(
            event_id=event.id, brief_text="Decision analysis", checkpoint="decision"
        )

        context_result = await get_match_brief(event_id=event.id, checkpoint="context")
        assert context_result["brief_count"] == 1
        assert context_result["briefs"][0]["brief_text"] == "Context analysis"

        decision_result = await get_match_brief(event_id=event.id, checkpoint="decision")
        assert decision_result["brief_count"] == 1
        assert decision_result["briefs"][0]["brief_text"] == "Decision analysis"

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
            await save_match_brief(event_id=event.id, brief_text=f"Brief {i}", checkpoint="context")

        result = await get_match_brief(event_id=event.id)

        assert result["brief_count"] == 3
        # Newest first: Brief 2 should be first
        assert result["briefs"][0]["brief_text"] == "Brief 2"
        assert result["briefs"][2]["brief_text"] == "Brief 0"

    @pytest.mark.asyncio
    async def test_event_not_found(self, patch_session_maker):
        """get_match_brief returns error for nonexistent event."""
        from odds_mcp.server import get_match_brief

        result = await get_match_brief(event_id="nonexistent_event")
        assert result == {"error": "Event 'nonexistent_event' not found"}


class TestSnapshotSharpPrices:
    """Tests for _snapshot_sharp_prices helper."""

    def test_per_outcome_fallback(self, partial_sharp_event):
        """Each outcome independently falls through to next sharp bookmaker."""
        from odds_analytics.sequence_loader import extract_odds_from_snapshot
        from odds_mcp.server import _snapshot_sharp_prices

        event, snapshot = partial_sharp_event
        odds = extract_odds_from_snapshot(snapshot, event.id, market="h2h")
        result = _snapshot_sharp_prices(odds, ["pinnacle", "betfair_exchange"])

        # Pinnacle only has Tottenham, so home should come from pinnacle
        assert result["Tottenham"]["bookmaker"] == "pinnacle"
        # Draw and Everton should fall through to betfair_exchange
        assert result["Draw"]["bookmaker"] == "betfair_exchange"
        assert result["Everton"]["bookmaker"] == "betfair_exchange"

    def test_all_outcomes_from_primary(self, epl_event_with_odds):
        """When primary sharp has all outcomes, all come from it."""
        from odds_analytics.sequence_loader import extract_odds_from_snapshot
        from odds_mcp.server import _snapshot_sharp_prices

        event, snapshot = epl_event_with_odds
        odds = extract_odds_from_snapshot(snapshot, event.id, market="h2h")
        result = _snapshot_sharp_prices(odds, ["pinnacle", "betfair_exchange"])

        for outcome_data in result.values():
            assert outcome_data["bookmaker"] == "pinnacle"

    def test_empty_odds_list(self):
        """Empty odds list returns empty dict."""
        from odds_mcp.server import _snapshot_sharp_prices

        result = _snapshot_sharp_prices([], ["pinnacle"])
        assert result == {}


class TestGetSharpSoftSpread:
    """Tests for the get_sharp_soft_spread MCP tool."""

    @pytest.mark.asyncio
    async def test_happy_path(self, patch_session_maker, epl_event_with_odds):
        """Sharp vs soft spread is computed correctly with divergence values."""
        from odds_mcp.server import get_sharp_soft_spread

        event, _ = epl_event_with_odds

        result = await get_sharp_soft_spread(event_id=event.id)

        assert "error" not in result
        assert result["event"]["id"] == event.id
        assert result["snapshot_time"] is not None
        spread = result["spread"]
        assert spread is not None

        # Check Arsenal outcome
        arsenal = spread["Arsenal"]
        assert arsenal["sharp"]["bookmaker"] == "pinnacle"
        assert arsenal["sharp"]["price"] == -120
        assert arsenal["sharp"]["implied_prob"] is not None

        # Check retail bookmakers present
        soft_bms = {s["bookmaker"] for s in arsenal["soft"]}
        assert "bet365" in soft_bms
        assert "betway" in soft_bms

        # Divergence should be computed
        for soft_entry in arsenal["soft"]:
            assert soft_entry["divergence"] is not None

    @pytest.mark.asyncio
    async def test_no_snapshot_graceful(self, patch_session_maker, epl_event_no_odds):
        """Event with no snapshots returns None spread with message."""
        from odds_mcp.server import get_sharp_soft_spread

        event = epl_event_no_odds

        result = await get_sharp_soft_spread(event_id=event.id)

        assert "error" not in result
        assert result["spread"] is None
        assert "message" in result

    @pytest.mark.asyncio
    async def test_event_not_found(self, patch_session_maker):
        """get_sharp_soft_spread returns error for nonexistent event."""
        from odds_mcp.server import get_sharp_soft_spread

        result = await get_sharp_soft_spread(event_id="nonexistent_event")
        assert result == {"error": "Event 'nonexistent_event' not found"}

    @pytest.mark.asyncio
    async def test_per_outcome_sharp_fallback(self, patch_session_maker, partial_sharp_event):
        """Sharp prices fall through per-outcome (reuses _snapshot_sharp_prices)."""
        from odds_mcp.server import get_sharp_soft_spread

        event, _ = partial_sharp_event

        result = await get_sharp_soft_spread(event_id=event.id)

        spread = result["spread"]
        # Tottenham from pinnacle (higher priority)
        assert spread["Tottenham"]["sharp"]["bookmaker"] == "pinnacle"
        # Draw/Everton fall through to betfair_exchange
        assert spread["Draw"]["sharp"]["bookmaker"] == "betfair_exchange"
        assert spread["Everton"]["sharp"]["bookmaker"] == "betfair_exchange"
