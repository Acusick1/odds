"""Integration tests for Phase 2 MCP tools (match briefs + sharp/soft spread).

Tests call the actual MCP tool handler functions end-to-end, with
async_session_maker patched to use the PGlite test database.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
from odds_core.match_brief_models import MatchBrief, SharpPriceResult
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


class TestGetSharpSoftSpread:
    """Tests for the get_sharp_soft_spread MCP tool."""

    @pytest.mark.asyncio
    async def test_happy_path(self, patch_session_maker, epl_event_with_odds):
        """Sharp vs soft spread is computed correctly with divergence values."""
        from odds_mcp.server import get_sharp_soft_spread

        event, _ = epl_event_with_odds

        result = await get_sharp_soft_spread(event_id=event.id, market="1x2")

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
        # Source snapshot metadata
        assert arsenal["sharp"]["snapshot_time"] is not None
        assert arsenal["sharp"]["age_seconds"] is not None

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

        result = await get_sharp_soft_spread(event_id=event.id, market="1x2")

        assert "error" not in result
        assert result["spread"] is None
        assert "message" in result

    @pytest.mark.asyncio
    async def test_event_not_found(self, patch_session_maker):
        """get_sharp_soft_spread returns error for nonexistent event."""
        from odds_mcp.server import get_sharp_soft_spread

        result = await get_sharp_soft_spread(event_id="nonexistent_event", market="1x2")
        assert result == {"error": "Event 'nonexistent_event' not found"}

    @pytest.mark.asyncio
    async def test_per_outcome_sharp_fallback(self, patch_session_maker, partial_sharp_event):
        """Sharp prices fall through per-outcome via lookback reader."""
        from odds_mcp.server import get_sharp_soft_spread

        event, _ = partial_sharp_event

        result = await get_sharp_soft_spread(event_id=event.id, market="1x2")

        spread = result["spread"]
        # Tottenham from pinnacle (higher priority)
        assert spread["Tottenham"]["sharp"]["bookmaker"] == "pinnacle"
        # Draw/Everton fall through to betfair_exchange
        assert spread["Draw"]["sharp"]["bookmaker"] == "betfair_exchange"
        assert spread["Everton"]["sharp"]["bookmaker"] == "betfair_exchange"


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


class TestSharpLookbackSpread:
    """Tests for get_sharp_soft_spread using sharp lookback."""

    @pytest.mark.asyncio
    async def test_lookback_finds_sharp_from_older_snapshot(
        self, patch_session_maker, sharp_lookback_event
    ):
        """get_sharp_soft_spread resolves sharp from older snapshot via lookback."""
        from odds_mcp.server import get_sharp_soft_spread

        event, snap_old, snap_new = sharp_lookback_event

        result = await get_sharp_soft_spread(
            event_id=event.id,
            market="1x2",
            sharp_lookback_hours=4.0,
        )

        assert "error" not in result
        spread = result["spread"]
        assert spread is not None

        # Sharp price should come from the older snapshot (pinnacle)
        brighton = spread["Brighton"]
        assert brighton["sharp"]["bookmaker"] == "pinnacle"
        assert brighton["sharp"]["price"] == -140
        assert brighton["sharp"]["snapshot_time"] is not None

        # Retail prices should come from the latest snapshot (bet365)
        assert len(brighton["soft"]) >= 1
        bet365_soft = [s for s in brighton["soft"] if s["bookmaker"] == "bet365"]
        assert bet365_soft[0]["price"] == -155

    @pytest.mark.asyncio
    async def test_no_sharp_returns_null_sharp(self, patch_session_maker, sharp_lookback_event):
        """When lookback is too short to find sharp, sharp fields are null."""
        from odds_mcp.server import get_sharp_soft_spread

        event, _snap_old, _snap_new = sharp_lookback_event

        # Very short lookback — only latest snapshot in window, no pinnacle
        result = await get_sharp_soft_spread(
            event_id=event.id,
            market="1x2",
            sharp_lookback_hours=0.01,
        )

        spread = result["spread"]
        assert spread is not None
        for outcome_data in spread.values():
            assert outcome_data["sharp"]["bookmaker"] is None
            assert outcome_data["sharp"]["snapshot_time"] is None
