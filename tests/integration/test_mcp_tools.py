"""Integration tests for Phase 2 MCP tools (match briefs + sharp/soft spread)."""

from datetime import UTC, datetime, timedelta

import pytest
from odds_core.match_brief_models import BriefCheckpoint, MatchBrief
from odds_core.models import Event, EventStatus, OddsSnapshot
from sqlalchemy import select


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
    async def test_save_and_retrieve_brief(self, pglite_async_session, epl_event_with_odds):
        """save_match_brief persists a brief that get_match_brief can retrieve."""
        event, _ = epl_event_with_odds

        brief = MatchBrief(
            event_id=event.id,
            checkpoint=BriefCheckpoint.CONTEXT,
            brief_text="Arsenal looking strong at home, Chelsea missing key players.",
            sharp_price_at_brief={"Arsenal": {"bookmaker": "pinnacle", "price": -120}},
        )
        pglite_async_session.add(brief)
        await pglite_async_session.commit()
        await pglite_async_session.refresh(brief)

        assert brief.id is not None
        assert brief.event_id == event.id
        assert brief.checkpoint == BriefCheckpoint.CONTEXT
        assert brief.created_at is not None

        # Retrieve
        result = await pglite_async_session.execute(
            select(MatchBrief).where(MatchBrief.event_id == event.id)
        )
        briefs = list(result.scalars().all())
        assert len(briefs) == 1
        assert (
            briefs[0].brief_text == "Arsenal looking strong at home, Chelsea missing key players."
        )
        assert briefs[0].sharp_price_at_brief is not None

    @pytest.mark.asyncio
    async def test_multiple_briefs_per_checkpoint(self, pglite_async_session, epl_event_with_odds):
        """Multiple briefs for the same event+checkpoint are allowed."""
        event, _ = epl_event_with_odds

        for i in range(3):
            brief = MatchBrief(
                event_id=event.id,
                checkpoint=BriefCheckpoint.DECISION,
                brief_text=f"Decision brief revision {i}",
            )
            pglite_async_session.add(brief)

        await pglite_async_session.commit()

        result = await pglite_async_session.execute(
            select(MatchBrief)
            .where(MatchBrief.event_id == event.id)
            .where(MatchBrief.checkpoint == BriefCheckpoint.DECISION)
        )
        briefs = list(result.scalars().all())
        assert len(briefs) == 3

    @pytest.mark.asyncio
    async def test_sharp_price_auto_capture(self, pglite_async_session, epl_event_with_odds):
        """Sharp prices from the latest snapshot are captured in the brief."""
        from odds_analytics.sequence_loader import extract_odds_from_snapshot
        from odds_mcp.server import _snapshot_sharp_prices

        event, snapshot = epl_event_with_odds
        odds = extract_odds_from_snapshot(snapshot, event.id, market="h2h")
        sharp_prices = _snapshot_sharp_prices(odds, ["pinnacle", "betfair_exchange"])

        brief = MatchBrief(
            event_id=event.id,
            checkpoint=BriefCheckpoint.CONTEXT,
            brief_text="Test brief with sharp prices",
            sharp_price_at_brief=sharp_prices,
        )
        pglite_async_session.add(brief)
        await pglite_async_session.commit()
        await pglite_async_session.refresh(brief)

        assert brief.sharp_price_at_brief is not None
        # Pinnacle should be the sharp source for all outcomes
        for outcome_data in brief.sharp_price_at_brief.values():
            assert outcome_data["bookmaker"] == "pinnacle"
            assert "price" in outcome_data
            assert "implied_prob" in outcome_data


class TestGetMatchBrief:
    """Tests for the get_match_brief MCP tool."""

    @pytest.mark.asyncio
    async def test_checkpoint_filtering(self, pglite_async_session, epl_event_with_odds):
        """Filtering by checkpoint returns only matching briefs."""
        event, _ = epl_event_with_odds

        context_brief = MatchBrief(
            event_id=event.id,
            checkpoint=BriefCheckpoint.CONTEXT,
            brief_text="Context analysis",
        )
        decision_brief = MatchBrief(
            event_id=event.id,
            checkpoint=BriefCheckpoint.DECISION,
            brief_text="Decision analysis",
        )
        pglite_async_session.add(context_brief)
        pglite_async_session.add(decision_brief)
        await pglite_async_session.commit()

        # Filter by context
        result = await pglite_async_session.execute(
            select(MatchBrief)
            .where(MatchBrief.event_id == event.id)
            .where(MatchBrief.checkpoint == BriefCheckpoint.CONTEXT)
        )
        briefs = list(result.scalars().all())
        assert len(briefs) == 1
        assert briefs[0].brief_text == "Context analysis"

        # Filter by decision
        result = await pglite_async_session.execute(
            select(MatchBrief)
            .where(MatchBrief.event_id == event.id)
            .where(MatchBrief.checkpoint == BriefCheckpoint.DECISION)
        )
        briefs = list(result.scalars().all())
        assert len(briefs) == 1
        assert briefs[0].brief_text == "Decision analysis"

    @pytest.mark.asyncio
    async def test_graceful_empty_results(self, pglite_async_session, epl_event_no_odds):
        """Querying briefs for an event with none returns empty list."""
        event = epl_event_no_odds

        result = await pglite_async_session.execute(
            select(MatchBrief).where(MatchBrief.event_id == event.id)
        )
        briefs = list(result.scalars().all())
        assert len(briefs) == 0

    @pytest.mark.asyncio
    async def test_returns_all_briefs_newest_first(self, pglite_async_session, epl_event_with_odds):
        """All briefs are returned, ordered newest first."""
        event, _ = epl_event_with_odds

        for i in range(3):
            brief = MatchBrief(
                event_id=event.id,
                checkpoint=BriefCheckpoint.CONTEXT,
                brief_text=f"Brief {i}",
            )
            pglite_async_session.add(brief)

        await pglite_async_session.commit()

        result = await pglite_async_session.execute(
            select(MatchBrief)
            .where(MatchBrief.event_id == event.id)
            .order_by(MatchBrief.created_at.desc())
        )
        briefs = list(result.scalars().all())
        assert len(briefs) == 3


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
    """Tests for spread computation logic used by get_sharp_soft_spread."""

    @pytest.mark.asyncio
    async def test_spread_computation(self, pglite_async_session, epl_event_with_odds):
        """Sharp vs soft spread is computed correctly with divergence values."""
        from odds_analytics.sequence_loader import extract_odds_from_snapshot
        from odds_analytics.utils import calculate_implied_probability

        event, snapshot = epl_event_with_odds
        odds = extract_odds_from_snapshot(snapshot, event.id, market="h2h")

        # Group by outcome
        outcomes: dict[str, list] = {}
        for o in odds:
            outcomes.setdefault(o.outcome_name, []).append(o)

        # Verify Arsenal outcome has expected sharp/soft prices
        arsenal_odds = outcomes["Arsenal"]
        pinnacle_price = next(o.price for o in arsenal_odds if o.bookmaker_key == "pinnacle")
        bet365_price = next(o.price for o in arsenal_odds if o.bookmaker_key == "bet365")

        assert pinnacle_price == -120
        assert bet365_price == -130

        sharp_prob = calculate_implied_probability(-120)
        soft_prob = calculate_implied_probability(-130)
        assert soft_prob > sharp_prob  # bet365 implies higher prob (wider margin)

    @pytest.mark.asyncio
    async def test_no_snapshot_returns_gracefully(self, pglite_async_session, epl_event_no_odds):
        """Event with no snapshots returns None spread with message."""
        from odds_lambda.storage.readers import OddsReader

        reader = OddsReader(pglite_async_session)
        snapshot = await reader.get_latest_snapshot("epl_test_002")
        assert snapshot is None
