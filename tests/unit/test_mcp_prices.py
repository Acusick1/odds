from dataclasses import replace
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest
from odds_core.models import Event
from odds_core.quotes import Quote
from odds_core.vocabulary import Market, PriceOutcome, Side, Source, TimeBasis
from odds_mcp.prices import load_market_window


class TestMarketWindow:
    @pytest.mark.asyncio
    async def test_cutoffs_recency_snapshot_ties_and_line_identity(self) -> None:
        anchor = datetime.now(UTC)
        event = Event(
            id=1,
            sport_key="basketball_nba",
            sport_title="NBA",
            home_team="Boston Celtics",
            away_team="Miami Heat",
            commence_time=anchor + timedelta(hours=4),
        )
        base = Quote(
            snapshot_id=2,
            event_id=1,
            snapshot_time=anchor,
            created_at=anchor,
            source=Source.ODDS_API,
            market=Market.TOTALS,
            time_basis=TimeBasis.OBSERVED,
            in_play=None,
            market_status=None,
            total_matched=None,
            market_start_time=None,
            bookmaker="betfair_exchange",
            outcome=PriceOutcome.OVER,
            line=200.5,
            side=Side.BACK,
            price=2.1,
            size=None,
            matched=None,
            changed_at=None,
        )
        found = [
            replace(
                base,
                snapshot_id=1,
                snapshot_time=anchor - timedelta(hours=2),
                bookmaker="pinnacle",
                price=1.8,
            ),
            replace(
                base,
                snapshot_id=1,
                snapshot_time=anchor - timedelta(hours=2),
                bookmaker="bet365",
                line=201.5,
                price=2.3,
            ),
            replace(base, snapshot_id=3, bookmaker="pinnacle", price=1.9),
            base,
            replace(base, bookmaker="pinnacle", price=2.0),
            replace(base, bookmaker="pinnacle", price=2.2),
        ]
        with patch("odds_mcp.prices.book_quotes", AsyncMock(return_value=found)):
            window = await load_market_window(AsyncMock(), event, Market.TOTALS, anchor, 2)
        latest = window.latest(0.5)
        assert latest[("pinnacle", "Over", 200.5)].price_decimal == 2.0
        assert ("bet365", "Over", 201.5) not in latest
        assert window.latest(2)[("bet365", "Over", 201.5)].price_decimal == 2.3
        sharp = window.sharp(["pinnacle", "betfair_exchange"], 2)
        assert sharp.prices[("Over", 200.5)]["price_decimal"] == 2.0
        assert sharp.meta[("Over", 200.5)].snapshot_id == 2
        assert (
            window.sharp(["betfair_exchange", "pinnacle"], 0.5).prices[("Over", 200.5)][
                "price_decimal"
            ]
            == 2.1
        )
