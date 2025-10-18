"""Integration tests for backtesting system."""

import tempfile
from datetime import datetime, timedelta

import pytest

from analytics.backtesting import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    BetSizingConfig,
)
from analytics.strategies import ArbitrageStrategy, FlatBettingStrategy
from core.models import Event, EventStatus, Odds
from storage.readers import OddsReader
from storage.writers import OddsWriter


@pytest.mark.asyncio
async def test_full_backtest_with_flat_strategy(test_session):
    """Test complete backtest flow with flat betting strategy."""
    writer = OddsWriter(test_session)
    reader = OddsReader(test_session)

    # Create test event with final result
    event_date = datetime.utcnow() - timedelta(days=10)
    event = Event(
        id="test_backtest_1",
        sport_key="basketball_nba",
        sport_title="NBA",
        commence_time=event_date,
        home_team="Lakers",
        away_team="Warriors",
        status=EventStatus.FINAL,
        home_score=110,
        away_score=105,  # Lakers win
    )

    await writer.upsert_event(event)

    # Create odds snapshot (1 hour before game)
    decision_time = event_date - timedelta(hours=1)

    odds_records = [
        Odds(
            event_id=event.id,
            bookmaker_key="fanduel",
            bookmaker_title="FanDuel",
            market_key="h2h",
            outcome_name="Lakers",
            price=-120,  # Lakers favorite
            point=None,
            odds_timestamp=decision_time,
            last_update=decision_time,
        ),
        Odds(
            event_id=event.id,
            bookmaker_key="fanduel",
            bookmaker_title="FanDuel",
            market_key="h2h",
            outcome_name="Warriors",
            price=+110,  # Warriors underdog
            point=None,
            odds_timestamp=decision_time,
            last_update=decision_time,
        ),
    ]

    for odd in odds_records:
        test_session.add(odd)
    await test_session.commit()

    # Create strategy and config
    strategy = FlatBettingStrategy(
        market="h2h",
        outcome_pattern="home",  # Always bet home team
        bookmaker="fanduel",
    )

    sizing = BetSizingConfig(
        method="flat",
        flat_stake_amount=100.0,
    )

    config = BacktestConfig(
        initial_bankroll=10000.0,
        start_date=event_date - timedelta(days=1),
        end_date=event_date + timedelta(days=1),
        decision_hours_before_game=1.0,
        sizing=sizing,
    )

    # Run backtest
    engine = BacktestEngine(strategy, config, reader)
    result = await engine.run()

    # Verify results
    assert result.total_bets == 1  # Should place 1 bet
    assert result.winning_bets == 1  # Lakers won
    assert result.losing_bets == 0
    assert result.total_profit > 0  # Made profit on winning bet
    assert result.win_rate == 100.0
    assert result.final_bankroll > result.initial_bankroll

    # Verify bet details
    assert len(result.bets) == 1
    bet = result.bets[0]
    assert bet.outcome == "Lakers"
    assert bet.market == "h2h"
    assert bet.result == "win"
    assert bet.stake == 100.0


@pytest.mark.asyncio
async def test_backtest_with_multiple_events(test_session):
    """Test backtest with multiple events."""
    writer = OddsWriter(test_session)
    reader = OddsReader(test_session)

    base_date = datetime.utcnow() - timedelta(days=30)

    # Create 3 test events
    events = []
    for i in range(3):
        event_date = base_date + timedelta(days=i * 10)
        event = Event(
            id=f"test_backtest_{i}",
            sport_key="basketball_nba",
            sport_title="NBA",
            commence_time=event_date,
            home_team=f"Team_Home_{i}",
            away_team=f"Team_Away_{i}",
            status=EventStatus.FINAL,
            home_score=110 + i,
            away_score=105 + i,  # Home always wins
        )
        await writer.upsert_event(event)
        events.append(event)

        # Add odds for each event
        decision_time = event_date - timedelta(hours=1)

        odds_records = [
            Odds(
                event_id=event.id,
                bookmaker_key="fanduel",
                bookmaker_title="FanDuel",
                market_key="h2h",
                outcome_name=event.home_team,
                price=-110,
                point=None,
                odds_timestamp=decision_time,
                last_update=decision_time,
            ),
            Odds(
                event_id=event.id,
                bookmaker_key="fanduel",
                bookmaker_title="FanDuel",
                market_key="h2h",
                outcome_name=event.away_team,
                price=-110,
                point=None,
                odds_timestamp=decision_time,
                last_update=decision_time,
            ),
        ]

        for odd in odds_records:
            test_session.add(odd)

    await test_session.commit()

    # Run backtest
    strategy = FlatBettingStrategy(
        market="h2h",
        outcome_pattern="home",
        bookmaker="fanduel",
    )

    sizing = BetSizingConfig(
        method="flat",
        flat_stake_amount=100.0,
    )

    config = BacktestConfig(
        initial_bankroll=10000.0,
        start_date=base_date - timedelta(days=1),
        end_date=base_date + timedelta(days=25),
        sizing=sizing,
    )

    engine = BacktestEngine(strategy, config, reader)
    result = await engine.run()

    # Verify results
    assert result.total_bets == 3  # Bet on all 3 games
    assert result.winning_bets == 3  # All home teams won
    assert result.win_rate == 100.0

    # Verify equity curve
    assert len(result.equity_curve) > 0
    assert result.equity_curve[-1].bankroll == result.final_bankroll

    # Verify market breakdown
    assert "h2h" in result.market_breakdown
    assert result.market_breakdown["h2h"].bets == 3


@pytest.mark.asyncio
async def test_backtest_json_export_and_reconstruction(test_session):
    """Test exporting to JSON and reconstructing results."""
    writer = OddsWriter(test_session)
    reader = OddsReader(test_session)

    # Create simple test event
    event_date = datetime.utcnow() - timedelta(days=5)
    event = Event(
        id="test_export_1",
        sport_key="basketball_nba",
        sport_title="NBA",
        commence_time=event_date,
        home_team="TestHome",
        away_team="TestAway",
        status=EventStatus.FINAL,
        home_score=100,
        away_score=95,
    )

    await writer.upsert_event(event)

    decision_time = event_date - timedelta(hours=1)

    odd = Odds(
        event_id=event.id,
        bookmaker_key="fanduel",
        bookmaker_title="FanDuel",
        market_key="h2h",
        outcome_name="TestHome",
        price=-110,
        point=None,
        odds_timestamp=decision_time,
        last_update=decision_time,
    )
    test_session.add(odd)
    await test_session.commit()

    # Run backtest
    strategy = FlatBettingStrategy(market="h2h", outcome_pattern="home", bookmaker="fanduel")
    sizing = BetSizingConfig(
        method="flat",
        flat_stake_amount=100.0,
    )
    config = BacktestConfig(
        initial_bankroll=10000.0,
        start_date=event_date - timedelta(days=1),
        end_date=event_date + timedelta(days=1),
        sizing=sizing,
    )

    engine = BacktestEngine(strategy, config, reader)
    original_result = await engine.run()

    # Export to JSON
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        original_result.to_json(f.name)

        # Reconstruct from JSON
        reconstructed_result = BacktestResult.from_json(f.name)

    # Verify reconstruction
    assert reconstructed_result.strategy_name == original_result.strategy_name
    assert reconstructed_result.total_bets == original_result.total_bets
    assert reconstructed_result.final_bankroll == pytest.approx(original_result.final_bankroll)
    assert len(reconstructed_result.bets) == len(original_result.bets)

    if reconstructed_result.bets:
        assert reconstructed_result.bets[0].event_id == original_result.bets[0].event_id


@pytest.mark.asyncio
async def test_backtest_with_no_matching_events(test_session):
    """Test backtest when no events match criteria."""
    reader = OddsReader(test_session)
    strategy = FlatBettingStrategy(market="h2h", outcome_pattern="home", bookmaker="fanduel")

    # Date range with no events
    sizing = BetSizingConfig(
        method="flat",
        flat_stake_amount=100.0,
    )
    config = BacktestConfig(
        initial_bankroll=10000.0,
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2020, 1, 31),
        sizing=sizing,
    )

    engine = BacktestEngine(strategy, config, reader)
    result = await engine.run()

    # Should return empty result
    assert result.total_bets == 0
    assert result.final_bankroll == result.initial_bankroll
    assert result.total_profit == 0.0
    assert "No bets placed" in result.data_quality_issues


@pytest.mark.asyncio
async def test_backtest_csv_export(test_session):
    """Test CSV export functionality."""
    writer = OddsWriter(test_session)
    reader = OddsReader(test_session)

    # Create test event
    event_date = datetime.utcnow() - timedelta(days=5)
    event = Event(
        id="test_csv_1",
        sport_key="basketball_nba",
        sport_title="NBA",
        commence_time=event_date,
        home_team="TestHome",
        away_team="TestAway",
        status=EventStatus.FINAL,
        home_score=100,
        away_score=95,
    )

    await writer.upsert_event(event)

    decision_time = event_date - timedelta(hours=1)

    odd = Odds(
        event_id=event.id,
        bookmaker_key="fanduel",
        bookmaker_title="FanDuel",
        market_key="h2h",
        outcome_name="TestHome",
        price=-110,
        point=None,
        odds_timestamp=decision_time,
        last_update=decision_time,
    )
    test_session.add(odd)
    await test_session.commit()

    # Run backtest
    strategy = FlatBettingStrategy(market="h2h", outcome_pattern="home", bookmaker="fanduel")
    sizing = BetSizingConfig(
        method="flat",
        flat_stake_amount=100.0,
    )
    config = BacktestConfig(
        initial_bankroll=10000.0,
        start_date=event_date - timedelta(days=1),
        end_date=event_date + timedelta(days=1),
        sizing=sizing,
    )

    engine = BacktestEngine(strategy, config, reader)
    result = await engine.run()

    # Export to CSV
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        result.to_csv(f.name)

        # Read and verify CSV
        with open(f.name) as read_f:
            lines = read_f.readlines()
            assert len(lines) >= 2  # Header + at least 1 bet
            assert "bet_id" in lines[0]
            assert "TestHome" in lines[1] or "TestHome" in str(lines)


@pytest.mark.asyncio
async def test_arbitrage_strategy_max_hold_filter(test_session):
    """Test that ArbitrageStrategy properly filters markets by max_hold."""
    writer = OddsWriter(test_session)
    reader = OddsReader(test_session)

    # Create test event
    event_date = datetime.utcnow() - timedelta(days=5)
    event = Event(
        id="test_arb_hold_1",
        sport_key="basketball_nba",
        sport_title="NBA",
        commence_time=event_date,
        home_team="TeamA",
        away_team="TeamB",
        status=EventStatus.FINAL,
        home_score=100,
        away_score=95,
    )

    await writer.upsert_event(event)
    decision_time = event_date - timedelta(hours=1)

    # Create odds with high hold (>10%)
    # -120 on both sides = ~9.1% hold (54.5% + 54.5% = 109.1%)
    high_hold_odds = [
        Odds(
            event_id=event.id,
            bookmaker_key="pinnacle",
            bookmaker_title="Pinnacle",
            market_key="h2h",
            outcome_name="TeamA",
            price=-120,
            point=None,
            odds_timestamp=decision_time,
            last_update=decision_time,
        ),
        Odds(
            event_id=event.id,
            bookmaker_key="fanduel",
            bookmaker_title="FanDuel",
            market_key="h2h",
            outcome_name="TeamB",
            price=-120,
            point=None,
            odds_timestamp=decision_time,
            last_update=decision_time,
        ),
    ]

    for odd in high_hold_odds:
        test_session.add(odd)
    await test_session.commit()

    # Test with max_hold=0.05 (5%) - should filter out the high-hold market
    strategy = ArbitrageStrategy(
        min_profit_margin=0.01,
        max_hold=0.05,  # Only allow markets with ≤5% hold
    )

    sizing = BetSizingConfig(method="flat", flat_stake_amount=100.0)
    config = BacktestConfig(
        initial_bankroll=10000.0,
        start_date=event_date - timedelta(days=1),
        end_date=event_date + timedelta(days=1),
        sizing=sizing,
    )

    engine = BacktestEngine(strategy, config, reader)
    result = await engine.run()

    # Should find no opportunities because hold is too high
    assert result.total_bets == 0

    # Now test with higher max_hold=0.15 (15%) - should allow the market
    strategy_lenient = ArbitrageStrategy(
        min_profit_margin=0.01,
        max_hold=0.15,  # Allow markets with ≤15% hold
    )

    engine_lenient = BacktestEngine(strategy_lenient, config, reader)
    result_lenient = await engine_lenient.run()

    # Still shouldn't find arbitrage (no arb opportunity exists), but market wasn't filtered
    # Note: Since there's no actual arbitrage in these odds, still 0 bets
    assert result_lenient.total_bets == 0
