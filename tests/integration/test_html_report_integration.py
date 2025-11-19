"""Integration tests for HTML report generation."""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from odds_analytics.backtesting import (
    BacktestConfig,
    BacktestEngine,
    BetSizingConfig,
)
from odds_analytics.reporting import HTMLReportGenerator
from odds_analytics.strategies import FlatBettingStrategy
from odds_core.models import Event, EventStatus, Odds
from odds_lambda.storage.readers import OddsReader
from odds_lambda.storage.writers import OddsWriter


@pytest.mark.asyncio
async def test_full_backtest_to_html_workflow(test_session):
    """Test complete workflow: backtest -> JSON -> HTML report."""
    writer = OddsWriter(test_session)
    reader = OddsReader(test_session)

    # Create test events with final results
    base_date = datetime.now(UTC) - timedelta(days=10)

    # Event 1: Lakers win (for flat betting on home team)
    event1 = Event(
        id="html_test_event_1",
        sport_key="basketball_nba",
        sport_title="NBA",
        commence_time=base_date,
        home_team="Lakers",
        away_team="Warriors",
        status=EventStatus.FINAL,
        home_score=110,
        away_score=105,
    )

    # Event 2: Celtics lose
    event2 = Event(
        id="html_test_event_2",
        sport_key="basketball_nba",
        sport_title="NBA",
        commence_time=base_date + timedelta(days=1),
        home_team="Celtics",
        away_team="Heat",
        status=EventStatus.FINAL,
        home_score=98,
        away_score=105,
    )

    # Event 3: Nuggets win
    event3 = Event(
        id="html_test_event_3",
        sport_key="basketball_nba",
        sport_title="NBA",
        commence_time=base_date + timedelta(days=2),
        home_team="Nuggets",
        away_team="Suns",
        status=EventStatus.FINAL,
        home_score=115,
        away_score=110,
    )

    await writer.upsert_event(event1)
    await writer.upsert_event(event2)
    await writer.upsert_event(event3)

    # Create odds for event 1
    decision_time1 = event1.commence_time - timedelta(hours=1)
    odds1 = [
        Odds(
            event_id=event1.id,
            bookmaker_key="fanduel",
            bookmaker_title="FanDuel",
            market_key="h2h",
            outcome_name="Lakers",
            price=-110,
            point=None,
            odds_timestamp=decision_time1,
            last_update=decision_time1,
        ),
        Odds(
            event_id=event1.id,
            bookmaker_key="fanduel",
            bookmaker_title="FanDuel",
            market_key="h2h",
            outcome_name="Warriors",
            price=-110,
            point=None,
            odds_timestamp=decision_time1,
            last_update=decision_time1,
        ),
    ]

    # Create odds for event 2
    decision_time2 = event2.commence_time - timedelta(hours=1)
    odds2 = [
        Odds(
            event_id=event2.id,
            bookmaker_key="draftkings",
            bookmaker_title="DraftKings",
            market_key="h2h",
            outcome_name="Celtics",
            price=-120,
            point=None,
            odds_timestamp=decision_time2,
            last_update=decision_time2,
        ),
        Odds(
            event_id=event2.id,
            bookmaker_key="draftkings",
            bookmaker_title="DraftKings",
            market_key="h2h",
            outcome_name="Heat",
            price=100,
            point=None,
            odds_timestamp=decision_time2,
            last_update=decision_time2,
        ),
    ]

    # Create odds for event 3
    decision_time3 = event3.commence_time - timedelta(hours=1)
    odds3 = [
        Odds(
            event_id=event3.id,
            bookmaker_key="pinnacle",
            bookmaker_title="Pinnacle",
            market_key="h2h",
            outcome_name="Nuggets",
            price=-105,
            point=None,
            odds_timestamp=decision_time3,
            last_update=decision_time3,
        ),
        Odds(
            event_id=event3.id,
            bookmaker_key="pinnacle",
            bookmaker_title="Pinnacle",
            market_key="h2h",
            outcome_name="Suns",
            price=-105,
            point=None,
            odds_timestamp=decision_time3,
            last_update=decision_time3,
        ),
    ]

    # Insert all odds
    for odds_record in odds1 + odds2 + odds3:
        test_session.add(odds_record)
    await test_session.commit()

    # Run backtest
    strategy = FlatBettingStrategy(
        market="h2h", outcome_pattern="home", bookmaker="fanduel"
    )

    sizing_config = BetSizingConfig(method="flat", flat_stake_amount=100.0)

    config = BacktestConfig(
        initial_bankroll=10000.0,
        start_date=base_date - timedelta(days=1),
        end_date=base_date + timedelta(days=3),
        sizing=sizing_config,
        decision_hours_before_game=0.5,  # 30 minutes before game
    )

    engine = BacktestEngine(strategy, config, reader)
    result = await engine.run()

    # Verify backtest ran successfully
    assert result.total_bets > 0
    assert len(result.bets) > 0
    assert len(result.equity_curve) > 0

    # Step 2: Export to JSON
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "backtest_result.json"
        result.to_json(str(json_path))

        # Verify JSON was created
        assert json_path.exists()

        # Verify JSON is valid and can be loaded
        with json_path.open() as f:
            json_data = json.load(f)
            assert "metadata" in json_data
            assert "summary" in json_data
            assert "bets" in json_data
            assert "equity_curve" in json_data

        # Step 3: Generate HTML report from JSON
        html_path = Path(tmpdir) / "backtest_report.html"

        generator = HTMLReportGenerator(result)
        generator.generate(str(html_path))

        # Verify HTML was created
        assert html_path.exists()
        assert html_path.stat().st_size > 0

        # Read and verify HTML content
        html_content = html_path.read_text(encoding="utf-8")

        # Check for essential HTML structure
        assert "<!DOCTYPE html>" in html_content
        assert "<html" in html_content
        assert "</html>" in html_content

        # Check for Bootstrap CSS
        assert "bootstrap" in html_content.lower()

        # Check for Plotly
        assert "plotly" in html_content.lower()

        # Check for strategy information
        assert "FlatBettingStrategy" in html_content or "Flat" in html_content

        # Check for key sections
        assert "Strategy Information" in html_content
        assert "Performance Visualizations" in html_content
        assert "Detailed Breakdowns" in html_content

        # Check for charts (div IDs)
        assert "equity-curve" in html_content
        assert "drawdown" in html_content
        assert "monthly-performance" in html_content

        # Check for tables
        assert "Bet Statistics" in html_content
        assert "Risk" in html_content or "Metrics" in html_content

        # Check for actual data values
        assert str(result.total_bets) in html_content or "Total Bets" in html_content
        # Just verify some bet data is present
        assert "Bet Statistics" in html_content


@pytest.mark.asyncio
async def test_html_report_from_json_reconstruction(test_session):
    """Test HTML report generation from reconstructed BacktestResult."""
    writer = OddsWriter(test_session)
    reader = OddsReader(test_session)

    # Create minimal test data
    base_date = datetime.now(UTC) - timedelta(days=5)

    event = Event(
        id="html_json_test_event",
        sport_key="basketball_nba",
        sport_title="NBA",
        commence_time=base_date,
        home_team="Bucks",
        away_team="76ers",
        status=EventStatus.FINAL,
        home_score=120,
        away_score=115,
    )

    await writer.upsert_event(event)

    decision_time = event.commence_time - timedelta(hours=1)
    odds = [
        Odds(
            event_id=event.id,
            bookmaker_key="betmgm",
            bookmaker_title="BetMGM",
            market_key="h2h",
            outcome_name="Bucks",
            price=-110,
            point=None,
            odds_timestamp=decision_time,
            last_update=decision_time,
        ),
        Odds(
            event_id=event.id,
            bookmaker_key="betmgm",
            bookmaker_title="BetMGM",
            market_key="h2h",
            outcome_name="76ers",
            price=-110,
            point=None,
            odds_timestamp=decision_time,
            last_update=decision_time,
        ),
    ]

    for odds_record in odds:
        test_session.add(odds_record)
    await test_session.commit()

    # Run backtest
    strategy = FlatBettingStrategy(market="h2h", outcome_pattern="home", bookmaker="betmgm")
    sizing_config = BetSizingConfig(method="flat", flat_stake_amount=100.0)
    config = BacktestConfig(
        initial_bankroll=10000.0,
        start_date=base_date - timedelta(days=1),
        end_date=base_date + timedelta(days=1),
        sizing=sizing_config,
    )

    engine = BacktestEngine(strategy, config, reader)
    result = await engine.run()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save to JSON
        json_path = Path(tmpdir) / "result.json"
        result.to_json(str(json_path))

        # Reconstruct from JSON
        from odds_analytics.backtesting.models import BacktestResult

        reconstructed = BacktestResult.from_json(str(json_path))

        # Generate HTML from reconstructed result
        html_path = Path(tmpdir) / "report.html"
        generator = HTMLReportGenerator(reconstructed)
        generator.generate(str(html_path))

        # Verify HTML was created successfully
        assert html_path.exists()

        html_content = html_path.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in html_content
        # Verify key sections are present
        assert "Strategy Information" in html_content
        assert "Bet Statistics" in html_content


@pytest.mark.asyncio
async def test_html_report_file_size(test_session):
    """Test that HTML report file size is reasonable."""
    writer = OddsWriter(test_session)
    reader = OddsReader(test_session)

    # Create test event
    base_date = datetime.now(UTC) - timedelta(days=3)

    event = Event(
        id="html_filesize_test",
        sport_key="basketball_nba",
        sport_title="NBA",
        commence_time=base_date,
        home_team="Knicks",
        away_team="Nets",
        status=EventStatus.FINAL,
        home_score=105,
        away_score=100,
    )

    await writer.upsert_event(event)

    decision_time = event.commence_time - timedelta(hours=1)
    odds = [
        Odds(
            event_id=event.id,
            bookmaker_key="caesars",
            bookmaker_title="Caesars",
            market_key="h2h",
            outcome_name="Knicks",
            price=-110,
            point=None,
            odds_timestamp=decision_time,
            last_update=decision_time,
        ),
        Odds(
            event_id=event.id,
            bookmaker_key="caesars",
            bookmaker_title="Caesars",
            market_key="h2h",
            outcome_name="Nets",
            price=-110,
            point=None,
            odds_timestamp=decision_time,
            last_update=decision_time,
        ),
    ]

    for odds_record in odds:
        test_session.add(odds_record)
    await test_session.commit()

    # Run backtest
    strategy = FlatBettingStrategy(market="h2h", outcome_pattern="home", bookmaker="caesars")
    sizing_config = BetSizingConfig(method="flat", flat_stake_amount=100.0)
    config = BacktestConfig(
        initial_bankroll=10000.0,
        start_date=base_date - timedelta(days=1),
        end_date=base_date + timedelta(days=1),
        sizing=sizing_config,
    )

    engine = BacktestEngine(strategy, config, reader)
    result = await engine.run()

    with tempfile.TemporaryDirectory() as tmpdir:
        html_path = Path(tmpdir) / "report.html"

        generator = HTMLReportGenerator(result)
        generator.generate(str(html_path))

        # Check file size is reasonable (should be < 500KB for single bet)
        file_size = html_path.stat().st_size
        assert file_size > 0
        assert file_size < 500_000, f"HTML file too large: {file_size} bytes"
