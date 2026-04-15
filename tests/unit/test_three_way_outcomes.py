"""Tests for 3-way outcome support in backtesting (football draw handling)."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
import structlog
from odds_analytics.backtesting.config import BacktestConfig, BetConstraintsConfig, BetSizingConfig
from odds_analytics.backtesting.models import BacktestEvent, BetOpportunity
from odds_analytics.backtesting.services import BacktestEngine, BettingStrategy
from odds_analytics.strategies import ArbitrageStrategy, BasicEVStrategy, FlatBettingStrategy
from odds_core.models import EventStatus, Odds

# -- Fixtures --


@pytest.fixture
def football_draw_event() -> BacktestEvent:
    return BacktestEvent(
        id="football_draw_1",
        commence_time=datetime(2025, 1, 15, 15, 0, tzinfo=UTC),
        home_team="Liverpool",
        away_team="Manchester City",
        home_score=1,
        away_score=1,
        status=EventStatus.FINAL,
    )


@pytest.fixture
def football_home_win_event() -> BacktestEvent:
    return BacktestEvent(
        id="football_home_1",
        commence_time=datetime(2025, 1, 15, 15, 0, tzinfo=UTC),
        home_team="Liverpool",
        away_team="Manchester City",
        home_score=2,
        away_score=1,
        status=EventStatus.FINAL,
    )


@pytest.fixture
def nba_tie_event() -> BacktestEvent:
    """NBA-style event with tied score (2-way market, should push)."""
    return BacktestEvent(
        id="nba_tie_1",
        commence_time=datetime(2025, 1, 15, 19, 0, tzinfo=UTC),
        home_team="Lakers",
        away_team="Warriors",
        home_score=105,
        away_score=105,
        status=EventStatus.FINAL,
    )


@pytest.fixture
def backtest_config() -> BacktestConfig:
    return BacktestConfig(
        initial_bankroll=10000.0,
        start_date=datetime(2025, 1, 1, tzinfo=UTC),
        end_date=datetime(2025, 1, 31, tzinfo=UTC),
        decision_hours_before_game=1.0,
        sizing=BetSizingConfig(method="flat", flat_stake_amount=100.0),
        constraints=BetConstraintsConfig(min_bet_size=10.0, max_bet_size=500.0),
    )


@pytest.fixture
def engine(backtest_config) -> BacktestEngine:
    class StubStrategy(BettingStrategy):
        def __init__(self):
            super().__init__("Stub")

        async def evaluate_opportunity(self, event, odds_snapshot, config, session=None):
            return []

    return BacktestEngine(
        strategy=StubStrategy(),
        config=backtest_config,
        reader=AsyncMock(),
        logger_instance=structlog.get_logger(),
    )


def _make_odds(
    event_id: str,
    bookmaker: str,
    market: str,
    outcome: str,
    price: int,
    point: float | None = None,
) -> Odds:
    return Odds(
        event_id=event_id,
        bookmaker_key=bookmaker,
        bookmaker_title=bookmaker.title(),
        market_key=market,
        outcome_name=outcome,
        price=price,
        point=point,
        odds_timestamp=datetime(2025, 1, 15, 14, 0, tzinfo=UTC),
        last_update=datetime(2025, 1, 15, 14, 0, tzinfo=UTC),
    )


# -- Moneyline outcome evaluation --


class TestThreeWayMoneylineOutcome:
    def test_draw_bet_on_draw_result_wins(self, engine, football_draw_event):
        result = engine._evaluate_moneyline_outcome("Draw", football_draw_event, is_three_way=True)
        assert result is True

    def test_home_bet_on_draw_result_loses(self, engine, football_draw_event):
        result = engine._evaluate_moneyline_outcome(
            "Liverpool", football_draw_event, is_three_way=True
        )
        assert result is False

    def test_away_bet_on_draw_result_loses(self, engine, football_draw_event):
        result = engine._evaluate_moneyline_outcome(
            "Manchester City", football_draw_event, is_three_way=True
        )
        assert result is False

    def test_draw_bet_on_home_win_loses(self, engine, football_home_win_event):
        result = engine._evaluate_moneyline_outcome(
            "Draw", football_home_win_event, is_three_way=True
        )
        assert result is False

    def test_home_bet_on_home_win_wins(self, engine, football_home_win_event):
        result = engine._evaluate_moneyline_outcome(
            "Liverpool", football_home_win_event, is_three_way=True
        )
        assert result is True

    def test_away_bet_on_home_win_loses(self, engine, football_home_win_event):
        result = engine._evaluate_moneyline_outcome(
            "Manchester City", football_home_win_event, is_three_way=True
        )
        assert result is False

    def test_draw_outcome_case_insensitive(self, engine, football_draw_event):
        for variant in ["Draw", "draw", "DRAW"]:
            result = engine._evaluate_moneyline_outcome(
                variant, football_draw_event, is_three_way=True
            )
            assert result is True, f"Failed for outcome={variant!r}"


class TestTwoWayBackwardCompat:
    def test_tie_returns_push_when_two_way(self, engine, nba_tie_event):
        result = engine._evaluate_moneyline_outcome("Lakers", nba_tie_event)
        assert result is None

    def test_tie_returns_push_explicit_false(self, engine, nba_tie_event):
        result = engine._evaluate_moneyline_outcome("Lakers", nba_tie_event, is_three_way=False)
        assert result is None


# -- Bet result evaluation with is_three_way --


class TestBetResultThreeWay:
    def test_draw_bet_wins_in_three_way(self, engine, football_draw_event):
        opp = BetOpportunity(
            event_id="football_draw_1",
            market="1x2",
            outcome="Draw",
            bookmaker="bet365",
            odds=250,
            line=None,
            confidence=0.30,
            rationale="Test",
        )
        result, profit = engine._evaluate_bet_result_for_opportunity(
            opp, football_draw_event, 100.0, is_three_way=True
        )
        assert result == "win"
        assert profit > 0

    def test_team_bet_loses_on_draw_in_three_way(self, engine, football_draw_event):
        opp = BetOpportunity(
            event_id="football_draw_1",
            market="1x2",
            outcome="Liverpool",
            bookmaker="bet365",
            odds=-120,
            line=None,
            confidence=0.50,
            rationale="Test",
        )
        result, profit = engine._evaluate_bet_result_for_opportunity(
            opp, football_draw_event, 100.0, is_three_way=True
        )
        assert result == "loss"
        assert profit == pytest.approx(-100.0)

    def test_two_way_tie_still_pushes(self, engine, nba_tie_event):
        opp = BetOpportunity(
            event_id="nba_tie_1",
            market="h2h",
            outcome="Lakers",
            bookmaker="fanduel",
            odds=-110,
            line=None,
            confidence=0.55,
            rationale="Test",
        )
        result, profit = engine._evaluate_bet_result_for_opportunity(
            opp, nba_tie_event, 100.0, is_three_way=False
        )
        assert result == "push"
        assert profit == 0.0

    def test_non_h2h_market_ignores_three_way_flag(self, engine, football_draw_event):
        """Spread/total evaluation should be unaffected by is_three_way."""
        event = BacktestEvent(
            id="football_draw_1",
            commence_time=datetime(2025, 1, 15, 15, 0, tzinfo=UTC),
            home_team="Liverpool",
            away_team="Manchester City",
            home_score=1,
            away_score=1,
            status=EventStatus.FINAL,
        )
        opp = BetOpportunity(
            event_id="football_draw_1",
            market="totals",
            outcome="Over",
            bookmaker="bet365",
            odds=-110,
            line=1.5,
            confidence=0.50,
            rationale="Test",
        )
        result, profit = engine._evaluate_bet_result_for_opportunity(
            opp, event, 100.0, is_three_way=True
        )
        assert result == "win"
        assert profit > 0


# -- FlatBettingStrategy draw pattern --


class TestFlatBettingDrawPattern:
    @pytest.fixture
    def three_way_odds(self) -> list[Odds]:
        eid = "football_draw_1"
        return [
            _make_odds(eid, "bet365", "1x2", "Liverpool", -150),
            _make_odds(eid, "bet365", "1x2", "Manchester City", 200),
            _make_odds(eid, "bet365", "1x2", "Draw", 250),
        ]

    @pytest.mark.asyncio
    async def test_draw_pattern_selects_draw_odds(
        self, football_draw_event, three_way_odds, backtest_config
    ):
        strategy = FlatBettingStrategy(market="1x2", outcome_pattern="draw", bookmaker="bet365")
        opps = await strategy.evaluate_opportunity(
            football_draw_event, three_way_odds, backtest_config
        )
        assert len(opps) == 1
        assert opps[0].outcome == "Draw"
        assert opps[0].odds == 250

    @pytest.mark.asyncio
    async def test_draw_pattern_no_draw_odds_returns_empty(
        self, football_draw_event, backtest_config
    ):
        two_way_odds = [
            _make_odds("football_draw_1", "bet365", "1x2", "Liverpool", -150),
            _make_odds("football_draw_1", "bet365", "1x2", "Manchester City", 200),
        ]
        strategy = FlatBettingStrategy(market="1x2", outcome_pattern="draw", bookmaker="bet365")
        opps = await strategy.evaluate_opportunity(
            football_draw_event, two_way_odds, backtest_config
        )
        assert len(opps) == 0


# -- BasicEVStrategy with 3-way odds --


class TestBasicEVThreeWay:
    @pytest.fixture
    def three_way_sharp_and_retail(self) -> list[Odds]:
        eid = "football_home_1"
        return [
            # Sharp book (pinnacle)
            _make_odds(eid, "pinnacle", "1x2", "Liverpool", -130),
            _make_odds(eid, "pinnacle", "1x2", "Manchester City", 200),
            _make_odds(eid, "pinnacle", "1x2", "Draw", 260),
            # Retail book (bet365) with inflated draw odds
            _make_odds(eid, "bet365", "1x2", "Liverpool", -125),
            _make_odds(eid, "bet365", "1x2", "Manchester City", 210),
            _make_odds(eid, "bet365", "1x2", "Draw", 350),
        ]

    @pytest.mark.asyncio
    async def test_finds_ev_on_draw_outcome(
        self, football_home_win_event, three_way_sharp_and_retail, backtest_config
    ):
        strategy = BasicEVStrategy(
            sharp_book="pinnacle",
            retail_books=["bet365"],
            min_ev_threshold=0.01,
            markets=["1x2"],
        )
        opps = await strategy.evaluate_opportunity(
            football_home_win_event, three_way_sharp_and_retail, backtest_config
        )
        draw_opps = [o for o in opps if o.outcome == "Draw"]
        assert len(draw_opps) == 1
        assert draw_opps[0].bookmaker == "bet365"
        assert draw_opps[0].odds == 350


# -- ArbitrageStrategy 3-way --


class TestArbitrageThreeWay:
    @pytest.fixture
    def three_way_arb_odds(self) -> list[Odds]:
        """Odds that create a 3-way arb opportunity."""
        eid = "football_home_1"
        return [
            # Best home odds at bookA
            _make_odds(eid, "bookA", "1x2", "Liverpool", 300),
            _make_odds(eid, "bookB", "1x2", "Liverpool", 250),
            # Best away odds at bookB
            _make_odds(eid, "bookA", "1x2", "Manchester City", 350),
            _make_odds(eid, "bookB", "1x2", "Manchester City", 400),
            # Best draw odds at bookA
            _make_odds(eid, "bookA", "1x2", "Draw", 500),
            _make_odds(eid, "bookB", "1x2", "Draw", 350),
        ]

    @pytest.fixture
    def two_way_arb_odds(self) -> list[Odds]:
        """2-way odds with an arb opportunity (no draw)."""
        eid = "nba_tie_1"
        return [
            _make_odds(eid, "bookA", "h2h", "Lakers", 150),
            _make_odds(eid, "bookB", "h2h", "Lakers", 100),
            _make_odds(eid, "bookA", "h2h", "Warriors", 100),
            _make_odds(eid, "bookB", "h2h", "Warriors", 150),
        ]

    @pytest.mark.asyncio
    async def test_three_way_arb_returns_three_legs(
        self, football_home_win_event, three_way_arb_odds, backtest_config
    ):
        strategy = ArbitrageStrategy(
            min_profit_margin=0.0,
            max_hold=1.0,
            bookmakers=["bookA", "bookB"],
        )
        opps = await strategy.evaluate_opportunity(
            football_home_win_event, three_way_arb_odds, backtest_config
        )
        h2h_opps = [o for o in opps if o.market == "1x2"]
        outcomes = {o.outcome for o in h2h_opps}
        assert outcomes == {"Liverpool", "Manchester City", "Draw"}
        assert all("3-way arb" in o.rationale for o in h2h_opps)

    @pytest.mark.asyncio
    async def test_two_way_arb_returns_two_legs(
        self, nba_tie_event, two_way_arb_odds, backtest_config
    ):
        strategy = ArbitrageStrategy(
            min_profit_margin=0.0,
            max_hold=1.0,
            bookmakers=["bookA", "bookB"],
        )
        opps = await strategy.evaluate_opportunity(nba_tie_event, two_way_arb_odds, backtest_config)
        h2h_opps = [o for o in opps if o.market == "h2h"]
        outcomes = {o.outcome for o in h2h_opps}
        assert outcomes == {"Lakers", "Warriors"}
        assert all("3-way" not in o.rationale for o in h2h_opps)

    @pytest.mark.asyncio
    async def test_three_way_arb_selects_best_odds_per_outcome(
        self, football_home_win_event, three_way_arb_odds, backtest_config
    ):
        strategy = ArbitrageStrategy(
            min_profit_margin=0.0,
            max_hold=1.0,
            bookmakers=["bookA", "bookB"],
        )
        opps = await strategy.evaluate_opportunity(
            football_home_win_event, three_way_arb_odds, backtest_config
        )
        h2h_opps = {o.outcome: o for o in opps if o.market == "1x2"}
        assert h2h_opps["Liverpool"].odds == 300  # bookA best
        assert h2h_opps["Liverpool"].bookmaker == "bookA"
        assert h2h_opps["Manchester City"].odds == 400  # bookB best
        assert h2h_opps["Manchester City"].bookmaker == "bookB"
        assert h2h_opps["Draw"].odds == 500  # bookA best
        assert h2h_opps["Draw"].bookmaker == "bookA"
