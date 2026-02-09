"""Unit tests for Polymarket API client and helper functions."""

from odds_core.polymarket_models import PolymarketMarketType
from odds_lambda.polymarket_fetcher import classify_market, process_order_book


class TestClassifyMarket:
    """Tests for classify_market function."""

    def test_moneyline_exact_match(self):
        """Test moneyline classification with exact question-title match."""
        market_type, point = classify_market(
            question="Lakers vs Celtics", event_title="Lakers vs Celtics"
        )

        assert market_type == PolymarketMarketType.MONEYLINE
        assert point is None

    def test_moneyline_with_different_casing(self):
        """Test moneyline only matches on exact string, not case-insensitive."""
        market_type, point = classify_market(
            question="lakers vs celtics", event_title="Lakers vs Celtics"
        )

        # Should NOT match as moneyline (case sensitive)
        assert market_type != PolymarketMarketType.MONEYLINE

    def test_spread_positive(self):
        """Test spread classification with positive point value."""
        market_type, point = classify_market(
            question="Lakers Spread: +6.5 (Over)", event_title="Lakers vs Celtics"
        )

        assert market_type == PolymarketMarketType.SPREAD
        assert point == 6.5

    def test_spread_negative(self):
        """Test spread classification with negative point value."""
        market_type, point = classify_market(
            question="Celtics Spread: -6.5 (Under)", event_title="Lakers vs Celtics"
        )

        assert market_type == PolymarketMarketType.SPREAD
        assert point == -6.5

    def test_spread_integer(self):
        """Test spread classification with integer point value."""
        market_type, point = classify_market(
            question="Mavericks Spread: -7 (Under)", event_title="Mavericks vs Bucks"
        )

        assert market_type == PolymarketMarketType.SPREAD
        assert point == -7.0

    def test_total_over_under(self):
        """Test total classification with O/U pattern."""
        market_type, point = classify_market(question="O/U 215.5", event_title="Lakers vs Celtics")

        assert market_type == PolymarketMarketType.TOTAL
        assert point == 215.5

    def test_total_lowercase(self):
        """Test total classification with lowercase o/u."""
        market_type, point = classify_market(question="o/u 220.5", event_title="Lakers vs Celtics")

        assert market_type == PolymarketMarketType.TOTAL
        assert point == 220.5

    def test_total_integer(self):
        """Test total classification with integer value."""
        market_type, point = classify_market(question="O/U 220", event_title="Lakers vs Celtics")

        assert market_type == PolymarketMarketType.TOTAL
        assert point == 220.0

    def test_player_prop_points(self):
        """Test player prop classification with points stat."""
        market_type, point = classify_market(
            question="LeBron James: Points Over 25.5", event_title="Lakers vs Celtics"
        )

        assert market_type == PolymarketMarketType.PLAYER_PROP
        assert point is None

    def test_player_prop_rebounds(self):
        """Test player prop classification with rebounds stat."""
        market_type, point = classify_market(
            question="Anthony Davis: Rebounds Over 10.5", event_title="Lakers vs Celtics"
        )

        assert market_type == PolymarketMarketType.PLAYER_PROP
        assert point is None

    def test_player_prop_assists(self):
        """Test player prop classification with assists stat."""
        market_type, point = classify_market(
            question="Chris Paul: Assists Over 8.5", event_title="Suns vs Warriors"
        )

        assert market_type == PolymarketMarketType.PLAYER_PROP
        assert point is None

    def test_player_prop_steals(self):
        """Test player prop classification with steals stat."""
        market_type, point = classify_market(
            question="Gary Payton II: Steals Over 2.5", event_title="Warriors vs Lakers"
        )

        assert market_type == PolymarketMarketType.PLAYER_PROP
        assert point is None

    def test_player_prop_blocks(self):
        """Test player prop classification with blocks stat."""
        market_type, point = classify_market(
            question="Rudy Gobert: Blocks Over 2.5", event_title="Jazz vs Lakers"
        )

        assert market_type == PolymarketMarketType.PLAYER_PROP
        assert point is None

    def test_player_prop_threes(self):
        """Test player prop classification with threes stat."""
        market_type, point = classify_market(
            question="Stephen Curry: Threes Made Over 4.5", event_title="Warriors vs Celtics"
        )

        assert market_type == PolymarketMarketType.PLAYER_PROP
        assert point is None

    def test_player_prop_abbreviation_pts(self):
        """Test player prop with abbreviated stat (pts)."""
        market_type, point = classify_market(
            question="Luka Doncic: Pts Over 28.5", event_title="Mavericks vs Lakers"
        )

        assert market_type == PolymarketMarketType.PLAYER_PROP
        assert point is None

    def test_player_prop_abbreviation_reb(self):
        """Test player prop with abbreviated stat (reb)."""
        market_type, point = classify_market(
            question="Giannis: Reb Over 12.5", event_title="Bucks vs Celtics"
        )

        assert market_type == PolymarketMarketType.PLAYER_PROP
        assert point is None

    def test_player_prop_abbreviation_ast(self):
        """Test player prop with abbreviated stat (ast)."""
        market_type, point = classify_market(
            question="Trae Young: Ast Over 9.5", event_title="Hawks vs Knicks"
        )

        assert market_type == PolymarketMarketType.PLAYER_PROP
        assert point is None

    def test_other_unrecognized_pattern(self):
        """Test fallback to OTHER for unrecognized patterns."""
        market_type, point = classify_market(
            question="Will overtime be required?", event_title="Lakers vs Celtics"
        )

        assert market_type == PolymarketMarketType.OTHER
        assert point is None

    def test_other_colon_without_stat_keyword(self):
        """Test OTHER classification when colon exists but no stat keyword."""
        market_type, point = classify_market(
            question="Quarter Winner: Lakers", event_title="Lakers vs Celtics"
        )

        assert market_type == PolymarketMarketType.OTHER
        assert point is None

    def test_other_empty_question(self):
        """Test OTHER classification with empty question."""
        market_type, point = classify_market(question="", event_title="Lakers vs Celtics")

        assert market_type == PolymarketMarketType.OTHER
        assert point is None

    def test_other_false_positive_quarter_points(self):
        """Test that 'First Quarter: Points Leader' is not classified as player prop."""
        market_type, point = classify_market(
            question="First Quarter: Points Leader", event_title="Lakers vs Celtics"
        )

        # Should be OTHER because no Over/Under keyword
        assert market_type == PolymarketMarketType.OTHER
        assert point is None

    def test_other_colon_with_stat_no_over_under(self):
        """Test that stat keyword after colon but no Over/Under is OTHER."""
        market_type, point = classify_market(
            question="Team with Most: Points", event_title="Lakers vs Celtics"
        )

        # Should be OTHER because no Over/Under keyword
        assert market_type == PolymarketMarketType.OTHER
        assert point is None


class TestProcessOrderBook:
    """Tests for process_order_book function."""

    def test_normal_order_book(self):
        """Test processing a normal order book with multiple levels."""
        raw_book = {
            "bids": [
                {"price": "0.60", "size": "100"},
                {"price": "0.59", "size": "50"},
                {"price": "0.58", "size": "25"},
            ],
            "asks": [
                {"price": "0.62", "size": "80"},
                {"price": "0.63", "size": "40"},
                {"price": "0.64", "size": "20"},
            ],
        }

        result = process_order_book(raw_book)

        assert result is not None
        assert result["best_bid"] == 0.60
        assert result["best_ask"] == 0.62
        assert abs(result["spread"] - 0.02) < 0.001
        assert abs(result["midpoint"] - 0.61) < 0.001
        assert result["bid_levels"] == 3
        assert result["ask_levels"] == 3
        assert result["bid_depth_total"] == 175.0
        assert result["ask_depth_total"] == 140.0

        # Check imbalance calculation: (175 - 140) / (175 + 140) = 35 / 315 ≈ 0.111
        assert abs(result["imbalance"] - 0.111) < 0.001

        # Check weighted midpoint: (0.60 * 80 + 0.62 * 100) / (100 + 80) = (48 + 62) / 180 = 0.611
        assert abs(result["weighted_mid"] - 0.6111) < 0.001

    def test_unsorted_order_book(self):
        """Test that unsorted order books are sorted correctly."""
        raw_book = {
            "bids": [
                {"price": "0.58", "size": "25"},
                {"price": "0.60", "size": "100"},
                {"price": "0.59", "size": "50"},
            ],
            "asks": [
                {"price": "0.64", "size": "20"},
                {"price": "0.62", "size": "80"},
                {"price": "0.63", "size": "40"},
            ],
        }

        result = process_order_book(raw_book)

        assert result is not None
        # Best bid should be highest price (0.60)
        assert result["best_bid"] == 0.60
        # Best ask should be lowest price (0.62)
        assert result["best_ask"] == 0.62

    def test_single_level_order_book(self):
        """Test processing order book with single level on each side."""
        raw_book = {
            "bids": [{"price": "0.55", "size": "200"}],
            "asks": [{"price": "0.57", "size": "150"}],
        }

        result = process_order_book(raw_book)

        assert result is not None
        assert result["best_bid"] == 0.55
        assert result["best_ask"] == 0.57
        assert abs(result["spread"] - 0.02) < 0.001
        assert abs(result["midpoint"] - 0.56) < 0.001
        assert result["bid_levels"] == 1
        assert result["ask_levels"] == 1
        assert result["bid_depth_total"] == 200.0
        assert result["ask_depth_total"] == 150.0

    def test_empty_bids(self):
        """Test that empty bids returns None."""
        raw_book = {
            "bids": [],
            "asks": [{"price": "0.62", "size": "80"}],
        }

        result = process_order_book(raw_book)

        assert result is None

    def test_empty_asks(self):
        """Test that empty asks returns None."""
        raw_book = {
            "bids": [{"price": "0.60", "size": "100"}],
            "asks": [],
        }

        result = process_order_book(raw_book)

        assert result is None

    def test_empty_order_book(self):
        """Test that empty order book returns None."""
        raw_book = {"bids": [], "asks": []}

        result = process_order_book(raw_book)

        assert result is None

    def test_missing_bids_key(self):
        """Test handling of missing bids key."""
        raw_book = {"asks": [{"price": "0.62", "size": "80"}]}

        result = process_order_book(raw_book)

        assert result is None

    def test_missing_asks_key(self):
        """Test handling of missing asks key."""
        raw_book = {"bids": [{"price": "0.60", "size": "100"}]}

        result = process_order_book(raw_book)

        assert result is None

    def test_imbalance_bid_heavy(self):
        """Test imbalance calculation when bids dominate."""
        raw_book = {
            "bids": [{"price": "0.60", "size": "1000"}],
            "asks": [{"price": "0.62", "size": "100"}],
        }

        result = process_order_book(raw_book)

        assert result is not None
        # Imbalance = (1000 - 100) / (1000 + 100) = 900 / 1100 ≈ 0.818
        assert abs(result["imbalance"] - 0.818) < 0.001

    def test_imbalance_ask_heavy(self):
        """Test imbalance calculation when asks dominate."""
        raw_book = {
            "bids": [{"price": "0.60", "size": "100"}],
            "asks": [{"price": "0.62", "size": "1000"}],
        }

        result = process_order_book(raw_book)

        assert result is not None
        # Imbalance = (100 - 1000) / (100 + 1000) = -900 / 1100 ≈ -0.818
        assert abs(result["imbalance"] - (-0.818)) < 0.001

    def test_imbalance_balanced(self):
        """Test imbalance calculation when perfectly balanced."""
        raw_book = {
            "bids": [{"price": "0.60", "size": "500"}],
            "asks": [{"price": "0.62", "size": "500"}],
        }

        result = process_order_book(raw_book)

        assert result is not None
        # Imbalance = (500 - 500) / (500 + 500) = 0
        assert result["imbalance"] == 0.0

    def test_float_precision(self):
        """Test that string prices are converted to floats correctly."""
        raw_book = {
            "bids": [{"price": "0.123456", "size": "100.5"}],
            "asks": [{"price": "0.234567", "size": "200.75"}],
        }

        result = process_order_book(raw_book)

        assert result is not None
        assert result["best_bid"] == 0.123456
        assert result["best_ask"] == 0.234567
        assert result["bid_depth_total"] == 100.5
        assert result["ask_depth_total"] == 200.75

    def test_crossed_book_bid_equals_ask(self):
        """Test that crossed book (bid == ask) returns None."""
        raw_book = {
            "bids": [{"price": "0.60", "size": "100"}],
            "asks": [{"price": "0.60", "size": "80"}],
        }

        result = process_order_book(raw_book)

        assert result is None

    def test_crossed_book_bid_greater_than_ask(self):
        """Test that crossed book (bid > ask) returns None."""
        raw_book = {
            "bids": [{"price": "0.62", "size": "100"}],
            "asks": [{"price": "0.60", "size": "80"}],
        }

        result = process_order_book(raw_book)

        assert result is None
