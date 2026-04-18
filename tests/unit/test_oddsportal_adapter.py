"""Tests for OddsHarvester upcoming → pipeline conversion."""

from __future__ import annotations

import pytest
from odds_lambda.oddsportal_adapter import (
    MatchOdds,
    _convert_1x2_match,
    _convert_home_away_match,
    _convert_over_under_match,
    convert_upcoming_matches,
    fractional_to_decimal,
    parse_betfair_odds,
)
from odds_lambda.oddsportal_common import (
    DRAW_OUTCOME,
    decimal_to_american,
    normalize_bookmaker_key,
    parse_over_under_line,
)


class TestFractionalToDecimal:
    def test_standard_fraction(self) -> None:
        assert fractional_to_decimal("9/10") == pytest.approx(1.9)

    def test_even_money(self) -> None:
        assert fractional_to_decimal("1/1") == pytest.approx(2.0)

    def test_evs(self) -> None:
        assert fractional_to_decimal("EVS") == pytest.approx(2.0)

    def test_evens(self) -> None:
        assert fractional_to_decimal("EVENS") == pytest.approx(2.0)

    def test_odds_on(self) -> None:
        # 1/2 = 0.5 + 1 = 1.5
        assert fractional_to_decimal("1/2") == pytest.approx(1.5)

    def test_long_odds(self) -> None:
        # 10/1 = 10 + 1 = 11.0
        assert fractional_to_decimal("10/1") == pytest.approx(11.0)

    def test_fractional_like_233_100(self) -> None:
        # 233/100 = 2.33 + 1 = 3.33
        assert fractional_to_decimal("233/100") == pytest.approx(3.33)

    def test_whole_number(self) -> None:
        # "3" is shorthand for 3/1 → 3 + 1 = 4.0
        assert fractional_to_decimal("3") == pytest.approx(4.0)

    def test_whitespace_stripped(self) -> None:
        assert fractional_to_decimal("  9/10  ") == pytest.approx(1.9)


class TestParseBetfairOdds:
    def test_betfair_format_with_liquidity(self) -> None:
        frac, liq = parse_betfair_odds("99/10099/100(300)")
        assert frac == "99/100"
        assert liq == 300

    def test_betfair_simple_fraction_repeated(self) -> None:
        frac, liq = parse_betfair_odds("7/27/2(227)")
        assert frac == "7/2"
        assert liq == 227

    def test_normal_fraction_passthrough(self) -> None:
        frac, liq = parse_betfair_odds("9/10")
        assert frac == "9/10"
        assert liq is None

    def test_betfair_13_5_format(self) -> None:
        frac, liq = parse_betfair_odds("13/513/5(277)")
        assert frac == "13/5"
        assert liq == 277

    def test_whitespace_stripped(self) -> None:
        frac, liq = parse_betfair_odds("  7/27/2(100)  ")
        assert frac == "7/2"
        assert liq == 100


class TestNormalizeBookmakerKey:
    def test_known_bookmaker(self) -> None:
        assert normalize_bookmaker_key("bet365") == "bet365"

    def test_known_bookmaker_mixed_case(self) -> None:
        assert normalize_bookmaker_key("Betway") == "betway"

    def test_betfair_exchange(self) -> None:
        assert normalize_bookmaker_key("Betfair Exchange") == "betfair_exchange"

    def test_unknown_slugified(self) -> None:
        key = normalize_bookmaker_key("Some New Book")
        assert key == "somenewbook"


class TestDecimalToAmerican:
    def test_plus_odds(self) -> None:
        assert decimal_to_american(3.0) == 200

    def test_minus_odds(self) -> None:
        assert decimal_to_american(1.5) == -200

    def test_even_money(self) -> None:
        assert decimal_to_american(2.0) == 100

    def test_edge_case_1_0(self) -> None:
        assert decimal_to_american(1.0) == -10000


class TestConvert1x2Match:
    @pytest.fixture()
    def sample_1x2_bookmakers(self) -> list[dict]:
        return [
            {
                "1": "9/10",
                "X": "12/5",
                "2": "3/1",
                "bookmaker_name": "10bet",
                "period": "FullTime",
            },
            {
                "1": "91/100",
                "X": "12/5",
                "2": "16/5",
                "bookmaker_name": "bet365",
                "period": "FullTime",
            },
            {
                "1": "99/10099/100(300)",
                "X": "13/513/5(277)",
                "2": "7/27/2(227)",
                "bookmaker_name": "Betfair Exchange",
                "period": "FullTime",
            },
        ]

    def test_basic_conversion(self, sample_1x2_bookmakers: list[dict]) -> None:
        result = _convert_1x2_match(sample_1x2_bookmakers, "Leeds", "Sunderland")
        assert result is not None
        assert "bookmakers" in result
        assert len(result["bookmakers"]) == 3

    def test_outcome_names(self, sample_1x2_bookmakers: list[dict]) -> None:
        result = _convert_1x2_match(sample_1x2_bookmakers, "Leeds", "Sunderland")
        assert result is not None
        bk = result["bookmakers"][0]
        outcomes = bk["markets"][0]["outcomes"]
        assert len(outcomes) == 3
        assert outcomes[0]["name"] == "Leeds"
        assert outcomes[1]["name"] == DRAW_OUTCOME
        assert outcomes[2]["name"] == "Sunderland"

    def test_odds_conversion(self, sample_1x2_bookmakers: list[dict]) -> None:
        result = _convert_1x2_match(sample_1x2_bookmakers, "Leeds", "Sunderland")
        assert result is not None
        # 10bet: home 9/10 = 1.9 decimal → -111 American
        bk_10bet = result["bookmakers"][0]
        home_price = bk_10bet["markets"][0]["outcomes"][0]["price"]
        assert home_price == decimal_to_american(1.9)

    def test_betfair_parsed(self, sample_1x2_bookmakers: list[dict]) -> None:
        result = _convert_1x2_match(sample_1x2_bookmakers, "Leeds", "Sunderland")
        assert result is not None
        betfair = next(b for b in result["bookmakers"] if b["key"] == "betfair_exchange")
        # 99/100 = 1.99 → -102
        home_price = betfair["markets"][0]["outcomes"][0]["price"]
        assert home_price == decimal_to_american(1.99)

    def test_betfair_liquidity_stored(self, sample_1x2_bookmakers: list[dict]) -> None:
        result = _convert_1x2_match(sample_1x2_bookmakers, "Leeds", "Sunderland")
        assert result is not None
        betfair = next(b for b in result["bookmakers"] if b["key"] == "betfair_exchange")
        assert "betfair_matched" in betfair
        assert betfair["betfair_matched"]["home"] == 300
        assert betfair["betfair_matched"]["draw"] == 277
        assert betfair["betfair_matched"]["away"] == 227

    def test_non_betfair_no_liquidity(self, sample_1x2_bookmakers: list[dict]) -> None:
        result = _convert_1x2_match(sample_1x2_bookmakers, "Leeds", "Sunderland")
        assert result is not None
        bet365 = next(b for b in result["bookmakers"] if b["key"] == "bet365")
        assert "betfair_matched" not in bet365

    def test_market_key_is_1x2(self, sample_1x2_bookmakers: list[dict]) -> None:
        result = _convert_1x2_match(sample_1x2_bookmakers, "Leeds", "Sunderland")
        assert result is not None
        assert result["bookmakers"][0]["markets"][0]["key"] == "1x2"

    def test_source_tag(self, sample_1x2_bookmakers: list[dict]) -> None:
        result = _convert_1x2_match(sample_1x2_bookmakers, "Leeds", "Sunderland")
        assert result is not None
        assert result["source"] == "oddsportal_live"

    def test_empty_bookmakers_returns_none(self) -> None:
        assert _convert_1x2_match([], "A", "B") is None

    def test_missing_odds_skipped(self) -> None:
        data = [{"bookmaker_name": "bet365", "1": "9/10", "X": "", "2": "3/1"}]
        assert _convert_1x2_match(data, "A", "B") is None


class TestConvertOverUnderMatch:
    @pytest.fixture()
    def sample_ou_bookmakers(self) -> list[dict]:
        return [
            {
                "odds_over": "11/10",
                "odds_under": "73/100",
                "bookmaker_name": "bet365",
                "period": "FullTime",
            },
            {
                "odds_over": "59/5059/50(159)",
                "odds_under": "81/10081/100(134)",
                "bookmaker_name": "Betfair Exchange",
                "period": "FullTime",
            },
        ]

    def test_basic_conversion(self, sample_ou_bookmakers: list[dict]) -> None:
        result = _convert_over_under_match(sample_ou_bookmakers, "Leeds", "Sunderland")
        assert result is not None
        assert len(result["bookmakers"]) == 2

    def test_outcome_names_and_point(self, sample_ou_bookmakers: list[dict]) -> None:
        result = _convert_over_under_match(sample_ou_bookmakers, "Leeds", "Sunderland")
        assert result is not None
        outcomes = result["bookmakers"][0]["markets"][0]["outcomes"]
        assert outcomes[0]["name"] == "Over"
        assert outcomes[0]["point"] == 2.5
        assert outcomes[1]["name"] == "Under"
        assert outcomes[1]["point"] == 2.5

    def test_market_key_is_totals(self, sample_ou_bookmakers: list[dict]) -> None:
        result = _convert_over_under_match(sample_ou_bookmakers, "Leeds", "Sunderland")
        assert result is not None
        assert result["bookmakers"][0]["markets"][0]["key"] == "totals"

    def test_betfair_parsed(self, sample_ou_bookmakers: list[dict]) -> None:
        result = _convert_over_under_match(sample_ou_bookmakers, "Leeds", "Sunderland")
        assert result is not None
        betfair = next(b for b in result["bookmakers"] if b["key"] == "betfair_exchange")
        # 59/50 = 1.18 + 1 = 2.18 → 118
        over_price = betfair["markets"][0]["outcomes"][0]["price"]
        assert over_price == decimal_to_american(59 / 50 + 1)

    def test_betfair_liquidity_stored(self, sample_ou_bookmakers: list[dict]) -> None:
        result = _convert_over_under_match(sample_ou_bookmakers, "Leeds", "Sunderland")
        assert result is not None
        betfair = next(b for b in result["bookmakers"] if b["key"] == "betfair_exchange")
        assert betfair["betfair_matched"]["over"] == 159
        assert betfair["betfair_matched"]["under"] == 134

    def test_empty_returns_none(self) -> None:
        assert _convert_over_under_match([], "A", "B") is None

    def test_custom_line_mlb(self, sample_ou_bookmakers: list[dict]) -> None:
        result = _convert_over_under_match(sample_ou_bookmakers, "Yankees", "Red Sox", line=8.5)
        assert result is not None
        outcomes = result["bookmakers"][0]["markets"][0]["outcomes"]
        assert outcomes[0]["point"] == 8.5
        assert outcomes[1]["point"] == 8.5

    def test_custom_line_whole_number(self, sample_ou_bookmakers: list[dict]) -> None:
        result = _convert_over_under_match(sample_ou_bookmakers, "Yankees", "Red Sox", line=9.0)
        assert result is not None
        outcomes = result["bookmakers"][0]["markets"][0]["outcomes"]
        assert outcomes[0]["point"] == 9.0


class TestParseOverUnderLine:
    def test_football_half_line(self) -> None:
        assert parse_over_under_line("over_under_2_5") == 2.5

    def test_mlb_half_line(self) -> None:
        assert parse_over_under_line("over_under_8_5") == 8.5

    def test_whole_number_short(self) -> None:
        assert parse_over_under_line("over_under_3") == 3.0

    def test_whole_number_with_zero(self) -> None:
        assert parse_over_under_line("over_under_8_0") == 8.0

    def test_quarter_line(self) -> None:
        assert parse_over_under_line("over_under_1_25") == 1.25

    def test_non_over_under_market(self) -> None:
        assert parse_over_under_line("1x2") is None

    def test_home_away(self) -> None:
        assert parse_over_under_line("home_away") is None

    def test_malformed(self) -> None:
        assert parse_over_under_line("over_under_abc") is None


class TestConvertUpcomingMatches:
    @pytest.fixture()
    def sample_match(self) -> list[dict]:
        return [
            {
                "scraped_date": "2026-03-02 18:50:32 UTC",
                "match_date": "2026-03-03 19:30:00 UTC",
                "match_link": "https://www.oddsportal.com/football/england/premier-league/leeds-sunderland-KYph380S/",
                "home_team": "Leeds",
                "away_team": "Sunderland",
                "league_name": "Premier League",
                "home_score": "",
                "away_score": "",
                "1x2_market": [
                    {
                        "1": "9/10",
                        "X": "12/5",
                        "2": "3/1",
                        "bookmaker_name": "bet365",
                        "period": "FullTime",
                    },
                ],
            }
        ]

    def test_returns_match_odds(self, sample_match: list[dict]) -> None:
        results = convert_upcoming_matches(sample_match, "1x2")
        assert len(results) == 1
        assert isinstance(results[0], MatchOdds)

    def test_match_metadata(self, sample_match: list[dict]) -> None:
        result = convert_upcoming_matches(sample_match, "1x2")[0]
        assert result.home_team == "Leeds"
        assert result.away_team == "Sunderland"
        assert result.match_date.year == 2026
        assert result.bookmaker_count == 1

    def test_raw_data_has_bookmakers(self, sample_match: list[dict]) -> None:
        result = convert_upcoming_matches(sample_match, "1x2")[0]
        assert "bookmakers" in result.raw_data
        assert len(result.raw_data["bookmakers"]) == 1

    def test_unsupported_market_raises(self, sample_match: list[dict]) -> None:
        with pytest.raises(ValueError, match="Unsupported market"):
            convert_upcoming_matches(sample_match, "asian_handicap")

    def test_malformed_over_under_falls_through_to_raise(self, sample_match: list[dict]) -> None:
        """Market starting with 'over_under_' but not numeric falls through to the map."""
        with pytest.raises(ValueError, match="Unsupported market"):
            convert_upcoming_matches(sample_match, "over_under_bad")

    def test_skips_incomplete_match(self) -> None:
        matches = [{"home_team": "Leeds", "away_team": "", "match_date": ""}]
        assert convert_upcoming_matches(matches, "1x2") == []

    def test_over_under_variable_line(self) -> None:
        """MLB-style over/under scraped at a non-2.5 line propagates the line."""
        matches = [
            {
                "scraped_date": "2026-06-15 20:00:00 UTC",
                "match_date": "2026-06-16 23:10:00 UTC",
                "match_link": "https://www.oddsportal.com/baseball/usa/mlb/yankees-red-sox-abc123/",
                "home_team": "New York Yankees",
                "away_team": "Boston Red Sox",
                "league_name": "MLB",
                "over_under_8_5_market": [
                    {
                        "odds_over": "11/10",
                        "odds_under": "73/100",
                        "bookmaker_name": "bet365",
                        "period": "FullTime",
                    },
                ],
            }
        ]
        results = convert_upcoming_matches(matches, "over_under_8_5")
        assert len(results) == 1
        outcomes = results[0].raw_data["bookmakers"][0]["markets"][0]["outcomes"]
        assert outcomes[0]["point"] == 8.5
        assert outcomes[1]["point"] == 8.5
        assert results[0].raw_data["bookmakers"][0]["markets"][0]["key"] == "totals"

    def test_over_under_2_5_still_supported(self) -> None:
        """Existing EPL totals keep working via the dynamic dispatch path."""
        matches = [
            {
                "scraped_date": "2026-03-02 18:50:32 UTC",
                "match_date": "2026-03-03 19:30:00 UTC",
                "match_link": "https://www.oddsportal.com/football/england/premier-league/leeds-sunderland-KYph380S/",
                "home_team": "Leeds",
                "away_team": "Sunderland",
                "league_name": "Premier League",
                "over_under_2_5_market": [
                    {
                        "odds_over": "11/10",
                        "odds_under": "73/100",
                        "bookmaker_name": "bet365",
                        "period": "FullTime",
                    },
                ],
            }
        ]
        results = convert_upcoming_matches(matches, "over_under_2_5")
        assert len(results) == 1
        outcomes = results[0].raw_data["bookmakers"][0]["markets"][0]["outcomes"]
        assert outcomes[0]["point"] == 2.5

    def test_home_away_market_converts(self) -> None:
        matches = [
            {
                "scraped_date": "2026-06-15 20:00:00 UTC",
                "match_date": "2026-06-16 23:10:00 UTC",
                "match_link": "https://www.oddsportal.com/baseball/usa/mlb/yankees-red-sox-abc123/",
                "home_team": "New York Yankees",
                "away_team": "Boston Red Sox",
                "league_name": "MLB",
                "home_away_market": [
                    {
                        "1": "5/6",
                        "2": "EVS",
                        "bookmaker_name": "bet365",
                        "period": "FullTime",
                    },
                ],
            }
        ]
        results = convert_upcoming_matches(matches, "home_away")
        assert len(results) == 1
        outcomes = results[0].raw_data["bookmakers"][0]["markets"][0]["outcomes"]
        assert len(outcomes) == 2
        assert outcomes[0]["name"] == "New York Yankees"
        assert outcomes[1]["name"] == "Boston Red Sox"


class TestConvertHomeAwayMatch:
    @pytest.fixture()
    def sample_ha_bookmakers(self) -> list[dict]:
        return [
            {
                "1": "5/6",
                "2": "EVS",
                "bookmaker_name": "bet365",
                "period": "FullTime",
            },
            {
                "1": "4/5",
                "2": "11/10",
                "bookmaker_name": "10bet",
                "period": "FullTime",
            },
            {
                "1": "9/109/10(500)",
                "2": "11/1011/10(400)",
                "bookmaker_name": "Betfair Exchange",
                "period": "FullTime",
            },
        ]

    def test_basic_conversion(self, sample_ha_bookmakers: list[dict]) -> None:
        result = _convert_home_away_match(sample_ha_bookmakers, "Yankees", "Red Sox")
        assert result is not None
        assert "bookmakers" in result
        assert len(result["bookmakers"]) == 3

    def test_outcome_names(self, sample_ha_bookmakers: list[dict]) -> None:
        result = _convert_home_away_match(sample_ha_bookmakers, "Yankees", "Red Sox")
        assert result is not None
        outcomes = result["bookmakers"][0]["markets"][0]["outcomes"]
        assert len(outcomes) == 2
        assert outcomes[0]["name"] == "Yankees"
        assert outcomes[1]["name"] == "Red Sox"

    def test_no_draw_outcome(self, sample_ha_bookmakers: list[dict]) -> None:
        result = _convert_home_away_match(sample_ha_bookmakers, "Yankees", "Red Sox")
        assert result is not None
        for bk in result["bookmakers"]:
            names = [o["name"] for o in bk["markets"][0]["outcomes"]]
            assert DRAW_OUTCOME not in names

    def test_odds_conversion(self, sample_ha_bookmakers: list[dict]) -> None:
        result = _convert_home_away_match(sample_ha_bookmakers, "Yankees", "Red Sox")
        assert result is not None
        # bet365: home 5/6 = 1.833... decimal
        home_price = result["bookmakers"][0]["markets"][0]["outcomes"][0]["price"]
        assert home_price == decimal_to_american(5 / 6 + 1)

    def test_betfair_parsed(self, sample_ha_bookmakers: list[dict]) -> None:
        result = _convert_home_away_match(sample_ha_bookmakers, "Yankees", "Red Sox")
        assert result is not None
        betfair = next(b for b in result["bookmakers"] if b["key"] == "betfair_exchange")
        # 9/10 = 1.9 decimal
        home_price = betfair["markets"][0]["outcomes"][0]["price"]
        assert home_price == decimal_to_american(1.9)

    def test_betfair_liquidity_stored(self, sample_ha_bookmakers: list[dict]) -> None:
        result = _convert_home_away_match(sample_ha_bookmakers, "Yankees", "Red Sox")
        assert result is not None
        betfair = next(b for b in result["bookmakers"] if b["key"] == "betfair_exchange")
        assert "betfair_matched" in betfair
        assert betfair["betfair_matched"]["home"] == 500
        assert betfair["betfair_matched"]["away"] == 400
        assert "draw" not in betfair["betfair_matched"]

    def test_market_key_is_h2h(self, sample_ha_bookmakers: list[dict]) -> None:
        result = _convert_home_away_match(sample_ha_bookmakers, "Yankees", "Red Sox")
        assert result is not None
        assert result["bookmakers"][0]["markets"][0]["key"] == "h2h"

    def test_source_tag(self, sample_ha_bookmakers: list[dict]) -> None:
        result = _convert_home_away_match(sample_ha_bookmakers, "Yankees", "Red Sox")
        assert result is not None
        assert result["source"] == "oddsportal_live"

    def test_empty_returns_none(self) -> None:
        assert _convert_home_away_match([], "A", "B") is None

    def test_missing_odds_skipped(self) -> None:
        data = [{"bookmaker_name": "bet365", "1": "5/6", "2": ""}]
        assert _convert_home_away_match(data, "A", "B") is None
