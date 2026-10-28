"""Each source's envelope read as market snapshots and price rows (#672)."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from odds_core.vocabulary import Market, MarketStatus, PriceOutcome, Side, Source
from odds_lambda.price_rows import (
    ExpandedEnvelope,
    MarketSnapshot,
    PriceSpec,
    RowIssueType,
    expand_envelope,
)

HOME, AWAY = "Arsenal", "Chelsea"


def _expand(raw_data: dict[str, Any], source: Source) -> ExpandedEnvelope:
    return expand_envelope(raw_data, source, HOME, AWAY)


def _rows(market: MarketSnapshot) -> set[tuple[str, str, Decimal | None, str, Decimal]]:
    return {(p.bookmaker, p.outcome.value, p.line, p.side.value, p.price) for p in market.prices}


def _issue_types(expanded: ExpandedEnvelope) -> list[str]:
    return [issue.issue_type for issue in expanded.issues]


def _three_way(key: str, prices: tuple[Any, Any, Any], **extra: Any) -> dict[str, Any]:
    return {
        "key": key,
        "title": key,
        "last_update": "2026-10-24T12:00:00Z",
        "markets": [
            {
                "key": "1x2",
                "outcomes": [
                    {"name": HOME, "price_decimal": prices[0]},
                    {"name": "Draw", "price_decimal": prices[1]},
                    {"name": AWAY, "price_decimal": prices[2]},
                ],
            }
        ],
        **extra,
    }


class TestOddsPortalLive:
    def test_books_become_back_rows_with_the_change_time(self) -> None:
        raw = {
            "bookmakers": [_three_way("bet365", (2.1, 3.4, 3.5))],
            "source": "oddsportal_live",
            "oddsportal_event_id": "Kl1AOZN9",
        }

        [market] = _expand(raw, Source.ODDSPORTAL).markets

        assert market.market is Market.ONE_X_TWO
        assert _rows(market) == {
            ("bet365", "home", None, "back", Decimal("2.1")),
            ("bet365", "draw", None, "back", Decimal("3.4")),
            ("bet365", "away", None, "back", Decimal("3.5")),
        }
        assert {p.changed_at for p in market.prices} == {datetime(2026, 10, 24, 12, tzinfo=UTC)}

    def test_betfair_matched_volume_goes_on_the_exchange_rows(self) -> None:
        raw = {
            "bookmakers": [
                _three_way(
                    "betfair_exchange",
                    (2.1, 3.4, 3.5),
                    betfair_matched={"home": 5117, "draw": 16, "away": 900.5},
                ),
                _three_way("bet365", (2.0, 3.3, 3.4)),
            ],
            "source": "oddsportal_live",
        }

        [market] = _expand(raw, Source.ODDSPORTAL).markets

        matched = {(p.bookmaker, p.outcome): p.matched for p in market.prices}
        assert matched[("betfair_exchange", PriceOutcome.HOME)] == Decimal("5117")
        assert matched[("betfair_exchange", PriceOutcome.AWAY)] == Decimal("900.5")
        assert matched[("bet365", PriceOutcome.HOME)] is None

    def test_totals_take_their_line(self) -> None:
        raw = {
            "bookmakers": [
                {
                    "key": "bet365",
                    "markets": [
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over", "price_decimal": 1.9, "point": 2.5},
                                {"name": "Under", "price_decimal": 1.95, "point": 2.5},
                            ],
                        }
                    ],
                }
            ],
            "source": "oddsportal_live",
        }

        [market] = _expand(raw, Source.ODDSPORTAL).markets

        assert market.market is Market.TOTALS
        assert _rows(market) == {
            ("bet365", "over", Decimal("2.5"), "back", Decimal("1.9")),
            ("bet365", "under", Decimal("2.5"), "back", Decimal("1.95")),
        }

    def test_alias_collision_keeps_the_first_entry_and_logs_the_rest(self) -> None:
        raw = {
            "bookmakers": [
                _three_way("7bet", (2.1, 3.4, 3.5)),
                _three_way("7betuk", (2.2, 3.5, 3.6)),
            ],
            "source": "oddsportal_live",
        }

        expanded = _expand(raw, Source.ODDSPORTAL)

        [market] = expanded.markets
        assert {p.price for p in market.prices if p.outcome is PriceOutcome.HOME} == {
            Decimal("2.1")
        }
        assert _issue_types(expanded) == ["bookmaker_alias_collision"]
        assert expanded.issues[0].context["dropped_key"] == "7betuk"

    def test_invalid_prices_are_logged_not_stored(self) -> None:
        raw = {
            "bookmakers": [_three_way("bet365", (1.0, "x", 3.5))],
            "source": "oddsportal_live",
        }

        expanded = _expand(raw, Source.ODDSPORTAL)

        assert _rows(expanded.markets[0]) == {("bet365", "away", None, "back", Decimal("3.5"))}
        assert _issue_types(expanded) == ["invalid_price", "invalid_price"]

    def test_unmatched_outcome_names_are_logged(self) -> None:
        raw = {
            "bookmakers": [
                {
                    "key": "bet365",
                    "markets": [
                        {
                            "key": "1x2",
                            "outcomes": [
                                {"name": "Arsenal FC Women", "price_decimal": 2.0},
                                {"name": AWAY, "price_decimal": 3.0},
                            ],
                        }
                    ],
                }
            ],
            "source": "oddsportal_live",
        }

        expanded = _expand(raw, Source.ODDSPORTAL)

        assert _rows(expanded.markets[0]) == {("bet365", "away", None, "back", Decimal("3.0"))}
        assert _issue_types(expanded) == ["unmatched_outcome"]

    def test_team_names_match_after_normalization(self) -> None:
        raw = {
            "bookmakers": [
                {
                    "key": "bet365",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Man Utd", "price_decimal": 2.0},
                                {"name": "Fulham", "price_decimal": 3.0},
                            ],
                        }
                    ],
                }
            ],
            "source": "oddsportal_live",
        }

        expanded = expand_envelope(raw, Source.ODDSPORTAL, "Manchester United", "Fulham")

        assert {p.outcome for p in expanded.markets[0].prices} == {
            PriceOutcome.HOME,
            PriceOutcome.AWAY,
        }


class TestBetfair:
    def _raw(self, **meta: Any) -> dict[str, Any]:
        runners = [
            {
                "selection_id": 1,
                "runner_name": HOME,
                "best_back": 2.1,
                "best_lay": 2.12,
                "back_size": 292.0,
                "lay_size": 314.0,
                "last_price_traded": 2.1,
            },
            {
                "selection_id": 2,
                "runner_name": "The Draw",
                "best_back": 3.4,
                "best_lay": 3.45,
                "back_size": 50.0,
                "lay_size": 60.0,
                "last_price_traded": None,
            },
            {
                "selection_id": 3,
                "runner_name": AWAY,
                "best_back": 3.5,
                "best_lay": None,
                "back_size": 10.0,
                "lay_size": None,
                "last_price_traded": 3.55,
            },
        ]
        return {
            "bookmakers": [
                {
                    "key": "betfair_exchange",
                    "title": "Betfair Exchange",
                    "last_update": "2026-10-24T12:00:00+00:00",
                    "markets": [{"key": "1x2", "outcomes": []}],
                    "betfair_meta": {
                        "market_id": "1.23",
                        "market_status": "OPEN",
                        "market_start_time": "2026-10-24T14:00:00+00:00",
                        "inplay": False,
                        "total_matched": 385702.5,
                        "runners": runners,
                        **meta,
                    },
                }
            ],
            "source": "betfair_api",
            "betfair_event_id": "35000001",
        }

    def test_runners_become_back_lay_and_traded_rows_with_sizes(self) -> None:
        [market] = _expand(self._raw(), Source.BETFAIR).markets

        sized = {(p.outcome.value, p.side.value): (p.price, p.size) for p in market.prices}
        assert sized == {
            ("home", "back"): (Decimal("2.1"), Decimal("292.0")),
            ("home", "lay"): (Decimal("2.12"), Decimal("314.0")),
            ("home", "traded"): (Decimal("2.1"), None),
            ("draw", "back"): (Decimal("3.4"), Decimal("50.0")),
            ("draw", "lay"): (Decimal("3.45"), Decimal("60.0")),
            ("away", "back"): (Decimal("3.5"), Decimal("10.0")),
            ("away", "traded"): (Decimal("3.55"), None),
        }
        assert {p.changed_at for p in market.prices} == {None}

    def test_market_metadata_goes_on_the_snapshot_lower_cased(self) -> None:
        [market] = _expand(
            self._raw(market_status="SUSPENDED", inplay=True), Source.BETFAIR
        ).markets

        assert market.meta.market_status is MarketStatus.SUSPENDED
        assert market.meta.in_play is True
        assert market.meta.total_matched == Decimal("385702.5")
        assert market.meta.market_start_time == datetime(2026, 10, 24, 14, tzinfo=UTC)

    def test_unknown_market_status_is_logged_and_left_empty(self) -> None:
        expanded = _expand(self._raw(market_status="WEIRD"), Source.BETFAIR)

        assert expanded.markets[0].meta.market_status is None
        assert _issue_types(expanded) == ["unknown_market_status"]

    def test_pitcher_annotations_on_runner_names_still_match(self) -> None:
        raw = self._raw()
        raw["bookmakers"][0]["markets"][0]["key"] = "h2h"
        runners = raw["bookmakers"][0]["betfair_meta"]["runners"]
        runners[0]["runner_name"] = f"{HOME} (G Col"
        del runners[1]

        [market] = expand_envelope(raw, Source.BETFAIR, HOME, AWAY).markets

        assert {p.outcome for p in market.prices} == {PriceOutcome.HOME, PriceOutcome.AWAY}


class TestPolymarket:
    def _raw(self, **quotes: Any) -> dict[str, Any]:
        return {
            "bookmakers": [
                {
                    "key": "polymarket",
                    "title": "Polymarket",
                    "last_update": "2026-10-24T12:00:00+00:00",
                    "markets": [{"key": "1x2", "outcomes": []}],
                    "polymarket_meta": {
                        "pm_event_id": "123",
                        "yes_probabilities": {"home": 0.5, "draw": 0.25, "away": 0.25},
                        **quotes,
                    },
                }
            ],
            "source": "polymarket",
        }

    def test_history_writes_mid_rows_as_decimal_odds(self) -> None:
        [market] = _expand(self._raw(), Source.POLYMARKET).markets

        assert _rows(market) == {
            ("polymarket", "home", None, "mid", Decimal("2.0")),
            ("polymarket", "draw", None, "mid", Decimal("4.0")),
            ("polymarket", "away", None, "mid", Decimal("4.0")),
        }

    def test_live_ask_is_back_and_bid_is_lay(self) -> None:
        raw = self._raw(
            best_ask={"home": 0.52, "draw": 0.26, "away": 0.26},
            best_bid={"home": 0.48, "draw": 0.24, "away": 0.25},
        )

        [market] = _expand(raw, Source.POLYMARKET).markets

        prices = {(p.outcome.value, p.side.value): p.price for p in market.prices}
        assert prices[("home", "back")] == Decimal(str(1 / 0.52))
        assert prices[("home", "lay")] == Decimal(str(1 / 0.48))
        assert prices[("away", "lay")] == Decimal("4.0")

    def test_certain_probabilities_are_invalid_prices(self) -> None:
        raw = self._raw(best_bid={"home": 1.0, "draw": 0.0, "away": 0.25})

        expanded = _expand(raw, Source.POLYMARKET)

        assert {(p.outcome, p.side) for p in expanded.markets[0].prices if p.side is Side.LAY} == {
            (PriceOutcome.AWAY, Side.LAY)
        }
        assert _issue_types(expanded) == ["invalid_price", "invalid_price"]


class TestOddsApi:
    def _raw(self) -> dict[str, Any]:
        return {
            "id": "abc",
            "bookmakers": [
                {
                    "key": "betmgm",
                    "title": "BetMGM",
                    "last_update": "2026-10-24T11:00:00Z",
                    "markets": [
                        {
                            "key": "1x2",
                            "last_update": "2026-10-24T11:30:00Z",
                            "outcomes": [
                                {"name": HOME, "price": -110},
                                {"name": "Draw", "price": 250},
                                {"name": AWAY, "price": 300},
                            ],
                        },
                        {
                            "key": "spreads",
                            "outcomes": [
                                {"name": HOME, "price": -105, "point": -0.5},
                                {"name": AWAY, "price": -115, "point": 0.5},
                            ],
                        },
                        {
                            "key": "totals",
                            "outcomes": [
                                {"name": "Over", "price": 100, "point": 2.5},
                                {"name": "Under", "price": -120, "point": 2.5},
                            ],
                        },
                    ],
                },
                {
                    "key": "betfair_exchange",
                    "markets": [
                        {
                            "key": "h2h_lay",
                            "outcomes": [
                                {"name": HOME, "price": 110},
                                {"name": "Draw", "price": 260},
                                {"name": AWAY, "price": 320},
                            ],
                        }
                    ],
                },
            ],
        }

    def test_each_market_is_its_own_snapshot_in_market_order(self) -> None:
        expanded = _expand(self._raw(), Source.ODDS_API)

        assert [m.market for m in expanded.markets] == [
            Market.ONE_X_TWO,
            Market.SPREADS,
            Market.TOTALS,
        ]
        spreads = expanded.markets[1]
        assert {(p.outcome.value, p.line) for p in spreads.prices} == {
            ("home", Decimal("-0.5")),
            ("away", Decimal("0.5")),
        }

    def test_american_prices_become_decimal_and_us_betmgm_is_aliased(self) -> None:
        one_x_two = _expand(self._raw(), Source.ODDS_API).markets[0]

        home = next(
            p for p in one_x_two.prices if p.outcome is PriceOutcome.HOME and p.side is Side.BACK
        )
        assert home.bookmaker == "betmgm_us"
        assert home.price == Decimal(str(100 / 110 + 1))
        assert home.changed_at == datetime(2026, 10, 24, 11, 30, tzinfo=UTC)

    def test_h2h_lay_becomes_lay_rows_of_the_1x2_snapshot_including_the_draw(self) -> None:
        one_x_two = _expand(self._raw(), Source.ODDS_API).markets[0]

        assert {p.outcome for p in one_x_two.prices if p.side is Side.LAY} == {
            PriceOutcome.HOME,
            PriceOutcome.DRAW,
            PriceOutcome.AWAY,
        }
        assert {p.bookmaker for p in one_x_two.prices if p.side is Side.LAY} == {"betfair_exchange"}

    def test_h2h_lay_without_a_draw_becomes_lay_rows_of_the_h2h_snapshot(self) -> None:
        raw = {
            "bookmakers": [
                {
                    "key": "fanduel",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": HOME, "price": 110},
                                {"name": AWAY, "price": -130},
                            ],
                        }
                    ],
                },
                {
                    "key": "betfair_ex_uk",
                    "markets": [
                        {
                            "key": "h2h_lay",
                            "outcomes": [
                                {"name": HOME, "price": 115},
                                {"name": AWAY, "price": -125},
                            ],
                        }
                    ],
                },
            ]
        }

        [h2h] = _expand(raw, Source.ODDS_API).markets

        assert h2h.market is Market.H2H
        assert {(p.bookmaker, p.side) for p in h2h.prices} == {
            ("fanduel", Side.BACK),
            ("betfair_exchange", Side.LAY),
        }

    def test_unknown_market_is_logged(self) -> None:
        raw = {"bookmakers": [{"key": "fanduel", "markets": [{"key": "outrights"}]}]}

        expanded = _expand(raw, Source.ODDS_API)

        assert expanded.markets == ()
        assert _issue_types(expanded) == [RowIssueType.UNKNOWN_MARKET]


class TestFootballDataUk:
    def test_aggregate_pseudo_books_are_not_written(self) -> None:
        raw = {
            "bookmakers": [
                _three_way("pinnacle", (2.1, 3.4, 3.5)),
                _three_way("market_max", (2.3, 3.6, 3.8)),
                _three_way("market_avg", (2.0, 3.3, 3.4)),
            ],
            "source": "football_data_uk",
            "timing": {"basis": "kickoff", "kickoff_time_assumed": False},
        }

        [market] = _expand(raw, Source.FOOTBALL_DATA_UK).markets

        assert {p.bookmaker for p in market.prices} == {"pinnacle"}
        assert {p.changed_at for p in market.prices} == {None}


class TestPriceSpec:
    def test_key_is_the_unique_columns(self) -> None:
        spec = PriceSpec("bet365", PriceOutcome.HOME, None, Side.BACK, Decimal("2"))

        assert spec.key == ("bet365", PriceOutcome.HOME, None, Side.BACK)
