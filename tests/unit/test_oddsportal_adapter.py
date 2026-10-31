"""Tests for convert_market (OddsPortal feed prices to pipeline raw_data)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from odds_lambda.oddsportal_adapter import convert_market
from odds_lambda.oddsportal_client import BookmakerPrices, ListingEvent

SCRAPED_AT = datetime(2026, 10, 17, 12, 0, tzinfo=UTC)


@pytest.fixture
def event() -> ListingEvent:
    return ListingEvent(
        event_id="abc12345",
        sport_id=1,
        home_team="Arsenal",
        away_team="Everton",
        start_time=datetime(2026, 10, 18, 15, 0, tzinfo=UTC),
        url="https://www.oddsportal.com/football/h2h/arsenal-x/everton-y/#abc12345",
    )


def _bk(
    name: str,
    prices: tuple[float, ...],
    *,
    volumes: tuple[int, ...] | None = None,
    last_change: datetime | None = None,
) -> BookmakerPrices:
    return BookmakerPrices("1", name, prices, volumes, last_change)


class TestConvertMarket:
    def test_1x2_shape(self, event: ListingEvent) -> None:
        change = SCRAPED_AT - timedelta(hours=3)
        match = convert_market(
            event, "1x2", [_bk("bet365", (2.0, 3.5, 1.5), last_change=change)], SCRAPED_AT
        )

        assert match is not None
        assert match.envelope == {
            "source": "oddsportal_live",
            "oddsportal_event_id": "abc12345",
            "bookmakers": [
                {
                    "key": "bet365",
                    "title": "bet365",
                    "last_update": "2026-10-17T09:00:00Z",
                    "markets": [
                        {
                            "key": "1x2",
                            "outcomes": [
                                {"name": "Arsenal", "price_decimal": 2.0},
                                {"name": "Draw", "price_decimal": 3.5},
                                {"name": "Everton", "price_decimal": 1.5},
                            ],
                        }
                    ],
                }
            ],
        }

    def test_match_metadata(self, event: ListingEvent) -> None:
        match = convert_market(event, "1x2", [_bk("bet365", (2.0, 3.5, 1.5))], SCRAPED_AT)

        assert match is not None
        assert match.home_team == "Arsenal"
        assert match.away_team == "Everton"
        assert match.match_date == event.start_time
        assert match.match_link == event.url
        assert match.market == "1x2"
        assert match.scraped_date == SCRAPED_AT
        assert match.bookmaker_count == 1

    def test_over_under_uses_totals_key_and_point(self, event: ListingEvent) -> None:
        match = convert_market(event, "over_under_2_5", [_bk("bet365", (2.0, 1.5))], SCRAPED_AT)

        assert match is not None
        market = match.envelope["bookmakers"][0]["markets"][0]
        assert market["key"] == "totals"
        assert market["outcomes"] == [
            {"name": "Over", "price_decimal": 2.0, "point": 2.5},
            {"name": "Under", "price_decimal": 1.5, "point": 2.5},
        ]

    def test_home_away_uses_h2h_key(self, event: ListingEvent) -> None:
        match = convert_market(event, "home_away", [_bk("bet365", (2.5, 1.5))], SCRAPED_AT)

        assert match is not None
        market = match.envelope["bookmakers"][0]["markets"][0]
        assert market["key"] == "h2h"
        assert market["outcomes"] == [
            {"name": "Arsenal", "price_decimal": 2.5},
            {"name": "Everton", "price_decimal": 1.5},
        ]

    def test_betfair_matched_only_with_volumes(self, event: ListingEvent) -> None:
        match = convert_market(
            event,
            "1x2",
            [
                _bk("Betfair Exchange", (2.0, 3.5, 4.0), volumes=(10, 20, 30)),
                _bk("bet365", (2.0, 3.5, 4.0)),
            ],
            SCRAPED_AT,
        )

        assert match is not None
        exchange, retail = match.envelope["bookmakers"]
        assert exchange["key"] == "betfair_exchange"
        assert exchange["betfair_matched"] == {"home": 10, "draw": 20, "away": 30}
        assert "betfair_matched" not in retail

    @pytest.mark.parametrize(
        ("market", "volumes", "expected"),
        [
            ("over_under_2_5", (5, 6), {"over": 5, "under": 6}),
            ("home_away", (7, 8), {"home": 7, "away": 8}),
        ],
    )
    def test_betfair_matched_labels(
        self,
        event: ListingEvent,
        market: str,
        volumes: tuple[int, ...],
        expected: dict[str, int],
    ) -> None:
        match = convert_market(
            event, market, [_bk("Betfair Exchange", (2.0, 2.0), volumes=volumes)], SCRAPED_AT
        )

        assert match is not None
        assert match.envelope["bookmakers"][0]["betfair_matched"] == expected

    def test_last_update_falls_back_to_scrape_time(self, event: ListingEvent) -> None:
        match = convert_market(event, "1x2", [_bk("bet365", (2.0, 3.5, 1.5))], SCRAPED_AT)

        assert match is not None
        assert match.envelope["bookmakers"][0]["last_update"] == "2026-10-17T12:00:00Z"

    def test_last_update_never_after_scrape_time(self, event: ListingEvent) -> None:
        future = SCRAPED_AT + timedelta(hours=1)
        match = convert_market(
            event, "1x2", [_bk("bet365", (2.0, 3.5, 1.5), last_change=future)], SCRAPED_AT
        )

        assert match is not None
        assert match.envelope["bookmakers"][0]["last_update"] == "2026-10-17T12:00:00Z"

    def test_no_bookmakers_returns_none(self, event: ListingEvent) -> None:
        assert convert_market(event, "1x2", [], SCRAPED_AT) is None

    def test_unsupported_market_raises(self, event: ListingEvent) -> None:
        with pytest.raises(ValueError, match="Unsupported market"):
            convert_market(event, "spreads", [_bk("bet365", (2.0, 2.0))], SCRAPED_AT)


class TestLadderMarkets:
    def test_totals_ladder_gives_each_bookmaker_every_line_on_both_outcomes(
        self, event: ListingEvent
    ) -> None:
        rows = [
            BookmakerPrices("1", "bet365", (1.9, 1.95), None, None, 220.5),
            BookmakerPrices("1", "bet365", (1.7, 2.2), None, None, 219.5),
        ]

        match = convert_market(event, "totals_ladder", rows, SCRAPED_AT)

        assert match is not None
        assert match.bookmaker_count == 1
        (entry,) = match.envelope["bookmakers"]
        (market,) = entry["markets"]
        assert market["key"] == "totals"
        assert [(o["name"], o["point"]) for o in market["outcomes"]] == [
            ("Over", 220.5),
            ("Under", 220.5),
            ("Over", 219.5),
            ("Under", 219.5),
        ]

    def test_spreads_ladder_signs_the_line_per_outcome(self, event: ListingEvent) -> None:
        rows = [BookmakerPrices("1", "bet365", (1.9, 1.95), None, None, -5.5)]

        match = convert_market(event, "spreads_ladder", rows, SCRAPED_AT)

        assert match is not None
        (market,) = match.envelope["bookmakers"][0]["markets"]
        assert market["key"] == "spreads"
        assert [(o["name"], o["point"]) for o in market["outcomes"]] == [
            ("Arsenal", -5.5),
            ("Everton", 5.5),
        ]

    def test_ladder_volumes_are_not_attached(self, event: ListingEvent) -> None:
        rows = [BookmakerPrices("1", "Betfair Exchange", (1.9, 1.95), (10, 20), None, 5.5)]

        match = convert_market(event, "spreads_ladder", rows, SCRAPED_AT)

        assert match is not None
        assert "betfair_matched" not in match.envelope["bookmakers"][0]

    def test_fixed_totals_keep_the_spec_point(self, event: ListingEvent) -> None:
        match = convert_market(event, "over_under_2_5", [_bk("bet365", (2.0, 1.5))], SCRAPED_AT)

        assert match is not None
        outcomes = match.envelope["bookmakers"][0]["markets"][0]["outcomes"]
        assert [o["point"] for o in outcomes] == [2.5, 2.5]

    def test_ladder_envelopes_expand_to_one_row_per_line_and_outcome(self) -> None:
        from odds_core.vocabulary import Market, PriceOutcome, Source
        from odds_lambda.price_rows import expand_envelope

        nba = ListingEvent(
            "hdJcPIgS",
            3,
            "Utah Jazz",
            "Portland Trail Blazers",
            datetime(2026, 10, 31, 19, 0, tzinfo=UTC),
            "https://www.oddsportal.com/basketball/h2h/x/y/#hdJcPIgS",
        )
        rows = [
            BookmakerPrices("1", "bet365", (1.9, 1.95), None, None, -5.5),
            BookmakerPrices("1", "bet365", (2.2, 1.7), None, None, -4.5),
        ]
        match = convert_market(nba, "spreads_ladder", rows, SCRAPED_AT)
        assert match is not None

        [expanded] = expand_envelope(
            match.envelope, Source.ODDSPORTAL, nba.home_team, nba.away_team
        ).markets

        assert expanded.market is Market.SPREADS
        assert sorted((float(p.line), p.outcome, float(p.price)) for p in expanded.prices) == [
            (-5.5, PriceOutcome.HOME, 1.9),
            (-4.5, PriceOutcome.HOME, 2.2),
            (4.5, PriceOutcome.AWAY, 1.7),
            (5.5, PriceOutcome.AWAY, 1.95),
        ]
