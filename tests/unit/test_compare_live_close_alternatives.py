"""The live-close alternatives comparison: picking each alternative's price, matching it with
the live close, and the report."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from odds_core.exchange_price import ExchangePrice
from odds_core.market_prices import Outcome
from odds_core.quotes import Quote
from odds_core.vocabulary import Market, Source

from scripts import compare_live_close_alternatives as cmp
from tests.test_helpers import betfair_exchange_bookmaker, make_epl_event, make_quotes, totals_book

KICKOFF = datetime(2026, 11, 1, 15, 0, tzinfo=UTC)
EVENT = make_epl_event(1, KICKOFF)
NARROW = (2.0, 3.6, 4.2)
LATER = (1.9, 3.8, 4.4)
WIDE = (2.0, 3.0, 2.0)
ONE_X_TWO = cmp.Pairing(Source.ODDSPORTAL, Market.ONE_X_TWO, None)
TOTALS = cmp.Pairing(Source.ODDSPORTAL, Market.TOTALS, cmp.TOTALS_LINE)
ODDS_API = cmp.Pairing(Source.ODDS_API, Market.ONE_X_TWO, None)


def at(minutes_before: float) -> datetime:
    return KICKOFF - timedelta(minutes=minutes_before)


def live(
    minutes_before: float, prices: tuple[float, float, float] = NARROW, snapshot_id: int = 10
) -> list[Quote]:
    return make_quotes(
        EVENT, snapshot_id, at(minutes_before), [betfair_exchange_bookmaker(prices)], "betfair_api"
    )


def oddsportal(
    minutes_before: float,
    prices: tuple[float, float, float] = NARROW,
    snapshot_id: int = 100,
) -> list[Quote]:
    book = betfair_exchange_bookmaker(prices)
    return make_quotes(EVENT, snapshot_id, at(minutes_before), [book], "oddsportal_live")


def compare(
    live_quotes: list[Quote],
    alternative: list[Quote],
    pairing: cmp.Pairing = ONE_X_TWO,
    max_age: timedelta | None = None,
) -> cmp.EventComparison:
    return cmp.compare_event(1, KICKOFF, pairing, live_quotes, alternative, max_age)


def fair(prices: tuple[float, ...]) -> list[float]:
    return [(1 / p) / sum(1 / q for q in prices) for p in prices]


class TestCompareEvent:
    def test_takes_the_latest_alternative_snapshot_at_or_before_kickoff(self) -> None:
        found = compare(
            live(5),
            [
                *oddsportal(20, NARROW, 100),
                *oddsportal(3, LATER, 101),
                *oddsportal(-2, WIDE, 102),
            ],
        )
        assert found.status is cmp.Status.COMPARED
        assert found.age_minutes == pytest.approx(3)
        assert found.alternative is not None and found.live is not None
        assert list(found.alternative.fair.values()) == pytest.approx(fair(LATER))
        assert found.live.time == at(5)

    def test_a_wide_latest_snapshot_falls_back_to_the_latest_narrow_one(self) -> None:
        found = compare(live(5), [*oddsportal(20, NARROW, 100), *oddsportal(3, WIDE, 101)])
        assert found.status is cmp.Status.COMPARED and found.age_minutes == pytest.approx(20)

    def test_an_in_play_snapshot_is_not_a_price(self) -> None:
        in_play = [replace(q, in_play=True) for q in oddsportal(1)]
        found = compare(live(5), in_play)
        assert found.status is cmp.Status.SNAPSHOT_UNUSABLE

    def test_an_incomplete_snapshot_is_not_a_price(self) -> None:
        partial = [q for q in oddsportal(3) if q.outcome is not Outcome.DRAW]
        assert compare(live(5), partial).status is cmp.Status.SNAPSHOT_UNUSABLE

    def test_no_alternative_snapshot_at_or_before_kickoff_is_missing(self) -> None:
        assert compare(live(5), oddsportal(-1)).status is cmp.Status.NO_SNAPSHOT
        assert compare(live(5), []).status is cmp.Status.NO_SNAPSHOT

    def test_max_age_leaves_out_older_snapshots(self) -> None:
        old = oddsportal(20)
        assert compare(live(5), old, max_age=timedelta(minutes=10)).status is cmp.Status.NO_SNAPSHOT
        assert compare(live(5), old, max_age=timedelta(minutes=20)).status is cmp.Status.COMPARED

    def test_only_another_sources_snapshots_are_missing(self) -> None:
        odds_api = make_quotes(EVENT, 200, at(3), [betfair_exchange_bookmaker(NARROW)], None)
        assert compare(live(5), odds_api).status is cmp.Status.SNAPSHOT_UNUSABLE
        assert compare(live(5), odds_api, ODDS_API).status is cmp.Status.COMPARED

    def test_without_a_live_close_there_is_no_comparison(self) -> None:
        # The Betfair job's last price is outside the live close window.
        found = compare(live(45), oddsportal(3))
        assert found.status is cmp.Status.NO_LIVE_CLOSE and found.alternative is None

    def test_totals_compare_on_the_line(self) -> None:
        live_totals = make_quotes(
            EVENT,
            10,
            at(5),
            [totals_book("betfair_exchange", {2.5: (1.9, 2.1)}, betfair_runners=True)],
            "betfair_api",
        )
        other_line = totals_book("betfair_exchange", {2.5: (1.8, 2.2), 3.5: (3.0, 1.4)})
        alternative = make_quotes(EVENT, 100, at(3), [other_line], "oddsportal_live")
        found = compare(live_totals, alternative, TOTALS)
        assert found.status is cmp.Status.COMPARED and found.alternative is not None
        assert list(found.alternative.fair.values()) == pytest.approx(fair((1.8, 2.2)))


def with_fair(price: ExchangePrice, fair_values: tuple[float, ...]) -> ExchangePrice:
    return replace(
        price, fair=dict(zip((Outcome.HOME, Outcome.DRAW, Outcome.AWAY), fair_values, strict=True))
    )


def compared(
    event_id: int, live_fair: tuple[float, ...], alt_fair: tuple[float, ...], age: float
) -> cmp.EventComparison:
    found = compare(live(5), oddsportal(age))
    assert found.live is not None and found.alternative is not None
    return replace(
        found,
        event_id=event_id,
        live=with_fair(found.live, live_fair),
        alternative=with_fair(found.alternative, alt_fair),
    )


class TestReport:
    LIVE_FAIR = (0.5, 0.3, 0.2)

    def test_a_section_has_the_statistics_coverage_and_ages(self) -> None:
        comparisons = [
            compared(1, self.LIVE_FAIR, (0.51, 0.295, 0.195), 4),
            compared(2, self.LIVE_FAIR, (0.53, 0.285, 0.185), 20),
            cmp.EventComparison(3, cmp.Status.NO_SNAPSHOT, live=compare(live(5), []).live),
            cmp.EventComparison(4, cmp.Status.NO_LIVE_CLOSE),
        ]
        section = cmp.build_section(comparisons, ONE_X_TWO)
        assert section["outcomes"]["home"]["n"] == 2
        assert section["outcomes"]["home"]["mean_pp"] == pytest.approx(2.0)
        assert section["bands"][">=0.5"]["n"] == 2
        assert section["verdict"] == "fail"
        assert section["matches"] == {
            "final_events": 4,
            "with_live_close": 3,
            "compared": 2,
            "coverage": 2 / 3,
            "missing": {
                "no_live_close": 1,
                "no_alternative_snapshot": 1,
                "alternative_unusable": 0,
            },
        }
        assert section["within_minutes_of_kickoff"]["5"] == {"matches": 1, "share": 1 / 3}
        assert section["within_minutes_of_kickoff"]["30"] == {"matches": 2, "share": 2 / 3}
        assert section["age_minutes"]["max"] == pytest.approx(20)

    def test_the_report_has_a_section_per_pairing_and_is_json(self) -> None:
        report = cmp.build_report({p: [] for p in cmp.PAIRINGS}, {"sport": "soccer_epl"})
        assert list(report["pairings"]) == [
            "oddsportal 1x2",
            "oddsportal totals 2.5",
            "odds_api 1x2",
        ]
        assert report["pairings"]["odds_api 1x2"]["verdict"] == "inconclusive"
        text = cmp.render(report)
        assert "== oddsportal totals 2.5: INCONCLUSIVE" in text and "over" in text
        json.dumps(report)
