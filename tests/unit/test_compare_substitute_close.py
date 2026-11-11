"""The substitute-close comparison: matching a match's two exchange prices, the statistics and
verdicts, and the report."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from odds_core.exchange_price import ExchangePrice
from odds_core.market_prices import Outcome
from odds_core.quotes import Quote
from odds_core.vocabulary import Source, TimeBasis

from scripts import compare_substitute_close as cmp
from tests.test_helpers import betfair_exchange_bookmaker, make_epl_event, make_quotes

KICKOFF = datetime(2025, 3, 1, 15, 0, tzinfo=UTC)
EVENT = make_epl_event(1, KICKOFF)
NARROW = (2.0, 3.6, 4.2)
WIDE = (2.0, 3.0, 2.0)


def fduk(prices: tuple[float, float, float]) -> list[Quote]:
    return make_quotes(
        EVENT, 10, KICKOFF, [betfair_exchange_bookmaker(prices)], "football_data_uk", "kickoff"
    )


def oddsapi(
    minutes_before: float, prices: tuple[float, float, float], snapshot_id: int = 100
) -> list[Quote]:
    return make_quotes(
        EVENT,
        snapshot_id,
        KICKOFF - timedelta(minutes=minutes_before),
        [betfair_exchange_bookmaker(prices)],
        None,
    )


def compare(
    fduk_quotes: list[Quote], oddsapi_quotes: list[Quote], max_age: timedelta | None = None
) -> cmp.EventComparison:
    return cmp.compare_event(1, KICKOFF, "soccer_epl", fduk_quotes, oddsapi_quotes, max_age)


def price(fair: tuple[float, float, float]) -> ExchangePrice:
    return ExchangePrice(
        time=KICKOFF,
        source=Source.ODDS_API,
        time_basis=TimeBasis.SOURCE_STAMPED,
        in_play=False,
        fair=dict(zip((Outcome.HOME, Outcome.DRAW, Outcome.AWAY), fair, strict=True)),
        overround=0.01,
    )


def compared(
    event_id: int,
    fduk_fair: tuple[float, float, float],
    api_fair: tuple[float, float, float],
    age: float,
) -> cmp.EventComparison:
    return cmp.EventComparison(
        event_id, cmp.Status.COMPARED, price(fduk_fair), price(api_fair), age
    )


class TestCompareEvent:
    def test_takes_the_latest_snapshot_at_or_before_kickoff(self) -> None:
        later_prices = (1.9, 3.8, 4.4)
        found = compare(
            fduk(NARROW),
            [
                *oddsapi(20, NARROW, 100),
                *oddsapi(3, later_prices, 101),
                *oddsapi(-2, (3.0, 3.0, 3.0), 102),
            ],
        )
        assert found.status is cmp.Status.COMPARED
        assert found.age_minutes == pytest.approx(3)
        assert found.oddsapi is not None and found.fduk is not None
        expected = [(1 / p) / sum(1 / q for q in later_prices) for p in later_prices]
        assert list(found.oddsapi.fair.values()) == pytest.approx(expected)
        assert found.fduk.time == KICKOFF

    def test_a_snapshot_at_kickoff_counts_with_age_zero(self) -> None:
        assert compare(fduk(NARROW), oddsapi(0, NARROW)).age_minutes == 0

    def test_a_wide_latest_snapshot_falls_back_to_the_latest_narrow_one(self) -> None:
        found = compare(fduk(NARROW), [*oddsapi(20, NARROW, 100), *oddsapi(3, WIDE, 101)])
        assert found.status is cmp.Status.COMPARED and found.age_minutes == pytest.approx(20)

    def test_an_incomplete_snapshot_is_not_a_price(self) -> None:
        partial = [q for q in oddsapi(3, NARROW) if q.outcome is not Outcome.DRAW]
        assert compare(fduk(NARROW), partial).status is cmp.Status.SNAPSHOT_UNUSABLE

    def test_only_wide_snapshots_are_unusable_not_missing(self) -> None:
        assert compare(fduk(NARROW), oddsapi(3, WIDE)).status is cmp.Status.SNAPSHOT_UNUSABLE

    def test_no_snapshot_at_or_before_kickoff_is_missing(self) -> None:
        assert compare(fduk(NARROW), oddsapi(-1, NARROW)).status is cmp.Status.NO_SNAPSHOT
        assert compare(fduk(NARROW), []).status is cmp.Status.NO_SNAPSHOT

    def test_max_age_leaves_out_older_snapshots(self) -> None:
        old = oddsapi(20, NARROW)
        assert compare(fduk(NARROW), old, timedelta(minutes=10)).status is cmp.Status.NO_SNAPSHOT
        assert compare(fduk(NARROW), old, timedelta(minutes=20)).status is cmp.Status.COMPARED

    def test_without_an_fduk_close_there_is_no_comparison(self) -> None:
        found = compare([], oddsapi(3, NARROW))
        assert found.status is cmp.Status.NO_FDUK_CLOSE and found.oddsapi is None

    def test_a_wide_fduk_close_is_unusable(self) -> None:
        assert compare(fduk(WIDE), oddsapi(3, NARROW)).status is cmp.Status.FDUK_UNUSABLE


class TestStatistics:
    def test_stat_is_the_mean_and_the_sample_sd(self) -> None:
        s = cmp.stat([1.0, 2.0, 4.0])
        assert (s.n, s.mean_pp) == (3, pytest.approx(7 / 3))
        assert s.sd_pp == pytest.approx(float(np.std([1.0, 2.0, 4.0], ddof=1)))

    def test_stat_of_few_values_has_no_sd(self) -> None:
        assert cmp.stat([]) == cmp.Stat(0, None, None)
        assert cmp.stat([1.5]) == cmp.Stat(1, 1.5, None)

    @pytest.mark.parametrize(
        ("mean", "sd", "verdict"),
        [
            (0.3, 1.0, cmp.Verdict.PASS),
            (-0.3, 1.0, cmp.Verdict.PASS),
            (0.31, 0.5, cmp.Verdict.FAIL),
            (0.0, 1.01, cmp.Verdict.FAIL),
        ],
    )
    def test_an_outcome_passes_within_both_thresholds(
        self, mean: float, sd: float, verdict: cmp.Verdict
    ) -> None:
        assert cmp.outcome_verdict(cmp.Stat(500, mean, sd)) is verdict

    def test_an_outcome_without_an_sd_is_inconclusive(self) -> None:
        assert cmp.outcome_verdict(cmp.Stat(1, 0.0, None)) is cmp.Verdict.INCONCLUSIVE

    @pytest.mark.parametrize(
        ("n", "mean", "verdict"),
        [
            (50, 0.5, cmp.Verdict.PASS),
            (50, -0.51, cmp.Verdict.FAIL),
            (49, 0.0, cmp.Verdict.INCONCLUSIVE),
            (49, 5.0, cmp.Verdict.INCONCLUSIVE),
            (0, None, cmp.Verdict.INCONCLUSIVE),
        ],
    )
    def test_a_band_needs_fifty_rows_and_a_mean_within_half_a_point(
        self, n: int, mean: float | None, verdict: cmp.Verdict
    ) -> None:
        assert cmp.band_verdict(cmp.Stat(n, mean, 0.5)) is verdict

    @pytest.mark.parametrize(
        ("verdicts", "expected"),
        [
            ([cmp.Verdict.PASS, cmp.Verdict.PASS], cmp.Verdict.PASS),
            ([cmp.Verdict.PASS, cmp.Verdict.INCONCLUSIVE], cmp.Verdict.INCONCLUSIVE),
            ([cmp.Verdict.FAIL, cmp.Verdict.INCONCLUSIVE], cmp.Verdict.FAIL),
            ([], cmp.Verdict.INCONCLUSIVE),
        ],
    )
    def test_overall_fails_on_any_failure_and_passes_only_when_all_pass(
        self, verdicts: list[cmp.Verdict], expected: cmp.Verdict
    ) -> None:
        assert cmp.overall_verdict(verdicts) is expected

    @pytest.mark.parametrize(
        ("p", "band"),
        [
            (0.0, "<0.25"),
            (0.2499, "<0.25"),
            (0.25, "0.25-0.5"),
            (0.4999, "0.25-0.5"),
            (0.5, ">=0.5"),
        ],
    )
    def test_bands_are_by_the_fduk_probability(self, p: float, band: str) -> None:
        assert cmp.band_of(p) == band


class TestReport:
    FDUK_FAIR = (0.5, 0.3, 0.2)

    def report(self, comparisons: list[cmp.EventComparison]) -> dict[str, object]:
        return cmp.build_report(comparisons, {"sport": "soccer_epl"})

    def test_per_outcome_and_per_band_statistics_are_over_the_deltas(self) -> None:
        # Home +1 pp, draw -0.5 pp, away -0.5 pp, then home +3 pp, draw -1.5, away -1.5.
        report = self.report(
            [
                compared(1, self.FDUK_FAIR, (0.51, 0.295, 0.195), 2),
                compared(2, self.FDUK_FAIR, (0.53, 0.285, 0.185), 4),
            ]
        )
        outcomes = report["outcomes"]
        assert isinstance(outcomes, dict)
        assert outcomes["home"]["n"] == 2
        assert outcomes["home"]["mean_pp"] == pytest.approx(2.0)
        assert outcomes["home"]["sd_pp"] == pytest.approx(np.std([1.0, 3.0], ddof=1))
        assert outcomes["home"]["verdict"] == "fail"
        bands = report["bands"]
        assert isinstance(bands, dict)
        assert (bands[">=0.5"]["n"], bands["0.25-0.5"]["n"], bands["<0.25"]["n"]) == (2, 2, 2)
        assert bands["0.25-0.5"]["mean_pp"] == pytest.approx(-1.0)
        assert bands["<0.25"]["verdict"] == "inconclusive"
        assert report["verdict"] == "fail"

    def test_a_clean_sample_with_thin_bands_is_inconclusive_not_a_pass(self) -> None:
        same = [compared(i, self.FDUK_FAIR, self.FDUK_FAIR, 3) for i in range(10)]
        report = self.report(same)
        assert report["outcomes"]["home"]["verdict"] == "pass"  # type: ignore[index]
        assert report["bands"]["<0.25"]["verdict"] == "inconclusive"  # type: ignore[index]
        assert report["verdict"] == "inconclusive"

    def test_a_large_clean_sample_passes(self) -> None:
        rng = np.random.default_rng(0)
        comparisons = []
        for i in range(120):
            home = float(rng.uniform(0.15, 0.7))
            draw = float(rng.uniform(0.15, 0.3))
            fair = (home, draw, 1 - home - draw)
            noise = rng.normal(0, 0.002, 2)
            api = (home + noise[0], draw + noise[1], 1 - home - draw - noise[0] - noise[1])
            comparisons.append(compared(i, fair, api, 3))
        assert self.report(comparisons)["verdict"] == "pass"

    def test_freshness_shares_count_matches_with_a_usable_fduk_close(self) -> None:
        comparisons = [
            compared(1, self.FDUK_FAIR, self.FDUK_FAIR, 4),
            compared(2, self.FDUK_FAIR, self.FDUK_FAIR, 8),
            compared(3, self.FDUK_FAIR, self.FDUK_FAIR, 45),
            cmp.EventComparison(4, cmp.Status.NO_SNAPSHOT, fduk=price(self.FDUK_FAIR)),
            cmp.EventComparison(5, cmp.Status.SNAPSHOT_UNUSABLE, fduk=price(self.FDUK_FAIR)),
            cmp.EventComparison(6, cmp.Status.NO_FDUK_CLOSE),
            cmp.EventComparison(7, cmp.Status.FDUK_UNUSABLE),
        ]
        report = self.report(comparisons)
        assert report["matches"] == {
            "final_events": 7,
            "with_fduk_close": 5,
            "compared": 3,
            "missing": {
                "no_fduk_close": 1,
                "fduk_unusable": 1,
                "no_oddsapi_snapshot": 1,
                "oddsapi_unusable": 1,
            },
        }
        within = report["within_minutes_of_kickoff"]
        assert isinstance(within, dict)
        assert within["5"] == {"matches": 1, "share": 1 / 5}
        assert within["10"] == {"matches": 2, "share": 2 / 5}
        assert within["60"] == {"matches": 3, "share": 3 / 5}

    def test_age_distribution_of_the_prices_used(self) -> None:
        ages = [1.0, 2.0, 3.0, 4.0, 100.0]
        report = self.report(
            [compared(i, self.FDUK_FAIR, self.FDUK_FAIR, a) for i, a in enumerate(ages)]
        )
        age = report["age_minutes"]
        assert isinstance(age, dict)
        assert age["n"] == 5 and age["min"] == 1.0 and age["max"] == 100.0
        assert age["p50"] == 3.0 and age["mean"] == pytest.approx(22.0)

    def test_an_empty_range_reports_without_failing_and_is_json(self) -> None:
        report = self.report([])
        assert report["verdict"] == "inconclusive"
        assert report["age_minutes"] == {"n": 0}
        assert "Verdict: INCONCLUSIVE" in cmp.render(report)
        json.dumps(report)

    def test_render_shows_the_tables(self) -> None:
        text = cmp.render(self.report([compared(1, self.FDUK_FAIR, (0.51, 0.295, 0.195), 2)]))
        for expected in ("home", "draw", "away", "<0.25", "0.25-0.5", ">=0.5", "within"):
            assert expected in text
