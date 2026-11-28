"""The close-agreement statistics, verdicts and bands pre-set on issue 730."""

from __future__ import annotations

import numpy as np
import pytest
from odds_analytics import close_agreement as ca


class TestStatistics:
    def test_stat_is_the_mean_and_the_sample_sd(self) -> None:
        s = ca.stat([1.0, 2.0, 4.0])
        assert (s.n, s.mean_pp) == (3, pytest.approx(7 / 3))
        assert s.sd_pp == pytest.approx(float(np.std([1.0, 2.0, 4.0], ddof=1)))

    def test_stat_of_few_values_has_no_sd(self) -> None:
        assert ca.stat([]) == ca.Stat(0, None, None)
        assert ca.stat([1.5]) == ca.Stat(1, 1.5, None)

    @pytest.mark.parametrize(
        ("mean", "sd", "verdict"),
        [
            (0.3, 1.0, ca.Verdict.PASS),
            (-0.3, 1.0, ca.Verdict.PASS),
            (0.31, 0.5, ca.Verdict.FAIL),
            (0.0, 1.01, ca.Verdict.FAIL),
        ],
    )
    def test_an_outcome_passes_within_both_thresholds(
        self, mean: float, sd: float, verdict: ca.Verdict
    ) -> None:
        assert ca.outcome_verdict(ca.Stat(500, mean, sd)) is verdict

    def test_an_outcome_without_an_sd_is_inconclusive(self) -> None:
        assert ca.outcome_verdict(ca.Stat(1, 0.0, None)) is ca.Verdict.INCONCLUSIVE

    @pytest.mark.parametrize(
        ("n", "mean", "verdict"),
        [
            (50, 0.5, ca.Verdict.PASS),
            (50, -0.51, ca.Verdict.FAIL),
            (49, 0.0, ca.Verdict.INCONCLUSIVE),
            (49, 5.0, ca.Verdict.INCONCLUSIVE),
            (0, None, ca.Verdict.INCONCLUSIVE),
        ],
    )
    def test_a_band_needs_fifty_rows_and_a_mean_within_half_a_point(
        self, n: int, mean: float | None, verdict: ca.Verdict
    ) -> None:
        assert ca.band_verdict(ca.Stat(n, mean, 0.5)) is verdict

    @pytest.mark.parametrize(
        ("verdicts", "expected"),
        [
            ([ca.Verdict.PASS, ca.Verdict.PASS], ca.Verdict.PASS),
            ([ca.Verdict.PASS, ca.Verdict.INCONCLUSIVE], ca.Verdict.INCONCLUSIVE),
            ([ca.Verdict.FAIL, ca.Verdict.INCONCLUSIVE], ca.Verdict.FAIL),
            ([], ca.Verdict.INCONCLUSIVE),
        ],
    )
    def test_overall_fails_on_any_failure_and_passes_only_when_all_pass(
        self, verdicts: list[ca.Verdict], expected: ca.Verdict
    ) -> None:
        assert ca.overall_verdict(verdicts) is expected

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
    def test_bands_are_by_the_reference_probability(self, p: float, band: str) -> None:
        assert ca.band_of(p) == band
