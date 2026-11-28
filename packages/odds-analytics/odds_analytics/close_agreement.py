"""Whether a candidate close agrees with a reference close: the statistics of
``dp = p_candidate - p_reference`` and the pass thresholds pre-set on issue 730."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

# Pre-set on issue 730, in percentage points.
OUTCOME_MAX_ABS_MEAN_PP = 0.3
OUTCOME_MAX_SD_PP = 1.0
BAND_MAX_ABS_MEAN_PP = 0.5
# A band with fewer rows is inconclusive.
MIN_BAND_ROWS = 50
# Bands of the reference probability: below the first edge, up to the second, and from it.
BAND_EDGES: tuple[float, ...] = (0.25, 0.5)
BAND_LABELS: tuple[str, ...] = ("<0.25", "0.25-0.5", ">=0.5")
# Minutes before kick-off that the candidate price must lie within, for the share reported.
FRESHNESS_MINUTES: tuple[int, ...] = (5, 10, 30, 60)
AGE_PERCENTILES: tuple[int, ...] = (25, 50, 75, 90)

THRESHOLDS: dict[str, float] = {
    "outcome_max_abs_mean_pp": OUTCOME_MAX_ABS_MEAN_PP,
    "outcome_max_sd_pp": OUTCOME_MAX_SD_PP,
    "band_max_abs_mean_pp": BAND_MAX_ABS_MEAN_PP,
    "min_band_rows": MIN_BAND_ROWS,
}


class Verdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class Stat:
    n: int
    mean_pp: float | None
    sd_pp: float | None


def stat(values_pp: Sequence[float]) -> Stat:
    n = len(values_pp)
    if n == 0:
        return Stat(0, None, None)
    return Stat(n, float(np.mean(values_pp)), float(np.std(values_pp, ddof=1)) if n > 1 else None)


def outcome_verdict(s: Stat) -> Verdict:
    if s.mean_pp is None or s.sd_pp is None:
        return Verdict.INCONCLUSIVE
    ok = abs(s.mean_pp) <= OUTCOME_MAX_ABS_MEAN_PP and s.sd_pp <= OUTCOME_MAX_SD_PP
    return Verdict.PASS if ok else Verdict.FAIL


def band_verdict(s: Stat) -> Verdict:
    if s.mean_pp is None or s.n < MIN_BAND_ROWS:
        return Verdict.INCONCLUSIVE
    return Verdict.PASS if abs(s.mean_pp) <= BAND_MAX_ABS_MEAN_PP else Verdict.FAIL


def overall_verdict(verdicts: Iterable[Verdict]) -> Verdict:
    """Fail if any slice fails; pass only when every slice passes; otherwise inconclusive."""
    seen = set(verdicts)
    if Verdict.FAIL in seen:
        return Verdict.FAIL
    if Verdict.INCONCLUSIVE in seen or not seen:
        return Verdict.INCONCLUSIVE
    return Verdict.PASS


def band_of(p_reference: float) -> str:
    return BAND_LABELS[bisect_right(BAND_EDGES, p_reference)]


@dataclass(frozen=True)
class Delta:
    """One outcome of one compared event: its reference and candidate probabilities."""

    event_id: int
    outcome: str
    p_reference: float
    p_candidate: float

    @property
    def dp_pp(self) -> float:
        return (self.p_candidate - self.p_reference) * 100


def agreement(rows: Sequence[Delta], outcomes: Sequence[str]) -> dict[str, Any]:
    """The overall verdict and, per outcome and per band of the reference probability, n, mean
    and sd in pp with a verdict."""

    def of(selected: Iterable[Delta]) -> Stat:
        return stat([d.dp_pp for d in selected])

    by_outcome = {str(o): of(d for d in rows if d.outcome == o) for o in outcomes}
    by_band = {b: of(d for d in rows if band_of(d.p_reference) == b) for b in BAND_LABELS}
    outcome_verdicts = {k: outcome_verdict(s) for k, s in by_outcome.items()}
    band_verdicts = {k: band_verdict(s) for k, s in by_band.items()}

    def table(stats: dict[str, Stat], verdicts: dict[str, Verdict]) -> dict[str, Any]:
        return {
            k: {"n": s.n, "mean_pp": s.mean_pp, "sd_pp": s.sd_pp, "verdict": verdicts[k]}
            for k, s in stats.items()
        }

    return {
        "verdict": overall_verdict([*outcome_verdicts.values(), *band_verdicts.values()]),
        "outcomes": table(by_outcome, outcome_verdicts),
        "bands": table(by_band, band_verdicts),
    }


def freshness(ages_minutes: Sequence[float], with_reference: int) -> dict[str, Any]:
    """The share of the ``with_reference`` events whose candidate price is within each of
    ``FRESHNESS_MINUTES`` before kick-off, and the age distribution of the prices used."""
    within = {
        str(m): {
            "matches": sum(a <= m for a in ages_minutes),
            "share": sum(a <= m for a in ages_minutes) / with_reference if with_reference else None,
        }
        for m in FRESHNESS_MINUTES
    }
    ages = (
        {
            "n": len(ages_minutes),
            "min": float(np.min(ages_minutes)),
            **{f"p{p}": float(np.percentile(ages_minutes, p)) for p in AGE_PERCENTILES},
            "max": float(np.max(ages_minutes)),
            "mean": float(np.mean(ages_minutes)),
        }
        if ages_minutes
        else {"n": 0}
    )
    return {"within_minutes_of_kickoff": within, "age_minutes": ages}


def _fmt(value: float | None, spec: str = "+.2f") -> str:
    return "-" if value is None else format(value, spec)


def render_agreement(report: dict[str, Any], band_title: str) -> list[str]:
    """The outcome and band tables of an ``agreement`` report, as text lines."""
    lines: list[str] = []
    for title, key in (("Outcome", "outcomes"), (band_title, "bands")):
        lines.append(f"{title:<28}{'n':>7}{'mean pp':>10}{'sd pp':>8}  verdict")
        for name, row in report[key].items():
            lines.append(
                f"{name:<28}{row['n']:>7}{_fmt(row['mean_pp']):>10}"
                f"{_fmt(row['sd_pp'], '.2f'):>8}  {row['verdict']}"
            )
        lines.append("")
    return lines


def render_freshness(report: dict[str, Any], reference: str, candidate: str) -> list[str]:
    """The freshness shares and age distribution of a ``freshness`` report, as text lines."""
    lines = [f"Share of matches with {reference} whose {candidate} price is within"]
    for minutes, row in report["within_minutes_of_kickoff"].items():
        share = "-" if row["share"] is None else f"{row['share']:.1%}"
        lines.append(f"  {minutes:>3} min before kick-off: {share} ({row['matches']})")
    age = report["age_minutes"]
    lines.append(
        f"Age of the {candidate} prices used (minutes before kick-off): "
        + ", ".join(f"{k} {v:.1f}" if k != "n" else f"n {v}" for k, v in age.items())
    )
    return lines
