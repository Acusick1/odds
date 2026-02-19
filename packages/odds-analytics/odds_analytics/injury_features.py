"""Injury report feature extraction for CLV prediction.

Provides an InjuryFeatures dataclass and extractor that converts NBA injury
report data into ML-ready features. Count features capture the injury
burden per team; timing features capture information recency.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime

import numpy as np
from odds_core.injury_models import InjuryReport, InjuryStatus
from odds_core.models import Event
from odds_lambda.polymarket_matching import normalize_team

__all__ = [
    "InjuryFeatures",
    "extract_injury_features",
]


@dataclass
class InjuryFeatures:
    """Point-in-time injury features for a single event.

    Count features are derived from the latest injury report snapshot
    filed before the decision time. All fields optional (None -> np.nan).
    """

    # Count of players ruled OUT per team
    num_out_home: float | None = None
    num_out_away: float | None = None

    # Count of game-time decisions (QUESTIONABLE + DOUBTFUL) per team
    num_gtd_home: float | None = None
    num_gtd_away: float | None = None

    # Hours between latest report_time and game commence_time
    report_hours_before_game: float | None = None

    # Hours between latest report_time and snapshot_time (staleness)
    injury_news_recency: float | None = None

    def to_array(self) -> np.ndarray:
        """Convert to numpy array. None -> np.nan."""
        return np.array(
            [
                getattr(self, f.name) if getattr(self, f.name) is not None else np.nan
                for f in fields(self)
            ],
            dtype=np.float64,
        )

    @classmethod
    def get_feature_names(cls) -> list[str]:
        return [f.name for f in fields(cls)]


def _classify_team(
    report: InjuryReport,
    home_team_canonical: str | None,
    away_team_canonical: str | None,
) -> str | None:
    """Classify an injury report as 'home', 'away', or None (unmatched)."""
    report_canonical = normalize_team(report.team)
    if report_canonical is None:
        return None
    if report_canonical == home_team_canonical:
        return "home"
    if report_canonical == away_team_canonical:
        return "away"
    return None


_GTD_STATUSES = frozenset({InjuryStatus.QUESTIONABLE, InjuryStatus.DOUBTFUL})


def extract_injury_features(
    reports: list[InjuryReport],
    event: Event,
    snapshot_time: datetime,
) -> InjuryFeatures:
    """Extract injury features from reports available at snapshot_time.

    Filters reports to those with report_time <= snapshot_time (look-ahead
    bias prevention), then uses the latest report snapshot for counts.

    Args:
        reports: All injury reports for this event (any time).
        event: Sportsbook event (provides home/away teams and commence_time).
        snapshot_time: Decision point time — only reports at or before this
            time are visible.

    Returns:
        InjuryFeatures with counts and timing, or all-None if no reports
        are available before snapshot_time.
    """
    # Filter to reports at or before snapshot_time
    eligible = [r for r in reports if r.report_time <= snapshot_time]
    if not eligible:
        return InjuryFeatures()

    # Find latest report_time
    latest_time = max(r.report_time for r in eligible)
    latest_reports = [r for r in eligible if r.report_time == latest_time]

    # Normalize event team names
    home_canonical = normalize_team(event.home_team)
    away_canonical = normalize_team(event.away_team)

    # Count by status and team
    out_home = 0
    out_away = 0
    gtd_home = 0
    gtd_away = 0

    for report in latest_reports:
        side = _classify_team(report, home_canonical, away_canonical)
        if side is None:
            continue

        if report.status == InjuryStatus.OUT:
            if side == "home":
                out_home += 1
            else:
                out_away += 1
        elif report.status in _GTD_STATUSES:
            if side == "home":
                gtd_home += 1
            else:
                gtd_away += 1

    # Timing features
    report_hours_before_game = (event.commence_time - latest_time).total_seconds() / 3600
    injury_news_recency = (snapshot_time - latest_time).total_seconds() / 3600

    return InjuryFeatures(
        num_out_home=float(out_home),
        num_out_away=float(out_away),
        num_gtd_home=float(gtd_home),
        num_gtd_away=float(gtd_away),
        report_hours_before_game=report_hours_before_game,
        injury_news_recency=injury_news_recency,
    )
