"""Injury report feature extraction for CLV prediction.

Provides an InjuryFeatures dataclass and extractor that converts NBA injury
report data into ML-ready features. Impact features weight players by their
on/off net rating; timing features capture information recency.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime

import numpy as np
from odds_core.injury_models import InjuryReport, InjuryStatus
from odds_core.models import Event
from odds_core.player_stats_models import NbaPlayerSeasonStats
from odds_lambda.polymarket_matching import normalize_team

__all__ = [
    "InjuryFeatures",
    "extract_injury_features",
    "date_to_nba_season",
]

_GTD_STATUSES = frozenset({InjuryStatus.QUESTIONABLE, InjuryStatus.DOUBTFUL})
_GTD_DISCOUNT = 0.5


@dataclass
class InjuryFeatures:
    """Point-in-time injury features for a single event.

    Impact features are derived from the latest injury report snapshot
    filed before the decision time, weighted by player on/off net rating.
    All fields optional (None -> np.nan).
    """

    # Impact-weighted sum of players ruled OUT per team
    impact_out_home: float | None = None
    impact_out_away: float | None = None

    # Impact-weighted sum of GTD (QUESTIONABLE + DOUBTFUL) per team, discounted 0.5x
    impact_gtd_home: float | None = None
    impact_gtd_away: float | None = None

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


def date_to_nba_season(dt: datetime) -> str:
    """Derive NBA season string from a datetime.

    NBA regular season runs Oct-June. A game in Jan 2025 -> '2024-25'.
    A game in Oct 2024 -> '2024-25'.
    """
    year = dt.year
    if dt.month >= 10:
        return f"{year}-{str(year + 1)[-2:]}"
    return f"{year - 1}-{str(year)[-2:]}"


def _compute_player_impact(stats: NbaPlayerSeasonStats | None) -> float:
    """Compute impact score for a player, or 1.0 if stats unavailable.

    Formula: (on_off_rtg - on_def_rtg) * (minutes_per_game / 48).
    Falls back to 1.0 (preserving headcount behavior) when stats are
    missing, ratings are None, or games_played is zero.
    """
    if stats is None:
        return 1.0
    if stats.on_off_rtg is None or stats.on_def_rtg is None:
        return 1.0
    if stats.games_played == 0:
        return 1.0
    net_rtg = stats.on_off_rtg - stats.on_def_rtg
    minutes_per_game = stats.minutes / stats.games_played
    return net_rtg * (minutes_per_game / 48.0)


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


def extract_injury_features(
    reports: list[InjuryReport],
    event: Event,
    snapshot_time: datetime,
    player_stats: dict[str, NbaPlayerSeasonStats] | None = None,
) -> InjuryFeatures:
    """Extract injury features from reports available at snapshot_time.

    Filters reports to those with report_time <= snapshot_time (look-ahead
    bias prevention), then uses the latest report snapshot for impact-weighted
    accumulation. Each player's impact is computed from their on/off net rating
    and minutes; GTD players are discounted by 0.5x.

    When player_stats is None or a player is not found, the player contributes
    1.0 (equivalent to old headcount behavior).

    Args:
        reports: All injury reports for this event (any time).
        event: Sportsbook event (provides home/away teams and commence_time).
        snapshot_time: Decision point time — only reports at or before this
            time are visible.
        player_stats: Optional dict of player name -> season stats for
            impact weighting. When None, falls back to count-based behavior.

    Returns:
        InjuryFeatures with impact-weighted sums and timing, or all-None
        if no reports are available before snapshot_time.
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

    # Accumulate impact-weighted sums
    impact_out_home = 0.0
    impact_out_away = 0.0
    impact_gtd_home = 0.0
    impact_gtd_away = 0.0

    for report in latest_reports:
        side = _classify_team(report, home_canonical, away_canonical)
        if side is None:
            continue

        stats = player_stats.get(report.player_name) if player_stats else None
        impact = _compute_player_impact(stats)

        if report.status == InjuryStatus.OUT:
            if side == "home":
                impact_out_home += impact
            else:
                impact_out_away += impact
        elif report.status in _GTD_STATUSES:
            if side == "home":
                impact_gtd_home += impact * _GTD_DISCOUNT
            else:
                impact_gtd_away += impact * _GTD_DISCOUNT

    # Timing features
    report_hours_before_game = (event.commence_time - latest_time).total_seconds() / 3600
    injury_news_recency = (snapshot_time - latest_time).total_seconds() / 3600

    return InjuryFeatures(
        impact_out_home=float(impact_out_home),
        impact_out_away=float(impact_out_away),
        impact_gtd_home=float(impact_gtd_home),
        impact_gtd_away=float(impact_gtd_away),
        report_hours_before_game=report_hours_before_game,
        injury_news_recency=injury_news_recency,
    )
