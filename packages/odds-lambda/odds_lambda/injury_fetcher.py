"""Wrapper around nbainjuries package for fetching NBA injury report data.

All nbainjuries imports are lazy (inside functions) because the package
starts a JVM at import time via jpype. This prevents JVM startup cost
when importing this module for unrelated code paths.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime

import structlog
from odds_core.injury_models import InjuryStatus
from odds_core.time import EASTERN, ensure_utc

logger = structlog.get_logger(__name__)

STATUS_MAP: dict[str, InjuryStatus] = {
    "out": InjuryStatus.OUT,
    "questionable": InjuryStatus.QUESTIONABLE,
    "doubtful": InjuryStatus.DOUBTFUL,
    "probable": InjuryStatus.PROBABLE,
    "available": InjuryStatus.AVAILABLE,
}


@dataclass(slots=True)
class InjuryRecord:
    """Parsed injury record ready for database storage."""

    report_time: datetime  # UTC
    game_date: date
    game_time_et: str
    matchup: str
    team: str
    player_name: str
    status: InjuryStatus
    reason: str


def _to_naive_et(target_utc: datetime) -> datetime:
    """Convert a UTC datetime to naive Eastern Time for nbainjuries API."""
    aware_et = ensure_utc(target_utc).astimezone(EASTERN)
    return aware_et.replace(tzinfo=None)


def _parse_records(raw_json: str, report_time_utc: datetime) -> list[InjuryRecord]:
    """Parse nbainjuries JSON output into InjuryRecord instances.

    Args:
        raw_json: JSON string from get_reportdata (orient='records' format).
        report_time_utc: UTC timestamp of the report snapshot.

    Returns:
        Parsed records with NOT YET SUBMITTED and unknown statuses filtered out.
    """
    rows: list[dict] = json.loads(raw_json)
    records: list[InjuryRecord] = []

    for row in rows:
        reason = str(row.get("Reason", "") or "").strip()

        # Filter NOT YET SUBMITTED entries (teams that haven't filed yet)
        if reason.casefold() == "not yet submitted":
            continue

        status_str = str(row.get("Current Status", "") or "").strip().lower()
        mapped_status = STATUS_MAP.get(status_str)
        if mapped_status is None:
            logger.warning(
                "injury_unknown_status",
                status=row.get("Current Status"),
                player=row.get("Player Name"),
            )
            continue

        # Parse game date from MM/DD/YYYY
        game_date_str = str(row.get("Game Date", "") or "").strip()
        try:
            game_date = datetime.strptime(game_date_str, "%m/%d/%Y").date()
        except ValueError:
            logger.warning("injury_invalid_game_date", date_str=game_date_str)
            continue

        player_name = str(row.get("Player Name", "") or "").strip()
        if not player_name:
            continue

        records.append(
            InjuryRecord(
                report_time=report_time_utc,
                game_date=game_date,
                game_time_et=str(row.get("Game Time", "") or "").strip(),
                matchup=str(row.get("Matchup", "") or "").strip(),
                team=str(row.get("Team", "") or "").strip(),
                player_name=player_name,
                status=mapped_status,
                reason=reason,
            )
        )

    return records


def fetch_injury_report(target_time_utc: datetime) -> list[InjuryRecord]:
    """Fetch injury report for a specific UTC time (synchronous).

    Converts UTC to naive ET for the nbainjuries API, then parses the
    result back into UTC-timestamped records.

    Args:
        target_time_utc: UTC timestamp for the desired report snapshot.

    Returns:
        Parsed InjuryRecord instances. Empty list if report unavailable.
    """
    from nbainjuries import injury  # Lazy import — triggers JVM start

    naive_et = _to_naive_et(target_time_utc)

    if not injury.check_reportvalid(naive_et):
        logger.info("injury_report_not_available", target_et=str(naive_et))
        return []

    raw_json = injury.get_reportdata(naive_et)
    if not raw_json:
        logger.info("injury_report_empty", target_et=str(naive_et))
        return []

    report_time_utc = ensure_utc(naive_et.replace(tzinfo=EASTERN))

    records = _parse_records(raw_json, report_time_utc)
    logger.info(
        "injury_report_fetched",
        count=len(records),
        target_time=str(target_time_utc),
    )
    return records
