"""Wrapper around nbainjuries package for fetching NBA injury report data.

All nbainjuries imports are lazy (inside functions) because the package
starts a JVM at import time via jpype. This prevents JVM startup cost
when importing this module for unrelated code paths.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import structlog
from odds_core.injury_models import InjuryStatus
from odds_core.time import EASTERN, ensure_utc, parse_date

logger = structlog.get_logger(__name__)

STATUS_MAP: dict[str, InjuryStatus] = {
    "out": InjuryStatus.OUT,
    "questionable": InjuryStatus.QUESTIONABLE,
    "doubtful": InjuryStatus.DOUBTFUL,
    "probable": InjuryStatus.PROBABLE,
    "available": InjuryStatus.AVAILABLE,
}


# Reports are published every 15 minutes (hourly before 2025-12-22).
REPORT_SLOT_MINUTES = 15


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


def _ensure_java_home() -> None:
    """Point JPype at the bundled JRE unless the host configures a JVM."""
    if "JAVA_HOME" in os.environ:
        return
    import jdk4py

    os.environ["JAVA_HOME"] = str(jdk4py.JAVA_HOME)


def latest_report_slot(now_utc: datetime) -> datetime:
    """The latest quarter-hour report slot at or before ``now_utc`` (UTC)."""
    now = ensure_utc(now_utc)
    return now.replace(
        minute=now.minute // REPORT_SLOT_MINUTES * REPORT_SLOT_MINUTES, second=0, microsecond=0
    )


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
            game_date = parse_date(game_date_str, "%m/%d/%Y")
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
    _ensure_java_home()
    from nbainjuries import injury  # Lazy import — triggers JVM start

    naive_et = _to_naive_et(target_time_utc)

    if not injury.check_reportvalid(naive_et):
        logger.info("injury_report_not_available", target_et=str(naive_et))
        return []

    raw_json = injury.get_reportdata(naive_et)
    if not raw_json:
        logger.info("injury_report_empty", target_et=str(naive_et))
        return []

    # Not re-derived from the naive ET, which is ambiguous at the 1 am repeated on fall-back day.
    report_time_utc = ensure_utc(target_time_utc)

    records = _parse_records(raw_json, report_time_utc)
    logger.info(
        "injury_report_fetched",
        count=len(records),
        target_time=str(target_time_utc),
    )
    return records


def fetch_latest_injury_report(
    now_utc: datetime, *, max_slots_back: int
) -> tuple[datetime, list[InjuryRecord]] | None:
    """The newest published report at or before ``now_utc`` (synchronous).

    A slot is published a few minutes after its time, so the latest slot may not exist yet;
    the search steps back one slot at a time, at most ``max_slots_back`` slots. A slot whose
    PDF fails to parse (one still uploading) is stepped over too.

    Returns:
        ``(slot_utc, records)`` of the first slot with a report, or None when none of the
        slots has one.

    Raises:
        Exception: The last parse error, when no slot had a report and a slot failed.
    """
    slot = latest_report_slot(now_utc)
    last_error: Exception | None = None
    for back in range(max_slots_back + 1):
        target = slot - timedelta(minutes=REPORT_SLOT_MINUTES * back)
        try:
            records = fetch_injury_report(target)
        except Exception as e:
            logger.warning("injury_report_unreadable", target=str(target), error=str(e))
            last_error = e
            continue
        if records:
            return target, records
    if last_error is not None:
        raise last_error
    return None


# Seven to ten seconds on the production host, almost all of it JVM start-up.
SUBPROCESS_TIMEOUT_SECONDS = 180


def _encode(slot: datetime, records: list[InjuryRecord]) -> str:
    return json.dumps(
        {
            "slot": slot.isoformat(),
            "records": [
                {
                    **asdict(r),
                    "report_time": r.report_time.isoformat(),
                    "game_date": r.game_date.isoformat(),
                    "status": r.status.value,
                }
                for r in records
            ],
        }
    )


def _decode(payload: str) -> tuple[datetime, list[InjuryRecord]]:
    data = json.loads(payload)
    records = [
        InjuryRecord(
            **{
                **r,
                "report_time": datetime.fromisoformat(r["report_time"]),
                "game_date": date.fromisoformat(r["game_date"]),
                "status": InjuryStatus(r["status"]),
            }
        )
        for r in data["records"]
    ]
    return datetime.fromisoformat(data["slot"]), records


async def fetch_latest_injury_report_isolated(
    now_utc: datetime, *, max_slots_back: int
) -> tuple[datetime, list[InjuryRecord]] | None:
    """``fetch_latest_injury_report`` in a child process.

    JPype cannot restart a JVM, and one started from an executor thread keeps the interpreter
    from exiting. A child keeps the JVM out of the long-running scheduler, frees its memory
    after each run and can be killed on a timeout.

    Raises:
        RuntimeError: If the child fails or exceeds ``SUBPROCESS_TIMEOUT_SECONDS``.
    """
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "report.json"
        child = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "odds_lambda.injury_fetcher",
            "--now",
            ensure_utc(now_utc).isoformat(),
            "--max-slots-back",
            str(max_slots_back),
            "--out",
            str(out),
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(child.communicate(), SUBPROCESS_TIMEOUT_SECONDS)
        except TimeoutError as e:
            raise RuntimeError(
                f"injury fetch exceeded {SUBPROCESS_TIMEOUT_SECONDS} s and was killed"
            ) from e
        finally:
            if child.returncode is None:
                child.kill()
                await child.wait()
        if child.returncode != 0:
            raise RuntimeError(
                f"injury fetch exited {child.returncode}: {stderr.decode(errors='replace')[-500:]}"
            )
        payload = out.read_text()
    return _decode(payload) if payload else None


def _child_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--now", required=True)
    parser.add_argument("--max-slots-back", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    found = fetch_latest_injury_report(
        datetime.fromisoformat(args.now), max_slots_back=args.max_slots_back
    )
    args.out.write_text(_encode(*found) if found else "")


if __name__ == "__main__":
    _child_main()
