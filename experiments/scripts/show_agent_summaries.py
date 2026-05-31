"""Print agent run `.result` summaries from logs/agent_runs/ JSONL traces.

Usage examples:
    uv run python experiments/scripts/show_agent_summaries.py
    uv run python experiments/scripts/show_agent_summaries.py --sport epl --since 24h
    uv run python experiments/scripts/show_agent_summaries.py --sport mlb --limit 3
    uv run python experiments/scripts/show_agent_summaries.py --file logs/agent_runs/soccer_epl_20260424T101044Z_1259597.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parents[2] / "logs" / "agent_runs"

SPORT_KEYS = {
    "epl": "soccer_epl",
    "mlb": "baseball_mlb",
}

FILENAME_RE = re.compile(r"^(?P<sport>[a-z_]+)_(?P<ts>\d{8}T\d{6}Z)_(?P<pid>\d+)\.jsonl$")


@dataclass
class SessionSummary:
    path: Path
    sport: str
    file_start: datetime
    session_idx: int  # 0-based position within the file
    duration_ms: int | None
    num_turns: int | None
    total_cost_usd: float | None
    subtype: str | None
    is_error: bool | None
    result: str


def parse_filename(path: Path) -> tuple[str, datetime] | None:
    m = FILENAME_RE.match(path.name)
    if not m:
        return None
    sport = m.group("sport")
    ts = datetime.strptime(m.group("ts"), "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    return sport, ts


def parse_since(raw: str) -> datetime:
    """Accepts `24h`, `7d`, or an ISO date/datetime. Returns an aware UTC datetime."""
    raw = raw.strip()
    m = re.fullmatch(r"(\d+)([hd])", raw)
    if m:
        value, unit = int(m.group(1)), m.group(2)
        delta = timedelta(hours=value) if unit == "h" else timedelta(days=value)
        return datetime.now(UTC) - delta
    # ISO date or datetime
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError as e:
        raise SystemExit(f"Could not parse --since {raw!r}: {e}") from e
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def iter_result_entries(path: Path):
    with path.open() as fh:
        for idx, line in enumerate(fh):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("type") == "result":
                yield idx, entry


def load_summaries(
    sport_filter: str | None,
    since: datetime | None,
    log_dir: Path,
    explicit_file: Path | None,
) -> list[SessionSummary]:
    if explicit_file is not None:
        files = [explicit_file]
    else:
        files = sorted(log_dir.glob("*.jsonl"))

    summaries: list[SessionSummary] = []
    for path in files:
        parsed = parse_filename(path)
        if parsed is None:
            continue
        sport, file_start = parsed
        if sport_filter is not None and sport != sport_filter:
            continue
        if since is not None and file_start < since:
            # Filename = first session start. Later sessions in a multi-session
            # file could still fall after `since`; walk the file anyway.
            pass

        session_idx = 0
        for _line_no, entry in iter_result_entries(path):
            result_text = entry.get("result") or ""
            if not result_text:
                session_idx += 1
                continue
            # Time-filter by file start when no per-session timestamp is available.
            if since is not None and file_start < since:
                session_idx += 1
                continue
            summaries.append(
                SessionSummary(
                    path=path,
                    sport=sport,
                    file_start=file_start,
                    session_idx=session_idx,
                    duration_ms=entry.get("duration_ms"),
                    num_turns=entry.get("num_turns"),
                    total_cost_usd=entry.get("total_cost_usd"),
                    subtype=entry.get("subtype"),
                    is_error=entry.get("is_error"),
                    result=result_text,
                )
            )
            session_idx += 1

    summaries.sort(key=lambda s: (s.file_start, s.session_idx), reverse=True)
    return summaries


def fmt_duration(ms: int | None) -> str:
    if ms is None:
        return "?"
    seconds = ms // 1000
    m, s = divmod(seconds, 60)
    return f"{m}m{s:02d}s"


def fmt_cost(cost: float | None) -> str:
    return f"${cost:.2f}" if cost is not None else "?"


def print_summary(s: SessionSummary) -> None:
    tag = f" session#{s.session_idx}" if s.session_idx > 0 else ""
    header = (
        f"[{s.file_start.strftime('%Y-%m-%d %H:%M UTC')}] {s.sport}{tag}  "
        f"turns={s.num_turns}  {fmt_duration(s.duration_ms)}  {fmt_cost(s.total_cost_usd)}"
    )
    if s.is_error:
        header += "  [ERROR]"
    elif s.subtype and s.subtype != "success":
        header += f"  [{s.subtype}]"
    print(header)
    print(f"  file: {s.path.name}")
    body = s.result
    indented = "\n".join(f"  {line}" for line in body.splitlines())
    print(indented)
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sport", choices=sorted(SPORT_KEYS), help="Filter by sport")
    parser.add_argument(
        "--since",
        help="Time filter: `24h`, `7d`, or ISO date (e.g. 2026-04-20).",
    )
    parser.add_argument("--limit", type=int, default=10, help="Max sessions to show (default 10).")
    parser.add_argument("--full", action="store_true", help="Print the full result text.")
    parser.add_argument("--file", type=Path, help="Target a specific JSONL file.")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=LOG_DIR,
        help=f"Agent-run log directory (default {LOG_DIR}).",
    )
    args = parser.parse_args(argv)

    sport_filter = SPORT_KEYS.get(args.sport) if args.sport else None
    since = parse_since(args.since) if args.since else None

    summaries = load_summaries(
        sport_filter=sport_filter,
        since=since,
        log_dir=args.log_dir,
        explicit_file=args.file,
    )

    if not summaries:
        print("No sessions matched.", file=sys.stderr)
        return 1

    total = len(summaries)
    shown = min(total, args.limit)
    if total > shown:
        print(f"Showing last {shown} of {total} — raise --limit to see more\n")

    for s in reversed(summaries[: args.limit]):
        print_summary(s)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
