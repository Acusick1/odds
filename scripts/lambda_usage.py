"""Analyze Lambda invocation patterns, costs, and failure rates.

Usage:
    uv run python scripts/lambda_usage.py                    # current month
    uv run python scripts/lambda_usage.py --days 7           # last 7 days
    uv run python scripts/lambda_usage.py --start 2026-04-01 --end 2026-04-07
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from datetime import UTC, datetime, timedelta

REGION = "eu-west-1"
LOG_GROUPS = {
    "odds-scheduler": "/aws/lambda/odds-scheduler",
    "odds-scheduler-scraper": "/aws/lambda/odds-scheduler-scraper",
}
# Memory in GB for GB-second calculation
MEMORY_GB = {
    "odds-scheduler": 0.5,
    "odds-scheduler-scraper": 2.0,
}
FREE_TIER_GB_SECONDS = 400_000


def aws_logs_query(
    log_group: str,
    start: datetime,
    end: datetime,
    filter_pattern: str,
) -> list[dict]:
    """Fetch log events from CloudWatch, handling pagination."""
    all_events: list[dict] = []
    next_token = None

    while True:
        cmd = [
            "aws",
            "logs",
            "filter-log-events",
            "--log-group-name",
            log_group,
            "--start-time",
            str(int(start.timestamp() * 1000)),
            "--end-time",
            str(int(end.timestamp() * 1000)),
            "--filter-pattern",
            filter_pattern,
            "--region",
            REGION,
            "--output",
            "json",
        ]
        if next_token:
            cmd.extend(["--next-token", next_token])

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        all_events.extend(data.get("events", []))

        next_token = data.get("nextToken")
        if not next_token:
            break

    return all_events


def parse_report(message: str) -> dict | None:
    """Extract fields from a Lambda REPORT line."""
    if "REPORT" not in message:
        return None
    try:
        rid = message.split("RequestId: ")[1].split()[0]
        dur_ms = float(message.split("Duration: ")[1].split(" ms")[0])
        billed_ms = float(message.split("Billed Duration: ")[1].split(" ms")[0])
        mem_mb = int(message.split("Memory Size: ")[1].split(" MB")[0])
        is_timeout = "status: timeout" in message.lower()
        has_init = "Init Duration:" in message
        return {
            "rid": rid,
            "dur_s": dur_ms / 1000,
            "billed_s": billed_ms / 1000,
            "mem_mb": mem_mb,
            "timeout": is_timeout,
            "cold_start": has_init,
        }
    except (IndexError, ValueError):
        return None


def analyze_function(
    function_name: str,
    log_group: str,
    start: datetime,
    end: datetime,
) -> list[dict]:
    """Fetch and parse all REPORT lines for a function."""
    events = aws_logs_query(log_group, start, end, "REPORT")
    reports = []
    for ev in events:
        parsed = parse_report(ev.get("message", ""))
        if parsed:
            parsed["timestamp"] = datetime.fromtimestamp(ev["timestamp"] / 1000, tz=UTC)
            parsed["function"] = function_name
            reports.append(parsed)
    return reports


def print_summary(reports: list[dict], start: datetime, end: datetime) -> None:
    """Print overall summary across all functions."""
    days = (end - start).days or 1
    print("=" * 80)
    print(
        f"Lambda Usage Report: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')} ({days} days)"
    )
    print("=" * 80)

    by_func = defaultdict(list)
    for r in reports:
        by_func[r["function"]].append(r)

    total_gb_seconds = 0.0

    for func in sorted(by_func):
        func_reports = by_func[func]
        mem_gb = MEMORY_GB.get(func, 0.128)
        total = len(func_reports)
        timeouts = sum(1 for r in func_reports if r["timeout"])
        successes = total - timeouts
        total_dur = sum(r["dur_s"] for r in func_reports)
        total_billed = sum(r["billed_s"] for r in func_reports)
        gb_sec = total_billed * mem_gb
        total_gb_seconds += gb_sec
        cold_starts = sum(1 for r in func_reports if r["cold_start"])

        pct_ok = (successes / total * 100) if total else 0
        print(f"\n  {func} ({mem_gb}GB, {total} invocations)")
        print(
            f"    Success: {successes} ({pct_ok:.0f}%)  Timeout: {timeouts}  Cold starts: {cold_starts}"
        )
        print(
            f"    Duration: {total_dur:,.0f}s total, {total_dur / total:.0f}s avg" if total else ""
        )
        print(f"    Billed:   {total_billed:,.0f}s  →  {gb_sec:,.0f} GB-seconds")

    projected = total_gb_seconds / days * 30
    print(f"\n  {'─' * 40}")
    print(f"  Total GB-seconds:     {total_gb_seconds:>12,.0f}")
    print(f"  Free tier:            {FREE_TIER_GB_SECONDS:>12,}")
    print(f"  Projected (30 days):  {projected:>12,.0f}")
    pct_free = total_gb_seconds / FREE_TIER_GB_SECONDS * 100
    print(f"  Free tier used:       {pct_free:>11.0f}%")


def print_daily_breakdown(reports: list[dict]) -> None:
    """Print per-day breakdown."""
    print("\n" + "=" * 80)
    print("Daily Breakdown")
    print("=" * 80)

    by_day: dict[str, list[dict]] = defaultdict(list)
    for r in reports:
        day_key = r["timestamp"].strftime("%Y-%m-%d")
        by_day[day_key].append(r)

    print(
        f"\n  {'Date':<12} {'Total':<7} {'OK':<7} {'Timeout':<9} {'OK%':<7} {'GB-sec':<10} {'Avg dur'}"
    )
    print(f"  {'─' * 12} {'─' * 6} {'─' * 6} {'─' * 8} {'─' * 6} {'─' * 9} {'─' * 7}")

    for day_key in sorted(by_day):
        day_reports = by_day[day_key]
        total = len(day_reports)
        timeouts = sum(1 for r in day_reports if r["timeout"])
        successes = total - timeouts
        pct = (successes / total * 100) if total else 0
        gb_sec = sum(r["billed_s"] * MEMORY_GB.get(r["function"], 0.128) for r in day_reports)
        avg_dur = sum(r["dur_s"] for r in day_reports) / total if total else 0
        print(
            f"  {day_key:<12} {total:<7} {successes:<7} {timeouts:<9} {pct:<6.0f}% {gb_sec:<10,.0f} {avg_dur:.0f}s"
        )


def print_retry_analysis(reports: list[dict]) -> None:
    """Analyze whether Lambda-level retries (same request ID) help."""
    print("\n" + "=" * 80)
    print("Retry Analysis (same request ID = Lambda auto-retry)")
    print("=" * 80)

    by_rid: dict[str, list[dict]] = defaultdict(list)
    for r in reports:
        by_rid[r["rid"]].append(r)

    attempt_stats: dict[int, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "timeout": 0, "success": 0}
    )
    for attempts in by_rid.values():
        # Sort by timestamp to determine attempt order
        attempts.sort(key=lambda x: x["timestamp"])
        for i, a in enumerate(attempts):
            n = i + 1
            attempt_stats[n]["total"] += 1
            if a["timeout"]:
                attempt_stats[n]["timeout"] += 1
            else:
                attempt_stats[n]["success"] += 1

    print(f"\n  {'Attempt':<10} {'Total':<8} {'OK':<8} {'Timeout':<9} {'OK%'}")
    print(f"  {'─' * 9} {'─' * 7} {'─' * 7} {'─' * 8} {'─' * 5}")
    for n in sorted(attempt_stats):
        s = attempt_stats[n]
        pct = (s["success"] / s["total"] * 100) if s["total"] else 0
        print(f"  Attempt {n:<2} {s['total']:<8} {s['success']:<8} {s['timeout']:<9} {pct:.0f}%")

    unique_jobs = len(by_rid)
    jobs_ok = sum(1 for attempts in by_rid.values() if any(not a["timeout"] for a in attempts))
    jobs_fail = unique_jobs - jobs_ok
    retry_saves = sum(
        1
        for attempts in by_rid.values()
        if len(attempts) > 1
        and attempts[0]["timeout"]
        and any(not a["timeout"] for a in attempts[1:])
    )
    wasted_gb_sec = sum(
        r["billed_s"] * MEMORY_GB.get(r["function"], 0.128)
        for attempts in by_rid.values()
        for r in attempts
        if r["timeout"]
    )

    print(f"\n  Unique jobs (request IDs): {unique_jobs}")
    print(f"  Jobs with at least 1 success: {jobs_ok}")
    print(f"  Jobs where ALL attempts timed out: {jobs_fail}")
    print(f"  Jobs rescued by retry: {retry_saves}")
    print(f"  GB-seconds wasted on timeouts: {wasted_gb_sec:,.0f}")


def print_hourly_heatmap(reports: list[dict]) -> None:
    """Print hourly success/timeout heatmap."""
    print("\n" + "=" * 80)
    print("Hourly Heatmap (S=success, T=timeout, .=no invocation)")
    print("=" * 80)

    # Collect unique days
    days = sorted({r["timestamp"].strftime("%Y-%m-%d") for r in reports})
    # Limit to 7 days for readability
    days = days[:7]

    hourly: dict[int, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"success": 0, "timeout": 0})
    )
    for r in reports:
        day_key = r["timestamp"].strftime("%Y-%m-%d")
        if day_key not in days:
            continue
        hour = r["timestamp"].hour
        if r["timeout"]:
            hourly[hour][day_key]["timeout"] += 1
        else:
            hourly[hour][day_key]["success"] += 1

    # Print header with short day labels
    day_labels = [d[5:] for d in days]  # MM-DD
    print(f"\n  {'Hour':<7}", end="")
    for label in day_labels:
        print(f"{label:<9}", end="")
    print()
    print(f"  {'─' * 6} ", end="")
    for _ in days:
        print(f"{'─' * 8} ", end="")
    print()

    for hour in range(24):
        print(f"  {hour:02d}:00  ", end="")
        for day_key in days:
            s = hourly[hour][day_key]["success"]
            t = hourly[hour][day_key]["timeout"]
            if s == 0 and t == 0:
                cell = "."
            elif s > 0 and t == 0:
                cell = f"{s}S"
            elif s == 0 and t > 0:
                cell = f"{t}T"
            else:
                cell = f"{s}S{t}T"
            print(f"{cell:<9}", end="")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze AWS Lambda usage and failure patterns")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, help="Look back N days from today")
    parser.add_argument(
        "--function",
        type=str,
        choices=list(LOG_GROUPS.keys()),
        help="Analyze a single function (default: all)",
    )
    parser.add_argument("--no-heatmap", action="store_true", help="Skip hourly heatmap")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    now = datetime.now(UTC)

    if args.start and args.end:
        start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=UTC)
        end = datetime.strptime(args.end, "%Y-%m-%d").replace(
            hour=23, minute=59, second=59, tzinfo=UTC
        )
    elif args.days:
        end = now
        start = now - timedelta(days=args.days)
    else:
        # Current month
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end = now

    functions = {args.function: LOG_GROUPS[args.function]} if args.function else LOG_GROUPS

    print(f"Fetching logs from CloudWatch ({REGION})...")
    all_reports: list[dict] = []
    for func_name, log_group in functions.items():
        print(f"  {func_name}...", end="", flush=True)
        func_reports = analyze_function(func_name, log_group, start, end)
        print(f" {len(func_reports)} REPORT lines")
        all_reports.extend(func_reports)

    if not all_reports:
        print("\nNo invocations found in the specified period.")
        return

    print_summary(all_reports, start, end)
    print_daily_breakdown(all_reports)

    # Only show retry/heatmap for scraper (scheduler doesn't timeout)
    scraper_reports = [r for r in all_reports if r["function"] == "odds-scheduler-scraper"]
    if scraper_reports:
        print_retry_analysis(scraper_reports)
        if not args.no_heatmap:
            print_hourly_heatmap(scraper_reports)


if __name__ == "__main__":
    main()
