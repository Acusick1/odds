"""Minimum snapshot-call grid for Odds API historical backfill (ticket #506).

Each (match, horizon) needs a snapshot T in [K-h-tol, K-h+tol]. One call at T serves every
(match, horizon) interval containing T, so the minimum call count is the minimum point set
stabbing all intervals (greedy by right endpoint is optimal). T is floored to the snapshot
grid (10 min before 2022-09-01, 5 min after).

Usage: uv run python research/odds-api-historical-grid/grid.py "$LOCAL_DATABASE_URL" (read-only)
"""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime, timedelta

M = timedelta(minutes=1)
H = timedelta(hours=1)
D = timedelta(days=1)
FULL = {
    "7d": 7 * D,
    "3d": 3 * D,
    "2d": 2 * D,
    "24h": 24 * H,
    "12h": 12 * H,
    "6h": 6 * H,
    "3h": 3 * H,
    "1h": H,
    "30m": 30 * M,
    "10m": 10 * M,
}
COARSE = {k: FULL[k] for k in ("24h", "6h", "1h", "10m")}
# 2020-21 starts 1 Aug so the 2019-20 restart (to 2020-07-26) is excluded.
SEASONS = [
    (
        f"{y}-{str(y + 1)[2:]}",
        datetime(y, 8 if y == 2020 else 7, 1, tzinfo=UTC),
        datetime(y + 1, 7, 1, tzinfo=UTC),
    )
    for y in range(2020, 2026)
]
FIVE_MIN_FROM = datetime(2022, 9, 1, tzinfo=UTC)
UNITS_PER_CALL = 10  # 10 markets-units x 1 market (h2h) x 1 region-equivalent (<=10 bookmakers)


def tol(h: timedelta) -> timedelta:
    return 15 * M if h <= 3 * H else H


def floor_to_grid(t: datetime) -> datetime:
    step = 5 if t >= FIVE_MIN_FROM else 10
    return t.replace(minute=t.minute - t.minute % step, second=0, microsecond=0)


def calls(kicks: list[datetime], horizons: dict[str, timedelta]) -> list[datetime]:
    ivs = sorted((k - h + tol(h), k - h - tol(h)) for k in kicks for h in horizons.values())
    out: list[datetime] = []
    for right, left in ivs:
        if out and out[-1] >= left:
            continue
        t = floor_to_grid(right)
        assert t >= left, (t, left)
        out.append(t)
    return out


def main(url: str) -> None:
    sql = (
        "select to_char(commence_time at time zone 'utc','YYYY-MM-DD\"T\"HH24:MI:SS'), home_team, away_team "
        "from events where sport_key='soccer_epl' order by 1"
    )
    out = subprocess.run(
        ["psql", url, "-At", "-F", "|", "-c", sql], check=True, capture_output=True, text=True
    ).stdout
    # Some fixtures exist twice with different ids/times; keep the later kick-off per (season, pairing).
    by_fixture: dict[tuple[str, str, str], datetime] = {}
    for line in out.splitlines():
        ts, home, away = line.split("|")
        k = datetime.fromisoformat(ts).replace(tzinfo=UTC)
        season = next((n for n, a, b in SEASONS if a <= k < b), "other")
        by_fixture[(season, home, away)] = max(k, by_fixture.get((season, home, away), k))
    kicks = sorted(by_fixture.values())
    print(
        "| season | matches | distinct kick-offs | 10-horizon calls | units | 4-horizon calls | units | per-match 10h | per-match 4h |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    tot = [0, 0, 0, 0]
    for name, a, b in SEASONS:
        ks = [k for k in kicks if a <= k < b]
        f, c = len(calls(ks, FULL)), len(calls(ks, COARSE))
        tot[0] += len(ks)
        tot[1] += f
        tot[2] += c
        print(
            f"| {name} | {len(ks)} | {len(set(ks))} | {f} | {f * UNITS_PER_CALL:,} | {c} | {c * UNITS_PER_CALL:,} | "
            f"{len(ks) * 10 * UNITS_PER_CALL:,} | {len(ks) * 4 * UNITS_PER_CALL:,} |"
        )
    print(
        f"| total | {tot[0]} | | {tot[1]} | {tot[1] * UNITS_PER_CALL:,} | {tot[2]} | {tot[2] * UNITS_PER_CALL:,} | "
        f"{tot[0] * 100:,} | {tot[0] * 40:,} |"
    )
    # cumulative newest-first, and 2022-09+ only
    print("\nCumulative newest-first (10h / 4h units):")
    c10 = c4 = 0
    for name, a, b in reversed(SEASONS):
        ks = [k for k in kicks if a <= k < b]
        c10 += len(calls(ks, FULL)) * UNITS_PER_CALL
        c4 += len(calls(ks, COARSE)) * UNITS_PER_CALL
        print(f"  back to {name}: {c10:,} / {c4:,}")
    # Horizons only (diagnostic): how many calls does each single horizon need?
    print("\nPer-horizon calls, 2024-25:")
    ks = [k for k in kicks if SEASONS[4][1] <= k < SEASONS[4][2]]
    for n, h in FULL.items():
        print(f"  {n}: {len(calls(ks, {n: h}))}")


if __name__ == "__main__":
    main(sys.argv[1])
