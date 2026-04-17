"""Pull MLB per-game events from Polymarket and summarize volume/liquidity distribution."""

import asyncio
import json
import statistics
from typing import Any

import aiohttp

GAMMA = "https://gamma-api.polymarket.com"
MLB_SERIES_ID = "3"
GAME_TAG_ID = "100639"


async def fetch_events(session: aiohttp.ClientSession, active: bool, closed: bool) -> list[dict]:
    all_events: list[dict] = []
    offset = 0
    while True:
        params = {
            "series_id": MLB_SERIES_ID,
            "tag_id": GAME_TAG_ID,
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            "limit": 200,
            "offset": offset,
        }
        async with session.get(
            f"{GAMMA}/events", params=params, timeout=aiohttp.ClientTimeout(total=30)
        ) as r:
            if r.status != 200:
                break
            batch: list[Any] = await r.json()
        if not batch:
            break
        all_events.extend(batch)
        if len(batch) < 200:
            break
        offset += 200
    return all_events


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return statistics.quantiles(values, n=100)[int(p) - 1] if len(values) > 1 else values[0]


def summarize(values: list[float], label: str) -> None:
    if not values:
        print(f"  {label}: no data")
        return
    print(f"  {label}:")
    print(f"    n={len(values)}")
    print(f"    min={min(values):.2f}")
    print(f"    p25={pct(values, 25):.2f}")
    print(f"    median={statistics.median(values):.2f}")
    print(f"    p75={pct(values, 75):.2f}")
    print(f"    p90={pct(values, 90):.2f}")
    print(f"    max={max(values):.2f}")
    print(f"    mean={statistics.mean(values):.2f}")
    print(f"    total={sum(values):.2f}")


async def main() -> None:
    async with aiohttp.ClientSession() as s:
        active = await fetch_events(s, active=True, closed=False)
        closed = await fetch_events(s, active=False, closed=True)
        print(f"Active/upcoming: {len(active)}")
        print(f"Closed/resolved: {len(closed)}")

        print("\n=== Sample active events ===")
        for e in active[:8]:
            print(
                f"  id={e.get('id')} {e.get('title')!r}"
                f" ticker={e.get('ticker')!r}"
                f" start={e.get('startDate')}"
                f" vol={e.get('volume')} liq={e.get('liquidity')}"
                f" n_markets={len(e.get('markets') or [])}"
            )

        # Volume/liquidity distribution across *all* MLB game events (active + closed)
        all_events = active + closed
        vols = [float(e.get("volume") or 0) for e in all_events]
        liqs = [float(e.get("liquidity") or 0) for e in all_events]

        print("\n=== Volume distribution (all MLB games) ===")
        summarize(vols, "volume (USDC)")

        print("\n=== Liquidity distribution (all MLB games) ===")
        summarize(liqs, "liquidity (USDC)")

        # Closed only — these have completed lifecycle, so volume is final
        closed_vols = [float(e.get("volume") or 0) for e in closed]
        print("\n=== Volume distribution (closed/resolved only) ===")
        summarize(closed_vols, "final volume")

        # Markets per game
        markets_per = [len(e.get("markets") or []) for e in all_events]
        summarize([float(m) for m in markets_per], "markets per event")

        # Top 10 highest-volume events
        print("\n=== Top 10 highest-volume MLB games ===")
        top = sorted(all_events, key=lambda e: float(e.get("volume") or 0), reverse=True)[:10]
        for e in top:
            print(
                f"  vol={float(e.get('volume') or 0):>10.0f}"
                f"  title={e.get('title')!r}"
                f"  start={e.get('startDate')}"
                f"  closed={e.get('closed')}"
            )

        # Dump one active game event to inspect market structure
        print("\n=== Sample active game event full markets ===")
        if active:
            e = sorted(active, key=lambda e: float(e.get("volume") or 0), reverse=True)[0]
            print(f"  Chosen: {e.get('title')} (vol={e.get('volume')})")
            for m in e.get("markets") or []:
                print(
                    f"    market={m.get('question')!r}"
                    f"  outcomes={m.get('outcomes')}"
                    f"  prices={m.get('outcomePrices')}"
                    f"  vol={m.get('volume')}"
                    f"  liq={m.get('liquidity')}"
                )
            # Dump first market fully for token_id inspection
            if e.get("markets"):
                print("\n  First market (raw):")
                print(json.dumps(e["markets"][0], indent=2, default=str)[:2500])


if __name__ == "__main__":
    asyncio.run(main())
