"""Discover MLB per-game tag/series on Polymarket via Gamma API."""

import asyncio
from collections import Counter
from typing import Any

import aiohttp

GAMMA = "https://gamma-api.polymarket.com"


async def fetch_json(session: aiohttp.ClientSession, path: str, params: dict | None = None) -> Any:
    url = f"{GAMMA}{path}"
    async with session.get(url, params=params or {}, timeout=aiohttp.ClientTimeout(total=30)) as r:
        if r.status != 200:
            return {"_status": r.status, "_body": (await r.text())[:500]}
        return await r.json()


async def main() -> None:
    async with aiohttp.ClientSession() as s:
        # Gather a large batch of MLB-tagged events; inspect tag/series distribution
        print("=== Pulling all MLB-tagged events (tag_slug=mlb) ===")
        all_events: list[dict] = []
        offset = 0
        while True:
            batch = await fetch_json(
                s,
                "/events",
                {"tag_slug": "mlb", "limit": 200, "offset": offset, "closed": "false"},
            )
            if not isinstance(batch, list) or not batch:
                break
            all_events.extend(batch)
            if len(batch) < 200:
                break
            offset += 200
        print(f"  total active/upcoming MLB events: {len(all_events)}")

        # Count tags across events — find which tag IDs appear on per-game events
        tag_counter: Counter[str] = Counter()
        series_counter: Counter[str] = Counter()
        for e in all_events:
            for t in e.get("tags") or []:
                tag_counter[f"{t.get('id')}|{t.get('label')}|{t.get('slug')}"] += 1
            for sr in e.get("series") or []:
                series_counter[f"{sr.get('id')}|{sr.get('title')}|{sr.get('ticker')}"] += 1

        print("\n  Top tags across events:")
        for tag_key, count in tag_counter.most_common(20):
            print(f"    {count:4d}  {tag_key}")

        print("\n  Series across events:")
        for s_key, count in series_counter.most_common(20):
            print(f"    {count:4d}  {s_key}")

        # Filter to what look like per-game events (ticker looks like team-team-date)
        per_game = [
            e
            for e in all_events
            if e.get("ticker")
            and "-" in e["ticker"]
            and any(c.isdigit() for c in e["ticker"][-10:])
            and "vs" not in e.get("title", "").lower()[:20]
        ]
        # Alternate: title contains " vs " pattern
        per_game_title = [e for e in all_events if " vs " in (e.get("title") or "").lower()]
        print(f"\n  Events with ticker ending in digits: {len(per_game)}")
        print(f"  Events with ' vs ' in title: {len(per_game_title)}")

        print("\n  Sample 'vs' titles:")
        for e in per_game_title[:10]:
            print(
                f"    id={e.get('id')} title={e.get('title')!r} "
                f"ticker={e.get('ticker')!r} "
                f"vol={e.get('volume')} liq={e.get('liquidity')}"
            )

        # Also try tag_slug=mlb-games, baseball
        for slug in ["mlb-games", "baseball", "baseball-games", "mlb-regular-season"]:
            print(f"\n=== tag_slug={slug} ===")
            ev = await fetch_json(s, "/events", {"tag_slug": slug, "limit": 10, "closed": "false"})
            if isinstance(ev, list):
                print(f"  returned {len(ev)} events")
                for e in ev[:3]:
                    print(f"    {e.get('title')!r} ticker={e.get('ticker')!r}")


if __name__ == "__main__":
    asyncio.run(main())
