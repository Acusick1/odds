"""Pull orderbooks for upcoming MLB games on Polymarket.

Measures: spread (cents), bid/ask depth, number of price levels, top-of-book size.
Compares to EPL ($10-100K thin/AMM) and NBA (~1¢ spread, 25-28 levels, $1,100 top).
"""

import asyncio
import json
import statistics
from datetime import UTC, datetime
from typing import Any

import aiohttp

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
MLB_SERIES_ID = "3"
GAME_TAG_ID = "100639"


async def fetch_json(session: aiohttp.ClientSession, url: str, params: dict | None = None) -> Any:
    async with session.get(url, params=params or {}, timeout=aiohttp.ClientTimeout(total=30)) as r:
        if r.status != 200:
            return {"_status": r.status, "_body": (await r.text())[:300]}
        return await r.json()


def process_book(raw: dict) -> dict | None:
    bids = raw.get("bids") or []
    asks = raw.get("asks") or []
    if not bids or not asks:
        return None
    bids_sorted = sorted(bids, key=lambda x: float(x["price"]), reverse=True)
    asks_sorted = sorted(asks, key=lambda x: float(x["price"]))
    best_bid = float(bids_sorted[0]["price"])
    best_ask = float(asks_sorted[0]["price"])
    if best_bid >= best_ask:
        return None
    return {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": best_ask - best_bid,
        "mid": (best_bid + best_ask) / 2,
        "bid_levels": len(bids),
        "ask_levels": len(asks),
        "bid_depth": sum(float(b["size"]) for b in bids),
        "ask_depth": sum(float(a["size"]) for a in asks),
        "top_bid_size": float(bids_sorted[0]["size"]),
        "top_ask_size": float(asks_sorted[0]["size"]),
    }


async def main() -> None:
    now = datetime.now(UTC)
    async with aiohttp.ClientSession() as s:
        # Active, upcoming events
        events = await fetch_json(
            s,
            f"{GAMMA}/events",
            {
                "series_id": MLB_SERIES_ID,
                "tag_id": GAME_TAG_ID,
                "active": "true",
                "closed": "false",
                "limit": 200,
            },
        )
        if not isinstance(events, list):
            print("Event fetch failed")
            return

        # Filter to events with gameStartTime in the future
        upcoming = []
        for e in events:
            for m in e.get("markets") or []:
                gst = m.get("gameStartTime")
                if gst and "closed" in m and not m.get("closed"):
                    try:
                        game_dt = datetime.fromisoformat(
                            gst.replace(" ", "T").replace("+00", "+00:00")
                        )
                        if game_dt > now:
                            upcoming.append((e, m, game_dt))
                            break
                    except Exception:
                        continue
        print(f"Upcoming MLB games with unclosed markets: {len(upcoming)}")

        # Sort by volume desc to pick busiest — but also make sure game hasn't started
        upcoming.sort(key=lambda x: float(x[0].get("volume") or 0), reverse=True)

        sample = upcoming[:10]
        print(f"\nSampling top {len(sample)} by volume:\n")

        all_books: list[dict] = []
        for e, m, game_dt in sample:
            title = e.get("title")
            vol = float(e.get("volume") or 0)
            liq = float(e.get("liquidity") or 0)
            # clobTokenIds is a JSON string
            try:
                token_ids = json.loads(m.get("clobTokenIds") or "[]")
            except Exception:
                token_ids = []
            if not token_ids:
                continue
            print(
                f"=== {title}  vol=${vol:,.0f}  liq=${liq:,.0f}  "
                f"game_start={game_dt.isoformat()} ==="
            )
            outcomes = json.loads(m.get("outcomes") or "[]")
            prices = json.loads(m.get("outcomePrices") or "[]")
            print(f"  outcomes={outcomes} last_prices={prices}")

            # Fetch book for each token (typically 2: home/away)
            for i, tok in enumerate(token_ids):
                raw = await fetch_json(s, f"{CLOB}/book", {"token_id": str(tok)})
                if not isinstance(raw, dict) or "_status" in raw:
                    print(f"  token[{i}] book fetch failed: {raw}")
                    continue
                processed = process_book(raw)
                if processed is None:
                    print(
                        f"  token[{i}] ({outcomes[i] if i < len(outcomes) else '?'}): empty/crossed"
                    )
                    continue
                all_books.append(
                    {
                        "event": title,
                        "outcome": outcomes[i] if i < len(outcomes) else "?",
                        **processed,
                    }
                )
                print(
                    f"  token[{i}] ({outcomes[i] if i < len(outcomes) else '?'}):"
                    f" bid={processed['best_bid']:.3f} ask={processed['best_ask']:.3f}"
                    f" spread={processed['spread']:.3f} ({processed['spread'] * 100:.2f}¢)"
                    f" bid_lvls={processed['bid_levels']} ask_lvls={processed['ask_levels']}"
                    f" bid_depth=${processed['bid_depth']:.0f}"
                    f" ask_depth=${processed['ask_depth']:.0f}"
                    f" top_bid_size=${processed['top_bid_size']:.0f}"
                    f" top_ask_size=${processed['top_ask_size']:.0f}"
                )
            print()

        # Aggregate stats
        if all_books:
            print("=== Aggregate order-book stats (moneyline tokens, all upcoming games) ===")
            print(f"  tokens sampled: {len(all_books)}")
            spreads = [b["spread"] * 100 for b in all_books]
            bid_lvls = [b["bid_levels"] for b in all_books]
            ask_lvls = [b["ask_levels"] for b in all_books]
            bid_depth = [b["bid_depth"] for b in all_books]
            ask_depth = [b["ask_depth"] for b in all_books]
            top_bid = [b["top_bid_size"] for b in all_books]
            top_ask = [b["top_ask_size"] for b in all_books]

            def summ(name: str, vals: list[float]) -> None:
                if not vals:
                    return
                print(
                    f"  {name}: n={len(vals)} min={min(vals):.2f}"
                    f" median={statistics.median(vals):.2f}"
                    f" mean={statistics.mean(vals):.2f}"
                    f" max={max(vals):.2f}"
                )

            summ("spread (cents)", spreads)
            summ("bid levels", bid_lvls)
            summ("ask levels", ask_lvls)
            summ("bid depth ($)", bid_depth)
            summ("ask depth ($)", ask_depth)
            summ("top bid size ($)", top_bid)
            summ("top ask size ($)", top_ask)


if __name__ == "__main__":
    asyncio.run(main())
