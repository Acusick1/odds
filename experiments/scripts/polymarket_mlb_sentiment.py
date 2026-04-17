"""Assess whether MLB Polymarket prices reflect genuine sentiment or AMM parameters.

Two probes:
1. Price-history shape: pull 5-min resolution for a recently-resolved game.
   AMMs produce smooth continuous curves; real trading produces discrete jumps.
2. Price comparison: for an upcoming game, compare Polymarket implied prob to
   latest sharp bookmaker odds in our DB (bet365 devigged).
"""

import asyncio
import json
import statistics
from datetime import UTC, datetime
from typing import Any

import aiohttp
from odds_core.database import async_session_maker
from sqlalchemy import text

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
MLB_SERIES_ID = "3"
GAME_TAG_ID = "100639"


async def fetch_json(session: aiohttp.ClientSession, url: str, params: dict | None = None) -> Any:
    async with session.get(url, params=params or {}, timeout=aiohttp.ClientTimeout(total=30)) as r:
        if r.status != 200:
            return {"_status": r.status, "_body": (await r.text())[:300]}
        return await r.json()


async def probe_price_history(session: aiohttp.ClientSession) -> None:
    """Pick a high-volume resolved MLB game, pull price history, assess shape."""
    print("=" * 80)
    print("PROBE 1: price-history shape (AMM smoothness check)")
    print("=" * 80)
    # Fetch closed events — they still return price history if within 30d retention
    events = await fetch_json(
        session,
        f"{GAMMA}/events",
        {
            "series_id": MLB_SERIES_ID,
            "tag_id": GAME_TAG_ID,
            "active": "false",
            "closed": "true",
            "limit": 100,
            "offset": 0,
            "order": "endDate",
            "ascending": "false",
        },
    )
    if not isinstance(events, list):
        print("  Failed to fetch closed events:", events)
        return

    # Recent (by endDate) high-volume event
    events_sorted = sorted(events, key=lambda e: float(e.get("volume") or 0), reverse=True)
    chosen = None
    for e in events_sorted:
        for m in e.get("markets") or []:
            if m.get("question") == e.get("title"):  # moneyline
                try:
                    toks = json.loads(m.get("clobTokenIds") or "[]")
                except Exception:
                    toks = []
                if toks:
                    chosen = (e, m, toks)
                    break
        if chosen:
            break

    if not chosen:
        print("  No suitable closed event with tokens")
        return
    e, m, toks = chosen
    print(f"\n  Chosen: {e['title']}  vol=${float(e.get('volume') or 0):,.0f}")
    print(f"  End: {e.get('endDate')}")

    # Pull history for first token
    hist = await fetch_json(
        session,
        f"{CLOB}/prices-history",
        {"market": str(toks[0]), "interval": "max", "fidelity": 5},
    )
    if not isinstance(hist, dict):
        print("  history fetch failed:", hist)
        return
    points = hist.get("history") or []
    print(f"  History points (5-min fidelity): {len(points)}")
    if len(points) < 10:
        return

    # Assess shape: count of distinct prices, avg step size, max step size
    prices = [float(p["p"]) for p in points]
    distinct = len(set(prices))
    steps = [abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))]
    nonzero_steps = [s for s in steps if s > 0]
    print(f"  distinct price levels: {distinct}")
    print(
        f"  steps: total={len(steps)} nonzero={len(nonzero_steps)} "
        f"({len(nonzero_steps) / len(steps) * 100:.1f}% moved)"
    )
    if nonzero_steps:
        print(
            f"  step size: median={statistics.median(nonzero_steps):.4f}"
            f" mean={statistics.mean(nonzero_steps):.4f}"
            f" max={max(nonzero_steps):.4f}"
        )
    # First & last 5 points
    print("  First 5 points:")
    for p in points[:5]:
        print(f"    t={datetime.fromtimestamp(p['t'], UTC).isoformat()}  p={p['p']}")
    print("  Last 5 points:")
    for p in points[-5:]:
        print(f"    t={datetime.fromtimestamp(p['t'], UTC).isoformat()}  p={p['p']}")
    # Check for discrete jumps (>1¢) — indicative of real trades clearing levels
    big_jumps = [s for s in nonzero_steps if s >= 0.01]
    print(f"  jumps >=1¢: {len(big_jumps)} ({len(big_jumps) / len(steps) * 100:.1f}% of ticks)")
    print(
        "  Interpretation: AMM produces continuous smooth curves (many tiny steps);"
        " real orderbooks show discrete clearing jumps."
    )


async def probe_sharp_vs_pm(session: aiohttp.ClientSession) -> None:
    """For today's MLB games, compare Polymarket implied prob to bet365/pinnacle stored odds."""
    print("\n" + "=" * 80)
    print("PROBE 2: Polymarket implied prob vs sharp bookmaker odds (our DB)")
    print("=" * 80)

    # Pull upcoming Polymarket events (with moneyline)
    events = await fetch_json(
        session,
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
        print("  fetch failed")
        return

    pm_rows: list[dict] = []
    for e in events:
        for m in e.get("markets") or []:
            if m.get("question") == e.get("title") and not m.get("closed"):
                try:
                    prices = json.loads(m.get("outcomePrices") or "[]")
                    outcomes = json.loads(m.get("outcomes") or "[]")
                    toks = json.loads(m.get("clobTokenIds") or "[]")
                except Exception:
                    continue
                if len(outcomes) == 2 and len(prices) == 2 and len(toks) == 2:
                    # Polymarket title format: "Away vs. Home" — verify
                    gst = m.get("gameStartTime")
                    pm_rows.append(
                        {
                            "title": e.get("title"),
                            "away": outcomes[0],
                            "home": outcomes[1],
                            "p_away": float(prices[0]),
                            "p_home": float(prices[1]),
                            "tok_away": toks[0],
                            "tok_home": toks[1],
                            "game_start": gst,
                            "vol": float(e.get("volume") or 0),
                        }
                    )
                break

    # Refresh prices from live midpoints for freshness
    print(f"  upcoming moneyline markets: {len(pm_rows)}")

    # For each, fetch live midpoint + compare to sharp odds in DB
    now = datetime.now(UTC)
    async with async_session_maker() as db:
        compared = 0
        deltas: list[float] = []
        for row in pm_rows[:20]:
            # Try to match event by team names + date (±24h)
            if not row["game_start"]:
                continue
            try:
                game_dt = datetime.fromisoformat(
                    row["game_start"].replace(" ", "T").replace("+00", "+00:00")
                )
            except Exception:
                continue
            if game_dt < now:
                continue
            from datetime import timedelta

            r = await db.execute(
                text(
                    """
                    SELECT id FROM events
                    WHERE sport_key='baseball_mlb'
                      AND home_team=:home AND away_team=:away
                      AND commence_time BETWEEN :lo AND :hi
                    LIMIT 1
                    """
                ),
                {
                    "home": row["home"],
                    "away": row["away"],
                    "lo": game_dt - timedelta(hours=24),
                    "hi": game_dt + timedelta(hours=24),
                },
            )
            matched_event = r.first()
            if not matched_event:
                print(f"  [no DB match] {row['away']} @ {row['home']}")
                continue
            event_id = matched_event[0]

            # Pull latest bet365 h2h odds
            q_odds = text(
                """
                SELECT bookmaker_key, outcome_name, price, last_update
                FROM odds
                WHERE event_id=:eid
                  AND market_key='h2h'
                  AND is_valid=true
                ORDER BY last_update DESC
                LIMIT 60
                """
            )
            r = await db.execute(q_odds, {"eid": event_id})
            odds_rows = list(r)
            if not odds_rows:
                continue

            # Get fresh Polymarket midpoints
            mid_away = await fetch_json(
                session, f"{CLOB}/midpoint", {"token_id": str(row["tok_away"])}
            )
            mid_home = await fetch_json(
                session, f"{CLOB}/midpoint", {"token_id": str(row["tok_home"])}
            )
            p_away = (
                float(mid_away["mid"])
                if isinstance(mid_away, dict) and "mid" in mid_away
                else row["p_away"]
            )
            p_home = (
                float(mid_home["mid"])
                if isinstance(mid_home, dict) and "mid" in mid_home
                else row["p_home"]
            )

            # Latest book prices per bookmaker for h2h — devig
            latest_by_book: dict[str, dict[str, float]] = {}
            for bk, name, price, _ in odds_rows:
                latest_by_book.setdefault(bk, {})
                if name == row["home"]:
                    latest_by_book[bk].setdefault("home", float(price))
                elif name == row["away"]:
                    latest_by_book[bk].setdefault("away", float(price))

            # Devig each book, prefer betfair_ex_eu/pinnacle/bet365
            book_priority = [
                "pinnacle",
                "betfair_ex_eu",
                "betfair_ex_uk",
                "bet365",
                "betmgm",
                "draftkings",
                "fanduel",
            ]
            devigged = None
            chosen_book = None
            for bk in book_priority:
                if (
                    bk in latest_by_book
                    and "home" in latest_by_book[bk]
                    and "away" in latest_by_book[bk]
                ):
                    # American odds or decimal? Check model — our odds table stores American
                    # For American: convert to implied prob
                    def ameri_to_prob(a: float) -> float:
                        return 100 / (a + 100) if a > 0 else -a / (-a + 100)

                    pa = ameri_to_prob(latest_by_book[bk]["away"])
                    ph = ameri_to_prob(latest_by_book[bk]["home"])
                    s = pa + ph
                    if 0.9 < s < 1.2:
                        devigged = {"home": ph / s, "away": pa / s}
                        chosen_book = bk
                        break
            if not devigged:
                continue

            delta_home = p_home - devigged["home"]
            deltas.append(delta_home)
            compared += 1
            print(f"  {row['away']} @ {row['home']} start={game_dt.isoformat()}")
            print(
                f"    Polymarket: p_home={p_home:.3f} p_away={p_away:.3f} (vol=${row['vol']:,.0f})"
            )
            print(
                f"    {chosen_book} devigged: p_home={devigged['home']:.3f}"
                f" p_away={devigged['away']:.3f}"
            )
            print(f"    delta (PM - sharp) on home: {delta_home:+.4f}")

        print(f"\n  compared {compared} games")
        if deltas:
            print(
                f"  delta stats: mean={statistics.mean(deltas):+.4f}"
                f" stdev={statistics.stdev(deltas) if len(deltas) > 1 else 0:.4f}"
                f" max_abs={max(abs(d) for d in deltas):.4f}"
            )


async def main() -> None:
    async with aiohttp.ClientSession() as s:
        await probe_price_history(s)
        await probe_sharp_vs_pm(s)


if __name__ == "__main__":
    asyncio.run(main())
