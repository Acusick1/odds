"""Betfair NBA Market Exploration.

Connects to the Betfair Exchange API using a delayed (free) or live app key
to assess NBA market availability, structure, and liquidity indicators.

Outputs:
  - Number of NBA events with active markets
  - Market types available per event (match odds, handicap, totals)
  - Back-lay spreads (primary liquidity indicator)
  - Matched volume where available (live key only)
  - Market open times relative to game start

Requires environment variables:
    BETFAIR_USERNAME: Betfair account username
    BETFAIR_PASSWORD: Betfair account password
    BETFAIR_APP_KEY: Application key (delayed or live)
    BETFAIR_CERT_FILE: Path to SSL cert (optional, for bot login)
    BETFAIR_CERT_KEY: Path to SSL key (optional, for bot login)

Usage:
    uv run python experiments/scripts/betfair_nba_explore.py
"""

from __future__ import annotations

import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import betfairlightweight
from betfairlightweight import filters

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "betfair_exploration"

BASKETBALL_EVENT_TYPE_ID = "7522"

MARKET_TYPES = [
    "MATCH_ODDS",
    "ASIAN_HANDICAP",
    "OVER_UNDER_215",
    "OVER_UNDER_220",
    "OVER_UNDER_225",
    "OVER_UNDER_230",
    "OVER_UNDER_235",
]


def create_client() -> betfairlightweight.APIClient:
    username = os.environ.get("BETFAIR_USERNAME")
    password = os.environ.get("BETFAIR_PASSWORD")
    app_key = os.environ.get("BETFAIR_APP_KEY")

    if not all([username, password, app_key]):
        print("ERROR: Set BETFAIR_USERNAME, BETFAIR_PASSWORD, BETFAIR_APP_KEY")
        sys.exit(1)

    cert_file = os.environ.get("BETFAIR_CERT_FILE")
    cert_key = os.environ.get("BETFAIR_CERT_KEY")

    kwargs: dict = {
        "username": username,
        "password": password,
        "app_key": app_key,
    }

    if cert_file and cert_key:
        kwargs["certs"] = [cert_file, cert_key]

    return betfairlightweight.APIClient(**kwargs)


def explore_competitions(client: betfairlightweight.APIClient) -> list[dict]:
    """List basketball competitions to find NBA."""
    market_filter = filters.market_filter(event_type_ids=[BASKETBALL_EVENT_TYPE_ID])
    competitions = client.betting.list_competitions(filter=market_filter)

    results = []
    for comp in competitions:
        results.append(
            {
                "id": comp.competition.id,
                "name": comp.competition.name,
                "market_count": comp.market_count,
            }
        )
        print(f"  {comp.competition.id}: {comp.competition.name} ({comp.market_count} markets)")

    return results


def explore_events(client: betfairlightweight.APIClient, competition_ids: list[str]) -> list[dict]:
    """List upcoming NBA events."""
    market_filter = filters.market_filter(
        event_type_ids=[BASKETBALL_EVENT_TYPE_ID],
        competition_ids=competition_ids,
    )
    events = client.betting.list_events(filter=market_filter)

    results = []
    for ev in sorted(events, key=lambda e: e.event.open_date):
        info = {
            "id": ev.event.id,
            "name": ev.event.name,
            "open_date": ev.event.open_date,
            "market_count": ev.market_count,
            "country_code": ev.event.country_code,
        }
        results.append(info)

        time_str = ev.event.open_date.strftime("%Y-%m-%d %H:%M UTC")
        print(f"  {ev.event.name} | {time_str} | {ev.market_count} markets")

    return results


def explore_markets(
    client: betfairlightweight.APIClient,
    event_ids: list[str],
) -> list[dict]:
    """Get market catalogue and book data for events."""
    market_filter = filters.market_filter(
        event_type_ids=[BASKETBALL_EVENT_TYPE_ID],
        event_ids=event_ids,
    )

    catalogues = client.betting.list_market_catalogue(
        filter=market_filter,
        market_projection=[
            "RUNNER_DESCRIPTION",
            "EVENT",
            "COMPETITION",
            "MARKET_START_TIME",
        ],
        max_results=200,
    )

    if not catalogues:
        print("  No markets found")
        return []

    market_ids = [c.market_id for c in catalogues]

    # Fetch prices in batches — delayed key may reject large requests
    price_proj = filters.price_projection(
        price_data=["EX_BEST_OFFERS"],
    )

    BATCH_SIZE = 10
    book_by_id: dict = {}
    for i in range(0, len(market_ids), BATCH_SIZE):
        batch = market_ids[i : i + BATCH_SIZE]
        books = client.betting.list_market_book(
            market_ids=batch,
            price_projection=price_proj,
        )
        for b in books:
            book_by_id[b.market_id] = b

    results = []
    for cat in catalogues:
        book = book_by_id.get(cat.market_id)
        if not book:
            continue

        event_name = cat.event.name if cat.event else "Unknown"
        market_start = cat.market_start_time

        hours_to_start = None
        if market_start:
            delta = (market_start - datetime.now(UTC)).total_seconds() / 3600
            hours_to_start = round(delta, 1)

        runners_info = []
        for runner in book.runners:
            runner_name = "Unknown"
            for cat_runner in cat.runners:
                if cat_runner.selection_id == runner.selection_id:
                    runner_name = cat_runner.runner_name
                    break

            ex = runner.ex
            back_offers = ex.available_to_back if ex else []
            lay_offers = ex.available_to_lay if ex else []

            best_back = back_offers[0].price if back_offers else None
            best_lay = lay_offers[0].price if lay_offers else None
            back_size = back_offers[0].size if back_offers else None
            lay_size = lay_offers[0].size if lay_offers else None
            last_price = runner.last_price_traded
            total_matched = runner.total_matched

            spread = None
            spread_pct = None
            if best_back and best_lay:
                spread = round(best_lay - best_back, 4)
                midpoint = (best_back + best_lay) / 2
                spread_pct = round(spread / midpoint * 100, 2)

            runners_info.append(
                {
                    "name": runner_name,
                    "selection_id": runner.selection_id,
                    "best_back": best_back,
                    "back_size": back_size,
                    "best_lay": best_lay,
                    "lay_size": lay_size,
                    "spread": spread,
                    "spread_pct": spread_pct,
                    "last_price_traded": last_price,
                    "total_matched": total_matched,
                }
            )

        overround = None
        back_prices = [r["best_back"] for r in runners_info if r["best_back"]]
        if len(back_prices) == 2:
            overround = round(sum(1 / p for p in back_prices) - 1, 4)

        market_info = {
            "market_id": cat.market_id,
            "market_name": cat.market_name,
            "market_type": cat.description.market_type if cat.description else None,
            "event_name": event_name,
            "market_start": market_start,
            "hours_to_start": hours_to_start,
            "status": book.status,
            "total_matched": book.total_matched,
            "runners": runners_info,
            "overround": overround,
        }
        results.append(market_info)

    return results


def print_market_summary(markets: list[dict]) -> None:
    """Print formatted summary of market data."""
    # Group by event
    by_event: dict[str, list[dict]] = {}
    for m in markets:
        by_event.setdefault(m["event_name"], []).append(m)

    for event_name, event_markets in sorted(by_event.items()):
        print(f"\n{'=' * 70}")
        print(f"  {event_name}")
        print(f"{'=' * 70}")

        for m in event_markets:
            status = m["status"]
            market_type = m["market_type"] or m["market_name"]
            hours = m["hours_to_start"]
            hours_str = f"{hours:.1f}h to start" if hours is not None else "unknown"
            matched = m["total_matched"]
            matched_str = f"£{matched:,.0f} matched" if matched else "no volume"

            print(f"\n  {market_type} | {status} | {hours_str} | {matched_str}")

            if m["overround"] is not None:
                print(f"  Overround (back): {m['overround'] * 100:.2f}%")

            for r in m["runners"]:
                back = f"{r['best_back']:.2f}" if r["best_back"] else "---"
                lay = f"{r['best_lay']:.2f}" if r["best_lay"] else "---"
                back_sz = f"£{r['back_size']:.0f}" if r["back_size"] else ""
                lay_sz = f"£{r['lay_size']:.0f}" if r["lay_size"] else ""
                spread = f"spread={r['spread_pct']:.1f}%" if r["spread_pct"] else ""
                ltp = f"LTP={r['last_price_traded']:.2f}" if r["last_price_traded"] else ""

                print(
                    f"    {r['name']:25s}  "
                    f"Back: {back:>6s} ({back_sz:>5s})  "
                    f"Lay: {lay:>6s} ({lay_sz:>5s})  "
                    f"{spread:>12s}  {ltp}"
                )


def save_results(markets: list[dict], output_dir: Path) -> None:
    """Save raw market data as CSV."""
    import pandas as pd

    rows = []
    for m in markets:
        for r in m["runners"]:
            rows.append(
                {
                    "event_name": m["event_name"],
                    "market_type": m["market_type"],
                    "market_start": m["market_start"],
                    "hours_to_start": m["hours_to_start"],
                    "status": m["status"],
                    "total_matched": m["total_matched"],
                    "overround": m["overround"],
                    "runner_name": r["name"],
                    "best_back": r["best_back"],
                    "back_size": r["back_size"],
                    "best_lay": r["best_lay"],
                    "lay_size": r["lay_size"],
                    "spread": r["spread"],
                    "spread_pct": r["spread_pct"],
                    "last_price_traded": r["last_price_traded"],
                    "runner_matched": r["total_matched"],
                }
            )

    if rows:
        df = pd.DataFrame(rows)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"nba_markets_{datetime.now(UTC).strftime('%Y%m%d_%H%M')}.csv"
        df.to_csv(path, index=False)
        print(f"\nSaved {len(rows)} rows to {path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Connecting to Betfair API...")
    client = create_client()

    cert_file = os.environ.get("BETFAIR_CERT_FILE")
    if cert_file:
        client.login()
    else:
        client.login_interactive()
    print("Logged in successfully\n")

    # Step 1: Find NBA competition
    print("Basketball competitions:")
    competitions = explore_competitions(client)

    nba_ids = [c["id"] for c in competitions if "NBA" in c["name"].upper()]
    if not nba_ids:
        print("\nNo NBA competition found. Available competitions listed above.")
        print("Try setting competition_ids manually if NBA is listed under a different name.")
        client.logout()
        return

    print(f"\nUsing NBA competition IDs: {nba_ids}")

    # Step 2: List upcoming events
    print("\nUpcoming NBA events:")
    events = explore_events(client, nba_ids)

    if not events:
        print("No upcoming NBA events found.")
        client.logout()
        return

    # Step 3: Get market data for all events
    event_ids = [e["id"] for e in events]
    print(f"\nFetching market data for {len(event_ids)} events...")
    markets = explore_markets(client, event_ids)

    # Step 4: Print summary
    print_market_summary(markets)

    # Step 5: Aggregate stats
    match_odds = [m for m in markets if m.get("market_type") == "MATCH_ODDS"]
    if match_odds:
        spreads = []
        for m in match_odds:
            for r in m["runners"]:
                if r["spread_pct"] is not None:
                    spreads.append(r["spread_pct"])

        overrounds = [m["overround"] for m in match_odds if m["overround"] is not None]
        matched_vols = [m["total_matched"] for m in match_odds if m["total_matched"]]

        print(f"\n{'=' * 70}")
        print("  MATCH ODDS SUMMARY")
        print(f"{'=' * 70}")
        print(f"  Markets found: {len(match_odds)}")

        if spreads:
            import numpy as np

            print(
                f"  Back-lay spread: {np.median(spreads):.2f}% median, {np.mean(spreads):.2f}% mean"
            )
            print(f"  Spread range: {min(spreads):.2f}% - {max(spreads):.2f}%")

        if overrounds:
            import numpy as np

            print(
                f"  Back overround: {np.median(overrounds) * 100:.2f}% median, "
                f"{np.mean(overrounds) * 100:.2f}% mean"
            )

        if matched_vols:
            import numpy as np

            print(
                f"  Matched volume: £{np.median(matched_vols):,.0f} median, "
                f"£{np.sum(matched_vols):,.0f} total"
            )

    # Save data
    save_results(markets, OUTPUT_DIR)

    client.logout()
    print("\nDone.")


if __name__ == "__main__":
    main()
