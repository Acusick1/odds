"""Polymarket data operations commands."""

import asyncio

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

app = typer.Typer(help="Polymarket prediction market operations")
console = Console()


@app.command("backfill")
def backfill(
    include_spreads: bool = typer.Option(
        False,
        "--include-spreads",
        help="Include spread market price histories",
    ),
    include_totals: bool = typer.Option(
        False,
        "--include-totals",
        help="Include total (over/under) market price histories",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Simulate execution without storing data",
    ),
):
    """
    Backfill historical Polymarket price data from closed NBA markets.

    This command captures historical price data before it expires. The CLOB
    /prices-history endpoint has a ~30-day rolling retention window.

    Behavior:
    - Fetches all closed NBA events from Gamma API
    - Checks which moneyline markets need backfill (skips if >10 snapshots exist)
    - Fetches 5-minute resolution price history from CLOB API
    - Bulk inserts into polymarket_price_snapshots table
    - Optionally backfills spread/total markets if flags enabled

    Examples:
        # Backfill moneyline markets only (default)
        odds polymarket backfill

        # Backfill all market types
        odds polymarket backfill --include-spreads --include-totals

        # Test without storing data
        odds polymarket backfill --dry-run

    Notes:
    - Safe to run repeatedly - skip logic prevents re-fetching
    - Gracefully handles markets with no available data (old games)
    - Individual market failures don't crash the job
    - Logged to polymarket_fetch_logs table
    """
    console.print("\n[bold cyan]Polymarket Historical Price Backfill[/bold cyan]\n")

    if dry_run:
        console.print("[yellow]DRY RUN - No data will be stored[/yellow]\n")

    asyncio.run(_backfill_async(include_spreads, include_totals, dry_run))


async def _backfill_async(
    include_spreads: bool,
    include_totals: bool,
    dry_run: bool,
):
    """Async implementation of backfill command."""
    # Display configuration
    config_table = Table(show_header=False, box=None, padding=(0, 2))
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="white")

    config_table.add_row("Include spreads", "Yes" if include_spreads else "No")
    config_table.add_row("Include totals", "Yes" if include_totals else "No")
    config_table.add_row("Dry run", "Yes" if dry_run else "No")

    console.print(config_table)
    console.print()

    # Import job module
    from odds_lambda.jobs import backfill_polymarket

    # Execute with progress indicator
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching and backfilling price histories...", total=None)

        try:
            await backfill_polymarket.main(
                include_spreads=include_spreads,
                include_totals=include_totals,
                dry_run=dry_run,
            )
            progress.update(task, completed=True)

        except Exception as e:
            progress.stop()
            console.print(f"\n[red]Backfill failed: {e}[/red]")
            raise typer.Exit(1) from e

    console.print("\n[green]Backfill completed successfully![/green]")
    console.print(
        "\n[dim]Check logs for detailed statistics (events processed, markets backfilled, etc.)[/dim]\n"
    )


@app.command("discover")
def discover():
    """List active NBA events currently on Polymarket."""
    asyncio.run(_discover_async())


async def _discover_async():
    """Async implementation of discover command."""
    from odds_lambda.polymarket_fetcher import PolymarketClient

    console.print("\n[bold cyan]Active Polymarket NBA Events[/bold cyan]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching events from Gamma API...", total=None)
        try:
            async with PolymarketClient() as client:
                events = await client.get_nba_events(active=True, closed=False)
            progress.update(task, completed=True)
        except Exception as e:
            progress.stop()
            console.print(f"\n[red]Failed to fetch events: {e}[/red]")
            raise typer.Exit(1) from e

    if not events:
        console.print("[yellow]No active NBA events found on Polymarket[/yellow]\n")
        return

    table = Table()
    table.add_column("Ticker", style="cyan")
    table.add_column("Title")
    table.add_column("Start Date")
    table.add_column("Markets", justify="right")
    table.add_column("Volume", justify="right")
    table.add_column("Liquidity", justify="right")

    for event in events:
        ticker = event.get("ticker", "")
        title = event.get("title", "")
        start_date = event.get("startDate", "")[:10] if event.get("startDate") else ""
        markets = event.get("markets", [])
        volume = event.get("volume")
        liquidity = event.get("liquidity")

        volume_str = f"${float(volume):,.0f}" if volume is not None else "-"
        liquidity_str = f"${float(liquidity):,.0f}" if liquidity is not None else "-"

        table.add_row(
            ticker,
            title,
            start_date,
            str(len(markets)),
            volume_str,
            liquidity_str,
        )

    console.print(table)
    console.print(f"\n[dim]{len(events)} active event(s)[/dim]\n")


@app.command("status")
def status():
    """Show Polymarket pipeline health and collection statistics."""
    asyncio.run(_status_async())


async def _status_async():
    """Async implementation of status command."""
    from datetime import UTC, datetime

    from odds_core.database import async_session_maker
    from odds_lambda.storage.polymarket_reader import PolymarketReader

    console.print("\n[bold cyan]Polymarket Pipeline Status[/bold cyan]\n")

    try:
        async with async_session_maker() as session:
            reader = PolymarketReader(session)
            stats = await reader.get_pipeline_stats()

        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Total Events", f"{stats['total_events']:,}")
        table.add_row("  Linked", f"[green]{stats['linked_events']:,}[/green]")
        table.add_row("  Unlinked", f"[yellow]{stats['unlinked_events']:,}[/yellow]")
        table.add_row("Total Markets", f"{stats['total_markets']:,}")
        table.add_row("Total Snapshots", f"{stats['total_snapshots']:,}")

        if stats["earliest_event"] and stats["latest_event"]:
            earliest = stats["earliest_event"].strftime("%Y-%m-%d")
            latest = stats["latest_event"].strftime("%Y-%m-%d")
            table.add_row("Date Coverage", f"{earliest} → {latest}")

        log = stats["latest_fetch_log"]
        if log:
            time_ago = datetime.now(UTC) - log.fetch_time
            minutes_ago = int(time_ago.total_seconds() / 60)
            status_icon = "[green]✓[/green]" if log.success else "[red]✗[/red]"
            table.add_row("Last Fetch", f"{status_icon} {minutes_ago}m ago ({log.job_type})")
        else:
            table.add_row("Last Fetch", "[yellow]No fetches yet[/yellow]")

        console.print(table)
        console.print()

    except Exception as e:
        console.print(f"\n[bold red]✗ Failed to get status: {e}[/bold red]")
        raise typer.Exit(1) from e


@app.command("link")
def link(
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show matches without updating the database",
    ),
):
    """Link unlinked Polymarket events to internal sportsbook Event records."""
    if dry_run:
        console.print("[yellow]DRY RUN - No changes will be written[/yellow]\n")
    asyncio.run(_link_async(dry_run))


async def _link_async(dry_run: bool):
    """Async implementation of link command."""
    from odds_core.database import async_session_maker
    from odds_lambda.polymarket_matching import match_polymarket_event
    from odds_lambda.storage.polymarket_reader import PolymarketReader

    console.print("\n[bold cyan]Polymarket Event Linking[/bold cyan]\n")

    try:
        async with async_session_maker() as session:
            reader = PolymarketReader(session)
            unlinked = await reader.get_unlinked_events()

            if not unlinked:
                console.print("[green]All Polymarket events are already linked.[/green]\n")
                return

            console.print(f"Found {len(unlinked)} unlinked event(s). Running matcher...\n")

            matched = 0
            unmatched = 0
            results_table = Table()
            results_table.add_column("Ticker", style="cyan")
            results_table.add_column("Title")
            results_table.add_column("Result")

            for pm_event in unlinked:
                event_id = await match_polymarket_event(
                    session, pm_event.ticker, pm_event.start_date
                )

                if event_id:
                    matched += 1
                    results_table.add_row(
                        pm_event.ticker,
                        pm_event.title,
                        f"[green]→ {event_id[:12]}...[/green]",
                    )
                    if not dry_run:
                        pm_event.event_id = event_id
                else:
                    unmatched += 1
                    results_table.add_row(
                        pm_event.ticker,
                        pm_event.title,
                        "[yellow]no match[/yellow]",
                    )

            if not dry_run and matched > 0:
                await session.commit()

        console.print(results_table)
        console.print(
            f"\n[green]Matched: {matched}[/green]  " f"[yellow]Unmatched: {unmatched}[/yellow]"
        )
        if dry_run:
            console.print("\n[dim]Dry run — no changes written.[/dim]")
        console.print()

    except Exception as e:
        console.print(f"\n[bold red]✗ Linking failed: {e}[/bold red]")
        raise typer.Exit(1) from e


@app.command("book")
def book(
    ticker: str = typer.Argument(help="Polymarket ticker, e.g. nba-dal-mil-2026-01-25"),
):
    """Show the current order book for a Polymarket game."""
    asyncio.run(_book_async(ticker))


async def _book_async(ticker: str):
    """Async implementation of book command."""
    from odds_core.database import async_session_maker
    from odds_lambda.polymarket_fetcher import PolymarketClient
    from odds_lambda.polymarket_matching import parse_ticker
    from odds_lambda.storage.polymarket_reader import PolymarketReader

    if not parse_ticker(ticker):
        console.print(
            f"[red]Invalid ticker format: {ticker!r}. "
            "Expected nba-{{away}}-{{home}}-{{yyyy}}-{{mm}}-{{dd}}[/red]\n"
        )
        raise typer.Exit(1)

    console.print(f"\n[bold cyan]Order Book: {ticker}[/bold cyan]\n")

    try:
        async with async_session_maker() as session:
            reader = PolymarketReader(session)
            pm_event = await reader.get_event_by_ticker(ticker)

            if not pm_event:
                console.print(f"[red]No Polymarket event found for ticker: {ticker}[/red]\n")
                raise typer.Exit(1)

            if pm_event.id is None:
                console.print("[red]Event record has no database ID[/red]\n")
                raise typer.Exit(1)

            market = await reader.get_moneyline_market(pm_event.id)

            if not market:
                console.print(f"[yellow]No moneyline market found for: {ticker}[/yellow]\n")
                raise typer.Exit(1)

            token_ids = list(market.clob_token_ids[:2])
            outcomes = (
                list(market.outcomes) if len(market.outcomes) >= 2 else ["Outcome 0", "Outcome 1"]
            )

        if len(token_ids) < 2:
            console.print("[red]Market has insufficient token IDs[/red]\n")
            raise typer.Exit(1)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Fetching order books...", total=None)
            async with PolymarketClient() as client:
                books = await client.get_order_books_batch(token_ids)
            progress.update(task, completed=True)

        for token_id, outcome in zip(token_ids, outcomes, strict=False):
            raw_book = books.get(token_id, {})
            bids = sorted(raw_book.get("bids", []), key=lambda x: float(x["price"]), reverse=True)
            asks = sorted(raw_book.get("asks", []), key=lambda x: float(x["price"]))

            console.print(f"[bold]{outcome}[/bold]")
            side_table = Table()
            side_table.add_column("Bid Price", justify="right", style="green")
            side_table.add_column("Bid Size", justify="right")
            side_table.add_column("Ask Price", justify="right", style="red")
            side_table.add_column("Ask Size", justify="right")

            max_levels = max(len(bids), len(asks), 1)
            for i in range(min(max_levels, 10)):
                bid_price = f"{float(bids[i]['price']):.4f}" if i < len(bids) else ""
                bid_size = f"{float(bids[i]['size']):.2f}" if i < len(bids) else ""
                ask_price = f"{float(asks[i]['price']):.4f}" if i < len(asks) else ""
                ask_size = f"{float(asks[i]['size']):.2f}" if i < len(asks) else ""
                side_table.add_row(bid_price, bid_size, ask_price, ask_size)

            console.print(side_table)

            if bids and asks:
                best_bid = float(bids[0]["price"])
                best_ask = float(asks[0]["price"])
                spread = best_ask - best_bid
                midpoint = (best_bid + best_ask) / 2
                console.print(
                    f"  Spread: [yellow]{spread:.4f}[/yellow]  "
                    f"Midpoint: [cyan]{midpoint:.4f}[/cyan]"
                )
            console.print()

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"\n[bold red]✗ Failed to fetch order book: {e}[/bold red]")
        raise typer.Exit(1) from e
