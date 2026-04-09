"""CLI commands for paper trade management."""

import asyncio

import typer
from odds_core.database import async_session_maker
from odds_lambda.paper_trading import (
    get_exposure_by_event,
    get_open_trades,
    get_portfolio_summary,
    get_settled_trades,
    settle_trades,
)
from rich.console import Console
from rich.table import Table

app = typer.Typer()
console = Console()


@app.command("list")
def list_trades(
    settled: bool = typer.Option(False, "--settled", "-s", help="Show settled trades"),
    limit: int = typer.Option(50, "--limit", "-n", help="Maximum trades to show"),
) -> None:
    """List open or settled paper trades."""
    asyncio.run(_list_trades(settled, limit))


async def _list_trades(settled: bool, limit: int) -> None:
    header = "Settled" if settled else "Open"
    console.print(f"\n[bold blue]{header} Paper Trades[/bold blue]\n")

    try:
        async with async_session_maker() as session:
            if settled:
                trades = await get_settled_trades(session, limit=limit)
            else:
                trades = await get_open_trades(session)

            if not trades:
                console.print(f"[yellow]No {header.lower()} trades found[/yellow]\n")
                return

            table = Table()
            table.add_column("ID", style="cyan", justify="right")
            table.add_column("Event")
            table.add_column("Selection")
            table.add_column("Bookmaker")
            table.add_column("Odds", justify="right")
            table.add_column("Stake", justify="right")
            if settled:
                table.add_column("Result")
                table.add_column("P&L", justify="right")

            for trade in trades:
                row = [
                    str(trade.id),
                    trade.event_id[:12] + "...",
                    trade.selection,
                    trade.bookmaker,
                    str(trade.odds),
                    f"{trade.stake:.2f}",
                ]
                if settled:
                    result_color = {
                        "win": "green",
                        "loss": "red",
                        "push": "yellow",
                        "void": "dim",
                    }.get(trade.result.value if trade.result else "", "white")
                    result_str = trade.result.value if trade.result else "-"
                    pnl_str = f"{trade.pnl:+.2f}" if trade.pnl is not None else "-"
                    row.append(f"[{result_color}]{result_str}[/{result_color}]")
                    row.append(pnl_str)
                table.add_row(*row)

            console.print(table)
            console.print(f"\n{len(trades)} trade(s)\n")

    except Exception as e:
        console.print(f"\n[bold red]Failed: {e}[/bold red]")
        raise typer.Exit(code=1) from e


@app.command("settle")
def settle() -> None:
    """Settle unsettled trades against final event scores."""
    asyncio.run(_settle())


async def _settle() -> None:
    console.print("\n[bold blue]Settling Paper Trades[/bold blue]\n")

    try:
        async with async_session_maker() as session:
            settled = await settle_trades(session)
            await session.commit()

            if not settled:
                console.print("[yellow]No trades to settle[/yellow]\n")
                return

            table = Table()
            table.add_column("ID", style="cyan", justify="right")
            table.add_column("Event")
            table.add_column("Selection")
            table.add_column("Result")
            table.add_column("P&L", justify="right")

            for trade in settled:
                result_color = {
                    "win": "green",
                    "loss": "red",
                    "push": "yellow",
                }.get(trade.result.value if trade.result else "", "white")
                result_str = trade.result.value if trade.result else "-"
                pnl_str = f"{trade.pnl:+.2f}" if trade.pnl is not None else "-"

                table.add_row(
                    str(trade.id),
                    trade.event_id[:12] + "...",
                    trade.selection,
                    f"[{result_color}]{result_str}[/{result_color}]",
                    pnl_str,
                )

            console.print(table)
            total_pnl = sum(t.pnl for t in settled if t.pnl is not None)
            console.print(f"\nSettled {len(settled)} trade(s), net P&L: {total_pnl:+.2f}\n")

    except Exception as e:
        console.print(f"\n[bold red]Failed: {e}[/bold red]")
        raise typer.Exit(code=1) from e


@app.command("summary")
def summary(
    bankroll: float = typer.Option(1000.0, "--bankroll", "-b", help="Initial bankroll"),
) -> None:
    """Show portfolio summary and exposure."""
    asyncio.run(_summary(bankroll))


async def _summary(initial_bankroll: float) -> None:
    console.print("\n[bold blue]Paper Trading Portfolio[/bold blue]\n")

    try:
        async with async_session_maker() as session:
            portfolio = await get_portfolio_summary(session, initial_bankroll=initial_bankroll)

            table = Table(show_header=False, box=None)
            table.add_column("Metric", style="cyan")
            table.add_column("Value")

            pnl_color = "green" if portfolio.total_pnl >= 0 else "red"
            table.add_row("Initial Bankroll", f"${initial_bankroll:,.2f}")
            table.add_row("Current Bankroll", f"${portfolio.current_bankroll:,.2f}")
            table.add_row(
                "Total P&L",
                f"[{pnl_color}]{portfolio.total_pnl:+,.2f}[/{pnl_color}]",
            )
            table.add_row(
                "ROI",
                f"[{pnl_color}]{portfolio.roi:+.2f}%[/{pnl_color}]",
            )
            table.add_row("Total Staked", f"${portfolio.total_staked:,.2f}")
            table.add_row("", "")
            table.add_row("Total Trades", str(portfolio.total_trades))
            table.add_row("Settled", str(portfolio.settled_trades))
            table.add_row("Open", str(portfolio.open_trades))
            table.add_row(
                "Record",
                f"{portfolio.win_count}W / {portfolio.loss_count}L / {portfolio.push_count}P",
            )

            console.print(table)

            # Exposure breakdown
            exposure = await get_exposure_by_event(session)
            if exposure:
                console.print("\n[bold]Open Exposure by Event:[/bold]")
                exp_table = Table()
                exp_table.add_column("Event ID")
                exp_table.add_column("Exposure", justify="right")

                for event_id, amount in exposure:
                    exp_table.add_row(event_id[:12] + "...", f"${amount:,.2f}")

                console.print(exp_table)

            console.print()

    except Exception as e:
        console.print(f"\n[bold red]Failed: {e}[/bold red]")
        raise typer.Exit(code=1) from e
