"""CLI commands for bet management."""

import asyncio

import typer
from odds_core.bet_models import Bet
from odds_core.database import async_session_maker
from odds_lambda.betting import (
    get_exposure_by_event,
    get_open_bets,
    get_portfolio_summary,
    get_settled_bets,
    settle_bets,
)
from rich.console import Console
from rich.table import Table

app = typer.Typer()
console = Console()


def _selection(bet: Bet) -> str:
    return bet.selection if bet.line is None else f"{bet.selection} {bet.line:g}"


@app.command("list")
def list_bets(
    settled: bool = typer.Option(False, "--settled", "-s", help="Show settled bets"),
    limit: int = typer.Option(50, "--limit", "-n", help="Maximum bets to show"),
) -> None:
    """List open or settled bets."""
    asyncio.run(_list_bets(settled, limit))


async def _list_bets(settled: bool, limit: int) -> None:
    header = "Settled" if settled else "Open"
    console.print(f"\n[bold blue]{header} Bets[/bold blue]\n")

    try:
        async with async_session_maker() as session:
            if settled:
                bets = await get_settled_bets(session, limit=limit)
            else:
                bets = await get_open_bets(session)

            if not bets:
                console.print(f"[yellow]No {header.lower()} bets found[/yellow]\n")
                return

            table = Table()
            table.add_column("ID", style="cyan", justify="right")
            table.add_column("Event")
            table.add_column("Selection")
            table.add_column("Bookmaker")
            table.add_column("Price", justify="right")
            table.add_column("Stake", justify="right")
            if settled:
                table.add_column("Result")
                table.add_column("P&L", justify="right")

            for bet in bets:
                row = [
                    str(bet.id),
                    str(bet.event_id),
                    _selection(bet),
                    bet.bookmaker,
                    f"{bet.price_decimal:.2f}",
                    f"{bet.stake:.2f}",
                ]
                if settled:
                    result_color = {
                        "win": "green",
                        "loss": "red",
                        "push": "yellow",
                        "void": "dim",
                    }.get(bet.result.value if bet.result else "", "white")
                    result_str = bet.result.value if bet.result else "-"
                    pnl_str = f"{bet.pnl:+.2f}" if bet.pnl is not None else "-"
                    row.append(f"[{result_color}]{result_str}[/{result_color}]")
                    row.append(pnl_str)
                table.add_row(*row)

            console.print(table)
            console.print(f"\n{len(bets)} bet(s)\n")

    except Exception as e:
        console.print(f"\n[bold red]Failed: {e}[/bold red]")
        raise typer.Exit(code=1) from e


@app.command("settle")
def settle() -> None:
    """Settle unsettled bets against final event scores."""
    asyncio.run(_settle())


async def _settle() -> None:
    console.print("\n[bold blue]Settling Bets[/bold blue]\n")

    try:
        async with async_session_maker() as session:
            settled = await settle_bets(session)
            await session.commit()

            if not settled:
                console.print("[yellow]No bets to settle[/yellow]\n")
                return

            table = Table()
            table.add_column("ID", style="cyan", justify="right")
            table.add_column("Event")
            table.add_column("Selection")
            table.add_column("Result")
            table.add_column("P&L", justify="right")

            for bet in settled:
                result_color = {
                    "win": "green",
                    "loss": "red",
                    "push": "yellow",
                }.get(bet.result.value if bet.result else "", "white")
                result_str = bet.result.value if bet.result else "-"
                pnl_str = f"{bet.pnl:+.2f}" if bet.pnl is not None else "-"

                table.add_row(
                    str(bet.id),
                    str(bet.event_id),
                    _selection(bet),
                    f"[{result_color}]{result_str}[/{result_color}]",
                    pnl_str,
                )

            console.print(table)
            total_pnl = sum(t.pnl for t in settled if t.pnl is not None)
            console.print(f"\nSettled {len(settled)} bet(s), net P&L: {total_pnl:+.2f}\n")

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
    console.print("\n[bold blue]Betting Portfolio[/bold blue]\n")

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
            table.add_row("Total Bets", str(portfolio.total_bets))
            table.add_row("Settled", str(portfolio.settled_bets))
            table.add_row("Open", str(portfolio.open_bets))
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
                    exp_table.add_row(str(event_id), f"${amount:,.2f}")

                console.print(exp_table)

            console.print()

    except Exception as e:
        console.print(f"\n[bold red]Failed: {e}[/bold red]")
        raise typer.Exit(code=1) from e
