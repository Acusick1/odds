"""CLI commands for system status and monitoring."""

import asyncio
from datetime import UTC, datetime, timedelta

import typer
from rich.console import Console
from rich.table import Table

from core.config import settings
from core.database import async_session_maker
from storage.readers import OddsReader

app = typer.Typer()
console = Console()


@app.command("show")
def show_status(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed statistics"),
):
    """Show system health and statistics."""
    asyncio.run(_show_status(verbose))


async def _show_status(verbose: bool):
    """Async implementation of show status."""
    console.print("\n[bold blue]Odds Pipeline Status[/bold blue]\n")

    try:
        async with async_session_maker() as session:
            reader = OddsReader(session)

            # Get database stats
            stats = await reader.get_database_stats()

            # Get latest fetch log
            fetch_logs = await reader.get_fetch_logs(limit=1)
            latest_fetch = fetch_logs[0] if fetch_logs else None

            # Create status table
            table = Table(show_header=False, box=None)
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="white")

            # Last fetch
            if latest_fetch:
                time_ago = datetime.now(UTC) - latest_fetch.fetch_time
                minutes_ago = int(time_ago.total_seconds() / 60)
                status_icon = "✓" if latest_fetch.success else "✗"
                status_color = "green" if latest_fetch.success else "red"

                table.add_row(
                    "Last Fetch",
                    f"[{status_color}]{status_icon}[/{status_color}] " f"{minutes_ago} minutes ago",
                )
            else:
                table.add_row("Last Fetch", "[yellow]No fetches yet[/yellow]")

            # Events
            events_by_status = stats.get("events_by_status", {})
            table.add_row("Events (Scheduled)", f"{events_by_status.get('scheduled', 0):,}")
            table.add_row("Events (Total)", f"{stats.get('total_events', 0):,}")

            # Odds records
            table.add_row("Odds Records", f"{stats.get('total_odds_records', 0):,}")
            table.add_row("Snapshots", f"{stats.get('total_snapshots', 0):,}")

            # API quota
            quota_remaining = stats.get("api_quota_remaining")
            if quota_remaining is not None:
                quota_percent = (quota_remaining / settings.odds_api_quota) * 100
                quota_color = (
                    "green" if quota_percent > 50 else ("yellow" if quota_percent > 20 else "red")
                )
                table.add_row(
                    "API Quota",
                    f"[{quota_color}]{quota_remaining:,} / {settings.odds_api_quota:,} "
                    f"({quota_percent:.1f}%)[/{quota_color}]",
                )

            # Fetch success rate
            success_rate = stats.get("fetch_success_rate_24h", 0)
            rate_color = (
                "green" if success_rate > 95 else ("yellow" if success_rate > 80 else "red")
            )
            table.add_row("Success Rate (24h)", f"[{rate_color}]{success_rate}%[/{rate_color}]")

            console.print(table)

            # Verbose mode
            if verbose:
                console.print("\n[bold]Events by Status:[/bold]")
                status_table = Table()
                status_table.add_column("Status", style="cyan")
                status_table.add_column("Count", justify="right")

                for status, count in events_by_status.items():
                    status_table.add_row(status.title(), f"{count:,}")

                console.print(status_table)

                # Recent data quality issues
                console.print("\n[bold]Recent Data Quality Issues (last 24h):[/bold]")
                day_ago = datetime.now(UTC) - timedelta(hours=24)
                quality_logs = await reader.get_data_quality_logs(start_time=day_ago, limit=10)

                if quality_logs:
                    quality_table = Table()
                    quality_table.add_column("Time")
                    quality_table.add_column("Severity")
                    quality_table.add_column("Type")
                    quality_table.add_column("Description")

                    for log in quality_logs:
                        time_ago = datetime.now(UTC) - log.created_at
                        minutes_ago = int(time_ago.total_seconds() / 60)

                        severity_color = {
                            "warning": "yellow",
                            "error": "red",
                            "critical": "bold red",
                        }.get(log.severity, "white")

                        quality_table.add_row(
                            f"{minutes_ago}m ago",
                            f"[{severity_color}]{log.severity}[/{severity_color}]",
                            log.issue_type,
                            log.description[:50] + ("..." if len(log.description) > 50 else ""),
                        )

                    console.print(quality_table)
                else:
                    console.print("[green]No issues in last 24 hours[/green]")

            console.print()

    except Exception as e:
        console.print(f"\n[bold red]✗ Failed to get status: {str(e)}[/bold red]")
        raise typer.Exit(code=1) from e


@app.command("quota")
def show_quota():
    """Show API quota usage."""
    asyncio.run(_show_quota())


async def _show_quota():
    """Async implementation of show quota."""
    console.print("\n[bold blue]API Quota Status[/bold blue]\n")

    try:
        async with async_session_maker() as session:
            reader = OddsReader(session)

            # Get recent fetch logs to see quota usage
            fetch_logs = await reader.get_fetch_logs(limit=10)

            if not fetch_logs:
                console.print("[yellow]No fetch logs available[/yellow]\n")
                return

            # Current quota
            latest = fetch_logs[0]
            if latest.api_quota_remaining is not None:
                quota_used = settings.odds_api_quota - latest.api_quota_remaining
                quota_percent = (latest.api_quota_remaining / settings.odds_api_quota) * 100

                table = Table(show_header=False, box=None)
                table.add_column("Metric", style="cyan")
                table.add_column("Value")

                quota_color = (
                    "green" if quota_percent > 50 else ("yellow" if quota_percent > 20 else "red")
                )

                table.add_row("Total Quota", f"{settings.odds_api_quota:,}")
                table.add_row("Used", f"{quota_used:,}")
                table.add_row(
                    "Remaining",
                    f"[{quota_color}]{latest.api_quota_remaining:,} ({quota_percent:.1f}%)[/{quota_color}]",
                )

                console.print(table)

                # Quota usage trend
                console.print("\n[bold]Recent Usage:[/bold]")
                trend_table = Table()
                trend_table.add_column("Time")
                trend_table.add_column("Remaining", justify="right")
                trend_table.add_column("Used", justify="right")

                for log in fetch_logs:
                    if log.api_quota_remaining is not None:
                        time_ago = datetime.now(UTC) - log.fetch_time
                        minutes_ago = int(time_ago.total_seconds() / 60)
                        used = settings.odds_api_quota - log.api_quota_remaining

                        trend_table.add_row(
                            f"{minutes_ago}m ago",
                            f"{log.api_quota_remaining:,}",
                            f"+{used - quota_used}" if used != quota_used else "0",
                        )

                console.print(trend_table)

            console.print()

    except Exception as e:
        console.print(f"\n[bold red]✗ Failed to get quota: {str(e)}[/bold red]")
        raise typer.Exit(code=1) from e


@app.command("events")
def show_events(
    days: int = typer.Option(7, "--days", "-d", help="Number of days to look back"),
    team: str = typer.Option(None, "--team", "-t", help="Filter by team name"),
):
    """Show recent events."""
    asyncio.run(_show_events(days, team))


async def _show_events(days: int, team: str | None):
    """Async implementation of show events."""
    console.print(f"\n[bold blue]Events (last {days} days)[/bold blue]\n")

    try:
        async with async_session_maker() as session:
            reader = OddsReader(session)

            if team:
                # Filter by team
                start_date = datetime.now(UTC) - timedelta(days=days)
                events = await reader.get_events_by_team(team_name=team, start_date=start_date)
            else:
                # Get all events in range
                end_date = datetime.now(UTC) + timedelta(days=1)
                start_date = end_date - timedelta(days=days)
                events = await reader.get_events_by_date_range(
                    start_date=start_date, end_date=end_date
                )

            if not events:
                console.print("[yellow]No events found[/yellow]\n")
                return

            # Display events
            table = Table()
            table.add_column("ID", style="cyan")
            table.add_column("Date/Time")
            table.add_column("Teams")
            table.add_column("Status")
            table.add_column("Score")

            for event in events[:50]:  # Limit to 50
                # Format time
                time_str = event.commence_time.strftime("%Y-%m-%d %H:%M")

                # Format matchup
                matchup = f"{event.away_team} @ {event.home_team}"

                # Format status
                status_color = {
                    "scheduled": "blue",
                    "live": "yellow",
                    "final": "green",
                    "cancelled": "red",
                    "postponed": "red",
                }.get(event.status.value, "white")

                # Format score
                score = ""
                if event.status.value == "final" and event.home_score is not None:
                    score = f"{event.away_score} - {event.home_score}"

                table.add_row(
                    event.id[:8] + "...",
                    time_str,
                    matchup,
                    f"[{status_color}]{event.status.value}[/{status_color}]",
                    score,
                )

            console.print(table)
            console.print(f"\nShowing {min(len(events), 50)} of {len(events)} events\n")

    except Exception as e:
        console.print(f"\n[bold red]✗ Failed to get events: {str(e)}[/bold red]")
        raise typer.Exit(code=1) from e
