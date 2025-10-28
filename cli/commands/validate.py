"""CLI commands for data validation."""

import asyncio
import json
import sys
from datetime import datetime, timedelta

import typer
from rich.console import Console
from rich.table import Table

from core.database import async_session_maker
from core.fetch_tier import FetchTier
from storage.tier_validator import TierCoverageValidator

app = typer.Typer()
console = Console()


@app.command("daily")
def validate_daily(
    target_date: str = typer.Option(
        None, "--date", "-d", help="Date to validate (YYYY-MM-DD), defaults to yesterday"
    ),
    strict: bool = typer.Option(
        True, "--strict/--no-strict", help="Require all 5 tiers (default: strict)"
    ),
    output_json: str = typer.Option(None, "--output-json", "-o", help="Save results to JSON file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed game breakdown"),
):
    """
    Validate tier coverage for a specific date.

    Examples:
        # Validate yesterday's data
        odds validate daily

        # Validate specific date
        odds validate daily --date 2024-10-24

        # Allow partial coverage (just show warnings)
        odds validate daily --date 2024-10-24 --no-strict

        # Save results to JSON
        odds validate daily --date 2024-10-24 --output-json results.json
    """
    asyncio.run(_validate_daily(target_date, strict, output_json, verbose))


async def _validate_daily(
    target_date_str: str | None, strict: bool, output_json: str | None, verbose: bool
):
    """Async implementation of daily validation."""
    # Parse target date (default to yesterday)
    if target_date_str:
        try:
            target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
        except ValueError:
            console.print(
                f"[red]Error: Invalid date format '{target_date_str}'. Use YYYY-MM-DD[/red]"
            )
            sys.exit(1)
    else:
        target_date = (datetime.now() - timedelta(days=1)).date()

    console.print(f"\n[bold blue]Tier Coverage Validation - {target_date}[/bold blue]\n")

    # Determine required tiers
    if strict:
        required_tiers = TierCoverageValidator.DEFAULT_REQUIRED_TIERS
    else:
        # Just require opening and closing for non-strict mode
        required_tiers = {FetchTier.OPENING, FetchTier.CLOSING}

    # Run validation
    async with async_session_maker() as session:
        validator = TierCoverageValidator(session)
        report = await validator.validate_date(target_date, required_tiers)

    # Display summary
    _display_summary(report, required_tiers)

    # Display tier breakdown
    _display_tier_breakdown(report)

    # Display incomplete games if any
    if report.incomplete_games > 0:
        _display_incomplete_games(report, verbose)

    # Save to JSON if requested
    if output_json:
        _save_to_json(report, output_json)
        console.print(f"\n[green]Results saved to {output_json}[/green]")

    # Exit with appropriate code
    if not report.is_valid:
        console.print(f"\n[red]Status: FAILED - {report.incomplete_games} games incomplete[/red]")
        sys.exit(1)
    else:
        console.print("\n[green]Status: PASSED - All games complete[/green]")
        sys.exit(0)


def _display_summary(report, required_tiers):
    """Display validation summary table."""
    table = Table(show_header=False, box=None)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Total Games", f"{report.total_games}")
    table.add_row(
        "Complete",
        f"[green]{report.complete_games} ({report.completion_rate:.1f}%)[/green]"
        if report.complete_games == report.total_games
        else f"{report.complete_games} ({report.completion_rate:.1f}%)",
    )
    if report.incomplete_games > 0:
        table.add_row("Incomplete", f"[red]{report.incomplete_games}[/red]")
    table.add_row("Required Tiers", f"{len(required_tiers)}/5")

    console.print(table)
    console.print()


def _display_tier_breakdown(report):
    """Display tier coverage breakdown."""
    console.print("[bold]Tier Coverage Breakdown:[/bold]")

    # Calculate coverage for each tier
    tier_table = Table(show_header=True)
    tier_table.add_column("Tier", style="cyan")
    tier_table.add_column("Games with Tier", justify="right")
    tier_table.add_column("Coverage %", justify="right")

    all_tiers = [
        FetchTier.OPENING,
        FetchTier.EARLY,
        FetchTier.SHARP,
        FetchTier.PREGAME,
        FetchTier.CLOSING,
    ]

    for tier in all_tiers:
        games_missing = report.missing_tier_breakdown.get(tier, 0)
        games_with = report.total_games - games_missing
        coverage_pct = (games_with / report.total_games * 100) if report.total_games > 0 else 0

        # Color coding
        if coverage_pct == 100:
            coverage_str = f"[green]{coverage_pct:.1f}%[/green]"
        elif coverage_pct >= 80:
            coverage_str = f"[yellow]{coverage_pct:.1f}%[/yellow]"
        else:
            coverage_str = f"[red]{coverage_pct:.1f}%[/red]"

        tier_table.add_row(tier.value.upper(), f"{games_with}/{report.total_games}", coverage_str)

    console.print(tier_table)
    console.print()


def _display_incomplete_games(report, verbose):
    """Display incomplete games."""
    console.print(f"[bold]Incomplete Games ({report.incomplete_games}):[/bold]\n")

    incomplete_reports = [r for r in report.game_reports if not r.is_complete]

    for game_report in incomplete_reports:
        status_icon = "✗"
        teams = f"{game_report.away_team} @ {game_report.home_team}"
        game_time = game_report.commence_time.strftime("%Y-%m-%d %H:%M")

        console.print(f"  {status_icon} [yellow]{teams}[/yellow] ({game_time})")

        # Show missing tiers
        missing = ", ".join(sorted([t.value.upper() for t in game_report.tiers_missing]))
        console.print(f"     Missing: [red]{missing}[/red]")

        if verbose:
            # Show present tiers
            present = ", ".join(sorted([t.value.upper() for t in game_report.tiers_present]))
            console.print(
                f"     Has: {present} ({len(game_report.tiers_present)}/{len(game_report.tiers_present) + len(game_report.tiers_missing)} tiers)"
            )
            console.print(f"     Total snapshots: {game_report.total_snapshots}")

        console.print()


def _save_to_json(report, output_path: str):
    """Save validation report to JSON file."""
    data = {
        "target_date": str(report.target_date),
        "validation_time": report.validation_time.isoformat(),
        "summary": {
            "total_games": report.total_games,
            "complete_games": report.complete_games,
            "incomplete_games": report.incomplete_games,
            "completion_rate": report.completion_rate,
            "is_valid": report.is_valid,
        },
        "tier_coverage": {
            tier.value: report.total_games - report.missing_tier_breakdown.get(tier, 0)
            for tier in [
                FetchTier.OPENING,
                FetchTier.EARLY,
                FetchTier.SHARP,
                FetchTier.PREGAME,
                FetchTier.CLOSING,
            ]
        },
        "incomplete_games": [
            {
                "event_id": r.event_id,
                "home_team": r.home_team,
                "away_team": r.away_team,
                "commence_time": r.commence_time.isoformat(),
                "tiers_present": [t.value for t in r.tiers_present],
                "tiers_missing": [t.value for t in r.tiers_missing],
                "total_snapshots": r.total_snapshots,
            }
            for r in report.game_reports
            if not r.is_complete
        ],
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


@app.command("game")
def validate_game(
    event_id: str = typer.Argument(..., help="Event ID to validate"),
    show_snapshots: bool = typer.Option(False, "--show-snapshots", help="Show all snapshots"),
):
    """
    Validate tier coverage for a specific game.

    Example:
        odds validate game abc123 --show-snapshots
    """
    asyncio.run(_validate_game(event_id, show_snapshots))


async def _validate_game(event_id: str, show_snapshots: bool):
    """Async implementation of game validation."""
    console.print(f"\n[bold blue]Tier Coverage for Event {event_id}[/bold blue]\n")

    async with async_session_maker() as session:
        validator = TierCoverageValidator(session)
        try:
            report = await validator.validate_game(event_id)
        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)

    # Display game info
    console.print(f"[bold]{report.away_team} @ {report.home_team}[/bold]")
    console.print(f"Game Time: {report.commence_time.strftime('%Y-%m-%d %H:%M %Z')}")
    console.print(f"Total Snapshots: {report.total_snapshots}\n")

    # Display tier coverage
    table = Table(show_header=True)
    table.add_column("Tier", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Snapshots", justify="right")

    all_tiers = [
        FetchTier.OPENING,
        FetchTier.EARLY,
        FetchTier.SHARP,
        FetchTier.PREGAME,
        FetchTier.CLOSING,
    ]

    for tier in all_tiers:
        if tier in report.tiers_present:
            status = "[green]✓[/green]"
            count = report.snapshots_by_tier.get(tier, 0)
        else:
            status = "[red]✗[/red]"
            count = 0

        table.add_row(tier.value.upper(), status, str(count))

    console.print(table)

    # Summary
    if report.is_complete:
        console.print("\n[green]Status: Complete - All tiers present[/green]")
    else:
        missing = ", ".join([t.value.upper() for t in report.tiers_missing])
        console.print(f"\n[yellow]Status: Incomplete - Missing: {missing}[/yellow]")


@app.command("tier-coverage")
def validate_tier_coverage(
    start_date: str = typer.Option(..., "--start", "-s", help="Start date (YYYY-MM-DD)"),
    end_date: str = typer.Option(..., "--end", "-e", help="End date (YYYY-MM-DD)"),
    output_json: str = typer.Option(None, "--output-json", "-o", help="Save results to JSON"),
):
    """
    Validate tier coverage for a date range.

    Example:
        odds validate tier-coverage --start 2024-10-01 --end 2024-10-31
    """
    asyncio.run(_validate_tier_coverage(start_date, end_date, output_json))


async def _validate_tier_coverage(start_date_str: str, end_date_str: str, output_json: str | None):
    """Async implementation of tier coverage validation."""
    # Parse dates
    try:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    except ValueError:
        console.print("[red]Error: Invalid date format. Use YYYY-MM-DD[/red]")
        sys.exit(1)

    console.print(f"\n[bold blue]Tier Coverage Analysis - {start_date} to {end_date}[/bold blue]\n")

    # Run validation
    async with async_session_maker() as session:
        validator = TierCoverageValidator(session)
        reports = await validator.validate_date_range(start_date, end_date)

    # Display summary table
    table = Table(show_header=True)
    table.add_column("Date", style="cyan")
    table.add_column("Games", justify="right")
    table.add_column("Complete", justify="right")
    table.add_column("Incomplete", justify="right")
    table.add_column("Coverage %", justify="right")

    total_games = 0
    total_complete = 0

    for report in reports:
        if report.total_games == 0:
            continue

        total_games += report.total_games
        total_complete += report.complete_games

        # Color code completion rate
        if report.completion_rate == 100:
            coverage_str = f"[green]{report.completion_rate:.1f}%[/green]"
        elif report.completion_rate >= 80:
            coverage_str = f"[yellow]{report.completion_rate:.1f}%[/yellow]"
        else:
            coverage_str = f"[red]{report.completion_rate:.1f}%[/red]"

        table.add_row(
            str(report.target_date),
            str(report.total_games),
            str(report.complete_games),
            str(report.incomplete_games),
            coverage_str,
        )

    console.print(table)

    # Overall summary
    overall_rate = (total_complete / total_games * 100) if total_games > 0 else 0
    console.print(
        f"\n[bold]Overall: {total_complete}/{total_games} games complete ({overall_rate:.1f}%)[/bold]"
    )

    # Save to JSON if requested
    if output_json:
        data = {
            "start_date": str(start_date),
            "end_date": str(end_date),
            "overall_summary": {
                "total_games": total_games,
                "complete_games": total_complete,
                "completion_rate": overall_rate,
            },
            "daily_reports": [
                {
                    "date": str(r.target_date),
                    "total_games": r.total_games,
                    "complete_games": r.complete_games,
                    "incomplete_games": r.incomplete_games,
                    "completion_rate": r.completion_rate,
                }
                for r in reports
                if r.total_games > 0
            ],
        }

        with open(output_json, "w") as f:
            json.dump(data, f, indent=2)

        console.print(f"\n[green]Results saved to {output_json}[/green]")
