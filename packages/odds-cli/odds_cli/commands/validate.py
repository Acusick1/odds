"""CLI commands for data validation."""

import asyncio
import json
import sys
from datetime import datetime, timedelta

import typer
from odds_core.database import async_session_maker
from odds_lambda.fetch_tier import FetchTier
from odds_lambda.storage.tier_validator import TierCoverageValidator
from rich.console import Console
from rich.table import Table

app = typer.Typer()
console = Console()


@app.command("daily")
def validate_daily(
    target_date: str = typer.Option(
        None,
        "--date",
        "-d",
        help="Date to validate (YYYY-MM-DD). Uses 24-hour window from noon UTC on this date to noon UTC next day. Defaults to yesterday.",
    ),
    strict: bool = typer.Option(
        True, "--strict/--no-strict", help="Require all 5 tiers (default: strict)"
    ),
    check_scores: bool = typer.Option(
        True,
        "--check-scores/--no-check-scores",
        help="Validate final scores exist for completed games",
    ),
    check_discovery: bool = typer.Option(
        False, "--check-discovery", help="Check for missing games via API (limited to last 3 days)"
    ),
    output_json: str = typer.Option(None, "--output-json", "-o", help="Save results to JSON file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed game breakdown"),
):
    """
    Validate tier coverage for a specific date.

    Uses a 24-hour window from noon UTC to noon UTC (next day) to capture a full
    NBA "game day" which spans two UTC calendar dates. For example, --date 2024-10-24
    validates games from noon Oct 24 to noon Oct 25 UTC, capturing all evening games
    played on Oct 24 in US timezones.

    Examples:
        # Validate yesterday's data (with score validation)
        odds validate daily

        # Validate specific date with game discovery check
        odds validate daily --date 2024-10-24 --check-discovery

        # Allow partial coverage (just show warnings)
        odds validate daily --date 2024-10-24 --no-strict

        # Skip score validation
        odds validate daily --no-check-scores

        # Save results to JSON
        odds validate daily --date 2024-10-24 --output-json results.json
    """
    asyncio.run(
        _validate_daily(target_date, strict, check_scores, check_discovery, output_json, verbose)
    )


async def _validate_daily(
    target_date_str: str | None,
    strict: bool,
    check_scores: bool,
    check_discovery: bool,
    output_json: str | None,
    verbose: bool,
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
        try:
            report = await validator.validate_date(
                target_date, required_tiers, check_scores, check_discovery
            )
        except Exception as e:
            console.print(f"\n[red]Validation failed with error: {e}[/red]")
            sys.exit(1)

    # Display summary
    _display_summary(report, required_tiers, check_scores, check_discovery)

    # Display tier breakdown
    _display_tier_breakdown(report)

    # Display score issues if any
    if check_scores and report.games_missing_scores > 0:
        _display_score_issues(report, verbose)

    # Display missing games if any (ERROR)
    if check_discovery and len(report.missing_games) > 0:
        _display_missing_games(report)

    # Display missing scores if any (WARNING)
    if check_discovery and len(report.missing_scores) > 0:
        _display_missing_scores(report)

    # Display incomplete games if any
    if report.incomplete_games > 0:
        _display_incomplete_games(report, verbose)

    # Save to JSON if requested
    if output_json:
        _save_to_json(report, output_json, check_scores, check_discovery)
        console.print(f"\n[green]Results saved to {output_json}[/green]")

    # Exit with appropriate code
    if check_scores and check_discovery:
        # Strict mode: fail if any issue detected
        if not report.is_fully_valid:
            console.print(
                f"\n[red]Status: FAILED - {report.incomplete_games} tier issues, "
                f"{report.games_missing_scores} score issues, "
                f"{len(report.missing_games)} missing games[/red]"
            )
            sys.exit(1)
        else:
            console.print("\n[green]Status: PASSED - All validations passed[/green]")
            sys.exit(0)
    else:
        # Legacy mode: only check tier coverage
        if not report.is_valid:
            console.print(
                f"\n[red]Status: FAILED - {report.incomplete_games} games incomplete[/red]"
            )
            sys.exit(1)
        else:
            console.print("\n[green]Status: PASSED - All games complete[/green]")
            sys.exit(0)


def _display_summary(report, required_tiers, check_scores=True, check_discovery=False):
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

    if check_scores:
        if report.games_missing_scores > 0:
            table.add_row("Missing Scores", f"[red]{report.games_missing_scores}[/red]")
        else:
            table.add_row("Missing Scores", "[green]0[/green]")

    if check_discovery:
        if len(report.missing_games) > 0:
            table.add_row("Missing Games (ERROR)", f"[red]{len(report.missing_games)}[/red]")
        else:
            table.add_row("Missing Games (ERROR)", "[green]0[/green]")

        if len(report.missing_scores) > 0:
            table.add_row(
                "Missing Scores (WARNING)", f"[yellow]{len(report.missing_scores)}[/yellow]"
            )
        else:
            table.add_row("Missing Scores (WARNING)", "[green]0[/green]")

    table.add_row("Required Tiers", f"{len(required_tiers)}/5")

    console.print(table)
    console.print()


def _display_tier_breakdown(report):
    """Display tier coverage breakdown."""
    from odds_analytics.utils import create_tier_coverage_table

    console.print("[bold]Tier Coverage Breakdown:[/bold]")

    tier_table = create_tier_coverage_table(
        total_games=report.total_games,
        missing_tier_breakdown=report.missing_tier_breakdown,
    )

    console.print(tier_table)
    console.print()


def _display_score_issues(report, verbose):
    """Display games with missing scores."""
    console.print(f"[bold]Games Missing Scores ({report.games_missing_scores}):[/bold]\n")

    games_with_score_issues = [
        r for r in report.game_reports if "Missing final scores" in r.validation_issues
    ]

    for game_report in games_with_score_issues:
        teams = f"{game_report.away_team} @ {game_report.home_team}"
        game_time = game_report.commence_time.strftime("%Y-%m-%d %H:%M")

        console.print(f"  ✗ [yellow]{teams}[/yellow] ({game_time})")
        console.print("     Issue: [red]Missing final scores[/red]")

        if verbose:
            console.print(f"     Event ID: {game_report.event_id}")
            console.print(f"     Home Score: {game_report.home_score}")
            console.print(f"     Away Score: {game_report.away_score}")

        console.print()


def _display_missing_games(report):
    """Display games from API that aren't in database (ERROR)."""
    console.print(
        f"[bold red]Missing Games from Database - ERROR ({len(report.missing_games)}):[/bold red]\n"
    )

    for missing_game in report.missing_games:
        teams = f"{missing_game['away_team']} @ {missing_game['home_team']}"
        commence_time = missing_game.get("commence_time", "Unknown")

        console.print(f"  ✗ [red]{teams}[/red] ({commence_time})")
        console.print(f"     Event ID: {missing_game['id']}")
        console.print(f"     Reason: {missing_game.get('reason', 'Unknown')}")

        home_score = missing_game.get("home_score")
        away_score = missing_game.get("away_score")
        if home_score is not None and away_score is not None:
            console.print(f"     Final Score: {away_score} - {home_score}")

        console.print()


def _display_missing_scores(report):
    """Display games in database without final scores (WARNING)."""
    console.print(
        f"[bold yellow]Games Missing Final Scores - WARNING ({len(report.missing_scores)}):[/bold yellow]\n"
    )

    for missing_score in report.missing_scores:
        teams = f"{missing_score['away_team']} @ {missing_score['home_team']}"
        commence_time = missing_score.get("commence_time", "Unknown")

        console.print(f"  ⚠ [yellow]{teams}[/yellow] ({commence_time})")
        console.print(f"     Event ID: {missing_score['id']}")
        console.print(f"     Status: {missing_score.get('status', 'Unknown')}")
        console.print(f"     Snapshots: {missing_score.get('snapshots_count', 0)}")
        console.print(f"     Reason: {missing_score.get('reason', 'Unknown')}")

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


def _save_to_json(report, output_path: str, check_scores=True, check_discovery=False):
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

    # Add score validation results if enabled
    if check_scores:
        data["summary"]["games_missing_scores"] = report.games_missing_scores
        data["summary"]["is_fully_valid"] = report.is_fully_valid
        data["games_missing_scores"] = [
            {
                "event_id": r.event_id,
                "home_team": r.home_team,
                "away_team": r.away_team,
                "commence_time": r.commence_time.isoformat(),
                "home_score": r.home_score,
                "away_score": r.away_score,
            }
            for r in report.game_reports
            if "Missing final scores" in r.validation_issues
        ]

    # Add game discovery results if enabled
    if check_discovery:
        data["summary"]["missing_games_count"] = len(report.missing_games)
        data["summary"]["missing_scores_count"] = len(report.missing_scores)
        data["missing_games"] = list(report.missing_games)  # ERROR: not in DB
        data["missing_scores"] = list(report.missing_scores)  # WARNING: in DB without scores

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
