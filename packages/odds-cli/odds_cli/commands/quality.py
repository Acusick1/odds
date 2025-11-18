"""CLI commands for data quality analysis."""

import asyncio
import sys
from datetime import UTC, datetime

import typer
from odds_analytics.quality_metrics import QualityMetrics
from odds_core.database import async_session_maker
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer()
console = Console()


def _parse_date(date_str: str, param_name: str) -> datetime:
    """
    Parse date string to datetime object.

    Args:
        date_str: Date string in YYYY-MM-DD format
        param_name: Parameter name for error message

    Returns:
        datetime object with UTC timezone

    Raises:
        SystemExit: If date format is invalid
    """
    try:
        # Parse to datetime and set to start of day in UTC
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=UTC)
    except ValueError:
        console.print(
            f"[red]Error: Invalid {param_name} date format '{date_str}'. Use YYYY-MM-DD[/red]"
        )
        sys.exit(1)


def _get_coverage_color(coverage_pct: float) -> str:
    """
    Get color based on coverage percentage thresholds.

    Args:
        coverage_pct: Coverage percentage (0-100)

    Returns:
        Rich color name: green (>90%), yellow (70-90%), or red (<70%)
    """
    if coverage_pct > 90:
        return "green"
    elif coverage_pct >= 70:
        return "yellow"
    else:
        return "red"


def _get_status_symbol(coverage_pct: float) -> str:
    """
    Get status symbol based on coverage percentage.

    Args:
        coverage_pct: Coverage percentage (0-100)

    Returns:
        Status symbol: ✓ (>90%) or ⚠ (<90%)
    """
    return "✓" if coverage_pct > 90 else "⚠"


@app.command("coverage")
def coverage(
    start: str = typer.Option(..., "--start", "-s", help="Start date (YYYY-MM-DD)"),
    end: str = typer.Option(..., "--end", "-e", help="End date (YYYY-MM-DD)"),
    sport: str = typer.Option(
        "basketball_nba", "--sport", help="Sport key (default: basketball_nba)"
    ),
):
    """
    Display comprehensive data quality coverage report.

    Analyzes data completeness for odds snapshots, final scores, tier coverage,
    and bookmaker availability across a specified date range.

    Examples:
        # Analyze coverage for October 2024
        odds quality coverage --start 2024-10-01 --end 2024-10-31

        # Analyze specific week
        odds quality coverage --start 2024-11-01 --end 2024-11-07

        # Different sport (when supported)
        odds quality coverage --start 2024-10-01 --end 2024-10-31 --sport basketball_ncaa
    """
    asyncio.run(_coverage(start, end, sport))


async def _coverage(start_date_str: str, end_date_str: str, sport_key: str):
    """
    Async implementation of coverage command.

    Args:
        start_date_str: Start date string (YYYY-MM-DD)
        end_date_str: End date string (YYYY-MM-DD)
        sport_key: Sport filter (e.g., "basketball_nba")
    """
    # Parse dates
    start_date = _parse_date(start_date_str, "start")
    end_date = _parse_date(end_date_str, "end")

    # Validate date range
    if end_date < start_date:
        console.print("[red]Error: End date must be after start date[/red]")
        sys.exit(1)

    # Create quality metrics instance and run queries
    async with async_session_maker() as session:
        metrics = QualityMetrics(session)

        # Fetch all required data (sequential due to AsyncSession limitations)
        try:
            game_counts = await metrics.get_game_counts(start_date, end_date, sport_key)
            games_with_odds = await metrics.get_games_with_odds(start_date, end_date, sport_key)
            games_with_scores = await metrics.get_games_with_scores(start_date, end_date, sport_key)
            games_missing_scores = await metrics.get_games_missing_scores(
                start_date, end_date, sport_key
            )
            tier_coverage = await metrics.get_tier_coverage(start_date, end_date, sport_key)
            bookmaker_coverage = await metrics.get_bookmaker_coverage(
                start_date, end_date, sport_key
            )
        except Exception as e:
            console.print(f"[red]Error fetching data: {e}[/red]")
            sys.exit(1)

    # Display results
    console.print()  # Blank line for spacing
    _display_summary(
        game_counts, len(games_with_odds), len(games_with_scores), start_date_str, end_date_str
    )
    console.print()  # Blank line for spacing

    _display_tier_coverage(tier_coverage)
    console.print()  # Blank line for spacing

    _display_bookmaker_coverage(bookmaker_coverage)
    console.print()  # Blank line for spacing

    _display_data_quality_issues(
        games_missing_scores, len(games_with_odds), game_counts.total_games
    )


def _display_summary(
    game_counts,
    games_with_odds_count: int,
    games_with_scores_count: int,
    start_date: str,
    end_date: str,
):
    """
    Display summary panel with key metrics.

    Args:
        game_counts: GameCountResult from quality_metrics
        games_with_odds_count: Number of games with odds snapshots
        games_with_scores_count: Number of games with final scores
        start_date: Start date string for display
        end_date: End date string for display
    """
    total_games = game_counts.total_games

    # Calculate completion percentages
    odds_pct = (games_with_odds_count / total_games * 100) if total_games > 0 else 0
    scores_pct = (games_with_scores_count / total_games * 100) if total_games > 0 else 0

    # Build summary content
    summary_lines = [
        f"[bold]Date Range:[/bold] {start_date} to {end_date}",
        f"[bold]Sport:[/bold] {game_counts.sport_key or 'All'}",
        "",
        f"[bold]Total Games:[/bold] {total_games}",
        f"[bold]Games with Odds Snapshots:[/bold] {games_with_odds_count} ({odds_pct:.1f}%)",
        f"[bold]Games with Final Scores:[/bold] {games_with_scores_count} ({scores_pct:.1f}%)",
    ]

    summary_text = "\n".join(summary_lines)

    # Create panel with appropriate styling
    panel = Panel(
        summary_text,
        title="[bold cyan]Data Quality Coverage Summary[/bold cyan]",
        border_style="cyan",
    )

    console.print(panel)


def _display_tier_coverage(tier_coverage: list):
    """
    Display tier coverage table with dynamic rows from FetchTier enum.

    Args:
        tier_coverage: List of TierCoverage objects from quality_metrics
    """
    if not tier_coverage:
        console.print("[yellow]No tier coverage data available[/yellow]")
        return

    table = Table(
        title="[bold cyan]Tier Coverage Analysis[/bold cyan]",
        show_header=True,
        header_style="bold cyan",
    )

    table.add_column("Tier", style="white")
    table.add_column("Hours Range", style="white")
    table.add_column("Expected Interval", justify="right", style="white")
    table.add_column("Games with Coverage", justify="right", style="white")
    table.add_column("Coverage %", justify="right", style="white")
    table.add_column("Avg Snapshots/Game", justify="right", style="white")
    table.add_column("Status", justify="center", style="white")

    for tier_info in tier_coverage:
        # Format tier name (capitalize)
        tier_name = tier_info.tier_name.upper()

        # Format expected interval
        interval_str = f"{tier_info.expected_interval_hours}h"

        # Format games with coverage
        games_str = f"{tier_info.games_with_tier_snapshots}/{tier_info.games_in_tier_range}"

        # Get color and status based on coverage percentage
        coverage_color = _get_coverage_color(tier_info.coverage_pct)
        status_symbol = _get_status_symbol(tier_info.coverage_pct)

        # Format coverage percentage with color
        coverage_str = f"[{coverage_color}]{tier_info.coverage_pct:.1f}%[/{coverage_color}]"

        # Format average snapshots per game
        avg_snapshots_str = f"{tier_info.avg_snapshots_per_game:.1f}"

        # Format status with color
        status_str = f"[{coverage_color}]{status_symbol}[/{coverage_color}]"

        table.add_row(
            tier_name,
            tier_info.hours_range,
            interval_str,
            games_str,
            coverage_str,
            avg_snapshots_str,
            status_str,
        )

    console.print(table)


def _display_bookmaker_coverage(bookmaker_coverage: list):
    """
    Display bookmaker coverage table sorted by coverage percentage.

    Args:
        bookmaker_coverage: List of BookmakerCoverage objects from quality_metrics
    """
    if not bookmaker_coverage:
        console.print("[yellow]No bookmaker coverage data available[/yellow]")
        return

    table = Table(
        title="[bold cyan]Bookmaker Coverage Analysis[/bold cyan]",
        show_header=True,
        header_style="bold cyan",
    )

    table.add_column("Bookmaker", style="white")
    table.add_column("Games with Odds", justify="right", style="white")
    table.add_column("Coverage %", justify="right", style="white")
    table.add_column("Total Snapshots", justify="right", style="white")
    table.add_column("Avg Snapshots/Game", justify="right", style="white")
    table.add_column("Status", justify="center", style="white")

    for book_info in bookmaker_coverage:
        # Format games with odds
        games_str = f"{book_info.games_with_odds}/{book_info.total_games}"

        # Get color and status based on coverage percentage
        coverage_color = _get_coverage_color(book_info.coverage_pct)
        status_symbol = _get_status_symbol(book_info.coverage_pct)

        # Format coverage percentage with color
        coverage_str = f"[{coverage_color}]{book_info.coverage_pct:.1f}%[/{coverage_color}]"

        # Format average snapshots per game
        avg_snapshots_str = f"{book_info.avg_snapshots_per_game:.1f}"

        # Format status with color
        status_str = f"[{coverage_color}]{status_symbol}[/{coverage_color}]"

        table.add_row(
            book_info.bookmaker_title,
            games_str,
            coverage_str,
            str(book_info.total_snapshots),
            avg_snapshots_str,
            status_str,
        )

    console.print(table)


def _display_data_quality_issues(
    games_missing_scores: list, games_with_odds_count: int, total_games: int
):
    """
    Display data quality issues section.

    Args:
        games_missing_scores: List of GameMissingScoresResult objects
        games_with_odds_count: Number of games with odds snapshots
        total_games: Total number of games in date range
    """
    issues = []

    # Check for missing final scores
    missing_scores_count = len(games_missing_scores)
    if missing_scores_count > 0:
        issues.append(f"⚠ [yellow]{missing_scores_count} game(s) missing final scores[/yellow]")

    # Check for games without any odds snapshots
    games_without_odds = total_games - games_with_odds_count
    if games_without_odds > 0:
        issues.append(f"⚠ [yellow]{games_without_odds} game(s) have no odds snapshots[/yellow]")

    # Display issues panel if any issues found
    if issues:
        issues_text = "\n".join(issues)
        panel = Panel(
            issues_text,
            title="[bold yellow]Data Quality Issues[/bold yellow]",
            border_style="yellow",
        )
        console.print(panel)
    else:
        # No issues - display success message
        panel = Panel(
            "[green]✓ No data quality issues detected[/green]",
            title="[bold green]Data Quality Issues[/bold green]",
            border_style="green",
        )
        console.print(panel)
