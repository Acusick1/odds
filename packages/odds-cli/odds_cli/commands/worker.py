"""CLI commands for the background scrape job worker."""

from __future__ import annotations

import asyncio
import dataclasses
from datetime import UTC, datetime, timedelta

import structlog
import typer
from odds_core.database import async_session_maker
from odds_core.scrape_job_models import ScrapeJob, ScrapeJobStatus
from odds_lambda.jobs.fetch_oddsportal import LEAGUE_SPEC_BY_NAME, ingest_league
from rich.console import Console
from sqlalchemy import select, update

app = typer.Typer()
console = Console()
logger = structlog.get_logger()

# Jobs stuck in 'running' longer than this are expired back to 'failed'
_STALE_TIMEOUT = timedelta(minutes=10)


@app.command("start")
def start(
    once: bool = typer.Option(False, "--once", help="Process one job and exit"),
    poll_interval: float = typer.Option(5.0, "--poll-interval", help="Seconds between polls"),
) -> None:
    """Poll for pending scrape jobs and execute them."""
    asyncio.run(_run_worker(once=once, poll_interval=poll_interval))


async def _run_worker(*, once: bool, poll_interval: float) -> None:
    console.print("[bold]Scrape worker started[/bold]")
    if once:
        console.print("  Mode: single job (--once)")
    else:
        console.print(f"  Poll interval: {poll_interval}s")

    while True:
        # Expire stale running jobs
        await _expire_stale_jobs()

        # Claim the oldest pending job atomically
        job_id: int | None = None
        async with async_session_maker() as session:
            # Find oldest pending job
            find_query = (
                select(ScrapeJob.id)
                .where(ScrapeJob.status == ScrapeJobStatus.PENDING)
                .order_by(ScrapeJob.created_at.asc())
                .limit(1)
                .with_for_update(skip_locked=True)
            )
            result = await session.execute(find_query)
            row = result.scalar_one_or_none()

            if row is not None:
                job_id = row
                # Atomically claim it
                await session.execute(
                    update(ScrapeJob)
                    .where(ScrapeJob.id == job_id, ScrapeJob.status == ScrapeJobStatus.PENDING)
                    .values(status=ScrapeJobStatus.RUNNING, started_at=datetime.now(UTC))
                )
                await session.commit()

        if job_id is not None:
            await _process_job(job_id)

            if once:
                return
        else:
            if once:
                console.print("No pending jobs found.")
                return

        await asyncio.sleep(poll_interval)


async def _expire_stale_jobs() -> None:
    """Mark running jobs older than the stale timeout as failed."""
    cutoff = datetime.now(UTC) - _STALE_TIMEOUT
    async with async_session_maker() as session:
        result = await session.execute(
            update(ScrapeJob)
            .where(
                ScrapeJob.status == ScrapeJobStatus.RUNNING,
                ScrapeJob.started_at < cutoff,
            )
            .values(
                status=ScrapeJobStatus.FAILED,
                completed_at=datetime.now(UTC),
                error_message=f"Expired: running for more than {int(_STALE_TIMEOUT.total_seconds() / 60)} minutes",
            )
            .returning(ScrapeJob.id)
        )
        expired_ids = list(result.scalars().all())
        await session.commit()

    if expired_ids:
        console.print(f"[yellow]Expired {len(expired_ids)} stale job(s): {expired_ids}[/yellow]")


async def _process_job(job_id: int) -> None:
    """Execute a single scrape job and write results back."""
    # Load the job
    async with async_session_maker() as session:
        result = await session.execute(select(ScrapeJob).where(ScrapeJob.id == job_id))
        job = result.scalar_one()
        league = job.league
        market = job.market

    console.print(f"Processing job {job_id}: {league} / {market}")

    # Build LeagueSpec
    known_spec = LEAGUE_SPEC_BY_NAME.get(league)
    if known_spec is None:
        await _fail_job(job_id, f"Unknown league '{league}'")
        return

    spec = dataclasses.replace(known_spec, markets=[market])

    try:
        stats = await ingest_league(spec)
    except Exception as e:
        logger.error("worker_job_failed", job_id=job_id, error=str(e), exc_info=True)
        await _fail_job(job_id, str(e)[:2000])
        return

    # Write results
    async with async_session_maker() as session:
        result = await session.execute(select(ScrapeJob).where(ScrapeJob.id == job_id))
        job = result.scalar_one()
        job.status = ScrapeJobStatus.COMPLETED
        job.completed_at = datetime.now(UTC)
        job.matches_scraped = stats.matches_scraped
        job.matches_converted = stats.matches_converted
        job.events_matched = stats.events_matched
        job.events_created = stats.events_created
        job.snapshots_stored = stats.snapshots_stored
        session.add(job)
        await session.commit()

    console.print(
        f"[green]Job {job_id} completed:[/green] "
        f"{stats.matches_scraped} scraped, {stats.events_matched} matched, "
        f"{stats.snapshots_stored} stored"
    )


async def _fail_job(job_id: int, error_message: str) -> None:
    """Mark a job as failed with an error message."""
    async with async_session_maker() as session:
        result = await session.execute(select(ScrapeJob).where(ScrapeJob.id == job_id))
        job = result.scalar_one()
        job.status = ScrapeJobStatus.FAILED
        job.completed_at = datetime.now(UTC)
        job.error_message = error_message[:2000]
        session.add(job)
        await session.commit()

    console.print(f"[red]Job {job_id} failed:[/red] {error_message}")
