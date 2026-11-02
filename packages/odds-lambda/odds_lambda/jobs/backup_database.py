"""Nightly backup of the production database, copied off the host with rclone (#734).

``scripts/backup-prod-db.sh nightly`` takes and checks the dump; this job uploads it to
``settings.backup.remote`` and prunes nightly dumps beyond the retention, on the remote and
locally. Only ``odds-nightly-*`` dumps are pruned, so hand-labelled dumps such as
``pre-deploy-*`` are never touched.
"""

from __future__ import annotations

import asyncio
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import structlog
from odds_core.alerts import job_alert_context
from odds_core.config import BackupConfig, get_settings

from odds_lambda.scheduling.jobs import JobContext

logger = structlog.get_logger()

JOB_NAME = "backup-database"
LABEL = "nightly"

_NIGHTLY_NAME = re.compile(rf"^odds-{LABEL}-(\d{{8}}T\d{{4}}Z)\.dump$")


@dataclass(frozen=True)
class NightlyDump:
    name: str
    taken_at: datetime


def parse_nightly(name: str) -> NightlyDump | None:
    """The nightly dump a file name denotes, or ``None`` for any other file."""
    match = _NIGHTLY_NAME.match(name)
    if match is None:
        return None
    taken_at = datetime.strptime(match.group(1), "%Y%m%dT%H%MZ").replace(tzinfo=UTC)
    return NightlyDump(name=name, taken_at=taken_at)


def select_kept(dumps: Iterable[NightlyDump], *, daily: int, weekly: int) -> set[str]:
    """Names of the dumps the retention keeps: the newest dump of each of the ``daily`` most
    recent UTC days that have one, and of each of the ``weekly`` most recent ISO weeks.

    Periods are counted among those holding a dump, so a run of failed nights never prunes the
    last good dumps.
    """
    newest_first = sorted(dumps, key=lambda d: d.taken_at, reverse=True)
    kept: set[str] = set()
    for limit, period in (
        (daily, lambda d: d.taken_at.date()),
        (weekly, lambda d: d.taken_at.isocalendar()[:2]),
    ):
        seen: set[object] = set()
        for dump in newest_first:
            key = period(dump)
            if key in seen:
                continue
            if len(seen) == limit:
                break
            seen.add(key)
            kept.add(dump.name)
    return kept


def _nightly(names: Iterable[str]) -> list[NightlyDump]:
    return [dump for name in names if (dump := parse_nightly(name)) is not None]


def _remote_path(remote: str, name: str) -> str:
    return f"{remote}{name}" if remote.endswith(":") else f"{remote.rstrip('/')}/{name}"


async def _run(*args: str, env: dict[str, str] | None = None) -> str:
    """Run a command and return its stdout; raise with its stderr when it fails."""
    process = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )
    stdout, stderr = await process.communicate()
    if process.returncode != 0:
        raise RuntimeError(
            f"{Path(args[0]).name} {args[1]} exited {process.returncode}: "
            f"{stderr.decode().strip()[-500:]}"
        )
    return stdout.decode()


async def take_dump(script: Path, config: BackupConfig) -> Path:
    """Run the backup script and return the dump it wrote."""
    env = {**os.environ, "ODDS_BACKUP_DIR": str(config.dir)}
    stdout = await _run(str(script), LABEL, env=env)
    return Path(stdout.strip().splitlines()[-1])


async def upload(dump: Path, remote: str) -> None:
    await _run("rclone", "copyto", str(dump), _remote_path(remote, dump.name))


async def prune_remote(remote: str, config: BackupConfig) -> list[str]:
    listing = await _run("rclone", "lsf", "--files-only", remote)
    dumps = _nightly(listing.splitlines())
    kept = select_kept(dumps, daily=config.keep_daily, weekly=config.keep_weekly)
    pruned = [dump.name for dump in dumps if dump.name not in kept]
    for name in pruned:
        await _run("rclone", "deletefile", _remote_path(remote, name))
    return pruned


def prune_local(config: BackupConfig) -> list[str]:
    dumps = _nightly(path.name for path in config.dir.iterdir())
    kept = select_kept(dumps, daily=config.keep_daily, weekly=config.keep_weekly)
    pruned = [dump.name for dump in dumps if dump.name not in kept]
    for name in pruned:
        (config.dir / name).unlink()
    return pruned


async def main(ctx: JobContext) -> None:
    settings = get_settings()
    config = settings.backup
    if config.remote is None:
        logger.warning("backup_remote_not_configured", job=JOB_NAME)
        return
    if settings.scheduler.dry_run:
        logger.info("backup_dry_run_skipped", remote=config.remote)
        return

    async with job_alert_context(JOB_NAME):
        dump = await take_dump(settings.project_root / "scripts" / "backup-prod-db.sh", config)
        await upload(dump, config.remote)
        pruned_remote = await prune_remote(config.remote, config)
        pruned_local = prune_local(config)
        logger.info(
            "backup_completed",
            dump=str(dump),
            remote=config.remote,
            pruned_remote=pruned_remote,
            pruned_local=pruned_local,
        )


if __name__ == "__main__":
    asyncio.run(main(JobContext()))
