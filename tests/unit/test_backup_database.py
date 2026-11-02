"""The nightly off-host backup job: retention, upload, and failure reporting (#734)."""

from __future__ import annotations

import asyncio
import stat
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from odds_core.config import AlertConfig, BackupConfig
from odds_lambda.health_monitor import HealthMonitor
from odds_lambda.jobs import backup_database
from odds_lambda.jobs.backup_database import NightlyDump, parse_nightly, select_kept
from odds_lambda.scheduling.jobs import JobContext, get_job_cron

START = datetime(2026, 11, 1, 5, 0, tzinfo=UTC)

# Stands in for scripts/backup-prod-db.sh: writes a dump named as the real script does.
FAKE_BACKUP_SCRIPT = """#!/usr/bin/env bash
set -euo pipefail
target="$ODDS_BACKUP_DIR/odds-$1-$FAKE_STAMP.dump"
echo dump >"$target"
echo "wrote $target" >&2
echo "$target"
"""

# Stands in for rclone with a local directory as the remote.
FAKE_RCLONE = """#!/usr/bin/env bash
set -euo pipefail
case "$1" in
    copyto) [[ -z "${FAKE_RCLONE_FAIL:-}" ]] || { echo "upload refused" >&2; exit 1; }; cp "$2" "$3" ;;
    lsf) ls -p "$3" | grep -v /$ || true ;;
    deletefile) rm "$2" ;;
    *) exit 2 ;;
esac
"""


def _nightly(taken_at: datetime) -> NightlyDump:
    return NightlyDump(name=f"odds-nightly-{taken_at:%Y%m%dT%H%MZ}.dump", taken_at=taken_at)


def _executable(path: Path, body: str) -> Path:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


class TestParseNightly:
    def test_reads_the_timestamp_of_a_nightly_dump(self) -> None:
        dump = parse_nightly("odds-nightly-20261101T0500Z.dump")

        assert dump == NightlyDump("odds-nightly-20261101T0500Z.dump", START)

    @pytest.mark.parametrize(
        "name",
        [
            "odds-pre-deploy-abc1234-20261101T0500Z.dump",
            "odds-nightly-20261101T0500Z.dump.partial",
            "odds-pre-712-20261029T1030Z.dump",
        ],
    )
    def test_ignores_every_other_file(self, name: str) -> None:
        assert parse_nightly(name) is None


class TestSelectKept:
    def test_keeps_the_newest_dump_of_each_recent_day(self) -> None:
        dumps = [_nightly(START - timedelta(days=n)) for n in range(20)]

        kept = select_kept(dumps, daily=14, weekly=0)

        assert kept == {d.name for d in dumps[:14]}

    def test_keeps_one_dump_per_day(self) -> None:
        early = _nightly(START)
        late = _nightly(START + timedelta(hours=3))

        assert select_kept([early, late], daily=14, weekly=8) == {late.name}

    def test_keeps_the_newest_dump_of_each_recent_week(self) -> None:
        dumps = [_nightly(START - timedelta(days=n)) for n in range(90)]

        kept = select_kept(dumps, daily=14, weekly=8)

        weekly = {d.name for d in dumps[14:] if d.name in kept}
        weeks = {d.taken_at.isocalendar()[:2] for d in dumps if d.name in kept}
        assert len(kept) == 14 + len(weekly)
        assert len(weeks) == 8
        # 2026-11-01 is a Sunday, so the newest dump of each earlier week is a Sunday too.
        assert all(d.taken_at.weekday() == 6 for d in dumps[14:] if d.name in weekly)

    def test_counts_only_days_that_have_a_dump(self) -> None:
        dumps = [_nightly(START - timedelta(days=30 + n)) for n in range(20)]

        kept = select_kept(dumps, daily=14, weekly=0)

        assert kept == {d.name for d in dumps[:14]}


@pytest.fixture
def alerts() -> Iterator[SimpleNamespace]:
    with (
        patch("odds_core.alerts.send_job_failure", new_callable=AsyncMock) as failure,
        patch("odds_core.alerts.record_to_alert_history", new_callable=AsyncMock) as record,
    ):
        yield SimpleNamespace(failure=failure, record=record)


@pytest.fixture
def host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """A repo with a fake backup script, a fake rclone on PATH, and local and remote dirs."""
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    _executable(root / "scripts" / "backup-prod-db.sh", FAKE_BACKUP_SCRIPT)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _executable(bin_dir / "rclone", FAKE_RCLONE)
    monkeypatch.setenv("PATH", f"{bin_dir}:{Path('/usr/bin')}:{Path('/bin')}")
    monkeypatch.setenv("FAKE_STAMP", f"{START:%Y%m%dT%H%MZ}")
    local = tmp_path / "backups"
    remote = tmp_path / "remote"
    local.mkdir()
    remote.mkdir()
    settings = SimpleNamespace(
        project_root=root,
        backup=BackupConfig(remote=str(remote), dir=local),
        scheduler=SimpleNamespace(dry_run=False),
    )
    monkeypatch.setattr(backup_database, "get_settings", lambda: settings)
    return SimpleNamespace(local=local, remote=remote, settings=settings)


def _files(path: Path) -> set[str]:
    return {p.name for p in path.iterdir()}


class TestBackupJob:
    def test_runs_nightly_from_cron(self) -> None:
        assert get_job_cron("backup-database") is not None

    async def test_uploads_the_dump_and_records_a_heartbeat(
        self, host: SimpleNamespace, alerts: SimpleNamespace
    ) -> None:
        await backup_database.main(JobContext())

        dump = f"odds-nightly-{START:%Y%m%dT%H%MZ}.dump"
        assert _files(host.local) == {dump}
        assert _files(host.remote) == {dump}
        assert [c.kwargs["alert_type"] for c in alerts.record.call_args_list] == [
            "heartbeat:backup-database"
        ]
        alerts.failure.assert_not_called()

    async def test_prunes_old_nightly_dumps_but_not_labelled_ones(
        self, host: SimpleNamespace, alerts: SimpleNamespace
    ) -> None:
        old = [_nightly(START - timedelta(days=n)).name for n in range(1, 20)]
        hand = "odds-pre-deploy-abc1234-20260101T0000Z.dump"
        for directory in (host.local, host.remote):
            for name in [*old, hand]:
                (directory / name).write_text("old")
        host.settings.backup = BackupConfig(
            remote=str(host.remote), dir=host.local, keep_daily=3, keep_weekly=0
        )

        await backup_database.main(JobContext())

        expected = {_nightly(START).name, *old[:2], hand}
        assert _files(host.local) == expected
        assert _files(host.remote) == expected

    async def test_failed_upload_alerts_and_writes_no_heartbeat(
        self, host: SimpleNamespace, alerts: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_RCLONE_FAIL", "1")

        with pytest.raises(RuntimeError, match="upload refused"):
            await backup_database.main(JobContext())

        alerts.failure.assert_awaited_once()
        assert alerts.failure.call_args.args[0] == "backup-database"
        alerts.record.assert_not_called()
        assert _files(host.remote) == set()

    async def test_failed_dump_alerts(
        self, host: SimpleNamespace, alerts: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _executable(
            host.settings.project_root / "scripts" / "backup-prod-db.sh",
            "#!/usr/bin/env bash\necho 'pg_dump: connection refused' >&2\nexit 1\n",
        )

        with pytest.raises(RuntimeError, match="connection refused"):
            await backup_database.main(JobContext())

        alerts.failure.assert_awaited_once()
        alerts.record.assert_not_called()

    async def test_timeout_stops_the_dump_and_its_children(
        self, host: SimpleNamespace, alerts: SimpleNamespace, tmp_path: Path
    ) -> None:
        pid_file = tmp_path / "child.pid"
        _executable(
            host.settings.project_root / "scripts" / "backup-prod-db.sh",
            f"#!/usr/bin/env bash\nsleep 60 &\necho $! >{pid_file}\nwait\n",
        )

        with pytest.raises(TimeoutError):
            async with asyncio.timeout(1):
                await backup_database.main(JobContext())

        stat_file = Path("/proc") / pid_file.read_text().strip() / "stat"
        for _ in range(50):
            if not stat_file.exists() or stat_file.read_text().split()[2] == "Z":
                break
            await asyncio.sleep(0.1)
        else:
            pytest.fail("the dump's child process outlived the timeout")

    async def test_does_nothing_until_a_remote_is_configured(
        self, host: SimpleNamespace, alerts: SimpleNamespace
    ) -> None:
        host.settings.backup = BackupConfig(dir=host.local)

        await backup_database.main(JobContext())

        assert _files(host.local) == set()
        alerts.record.assert_not_called()
        alerts.failure.assert_not_called()

    async def test_dry_run_takes_no_dump(
        self, host: SimpleNamespace, alerts: SimpleNamespace
    ) -> None:
        host.settings.scheduler.dry_run = True

        await backup_database.main(JobContext())

        assert _files(host.local) == set()
        alerts.record.assert_not_called()


class TestRemotePath:
    @pytest.mark.parametrize(
        ("remote", "expected"),
        [
            ("r2:odds-backups", "r2:odds-backups/f.dump"),
            ("r2:odds-backups/", "r2:odds-backups/f.dump"),
            ("local:", "local:f.dump"),
        ],
    )
    def test_joins_the_file_name(self, remote: str, expected: str) -> None:
        assert backup_database._remote_path(remote, "f.dump") == expected


class TestStaleBackup:
    async def test_health_check_reports_no_backup_in_the_last_day(self) -> None:
        session = AsyncMock()
        result = MagicMock()
        result.scalar_one_or_none.return_value = datetime.now(UTC) - timedelta(hours=27)
        session.execute.return_value = result
        settings = SimpleNamespace(
            alerts=AlertConfig(heartbeat_expectations={"backup-database": 26})
        )
        monitor = HealthMonitor(session, settings)  # type: ignore[arg-type]

        with patch.object(monitor, "_send_alert", new_callable=AsyncMock) as send:
            issues = await monitor.check_job_heartbeats()

        assert len(issues) == 1
        assert "backup-database" in issues[0]
        assert send.call_args.kwargs["alert_type"] == "missing_heartbeat:backup-database"

    def test_default_expectation_covers_one_nightly_run(self) -> None:
        assert 24 < AlertConfig().heartbeat_expectations["backup-database"] <= 26
