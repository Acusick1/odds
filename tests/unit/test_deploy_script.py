"""deploy.sh against stand-in git, uv, sudo and systemctl commands that log what they are asked."""

from __future__ import annotations

import os
import pty
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

DEPLOY = Path(__file__).parents[2] / "deploy.sh"

SYSTEMCTL = "/usr/bin/systemctl"
RULES = "\n".join(
    f"    (root) NOPASSWD: {SYSTEMCTL} {action} {service}"
    for service in ("odds-scheduler", "odds-web")
    for action in ("restart", "stop")
)

# One script stands in for every command; it is installed under each name.
SHIM = r"""#!/usr/bin/env bash
name="$(basename "$0")"
echo "$name $*" >>"$SHIM_DIR/calls.log"
state() { cat "$SHIM_DIR/state.$1" 2>/dev/null || echo inactive; }
case "$name" in
git)
    [[ "$1" == "rev-parse" ]] && echo abc1234
    exit 0 ;;
uv)
    if [[ "$*" == "run alembic current" ]]; then
        echo "${ALEMBIC_CURRENT:-abc123 (head)}"
    fi
    [[ -n "${FAIL_UV:-}" && "$*" == *"$FAIL_UV"* ]] && exit 1
    exit 0 ;;
sudo)
    if [[ "$1" == "-n" ]]; then echo "$SUDO_RULES"; exit 0; fi
    # sudo <systemctl> <action> <service>
    action="$2"; service="$3"
    if [[ "$action" == "restart" && "$service" == "${FAIL_RESTART:-}" ]]; then exit 1; fi
    if [[ "$action" == "stop" ]]; then echo inactive >"$SHIM_DIR/state.$service"; fi
    if [[ "$action" == "restart" ]]; then echo active >"$SHIM_DIR/state.$service"; fi
    exit 0 ;;
sleep)
    exit 0 ;;
systemctl)
    case "$1" in
    cat) [[ "$2" == "odds-web.service" && -n "${WEB_UNIT_MISSING:-}" ]] && exit 1; exit 0 ;;
    is-active)
        if [[ "$2" == "--quiet" ]]; then [[ "$(state "$3")" == active ]]; exit $?; fi
        state "$2"; [[ "$(state "$2")" == active ]]; exit $? ;;
    show)
        service="$2"
        if [[ "$*" == *NRestarts* ]]; then
            n=0; [[ "$service" == "${CRASH_LOOP:-}" ]] && n="$(cat "$SHIM_DIR/nrestarts" 2>/dev/null || echo 0)"
            if [[ "$service" == "${CRASH_LOOP:-}" ]]; then echo $((n + 1)) >"$SHIM_DIR/nrestarts"; fi
            echo "$n"
        else
            echo "@$(date +%s)"
        fi
        exit 0 ;;
    status) echo "status of $2"; exit 0 ;;
    esac ;;
esac
exit 0
"""


@pytest.fixture
def host(tmp_path: Path) -> Path:
    """A repo checkout holding deploy.sh, and a directory of stand-in commands in front of PATH."""
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy(DEPLOY, repo / "deploy.sh")
    backup = repo / "scripts" / "backup-prod-db.sh"
    backup.write_text('#!/usr/bin/env bash\necho "backup-prod-db.sh $*" >>"$SHIM_DIR/calls.log"\n')
    backup.chmod(backup.stat().st_mode | stat.S_IXUSR)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("git", "uv", "sudo", "systemctl", "sleep"):
        shim = bin_dir / name
        shim.write_text(SHIM)
        shim.chmod(0o755)
    for service in ("odds-scheduler", "odds-web"):
        (tmp_path / f"state.{service}").write_text("active")
    return tmp_path


def deploy(host: Path, *, terminal: bool = False, **env: str) -> subprocess.CompletedProcess[str]:
    # The script resolves systemctl by PATH and requires the sudo rules to name that path.
    full_env = {
        "PATH": f"{host / 'bin'}:/usr/bin:/bin",
        "HOME": str(host),
        "SHIM_DIR": str(host),
        "SUDO_RULES": RULES.replace(SYSTEMCTL, str(host / "bin" / "systemctl")),
        **env,
    }
    command = ["bash", str(host / "repo" / "deploy.sh")]
    if not terminal:
        return subprocess.run(
            command,
            env=full_env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    master, slave = pty.openpty()
    try:
        return subprocess.run(
            command,
            env=full_env,
            stdin=slave,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    finally:
        os.close(master)
        os.close(slave)


def calls(host: Path) -> list[str]:
    log = host / "calls.log"
    return log.read_text().splitlines() if log.exists() else []


def mutating(host: Path) -> list[str]:
    """Every call that could change the host: sudo, and git other than reading the head."""
    return [c for c in calls(host) if c.split()[0] in ("sudo", "git") and "rev-parse" not in c]


class TestPreflight:
    def test_a_missing_web_unit_exits_with_nothing_changed(self, host: Path) -> None:
        result = deploy(host, WEB_UNIT_MISSING="1")

        assert result.returncode == 1
        assert "odds-web.service is not installed" in result.stderr
        assert "nothing was changed" in result.stderr
        assert mutating(host) == []

    @pytest.mark.parametrize("missing", ["restart odds-web", "stop odds-web"])
    def test_a_missing_web_sudo_rule_exits_with_nothing_changed(
        self, host: Path, missing: str
    ) -> None:
        rules = "\n".join(
            line
            for line in RULES.replace(SYSTEMCTL, str(host / "bin" / "systemctl")).splitlines()
            if not line.endswith(missing)
        )

        result = deploy(host, SUDO_RULES=rules)

        assert result.returncode == 1
        assert "no NOPASSWD sudo rule for" in result.stderr and missing in result.stderr
        assert "nothing was changed" in result.stderr
        assert mutating(host) == ["sudo -n -l"]

    @pytest.mark.parametrize("rule_suffix", ["-other", "-other, /usr/bin/true"])
    def test_a_rule_for_a_similarly_named_service_does_not_satisfy_the_check(
        self, host: Path, rule_suffix: str
    ) -> None:
        systemctl = str(host / "bin" / "systemctl")
        rules = "\n".join(
            [
                f"    (root) NOPASSWD: {systemctl} {action} {service}"
                for service in ("odds-scheduler",)
                for action in ("restart", "stop")
            ]
            + [
                f"    (root) NOPASSWD: {systemctl} restart odds-web{rule_suffix}",
                f"    (root) NOPASSWD: {systemctl} stop odds-web{rule_suffix}",
            ]
        )

        result = deploy(host, SUDO_RULES=rules)

        assert result.returncode == 1
        assert "no NOPASSWD sudo rule for" in result.stderr and "odds-web'" in result.stderr
        assert mutating(host) == ["sudo -n -l"]

    def test_comma_separated_rules_on_one_line_are_each_honoured(self, host: Path) -> None:
        systemctl = str(host / "bin" / "systemctl")
        commands = [
            f"{systemctl} {action} {service}"
            for service in ("odds-scheduler", "odds-web")
            for action in ("restart", "stop")
        ]

        result = deploy(host, SUDO_RULES=f"    (root) NOPASSWD: {', '.join(commands)}")

        assert result.returncode == 0, result.stderr

    def test_with_a_terminal_the_sudo_rules_are_not_checked(self, host: Path) -> None:
        result = deploy(host, terminal=True, SUDO_RULES="")

        assert result.returncode == 0, result.stderr
        assert "sudo -n -l" not in calls(host)

    def test_with_a_terminal_a_missing_web_unit_still_exits_before_the_pull(
        self, host: Path
    ) -> None:
        result = deploy(host, terminal=True, WEB_UNIT_MISSING="1")

        assert result.returncode == 1
        assert "odds-web.service is not installed" in result.stderr
        assert mutating(host) == []


class TestDeploy:
    def test_restarts_the_scheduler_then_the_web_service(self, host: Path) -> None:
        result = deploy(host)

        assert result.returncode == 0, result.stderr
        restarts = [c.split()[-1] for c in calls(host) if c.startswith("sudo") and " restart " in c]
        assert restarts == ["odds-scheduler", "odds-web"]
        assert not any(" stop " in c for c in calls(host))

    def test_a_pending_migration_stops_both_services_before_it_and_dumps_first(
        self, host: Path
    ) -> None:
        result = deploy(host, ALEMBIC_CURRENT="abc123")

        assert result.returncode == 0, result.stderr
        log = calls(host)
        stops = [i for i, c in enumerate(log) if " stop " in c]
        assert [log[i].split()[-1] for i in stops] == ["odds-scheduler", "odds-web"]
        upgrade = next(i for i, c in enumerate(log) if c == "uv run alembic upgrade head")
        backup = next(i for i, c in enumerate(log) if c.startswith("backup-prod-db.sh"))
        assert max(stops) < backup < upgrade

    def test_applies_the_grants_then_checks_the_role_after_the_migration_and_before_any_restart(
        self, host: Path
    ) -> None:
        result = deploy(host, ALEMBIC_CURRENT="abc123")

        assert result.returncode == 0, result.stderr
        log = calls(host)
        upgrade = log.index("uv run alembic upgrade head")
        grants = log.index("uv run python -m odds_web.grants")
        check = log.index("uv run python -m odds_web.role_check")
        first_restart = next(i for i, c in enumerate(log) if " restart " in c)
        assert upgrade < grants < check < first_restart

    @pytest.mark.parametrize("failing", ["odds_web.grants", "odds_web.role_check"])
    def test_a_failed_grants_or_role_check_aborts_before_any_restart(
        self, host: Path, failing: str
    ) -> None:
        result = deploy(host, ALEMBIC_CURRENT="abc123", FAIL_UV=failing)

        assert result.returncode != 0
        assert not any(" restart " in c for c in calls(host))
        assert "FAILED with odds-scheduler odds-web STOPPED" in result.stderr
        assert "alembic downgrade abc123" in result.stderr
        if failing == "odds_web.grants":
            assert not any("odds_web.role_check" in c for c in calls(host))

    def test_a_failed_role_check_without_a_migration_leaves_the_services_running(
        self, host: Path
    ) -> None:
        result = deploy(host, FAIL_UV="odds_web.role_check")

        assert result.returncode != 0
        assert not any(" restart " in c or " stop " in c for c in calls(host))
        assert (host / "state.odds-scheduler").read_text().strip() == "active"
        assert (host / "state.odds-web").read_text().strip() == "active"

    def test_a_web_restart_failure_fails_the_deploy_and_leaves_the_scheduler_running(
        self, host: Path
    ) -> None:
        result = deploy(host, FAIL_RESTART="odds-web")

        assert result.returncode == 1
        assert "systemctl restart odds-web failed" in result.stderr
        assert "odds-scheduler: active" in result.stderr
        assert "odds-web: " in result.stderr
        log = calls(host)
        assert not any(" stop odds-scheduler" in c for c in log)
        assert [c.split()[-1] for c in log if " restart " in c] == ["odds-scheduler", "odds-web"]
        assert (host / "state.odds-scheduler").read_text().strip() == "active"

    def test_a_crash_looping_web_service_fails_the_deploy_and_reports_both_states(
        self, host: Path
    ) -> None:
        result = deploy(host, CRASH_LOOP="odds-web")

        assert result.returncode == 1
        assert "odds-web restarted 1 time(s)" in result.stderr
        assert "odds-scheduler: active" in result.stderr

    def test_a_web_failure_after_a_migration_says_the_web_service_is_stopped(
        self, host: Path
    ) -> None:
        result = deploy(host, ALEMBIC_CURRENT="abc123", FAIL_RESTART="odds-web")

        assert result.returncode == 1
        assert "FAILED with odds-web STOPPED" in result.stderr
        assert "odds-scheduler STOPPED" not in result.stderr

    def test_a_web_only_failure_after_a_migration_leaves_the_scheduler_and_the_database_alone(
        self, host: Path
    ) -> None:
        result = deploy(host, ALEMBIC_CURRENT="abc123", FAIL_RESTART="odds-web")

        assert result.returncode == 1
        assert "odds-scheduler is running on the new code and schema" in result.stderr
        assert "journalctl -u odds-web" in result.stderr
        assert "sudo " in result.stderr and "restart odds-web" in result.stderr
        assert "alembic downgrade" not in result.stderr
        assert "git checkout" not in result.stderr
        assert (host / "state.odds-scheduler").read_text().strip() == "active"

    def test_a_scheduler_failure_after_a_migration_still_prints_the_downgrade_recovery(
        self, host: Path
    ) -> None:
        result = deploy(host, ALEMBIC_CURRENT="abc123", FAIL_RESTART="odds-scheduler")

        assert result.returncode == 1
        assert "FAILED with odds-scheduler odds-web STOPPED" in result.stderr
        assert "alembic downgrade abc123" in result.stderr
        assert "git checkout abc1234" in result.stderr

    def test_a_scheduler_restart_failure_never_restarts_the_web_service(self, host: Path) -> None:
        result = deploy(host, FAIL_RESTART="odds-scheduler")

        assert result.returncode == 1
        assert [c.split()[-1] for c in calls(host) if " restart " in c] == ["odds-scheduler"]
