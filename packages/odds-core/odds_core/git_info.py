"""The commit the running code was checked out at."""

import subprocess

from odds_core.config import get_settings


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=get_settings().project_root,
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    ).stdout.strip()


def resolve_git_sha() -> str | None:
    """``HEAD`` of the project checkout, suffixed ``-dirty`` when the tree has uncommitted changes.

    None when git or the repository is unavailable.
    """
    try:
        sha = _git("rev-parse", "HEAD")
        if not sha:
            return None
        return f"{sha}-dirty" if _git("status", "--porcelain") else sha
    except (OSError, subprocess.SubprocessError):
        return None


# Resolved at process start so a checkout edited while the process runs is not misattributed.
GIT_SHA: str | None = resolve_git_sha()


def current_git_sha() -> str | None:
    return GIT_SHA
