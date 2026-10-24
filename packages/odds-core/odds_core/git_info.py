"""The commit the running code was checked out at."""

import subprocess
from functools import cache

from odds_core.config import get_settings


@cache
def current_git_sha() -> str | None:
    """``HEAD`` of the project checkout, or None when git or the repository is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=get_settings().project_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None
