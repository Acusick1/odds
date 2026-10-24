"""Fail when a candidate file changes without a version bump, or is deleted.

Compares candidates/*.yaml in the working tree with the base: --base, else
$CANDIDATE_BASE_REF, else the merge-base with origin/$GITHUB_BASE_REF, else HEAD.
Only stdlib and yaml are imported: the pre-commit hook env has no project install.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

CANDIDATES_DIR = "candidates"
# Neither changes what a candidate bets on.
UNVERSIONED_KEYS = frozenset({"status", "description"})


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def _resolve_base(arg: str | None) -> str:
    if arg:
        return arg
    if ref := os.environ.get("CANDIDATE_BASE_REF"):
        return _git("merge-base", "HEAD", ref).strip()
    if base_branch := os.environ.get("GITHUB_BASE_REF"):
        return _git("merge-base", "HEAD", f"origin/{base_branch}").strip()
    return "HEAD"


def _load(text: str) -> dict[str, Any]:
    loaded = yaml.safe_load(text)
    return loaded if isinstance(loaded, dict) else {}


def check(base: str) -> list[str]:
    root = Path(_git("rev-parse", "--show-toplevel").strip())
    listing = _git("ls-tree", "-r", "--name-only", base, "--", f"{CANDIDATES_DIR}/")
    errors: list[str] = []
    for rel in listing.splitlines():
        if not rel.endswith(".yaml"):
            continue
        path = root / rel
        if not path.exists():
            errors.append(
                f"{rel}: deleted. Kill a candidate by setting status: killed, don't delete it"
            )
            continue
        old = _load(_git("show", f"{base}:{rel}"))
        new = _load(path.read_text())
        old_version, new_version = old.get("version"), new.get("version")
        if isinstance(old_version, int) and isinstance(new_version, int):
            if new_version < old_version:
                errors.append(f"{rel}: version decreased from {old_version} to {new_version}")
                continue
            if new_version > old_version:
                continue
        elif old_version != new_version:
            errors.append(f"{rel}: version changed to a non-integer value")
            continue
        changed = sorted(
            k
            for k in old.keys() | new.keys()
            if k not in UNVERSIONED_KEYS and old.get(k) != new.get(k)
        )
        if changed:
            errors.append(
                f"{rel}: {', '.join(changed)} changed without a version bump "
                f"(still {old_version}); only status and description may change at the same version"
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", help="Git ref to compare against")
    errors = check(_resolve_base(parser.parse_args().base))
    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
