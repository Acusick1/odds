"""Fail when a candidate file changes without a version bump, or is deleted.

A candidate whose start_date at the base is later than today (UTC) has made no
decisions yet, so it may change in place at the same version. A rule_sha256 may also
change in place when the rule and both hashes are a complete row of the rehash ledger,
candidates/rehashes.jsonl. The hook checks a row's shape only: the ledger is a record the
rehash command makes, and nothing here re-runs the replay it records.

The holdout ledger, candidates/holdout.jsonl, is append-only: the base's rows must be the
first rows of the working tree's, and no idea version may be reserved twice. A file under
candidates/preregistrations/ or candidates/selections/ that exists at the base may not change,
be renamed or be deleted.

Compares candidates/*.yaml in the working tree with the base: --base, else the
merge-base of HEAD with origin/main, else with main, else HEAD.
Only stdlib, yaml and pydantic are imported: the pre-commit hook env has no project install.
"""

import argparse
import importlib.util
import json
import subprocess
import sys
from datetime import UTC, date, datetime
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

CANDIDATES_DIR = "candidates"
# Neither changes what a candidate bets on.
UNVERSIONED_KEYS = frozenset({"status", "description"})
REHASH_LEDGER = "rehashes.jsonl"
HOLDOUT_LEDGER = "holdout.jsonl"
SEALED_DIRS = ("preregistrations", "selections")


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def _ref_exists(ref: str) -> bool:
    return (
        subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"], capture_output=True
        ).returncode
        == 0
    )


def _resolve_base(arg: str | None) -> str:
    if arg:
        return arg
    for ref in ("origin/main", "main"):
        if _ref_exists(ref):
            return _git("merge-base", "HEAD", ref).strip()
    return "HEAD"


def _load(text: str) -> dict[str, Any]:
    loaded = yaml.safe_load(text)
    return loaded if isinstance(loaded, dict) else {}


def _not_started(candidate: dict[str, Any], today: date) -> bool:
    start = candidate.get("start_date")
    # yaml parses an unquoted ISO date to a date; anything else is malformed.
    return isinstance(start, date) and not isinstance(start, datetime) and start > today


# The ledger's models, loaded from their file: importing the package would need the project's
# dependencies, which the pre-commit hook env lacks.
_LEDGER_MODULE = (
    Path(__file__).resolve().parents[1] / "packages/odds-core/odds_core/rehash_ledger.py"
)


@cache
def _ledger_models() -> ModuleType:
    name = "odds_core.rehash_ledger"
    if (loaded := sys.modules.get(name)) is not None:
        return loaded
    spec = importlib.util.spec_from_file_location(name, _LEDGER_MODULE)
    assert spec is not None and spec.loader is not None  # noqa: S101
    module = importlib.util.module_from_spec(spec)
    # pydantic resolves the models' annotations through the registered module.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _row_problem(line: str) -> str | None:
    """Why a ledger line is not a complete record, or None. The hook checks the record's
    shape; it cannot re-run the replay the row records."""
    try:
        _ledger_models().Rehash.model_validate_json(line)
    except ValueError as exc:
        return " ".join(str(exc).split())
    return None


def _rehashes(root: Path) -> tuple[set[tuple[str, str, str]], list[str]]:
    """The (rule, old hash, new hash) of each complete row of the working tree's rehash
    ledger, and a problem for each other row."""
    path = root / CANDIDATES_DIR / REHASH_LEDGER
    if not path.exists():
        return set(), []
    accepted: set[tuple[str, str, str]] = set()
    problems: list[str] = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        if (problem := _row_problem(line)) is not None:
            problems.append(f"{CANDIDATES_DIR}/{REHASH_LEDGER}:{number}: {problem}")
        else:
            row = json.loads(line)
            accepted.add((row["rule"], row["old"], row["new"]))
    return accepted, problems


def _holdout_problems(base: str, root: Path) -> list[str]:
    """Why the holdout ledger or a pre-registration changed other than by appending."""
    problems: list[str] = []
    ledger = f"{CANDIDATES_DIR}/{HOLDOUT_LEDGER}"
    current = root / ledger
    now = current.read_text().splitlines() if current.exists() else []
    shown = subprocess.run(["git", "show", f"{base}:{ledger}"], capture_output=True, text=True)
    before = shown.stdout.splitlines() if shown.returncode == 0 else []
    if now[: len(before)] != before:
        problems.append(f"{ledger}: a committed row was modified or removed; it is append-only")
    reserved: set[tuple[Any, Any, Any]] = set()
    for number, line in enumerate(now, start=1):
        try:
            row = json.loads(line)
        except ValueError:
            problems.append(f"{ledger}:{number}: not JSON")
            continue
        if isinstance(row, dict) and row.get("kind") == "reserved":
            key = (row.get("issue"), row.get("market"), row.get("version"))
            if key in reserved:
                problems.append(
                    f"{ledger}:{number}: issue {key[0]} {key[1]} version {key[2]} reserved twice"
                )
            reserved.add(key)
    for folder in SEALED_DIRS:
        listing = _git("ls-tree", "-r", "--name-only", base, "--", f"{CANDIDATES_DIR}/{folder}/")
        for rel in listing.splitlines():
            path = root / rel
            if not path.exists():
                problems.append(f"{rel}: deleted or renamed; a pre-registration file never moves")
            elif path.read_text() != _git("show", f"{base}:{rel}"):
                problems.append(f"{rel}: changed; a change is a new version with a new file")
    return problems


def check(base: str, today: date | None = None) -> list[str]:
    today = today or datetime.now(UTC).date()
    root = Path(_git("rev-parse", "--show-toplevel").strip())
    rehashes, ledger_problems = _rehashes(root)
    listing = _git("ls-tree", "-r", "--name-only", base, "--", f"{CANDIDATES_DIR}/")
    errors: list[str] = [*ledger_problems, *_holdout_problems(base, root)]
    for rel in listing.splitlines():
        if not rel.endswith(".yaml"):
            continue
        path = root / rel
        if not path.exists():
            errors.append(
                f"{rel}: deleted. Kill a candidate by setting status: killed, don't delete it"
            )
            continue
        try:
            old = _load(_git("show", f"{base}:{rel}"))
            new = _load(path.read_text())
        except yaml.YAMLError as exc:
            reason = getattr(exc, "problem", None) or str(exc).splitlines()[0]
            errors.append(f"{rel}: invalid YAML ({reason})")
            continue
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
        if _not_started(old, today):
            continue
        changed = sorted(
            k
            for k in old.keys() | new.keys()
            if k not in UNVERSIONED_KEYS
            and old.get(k) != new.get(k)
            and not (
                k == "rule_sha256"
                and old.get("rule") == new.get("rule")
                and (new.get("rule"), old.get(k), new.get(k)) in rehashes
            )
        )
        if changed:
            errors.append(
                f"{rel}: {', '.join(changed)} changed without a version bump "
                f"(still {old_version}); only status and description may change at the same version once the start date has arrived"
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
