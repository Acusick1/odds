"""Fail when a candidate file changes without a version bump, or is deleted.

A candidate whose start_date at the base is later than today (UTC) has made no
decisions yet, so it may change in place at the same version. A rule_sha256 may also
change in place when the rule and both hashes are a complete row of the rehash ledger,
candidates/rehashes.jsonl. The hook checks a row's shape only: the ledger is a record the
rehash command makes, and nothing here re-runs the replay it records.

Compares candidates/*.yaml in the working tree with the base: --base, else the
merge-base of HEAD with origin/main, else with main, else HEAD.
Only stdlib and yaml are imported: the pre-commit hook env has no project install.
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import yaml

CANDIDATES_DIR = "candidates"
# Neither changes what a candidate bets on.
UNVERSIONED_KEYS = frozenset({"status", "description"})
REHASH_LEDGER = "rehashes.jsonl"


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


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_ROW_KEYS = {
    "rule",
    "old",
    "new",
    "base",
    "head",
    "start",
    "end",
    "database",
    "replay_sha256",
    "link",
    "candidates",
}
_CANDIDATE_KEYS = {"name", "version", "decisions", "bets", "parity_matched", "parity_total"}


def _count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _row_problem(row: Any) -> str | None:
    """Why a ledger row is not a complete record, or None. The hook checks the record's shape;
    it cannot re-run the replay the row records."""
    if not isinstance(row, dict) or set(row) != _ROW_KEYS:
        return f"needs exactly the keys {sorted(_ROW_KEYS)}"
    for key in ("old", "new", "replay_sha256"):
        if not isinstance(row[key], str) or not _SHA256.match(row[key]):
            return f"{key} is not a sha256"
    for key in ("base", "head"):
        if not isinstance(row[key], str) or not _COMMIT.match(row[key]):
            return f"{key} is not a commit SHA"
    for key in ("rule", "database"):
        if not isinstance(row[key], str) or not row[key]:
            return f"{key} is empty"
    if not isinstance(row["link"], str) or not row["link"].startswith("https://github.com/"):
        return "link is not a GitHub link"
    try:
        start, end = date.fromisoformat(row["start"]), date.fromisoformat(row["end"])
    except (TypeError, ValueError):
        return "start and end are not dates"
    if start > end:
        return "the replay range is empty"
    candidates = row["candidates"]
    if not isinstance(candidates, list) or not candidates:
        return "it lists no candidate"
    for c in candidates:
        if not isinstance(c, dict) or set(c) != _CANDIDATE_KEYS:
            return f"each candidate needs exactly the keys {sorted(_CANDIDATE_KEYS)}"
        if not isinstance(c["name"], str) or not c["name"]:
            return "a candidate has no name"
        if not all(_count(c[k]) for k in _CANDIDATE_KEYS - {"name"}):
            return f"{c['name']}: counts must be non-negative integers"
        if c["version"] == 0 or c["decisions"] == 0:
            return f"{c['name']}: no version or no decision in the replay"
        if c["parity_matched"] != c["parity_total"]:
            return f"{c['name']}: parity did not hold"
    if not any(c["bets"] for c in candidates):
        return "the replay placed no bet"
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
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            row = None
        if (problem := _row_problem(row)) is not None:
            problems.append(f"{CANDIDATES_DIR}/{REHASH_LEDGER}:{number}: {problem}")
        else:
            accepted.add((row["rule"], row["old"], row["new"]))
    return accepted, problems


def check(base: str, today: date | None = None) -> list[str]:
    today = today or datetime.now(UTC).date()
    root = Path(_git("rev-parse", "--show-toplevel").strip())
    rehashes, ledger_problems = _rehashes(root)
    listing = _git("ls-tree", "-r", "--name-only", base, "--", f"{CANDIDATES_DIR}/")
    errors: list[str] = list(ledger_problems)
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
