"""Rehash a rule whose hash changed without changing a bet, with the evidence that it didn't."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections.abc import AsyncIterator, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml
from odds_core.candidates import MechanicalCandidate, load_candidate_file, load_candidates
from odds_core.rehash_ledger import Rehash, RehashCandidate, append_rehash
from odds_lambda.candidate_rules.pinning import rule_sha256
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession, create_async_engine

from odds_analytics.candidate_parity import ParityStatus, check_parity
from odds_analytics.holdout import Window, development_segments, holdout_windows

REPLAY_SCRIPT = Path("scripts") / "replay_candidates.py"
LOCK_FILE = "odds-rehash.lock"


@dataclass(frozen=True)
class RehashResult:
    rule: str
    new: str
    rehash: Rehash | None
    # Why nothing was written; None when the files and ledger were updated or already current.
    refused: str | None = None


class Refused(Exception):
    """A rehash the evidence does not support."""


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def database_label(url: str) -> str:
    """The database a URL names, without its password."""
    parsed = make_url(url)
    return f"{parsed.username}@{parsed.host}:{parsed.port}/{parsed.database}"


@contextmanager
def rehash_lock(root: Path) -> Iterator[None]:
    """Hold the repository's rehash lock, or raise ``Refused`` when another run holds it."""
    path = Path(_git(root, "rev-parse", "--path-format=absolute", "--git-common-dir")) / LOCK_FILE
    with path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise Refused("another rehash holds the lock") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


@dataclass(frozen=True)
class Replay:
    output: bytes
    rule_sha256: str

    def decisions_and_bets(self, name: str) -> tuple[int, int]:
        """A candidate's decisions with inputs and its bets."""
        replayed = json.loads(self.output)[name]
        return replayed["counts"]["with_input"], len(replayed["bets"])


def _leagues(path: Path) -> tuple[str, ...]:
    candidate = load_candidate_file(path)
    assert isinstance(candidate, MechanicalCandidate)  # noqa: S101
    return candidate.sports


# Prints the holdout windows a checkout's code seals, as JSON.
_WINDOWS_PROBE = (
    "import json; from odds_analytics.holdout import holdout_windows; "
    "print(json.dumps([w.model_dump(mode='json') for w in holdout_windows()]))"
)


def windows_under(checkout: Path) -> list[Window]:
    """The holdout windows ``checkout``'s code seals; none for a revision without a holdout."""
    result = subprocess.run(
        ["uv", "run", "--project", str(checkout), "python", "-c", _WINDOWS_PROBE],
        cwd=checkout,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        if "No module named 'odds_analytics.holdout'" in result.stderr:
            return []
        raise subprocess.CalledProcessError(
            result.returncode, result.args, result.stdout, result.stderr
        )
    return [Window.model_validate(w) for w in json.loads(result.stdout)]


def development_ranges(
    paths: Sequence[Path], start: date, end: date, windows: Sequence[Window]
) -> dict[str, list[tuple[str, str]]]:
    """Each league's days of ``start``..``end`` that no window in ``windows`` seals."""
    return {
        sport: [
            (a.isoformat(), b.isoformat())
            for a, b in development_segments(sport, start, end, windows)
        ]
        for sport in {league for p in paths for league in _leagues(p)}
    }


def replay_under(
    checkout: Path,
    root: Path,
    rule: str,
    paths: Sequence[Path],
    start: date,
    end: date,
    segments: dict[str, list[tuple[str, str]]],
    database_url: str,
    snapshot: str,
) -> Replay:
    """The replay of the candidate files under ``checkout``'s code from the exported database
    ``snapshot``, over each league's ``segments``, and the rule's hash under that code."""
    with tempfile.TemporaryDirectory() as tmp:
        out, hash_out = Path(tmp) / "replay.json", Path(tmp) / "hash"
        subprocess.run(
            [
                "uv",
                "run",
                "--project",
                str(checkout),
                "python",
                str(root / REPLAY_SCRIPT),
                "--from",
                start.isoformat(),
                "--to",
                end.isoformat(),
                "--out",
                str(out),
                "--segments",
                json.dumps(segments),
                "--snapshot",
                snapshot,
                "--rule",
                rule,
                "--hash-out",
                str(hash_out),
                *(str(p) for p in paths),
            ],
            cwd=checkout,
            check=True,
            capture_output=True,
            env={**os.environ, "DATABASE_URL": database_url},
        )
        return Replay(out.read_bytes(), hash_out.read_text().strip())


async def parity_counts(
    connection: AsyncConnection, candidate: MechanicalCandidate
) -> tuple[int, int]:
    """The candidate's recorded decisions whose replay matched, and all of them."""
    session = AsyncSession(bind=connection)
    report = await check_parity(session, candidate)
    return len(report.of(ParityStatus.MATCH)), report.checked


@asynccontextmanager
async def frozen_snapshot(database_url: str) -> AsyncIterator[tuple[AsyncConnection, str]]:
    """A read-only repeatable-read transaction held open, and the id of its exported snapshot,
    which other sessions read the same data through."""
    engine = create_async_engine(database_url)
    try:
        async with engine.connect() as connection:
            await connection.execute(
                text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
            )
            snapshot = (await connection.execute(text("SELECT pg_export_snapshot()"))).scalar_one()
            yield connection, snapshot
            await connection.rollback()
    finally:
        await engine.dispose()


@contextmanager
def base_checkout(root: Path, base: str) -> Iterator[Path]:
    checkout = root / "worktrees" / f"rehash-{base[:12]}"
    _git(root, "worktree", "add", "--detach", str(checkout), base)
    try:
        yield checkout
    finally:
        _git(root, "worktree", "remove", "--force", str(checkout))


def staged_file(path: Path, new: str) -> str:
    """The candidate file at ``path`` with its ``rule_sha256`` set to ``new``, keeping every
    other line; refuses unless exactly one value changes and the result parses to ``new``."""
    content, count = re.subn(
        r"^(rule_sha256:[ \t]*)[0-9a-f]{64}(?![0-9a-f])",
        rf"\g<1>{new}",
        path.read_text(),
        flags=re.M,
    )
    if count != 1:
        raise Refused(f"{path.name}: no single rule_sha256 line to update")
    parsed = MechanicalCandidate.model_validate(yaml.safe_load(content))
    if parsed.rule_sha256 != new:
        raise Refused(f"{path.name}: the updated file records {parsed.rule_sha256}, not {new}")
    return content


def _write_atomically(path: Path, content: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(content)
    os.replace(tmp, path)


async def _evidence(
    root: Path,
    rule: str,
    new: str,
    candidates: list[MechanicalCandidate],
    database_url: str,
    start: date,
    end: date,
    link: str,
    base: str,
    head: str,
) -> Rehash:
    directory = root / "candidates"
    paths = [directory / f"{c.name}.yaml" for c in candidates]
    # One snapshot, held open, is what both replays and parity read.
    async with frozen_snapshot(database_url) as (connection, snapshot):
        with base_checkout(root, base) as checkout:
            # Both replays read only days neither revision seals, so neither refuses a range.
            segments = development_ranges(
                paths, start, end, [*holdout_windows(), *windows_under(checkout)]
            )
            before = replay_under(
                checkout, root, rule, paths, start, end, segments, database_url, snapshot
            )
        olds = {c.rule_sha256 for c in candidates}
        if olds != {before.rule_sha256}:
            raise Refused(
                f"the candidates record {sorted(str(o) for o in olds)}, not the hash at base "
                f"{base}, {before.rule_sha256}"
            )
        after = replay_under(root, root, rule, paths, start, end, segments, database_url, snapshot)
        if after.rule_sha256 != new:
            raise Refused(f"the working tree's hash {after.rule_sha256} is not {new}")
        if before.output != after.output:
            raise Refused("the replays before and after differ")
        rows: list[RehashCandidate] = []
        for candidate in candidates:
            decisions, bets = after.decisions_and_bets(candidate.name)
            if decisions == 0:
                raise Refused(f"{candidate.name} has no decision with inputs in the replay")
            matched, total = await parity_counts(connection, candidate)
            if matched != total:
                raise Refused(f"{candidate.name}: parity matched {matched} of {total}")
            rows.append(
                RehashCandidate(
                    name=candidate.name,
                    version=candidate.version,
                    decisions=decisions,
                    bets=bets,
                    parity_matched=matched,
                    parity_total=total,
                )
            )
    if not any(row.bets for row in rows):
        raise Refused("the replay placed no bet")
    return Rehash(
        rule=rule,
        old=before.rule_sha256,
        new=new,
        base=base,
        head=head,
        start=start,
        end=end,
        database=database_label(database_url),
        replay_sha256=hashlib.sha256(after.output).hexdigest(),
        link=link,
        candidates=rows,
    )


async def rehash(
    root: Path,
    rule: str,
    database_url: str,
    start: date,
    end: date,
    link: str,
    base: str,
) -> RehashResult:
    """Record ``rule``'s current hash for every registered candidate of it, when they all
    record its hash at ``base``, their replays from ``start`` to ``end`` under ``base``'s code
    and the committed working tree's are byte-identical on one database snapshot, each has a
    decision with inputs, some bet is placed and parity holds. Refuse otherwise, writing
    nothing.

    The candidate files are written before the ledger, so a failure between them leaves a hash
    change without its ledger row, which the version check rejects.
    """
    directory = root / "candidates"
    new = rule_sha256(rule)
    try:
        if start > end:
            raise Refused(f"--from {start} is after --to {end}")
        with rehash_lock(root):
            if _git(root, "status", "--porcelain", "--", ".", ":(exclude)candidates"):
                raise Refused("commit the code change first: the working tree has other changes")
            candidates = [
                c
                for c in load_candidates(directory)
                if isinstance(c, MechanicalCandidate) and c.rule == rule
            ]
            if not candidates:
                raise Refused(f"no registered candidate runs rule {rule!r}")
            if all(c.rule_sha256 == new for c in candidates):
                return RehashResult(rule, new, None)
            commit = _git(root, "rev-parse", "--verify", f"{base}^{{commit}}")
            head = _git(root, "rev-parse", "HEAD")
            entry = await _evidence(
                root, rule, new, candidates, database_url, start, end, link, commit, head
            )
            updated = {
                path: staged_file(path, new)
                for path in (directory / f"{c.name}.yaml" for c in candidates)
            }
            for path, content in updated.items():
                _write_atomically(path, content)
            for path in updated:
                written = load_candidate_file(path)
                if not isinstance(written, MechanicalCandidate) or written.rule_sha256 != new:
                    raise Refused(f"{path.name} does not record {new} after writing")
            append_rehash(directory, entry)
            return RehashResult(rule, new, entry)
    except Refused as exc:
        return RehashResult(rule, new, None, refused=str(exc))
