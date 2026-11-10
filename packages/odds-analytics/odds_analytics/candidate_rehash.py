"""Rehash a rule whose hash changed without changing a bet, with the evidence that it didn't."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from odds_core.candidates import (
    MechanicalCandidate,
    Rehash,
    RehashEvidence,
    append_rehash,
    load_candidates,
)
from odds_lambda.candidate_rules import rule_sha256
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from odds_analytics.candidate_parity import ParityStatus, check_parity

REPLAY_SCRIPT = Path("scripts") / "replay_candidates.py"


@dataclass(frozen=True)
class RehashResult:
    rule: str
    new: str
    rehashes: list[Rehash]
    # Why nothing was written; None when the ledger and files were updated.
    refused: str | None = None


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def stale_candidates(directory: Path, rule: str, new: str) -> list[MechanicalCandidate]:
    """The registered candidates of ``rule`` whose recorded hash is not ``new``."""
    return [
        c
        for c in load_candidates(directory)
        if isinstance(c, MechanicalCandidate) and c.rule == rule and c.rule_sha256 != new
    ]


def replay_under(
    checkout: Path, root: Path, paths: Sequence[Path], start: date, end: date, database_url: str
) -> bytes:
    """The replay of the candidate files under ``checkout``'s code, as canonical JSON."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "replay.json"
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
                *(str(p) for p in paths),
            ],
            cwd=checkout,
            check=True,
            capture_output=True,
            env={**os.environ, "DATABASE_URL": database_url},
        )
        return out.read_bytes()


async def parity_summary(
    candidates: Sequence[MechanicalCandidate], database_url: str
) -> tuple[bool, str]:
    """Whether every recorded decision of the candidates replays to its bets, and a summary."""
    engine = create_async_engine(database_url)
    try:
        lines: list[str] = []
        ok = True
        for candidate in candidates:
            async with async_sessionmaker(engine)() as session:
                await session.execute(text("SET TRANSACTION READ ONLY"))
                report = await check_parity(session, candidate)
            matched = len(report.of(ParityStatus.MATCH))
            ok &= report.ok
            lines.append(f"{report.name} v{report.version}: {matched}/{report.checked} matched")
        return ok, "; ".join(lines)
    finally:
        await engine.dispose()


async def rehash(
    root: Path,
    rule: str,
    database_url: str,
    start: date,
    end: date,
    link: str,
    base: str,
) -> RehashResult:
    """Record ``rule``'s current hash for its registered candidates when their replays from
    ``start`` to ``end`` under ``base``'s code and the working tree's are byte-identical and
    parity holds; refuse otherwise, writing nothing."""
    directory = root / "candidates"
    new = rule_sha256(rule)
    stale = stale_candidates(directory, rule, new)
    if not stale:
        return RehashResult(rule, new, [])
    paths = [directory / f"{c.name}.yaml" for c in stale]
    checkout = root / "worktrees" / f"rehash-{_git(root, 'rev-parse', '--short', base)}"
    _git(root, "worktree", "add", "--detach", str(checkout), base)
    try:
        before = replay_under(checkout, root, paths, start, end, database_url)
    finally:
        _git(root, "worktree", "remove", "--force", str(checkout))
    after = replay_under(root, root, paths, start, end, database_url)
    if before != after:
        return RehashResult(rule, new, [], refused="the replays before and after differ")
    parity_ok, parity = await parity_summary(stale, database_url)
    if not parity_ok:
        return RehashResult(rule, new, [], refused=f"parity mismatches: {parity}")
    replay = (
        f"{len(stale)} candidates replayed {start}..{end} under {base} and the change: "
        f"byte-identical (sha256 {hashlib.sha256(after).hexdigest()})"
    )
    rehashes = [
        Rehash(
            rule=rule,
            old=old,
            new=new,
            evidence=RehashEvidence(link=link, replay=replay, parity=parity),
        )
        for old in sorted({c.rule_sha256 for c in stale if c.rule_sha256 is not None})
    ]
    for entry in rehashes:
        append_rehash(directory, entry)
    for path in paths:
        path.write_text(
            re.sub(r"^rule_sha256: \S+$", f"rule_sha256: {new}", path.read_text(), flags=re.M)
        )
    return RehashResult(rule, new, rehashes)
