"""The sealed holdout: its windows, the pre-registration that opens it, the one-shot ledger of
reservations and results, and the gate verdict.

``odds_analytics.candidate_backtest.final_events`` calls ``require_clear``, so no replay of a
sealed range runs without a ``HoldoutAccess``, which only a ``Reservation`` issues. The module
imports nothing from the replay, which imports it, and lies outside the rule import closure.
"""

from __future__ import annotations

import fcntl
import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from odds_analytics.candidate_report import Checkpoint

HOLDOUT_LEDGER = "holdout.jsonl"
PREREGISTRATION_DIR = "preregistrations"
COMMIT_PATTERN = r"^[0-9a-f]{40}$"
SELECTION_RULE_PATTERN = r"^https://github\.com/[^/\s]+/[^/\s]+/(issues|pull)/[0-9]+#\S+$"
# Below this many holdout bets the gate asks only that the cell is not killed, and flags it.
MIN_GATE_BETS = 30

# Whether EPL 2022-23 and 2023-24 are sealed. They join the holdout once the substitute close is
# accepted (issue 730); until then no close exists to score them, so sealing them costs nothing
# and errs safe. Set False only if the substitute close is rejected.
EPL_2022_24_SEALED = True

# Seasons run 1 August to 31 July.
_FIVE_LEAGUES = (
    "soccer_efl_champ",
    "soccer_spain_la_liga",
    "soccer_italy_serie_a",
    "soccer_germany_bundesliga",
    "soccer_france_ligue_one",
)


class HoldoutRefused(ValueError):
    """A holdout replay or run that is not allowed; nothing sealed was read."""


class Window(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    sport: str
    start: date
    end: date

    def __str__(self) -> str:
        return f"{self.sport} {self.start}..{self.end}"


def holdout_windows() -> list[Window]:
    windows = [
        Window(sport=s, start=date(2025, 8, 1), end=date(2026, 7, 31)) for s in _FIVE_LEAGUES
    ]
    if EPL_2022_24_SEALED:
        windows.append(Window(sport="soccer_epl", start=date(2022, 8, 1), end=date(2024, 7, 31)))
    return windows


def windows_of(sport: str) -> list[Window]:
    return [w for w in holdout_windows() if w.sport == sport]


def intersecting(sport: str, start: date, end: date) -> list[Window]:
    """The holdout windows of ``sport`` that the days ``start``..``end`` (inclusive) touch."""
    return [w for w in windows_of(sport) if start <= w.end and end >= w.start]


def development_segments(sport: str, start: date, end: date) -> list[tuple[date, date]]:
    """``start``..``end`` less the sport's holdout windows, as ordered day ranges."""
    segments = [(start, end)]
    for w in sorted(windows_of(sport), key=lambda w: w.start):
        clipped: list[tuple[date, date]] = []
        for a, b in segments:
            if b < w.start or a > w.end:
                clipped.append((a, b))
                continue
            if a < w.start:
                clipped.append((a, date.fromordinal(w.start.toordinal() - 1)))
            if b > w.end:
                clipped.append((date.fromordinal(w.end.toordinal() + 1), b))
        segments = clipped
    return segments


_ISSUER = object()


@dataclass(frozen=True)
class HoldoutAccess:
    """The capability to replay one sport's sealed window. Only ``Reservation.access`` issues
    one, so a replay of sealed days is reachable only from a reserved holdout run."""

    window: Window
    _token: object = field(repr=False)

    def __post_init__(self) -> None:
        if self._token is not _ISSUER:
            raise HoldoutRefused("a HoldoutAccess is issued by a holdout reservation only")

    def covers(self, sport: str, start: date, end: date) -> bool:
        w = self.window
        return sport == w.sport and w.start <= start and end <= w.end


def require_clear(sport: str, start: date, end: date, access: HoldoutAccess | None = None) -> None:
    """Raise ``HoldoutRefused`` for a range that touches the sport's sealed windows, unless
    ``access`` covers it."""
    touched = intersecting(sport, start, end)
    if not touched or (access is not None and access.covers(sport, start, end)):
        return
    raise HoldoutRefused(
        f"{sport} {start}..{end} intersects the sealed holdout ({', '.join(map(str, touched))}); "
        "only `odds candidates holdout <preregistration>` replays it"
    )


class Preregistration(BaseModel):
    """What an idea fixes before the backfill lands and before its one holdout run: the
    strategy (name, version, rule, market, leagues), its whole grid, how the cell was to be
    selected (a link to the rule's text) and the selected cell. A file
    ``preregistrations/<issue>.yaml`` added in one commit and never changed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    issue: Annotated[int, Field(strict=True, gt=0)]
    name: str = Field(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")
    version: Annotated[int, Field(strict=True, gt=0)]
    rule: str = Field(min_length=1)
    market: str = Field(min_length=1)
    leagues: list[str] = Field(min_length=1)
    grid: dict[str, list[object]] = Field(min_length=1)
    selection_rule: str = Field(pattern=SELECTION_RULE_PATTERN)
    cell: dict[str, object] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> Preregistration:
        if len(set(self.leagues)) != len(self.leagues):
            raise ValueError("a league is listed twice")
        if any(not values for values in self.grid.values()):
            raise ValueError("a grid parameter has no values")
        if self.cell.keys() != self.grid.keys():
            raise ValueError(
                f"the cell's parameters {sorted(self.cell)} are not the grid's {sorted(self.grid)}"
            )
        for key, value in self.cell.items():
            if value not in self.grid[key]:
                raise ValueError(
                    f"the cell's {key}={value!r} is not in the grid {self.grid[key]!r}"
                )
        return self


class GateVerdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"


class Gate(BaseModel):
    """The holdout gate: mean EV above 0 and not killed (the 95% interval's upper bound above
    0). Under ``MIN_GATE_BETS`` bets, not killed alone passes, flagged ``low_bets``. A result
    never promotes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    verdict: GateVerdict
    bets: NonNegativeInt
    mean_ev: float | None
    ci_lower: float | None
    ci_upper: float | None
    killed: bool
    low_bets: bool


def gate(overall: Checkpoint | None) -> Gate:
    """The verdict over every bet with a close. No bets is a fail, flagged: nothing was scored."""
    if overall is None or overall.n == 0:
        return Gate(
            verdict=GateVerdict.FAIL,
            bets=0,
            mean_ev=None,
            ci_lower=None,
            ci_upper=None,
            killed=False,
            low_bets=True,
        )
    killed = overall.ci95[1] <= 0
    low = overall.n < MIN_GATE_BETS
    passed = not killed and (low or overall.mean_ev > 0)
    return Gate(
        verdict=GateVerdict.PASS if passed else GateVerdict.FAIL,
        bets=overall.n,
        mean_ev=overall.mean_ev,
        ci_lower=overall.ci95[0],
        ci_upper=overall.ci95[1],
        killed=killed,
        low_bets=low,
    )


class Reserved(BaseModel):
    """The first ledger row of a run, written and synced before any sealed data is read. It
    consumes the idea's version whatever follows."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["reserved"] = "reserved"
    issue: Annotated[int, Field(strict=True, gt=0)]
    version: Annotated[int, Field(strict=True, gt=0)]
    git_sha: str = Field(pattern=COMMIT_PATTERN)
    started_at: datetime


class HoldoutResult(BaseModel):
    """The pooled result of a reserved run. ``git_sha`` is the code that ran and
    ``preregistration_commit`` the one commit that added the pre-registration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["result"] = "result"
    issue: Annotated[int, Field(strict=True, gt=0)]
    version: Annotated[int, Field(strict=True, gt=0)]
    name: str = Field(min_length=1)
    rule: str = Field(min_length=1)
    rule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cell: dict[str, object] = Field(min_length=1)
    windows: list[Window] = Field(min_length=1)
    ran_on: date
    git_sha: str = Field(pattern=COMMIT_PATTERN)
    preregistration_commit: str = Field(pattern=COMMIT_PATTERN)
    result: Gate


LedgerRow = Annotated[Reserved | HoldoutResult, Field(discriminator="kind")]
_row_adapter: TypeAdapter[Reserved | HoldoutResult] = TypeAdapter(LedgerRow)


def parse_ledger(text: str) -> list[Reserved | HoldoutResult]:
    """The rows of a ledger, or ``ValueError``: each idea version has one ``reserved`` row, and
    at most one ``result`` row, after it."""
    rows = [_row_adapter.validate_json(line) for line in text.splitlines() if line.strip()]
    reserved: set[tuple[int, int]] = set()
    resulted: set[tuple[int, int]] = set()
    for row in rows:
        key = (row.issue, row.version)
        if isinstance(row, Reserved):
            if key in reserved:
                raise ValueError(f"issue {key[0]} version {key[1]} is reserved twice")
            reserved.add(key)
        else:
            if key not in reserved or key in resulted:
                raise ValueError(
                    f"issue {key[0]} version {key[1]}: a result without one reservation"
                )
            resulted.add(key)
    return rows


def load_ledger(directory: Path) -> list[Reserved | HoldoutResult]:
    path = directory / HOLDOUT_LEDGER
    return parse_ledger(path.read_text()) if path.exists() else []


def _append(directory: Path, row: Reserved | HoldoutResult) -> None:
    with (directory / HOLDOUT_LEDGER).open("a") as ledger:
        ledger.write(row.model_dump_json() + "\n")
        ledger.flush()
        os.fsync(ledger.fileno())


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)


def _git_out(root: Path, *args: str) -> str:
    done = _git(root, *args)
    if done.returncode != 0:
        raise HoldoutRefused(f"git {' '.join(args)} failed: {done.stderr.strip()}")
    return done.stdout.strip()


@contextmanager
def exclusive_lock(root: Path) -> Iterator[None]:
    """An exclusive lock shared by every worktree of the repository, held for a whole run."""
    common = Path(_git_out(root, "rev-parse", "--path-format=absolute", "--git-common-dir"))
    with (common / "odds-holdout.lock").open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise HoldoutRefused("another holdout run holds the lock") from exc
        yield


def load_preregistration(path: Path, directory: Path) -> Preregistration:
    relative = f"{directory.name}/{PREREGISTRATION_DIR}"
    if path.resolve().parent != (directory / PREREGISTRATION_DIR).resolve():
        raise HoldoutRefused(f"{path} is not a file in {relative}/")
    try:
        prereg = Preregistration.model_validate(yaml.safe_load(path.read_text()))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise HoldoutRefused(f"{path} is invalid: {exc}") from exc
    if path.stem != str(prereg.issue):
        raise HoldoutRefused(f"{path.name} must be named {prereg.issue}.yaml")
    for league in prereg.leagues:
        if not windows_of(league):
            raise HoldoutRefused(f"{league} has no sealed window, so it has no holdout to run")
    return prereg


def preregistration_commit(path: Path, root: Path, earliest_backfill: datetime | None) -> str:
    """The one commit that added ``path``, after checking it is committed, unchanged and older
    than ``earliest_backfill``, the first Odds API backfill snapshot the database holds."""
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    if _git(root, "cat-file", "-e", f"HEAD:{relative}").returncode != 0:
        raise HoldoutRefused(f"{relative} is not committed")
    if _git(root, "diff", "--quiet", "HEAD", "--", relative).returncode != 0:
        raise HoldoutRefused(f"{relative} differs from its committed version")
    log = _git_out(root, "log", "--format=%H %cI", "HEAD", "--", relative).splitlines()
    if len(log) != 1:
        raise HoldoutRefused(
            f"{relative} must be touched by exactly one commit, it has {len(log)}: a change "
            "to a pre-registration is a new version with a new file"
        )
    commit, committed = log[0].split(" ")
    if earliest_backfill is not None and datetime.fromisoformat(committed) >= earliest_backfill:
        raise HoldoutRefused(
            f"{relative} was committed {committed}, not before the first backfill snapshot "
            f"({earliest_backfill.isoformat()}), so its grid was not frozen first"
        )
    return commit


def _committed_ledger(root: Path, directory: Path, ref: str) -> str | None:
    done = _git(root, "show", f"{ref}:{directory.name}/{HOLDOUT_LEDGER}")
    return done.stdout if done.returncode == 0 else None


def require_committed_ledger(root: Path, directory: Path, issue: int, version: int) -> None:
    """The tree is clean, the ledger at HEAD extends the one at ``origin/main`` (when there is
    one) and holds no row for ``issue`` and ``version``."""
    if _git_out(root, "status", "--porcelain", "--untracked-files=no"):
        raise HoldoutRefused("tracked files have uncommitted changes, so HEAD is not what runs")
    committed = _committed_ledger(root, directory, "HEAD")
    if committed is None:
        raise HoldoutRefused(f"{directory.name}/{HOLDOUT_LEDGER} is not committed")
    if _git(root, "rev-parse", "--verify", "--quiet", "origin/main").returncode == 0:
        upstream = _committed_ledger(root, directory, "origin/main") or ""
        if committed.splitlines()[: len(upstream.splitlines())] != upstream.splitlines():
            raise HoldoutRefused("the committed ledger does not extend origin/main's: rebase first")
    try:
        rows = parse_ledger(committed)
    except (ValueError, ValidationError) as exc:
        raise HoldoutRefused(f"the committed ledger is invalid: {exc}") from exc
    if any((r.issue, r.version) == (issue, version) for r in rows):
        raise HoldoutRefused(
            f"issue {issue} version {version} already ran on the holdout; a new version "
            "needs a new pre-registration"
        )


@dataclass(frozen=True)
class Reservation:
    """The idea's one run, reserved in the ledger. Holds the lock until closed by the caller's
    ``exclusive_lock`` block."""

    prereg: Preregistration
    directory: Path
    git_sha: str
    preregistration_commit: str

    def access(self, sport: str) -> HoldoutAccess:
        windows = windows_of(sport)
        if sport not in self.prereg.leagues or len(windows) != 1:
            raise HoldoutRefused(f"{sport} is not a league of this pre-registration's holdout")
        return HoldoutAccess(windows[0], _ISSUER)

    def record(
        self, rule_sha: str, windows: list[Window], ran_on: date, result: Gate
    ) -> HoldoutResult:
        row = HoldoutResult(
            issue=self.prereg.issue,
            version=self.prereg.version,
            name=self.prereg.name,
            rule=self.prereg.rule,
            rule_sha256=rule_sha,
            cell=self.prereg.cell,
            windows=windows,
            ran_on=ran_on,
            git_sha=self.git_sha,
            preregistration_commit=self.preregistration_commit,
            result=result,
        )
        _append(self.directory, row)
        return row


def reserve(
    prereg: Preregistration,
    path: Path,
    directory: Path,
    earliest_backfill: datetime | None,
    started_at: datetime,
) -> Reservation:
    """Check the pre-registration and the committed ledger, then append and sync the
    ``reserved`` row. Call inside ``exclusive_lock`` and before any sealed data is read."""
    root = directory.parent
    commit = preregistration_commit(path, root, earliest_backfill)
    require_committed_ledger(root, directory, prereg.issue, prereg.version)
    sha = _git_out(root, "rev-parse", "HEAD")
    _append(
        directory,
        Reserved(issue=prereg.issue, version=prereg.version, git_sha=sha, started_at=started_at),
    )
    return Reservation(prereg, directory, sha, commit)
