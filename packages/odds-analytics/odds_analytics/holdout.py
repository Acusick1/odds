"""The sealed holdout: its windows, the pre-registration that opens it, the one-shot ledger of
reservations and results, and the gate verdict.

``odds_analytics.candidate_backtest.final_events`` calls ``require_clear``, so no replay of a
sealed range runs without a ``HoldoutAccess``, which only a ``Reservation`` issues. The module
imports nothing from the replay, which imports it, and lies outside the rule import closure.
"""

from __future__ import annotations

import fcntl
import hashlib
import itertools
import json
import os
import subprocess
from collections.abc import Iterator, Sequence
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
SELECTION_DIR = "selections"
SPENT_FILE = "odds-holdout-spent.jsonl"
COMMIT_PATTERN = r"^[0-9a-f]{40}$"
MARKET_PATTERN = r"^[a-z0-9_]+$"
SELECTION_RULE_PATTERN = r"^https://github\.com/[^/\s]+/[^/\s]+/(issues|pull)/[0-9]+#\S+$"
# Below this many holdout bets the gate asks only that the cell is not killed, and flags it.
MIN_GATE_BETS = 30

# Seasons run 1 August to 31 July.
FIVE_LEAGUES = (
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
    return [Window(sport=s, start=date(2025, 8, 1), end=date(2026, 7, 31)) for s in FIVE_LEAGUES]


def windows_of(sport: str) -> list[Window]:
    return [w for w in holdout_windows() if w.sport == sport]


def intersecting(sport: str, start: date, end: date) -> list[Window]:
    """The holdout windows of ``sport`` that the days ``start``..``end`` (inclusive) touch."""
    return [w for w in windows_of(sport) if start <= w.end and end >= w.start]


def development_segments(
    sport: str, start: date, end: date, windows: Sequence[Window] | None = None
) -> list[tuple[date, date]]:
    """``start``..``end`` less the sport's holdout windows, or the sport's among ``windows``
    when given, as ordered day ranges."""
    sealed = windows_of(sport) if windows is None else [w for w in windows if w.sport == sport]
    segments = [(start, end)]
    for w in sorted(sealed, key=lambda w: w.start):
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


def stem(issue: int, market: str, version: int) -> str:
    """The identity of one idea family's version, and the name of its files."""
    return f"{issue}-{market}-v{version}"


class Preregistration(BaseModel):
    """The frozen cells of an idea's family and version: its strategy (name, rule, market,
    leagues), either a ``grid`` of every parameter's values (decision times included) or, for
    parameters that vary together, an explicit list of ``cells``, and a link to the text of the
    rule that selects a cell. It holds no selected cell. A file ``preregistrations/<issue>-<market>-v<version>.yaml``
    added in one commit, before the backfill lands, and never changed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    issue: Annotated[int, Field(strict=True, gt=0)]
    name: str = Field(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")
    version: Annotated[int, Field(strict=True, gt=0)]
    rule: str = Field(min_length=1)
    market: str = Field(pattern=MARKET_PATTERN)
    leagues: list[str] = Field(min_length=1)
    grid: dict[str, list[object]] | None = None
    cells: list[dict[str, object]] | None = None
    selection_rule: str = Field(pattern=SELECTION_RULE_PATTERN)

    @model_validator(mode="after")
    def _consistent(self) -> Preregistration:
        if len(set(self.leagues)) != len(self.leagues):
            raise ValueError("a league is listed twice")
        if (self.grid is None) == (self.cells is None):
            raise ValueError("a pre-registration has exactly one of `grid` and `cells`")
        if self.grid is not None and (
            not self.grid or any(not values for values in self.grid.values())
        ):
            raise ValueError("a grid needs parameters and each needs values")
        if self.cells is not None:
            if not self.cells or any(not cell for cell in self.cells):
                raise ValueError("`cells` needs cells with parameters")
            if any(cell.keys() != self.cells[0].keys() for cell in self.cells):
                raise ValueError("every cell has the same parameters")
            if any(self.cells[i] in self.cells[:i] for i in range(len(self.cells))):
                raise ValueError("a cell is listed twice")
        return self

    def candidate_cells(self) -> list[dict[str, object]]:
        """Every cell the file pre-registers: the grid's cartesian product, or its listed cells."""
        if self.cells is not None:
            return self.cells
        assert self.grid is not None  # noqa: S101
        keys = list(self.grid)
        return [
            dict(zip(keys, values, strict=True))
            for values in itertools.product(*self.grid.values())
        ]

    @property
    def stem(self) -> str:
        return stem(self.issue, self.market, self.version)


class Selection(BaseModel):
    """The cell the pre-registered selection rule picked on development data, bound to the
    grid file by its sha256. A file ``selections/<issue>-<market>-v<version>.yaml`` added in
    one commit after the grid's and never changed; no deadline applies to it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    issue: Annotated[int, Field(strict=True, gt=0)]
    market: str = Field(pattern=MARKET_PATTERN)
    version: Annotated[int, Field(strict=True, gt=0)]
    grid_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cell: dict[str, object] = Field(min_length=1)

    @property
    def stem(self) -> str:
        return stem(self.issue, self.market, self.version)


@dataclass(frozen=True)
class Plan:
    """A grid file and the selection of its cell, checked against each other."""

    prereg: Preregistration
    selection: Selection
    grid_path: Path
    selection_path: Path

    @property
    def cell(self) -> dict[str, object]:
        return self.selection.cell

    @property
    def holdout_leagues(self) -> list[str]:
        """The pre-registered leagues that have a sealed window, which the run replays."""
        return [league for league in self.prereg.leagues if windows_of(league)]

    @property
    def windowless_leagues(self) -> list[str]:
        """The pre-registered leagues with no sealed window, which contribute no holdout bets."""
        return [league for league in self.prereg.leagues if not windows_of(league)]


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
    market: str = Field(pattern=MARKET_PATTERN)
    version: Annotated[int, Field(strict=True, gt=0)]
    git_sha: str = Field(pattern=COMMIT_PATTERN)
    started_at: datetime


class HoldoutResult(BaseModel):
    """The pooled result of a reserved run. ``git_sha`` is the code that ran;
    ``preregistration_commit`` and ``selection_commit`` the one commit of each file.
    ``leagues_without_window`` are the pre-registered leagues that had no sealed window and so
    contributed no bets."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["result"] = "result"
    issue: Annotated[int, Field(strict=True, gt=0)]
    market: str = Field(pattern=MARKET_PATTERN)
    version: Annotated[int, Field(strict=True, gt=0)]
    name: str = Field(min_length=1)
    rule: str = Field(min_length=1)
    rule_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cell: dict[str, object] = Field(min_length=1)
    windows: list[Window] = Field(min_length=1)
    leagues_without_window: list[str]
    ran_on: date
    git_sha: str = Field(pattern=COMMIT_PATTERN)
    preregistration_commit: str = Field(pattern=COMMIT_PATTERN)
    selection_commit: str = Field(pattern=COMMIT_PATTERN)
    result: Gate


LedgerRow = Annotated[Reserved | HoldoutResult, Field(discriminator="kind")]
_row_adapter: TypeAdapter[Reserved | HoldoutResult] = TypeAdapter(LedgerRow)


def parse_ledger(text: str) -> list[Reserved | HoldoutResult]:
    """The rows of a ledger, or ``ValueError``: each idea version has one ``reserved`` row, and
    at most one ``result`` row, after it."""
    rows = [_row_adapter.validate_json(line) for line in text.splitlines() if line.strip()]
    reserved: set[str] = set()
    resulted: set[str] = set()
    for row in rows:
        key = stem(row.issue, row.market, row.version)
        if isinstance(row, Reserved):
            if key in reserved:
                raise ValueError(f"{key} is reserved twice")
            reserved.add(key)
        else:
            if key not in reserved or key in resulted:
                raise ValueError(f"{key}: a result without one reservation")
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


def _common_dir(root: Path) -> Path:
    return Path(_git_out(root, "rev-parse", "--path-format=absolute", "--git-common-dir"))


def _load_file(path: Path, directory: Path, folder: str, model: type[BaseModel]) -> BaseModel:
    if path.resolve().parent != (directory / folder).resolve():
        raise HoldoutRefused(f"{path} is not a file in {directory.name}/{folder}/")
    try:
        return model.model_validate(yaml.safe_load(path.read_text()))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise HoldoutRefused(f"{path} is invalid: {exc}") from exc


def check_cell(prereg: Preregistration, cell: dict[str, object]) -> None:
    """``ValueError`` unless ``cell`` is one of the listed ``cells``, or has the grid's
    parameters and a grid value for each."""
    if prereg.cells is not None:
        if cell not in prereg.cells:
            raise ValueError(f"the cell {cell!r} is not one of the pre-registered cells")
        return
    assert prereg.grid is not None  # noqa: S101
    if cell.keys() != prereg.grid.keys():
        raise ValueError(
            f"the cell's parameters {sorted(cell)} are not the grid's {sorted(prereg.grid)}"
        )
    for key, value in cell.items():
        if value not in prereg.grid[key]:
            raise ValueError(f"the cell's {key}={value!r} is not in the grid {prereg.grid[key]!r}")


def load_preregistration(path: Path, directory: Path) -> Preregistration:
    """The grid file at ``path``, which must be in ``directory/preregistrations`` and named by
    its identity."""
    prereg = _load_file(path, directory, PREREGISTRATION_DIR, Preregistration)
    assert isinstance(prereg, Preregistration)  # noqa: S101
    if path.name != f"{prereg.stem}.yaml":
        raise HoldoutRefused(f"{path.name} must be named {prereg.stem}.yaml")
    return prereg


def load_plan(selection_path: Path, directory: Path) -> Plan:
    """The selection at ``selection_path`` and the grid file of its identity, checked: file
    names are the identity, the selection's ``grid_sha256`` is the grid file's content, the cell
    belongs to the grid and at least one league has a sealed window."""
    selection = _load_file(selection_path, directory, SELECTION_DIR, Selection)
    assert isinstance(selection, Selection)  # noqa: S101
    if selection_path.name != f"{selection.stem}.yaml":
        raise HoldoutRefused(f"{selection_path.name} must be named {selection.stem}.yaml")
    grid_path = directory / PREREGISTRATION_DIR / f"{selection.stem}.yaml"
    if not grid_path.is_file():
        raise HoldoutRefused(f"no pre-registration {PREREGISTRATION_DIR}/{grid_path.name}")
    prereg = load_preregistration(grid_path, directory)
    if hashlib.sha256(grid_path.read_bytes()).hexdigest() != selection.grid_sha256:
        raise HoldoutRefused(f"{selection_path.name} was selected from another grid file")
    try:
        check_cell(prereg, selection.cell)
    except ValueError as exc:
        raise HoldoutRefused(f"{selection_path.name}: {exc}") from exc
    plan = Plan(prereg, selection, grid_path, selection_path)
    if not plan.holdout_leagues:
        raise HoldoutRefused(
            f"none of {', '.join(prereg.leagues)} has a sealed window, so there is no holdout to run"
        )
    return plan


def committed_once(path: Path, root: Path, before: datetime | None = None) -> str:
    """The one commit that added ``path``, after checking it is committed, unchanged, has one
    commit in its history (renames followed, so a rename cannot hide edits) and, when
    ``before`` is given, was committed before it."""
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    if _git(root, "cat-file", "-e", f"HEAD:{relative}").returncode != 0:
        raise HoldoutRefused(f"{relative} is not committed")
    if _git(root, "diff", "--quiet", "HEAD", "--", relative).returncode != 0:
        raise HoldoutRefused(f"{relative} differs from its committed version")
    log = _git_out(root, "log", "--follow", "--format=%H %cI", "HEAD", "--", relative).splitlines()
    if len(log) != 1:
        raise HoldoutRefused(
            f"{relative} must be touched by exactly one commit, it has {len(log)}: a change "
            "is a new version with a new file"
        )
    commit, committed = log[0].split(" ")
    if before is not None and datetime.fromisoformat(committed) >= before:
        raise HoldoutRefused(
            f"{relative} was committed {committed}, not before the first backfill snapshot "
            f"({before.isoformat()}), so its grid was not frozen first"
        )
    return commit


def _committed_ledger(root: Path, directory: Path, ref: str) -> str | None:
    done = _git(root, "show", f"{ref}:{directory.name}/{HOLDOUT_LEDGER}")
    return done.stdout if done.returncode == 0 else None


def require_committed_ledger(root: Path, directory: Path, key: str) -> None:
    """The tree is clean, the ledger at HEAD extends the one at ``origin/main`` (when there is
    one) and holds no row for the identity ``key``."""
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
    if any(stem(r.issue, r.market, r.version) == key for r in rows):
        raise HoldoutRefused(f"{key} already ran on the holdout; a new version needs a new grid")


def _spend(root: Path, key: str, sha: str, started_at: datetime) -> None:
    """Record ``key`` as spent in state every worktree shares, synced before anything else is
    written. Under the lock, so two checkouts cannot both pass the check."""
    path = _common_dir(root) / SPENT_FILE
    spent = path.read_text().splitlines() if path.exists() else []
    if key in (json.loads(line)["key"] for line in spent if line.strip()):
        raise HoldoutRefused(f"{key} was already reserved by a run in this repository")
    with path.open("a") as handle:
        handle.write(
            json.dumps({"key": key, "git_sha": sha, "started_at": started_at.isoformat()}) + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
    # A newly created file's name survives a power loss only once its directory is synced.
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


@dataclass(frozen=True)
class Reservation:
    """The idea's one run, reserved in the ledger and in the repository's shared spent file.

    Issued by ``reserve`` only: the constructor refuses any other token. That is a guard
    against an accidental holdout read, not enforcement, since code that wants to read the
    holdout can read the database or forge the token."""

    plan: Plan
    directory: Path
    git_sha: str
    preregistration_commit: str
    selection_commit: str
    _token: object = field(repr=False)

    def __post_init__(self) -> None:
        if self._token is not _ISSUER:
            raise HoldoutRefused("a Reservation is issued by reserve() only")

    def access(self, sport: str) -> HoldoutAccess:
        windows = windows_of(sport)
        if sport not in self.plan.prereg.leagues or len(windows) != 1:
            raise HoldoutRefused(f"{sport} is not a league of this pre-registration's holdout")
        return HoldoutAccess(windows[0], _ISSUER)

    def record(
        self, rule_sha: str, windows: list[Window], ran_on: date, result: Gate
    ) -> HoldoutResult:
        prereg = self.plan.prereg
        row = HoldoutResult(
            issue=prereg.issue,
            market=prereg.market,
            version=prereg.version,
            name=prereg.name,
            rule=prereg.rule,
            rule_sha256=rule_sha,
            cell=self.plan.cell,
            windows=windows,
            leagues_without_window=self.plan.windowless_leagues,
            ran_on=ran_on,
            git_sha=self.git_sha,
            preregistration_commit=self.preregistration_commit,
            selection_commit=self.selection_commit,
            result=result,
        )
        _append(self.directory, row)
        return row


def reserve(
    plan: Plan,
    directory: Path,
    earliest_backfill: datetime | None,
    started_at: datetime,
) -> Reservation:
    """Check both files and the committed ledger, then spend the version in the shared spent
    file and append and sync the ``reserved`` ledger row. Call inside ``exclusive_lock`` and
    before any sealed data is read. A crash after the spend leaves the version spent."""
    root = directory.parent
    prereg = plan.prereg
    grid_commit = committed_once(plan.grid_path, root, earliest_backfill)
    selection_commit = committed_once(plan.selection_path, root)
    require_committed_ledger(root, directory, prereg.stem)
    sha = _git_out(root, "rev-parse", "HEAD")
    _spend(root, prereg.stem, sha, started_at)
    _append(
        directory,
        Reserved(
            issue=prereg.issue,
            market=prereg.market,
            version=prereg.version,
            git_sha=sha,
            started_at=started_at,
        ),
    )
    return Reservation(plan, directory, sha, grid_commit, selection_commit, _ISSUER)
