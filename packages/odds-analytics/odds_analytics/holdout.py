"""The sealed holdout: which ranges it covers, the pre-registration that opens it, the one-shot
ledger of runs and the gate verdict.

Lives outside the rule import closure, so editing it changes no rule hash.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import yaml
from odds_core.candidates import MechanicalCandidate
from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, model_validator

from odds_analytics.candidate_report import Checkpoint

HOLDOUT_LEDGER = "holdout.jsonl"
PREREGISTRATION_DIR = "preregistrations"
COMMIT_PATTERN = r"^[0-9a-f]{40}$"
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


@dataclass(frozen=True)
class Window:
    sport: str
    start: date
    end: date

    def __str__(self) -> str:
        return f"{self.sport} {self.start}..{self.end}"


def holdout_windows() -> list[Window]:
    windows = [Window(sport, date(2025, 8, 1), date(2026, 7, 31)) for sport in _FIVE_LEAGUES]
    if EPL_2022_24_SEALED:
        windows.append(Window("soccer_epl", date(2022, 8, 1), date(2024, 7, 31)))
    return windows


def intersecting(sport: str, start: date, end: date) -> list[Window]:
    """The holdout windows of ``sport`` that the days ``start``..``end`` (inclusive) touch."""
    return [w for w in holdout_windows() if w.sport == sport and start <= w.end and end >= w.start]


class HoldoutRefused(ValueError):
    """A holdout run that is not allowed; nothing was replayed or written."""


class Preregistration(BaseModel):
    """What an idea fixed before its first holdout run: its issue, the selected cell's params
    and the commit at which the grid was frozen. A file ``preregistrations/<issue>.yaml``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    issue: Annotated[int, Field(strict=True, gt=0)]
    grid_sha: str = Field(pattern=COMMIT_PATTERN)
    cell: dict[str, object] = Field(min_length=1)


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


class HoldoutRun(BaseModel):
    """One holdout run, a line of the ledger. ``issue`` and ``version`` are the idea; a second
    run of the pair is refused. ``git_sha`` is the code that ran, ``grid_sha`` the freeze."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    issue: Annotated[int, Field(strict=True, gt=0)]
    version: Annotated[int, Field(strict=True, gt=0)]
    name: str = Field(min_length=1)
    sport: str = Field(min_length=1)
    cell: dict[str, object] = Field(min_length=1)
    start: date
    end: date
    ran_on: date
    git_sha: str = Field(pattern=COMMIT_PATTERN)
    grid_sha: str = Field(pattern=COMMIT_PATTERN)
    result: Gate

    @model_validator(mode="after")
    def _range(self) -> HoldoutRun:
        if self.start > self.end:
            raise ValueError(f"range {self.start}..{self.end} is empty")
        return self


def load_runs(directory: Path) -> list[HoldoutRun]:
    path = directory / HOLDOUT_LEDGER
    if not path.exists():
        return []
    return [
        HoldoutRun.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def append_run(directory: Path, run: HoldoutRun) -> None:
    with (directory / HOLDOUT_LEDGER).open("a") as ledger:
        ledger.write(run.model_dump_json() + "\n")


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)


def head_sha(root: Path) -> str:
    done = _git(root, "rev-parse", "HEAD")
    if done.returncode != 0:
        raise HoldoutRefused(f"cannot read HEAD of {root}: {done.stderr.strip()}")
    return done.stdout.strip()


def admit(
    candidate: MechanicalCandidate, start: date, end: date, directory: Path
) -> Preregistration | None:
    """None when the range is outside the holdout. Otherwise the candidate's pre-registration,
    or ``HoldoutRefused`` unless: the file ``preregistrations/<issue>.yaml`` in ``directory``
    (the ``candidates/`` directory of a git checkout) is committed and unmodified, its grid
    commit is an ancestor of HEAD, the draft's params equal its cell, the tracked files are
    clean (the ledger apart) and the idea and version have not run on the holdout."""
    touched = intersecting(candidate.sport, start, end)
    if not touched:
        return None
    where = ", ".join(map(str, touched))
    root = directory.parent
    relative = f"{directory.name}/{PREREGISTRATION_DIR}/{candidate.issue}.yaml"
    path = root / relative

    def refuse(reason: str) -> HoldoutRefused:
        return HoldoutRefused(f"{start}..{end} intersects the sealed holdout ({where}): {reason}")

    if not path.is_file():
        raise refuse(f"no pre-registration {relative}")
    if _git(root, "cat-file", "-e", f"HEAD:{relative}").returncode != 0:
        raise refuse(f"{relative} is not committed")
    if _git(root, "diff", "--quiet", "HEAD", "--", relative).returncode != 0:
        raise refuse(f"{relative} differs from its committed version")
    try:
        prereg = Preregistration.model_validate(yaml.safe_load(path.read_text()))
    except (ValueError, yaml.YAMLError) as exc:
        raise refuse(f"{relative} is invalid: {exc}") from exc
    if prereg.issue != candidate.issue:
        raise refuse(f"{relative} is for issue {prereg.issue}, the draft's is {candidate.issue}")
    if _git(root, "merge-base", "--is-ancestor", prereg.grid_sha, "HEAD").returncode != 0:
        raise refuse(f"grid commit {prereg.grid_sha} is not an ancestor of HEAD")
    if candidate.params != prereg.cell:
        raise refuse(
            f"the draft's params {candidate.params} are not the selected cell {prereg.cell}"
        )
    dirty = [
        line
        for line in _git(root, "status", "--porcelain", "--untracked-files=no").stdout.splitlines()
        if line[3:] != f"{directory.name}/{HOLDOUT_LEDGER}"
    ]
    if dirty:
        raise refuse("tracked files have uncommitted changes, so HEAD is not the code that runs")
    for run in load_runs(directory):
        if (run.issue, run.version) == (candidate.issue, candidate.version):
            raise refuse(
                f"issue {candidate.issue} version {candidate.version} already ran on "
                f"{run.start}..{run.end}; a new version needs a new grid"
            )
    return prereg


def record(
    candidate: MechanicalCandidate,
    prereg: Preregistration,
    start: date,
    end: date,
    overall: Checkpoint | None,
    directory: Path,
    today: date,
) -> HoldoutRun:
    """Append this run to the ledger and return it."""
    run = HoldoutRun(
        issue=candidate.issue,
        version=candidate.version,
        name=candidate.name,
        sport=candidate.sport,
        cell=prereg.cell,
        start=start,
        end=end,
        ran_on=today,
        git_sha=head_sha(directory.parent),
        grid_sha=prereg.grid_sha,
        result=gate(overall),
    )
    append_run(directory, run)
    return run
