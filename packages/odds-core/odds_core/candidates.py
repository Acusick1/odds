"""Betting candidate registry: one YAML file per candidate."""

from __future__ import annotations

from collections import Counter
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    NonNegativeInt,
    PositiveInt,
    TypeAdapter,
    model_validator,
)

from odds_core.config import get_settings
from odds_core.sports import SportKey
from odds_core.vocabulary import Market

NAME_PATTERN = r"^[a-z0-9]+(-[a-z0-9]+)*$"
SHA256_PATTERN = r"^[0-9a-f]{64}$"


class CandidateStatus(StrEnum):
    ACTIVE = "active"
    PROMOTED = "promoted"
    KILLED = "killed"


class _CandidateBase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=NAME_PATTERN)
    description: str
    sport: SportKey
    version: Annotated[int, Field(strict=True, gt=0)]
    issue: PositiveInt
    start_date: date
    status: CandidateStatus


class MechanicalCandidate(_CandidateBase):
    kind: Literal["mechanical"]
    market: Market
    rule: str
    # Absent only in a draft file; the registry requires it.
    rule_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    params: dict[str, JsonValue] = {}


class AgentCandidate(_CandidateBase):
    kind: Literal["agent"]


Candidate = Annotated[MechanicalCandidate | AgentCandidate, Field(discriminator="kind")]

_candidate_adapter: TypeAdapter[Candidate] = TypeAdapter(Candidate)


def candidates_dir() -> Path:
    return get_settings().project_root / "candidates"


# The rehash ledger, beside the candidate files; not a YAML file, so not a candidate.
REHASH_LEDGER = "rehashes.jsonl"


COMMIT_PATTERN = r"^[0-9a-f]{40}$"


class RehashCandidate(BaseModel):
    """One candidate's replay and parity in a rehash: its decisions with inputs and bets in the
    replay, and its recorded decisions the parity check matched, of ``parity_total``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=NAME_PATTERN)
    version: Annotated[int, Field(strict=True, gt=0)]
    decisions: PositiveInt
    bets: NonNegativeInt
    parity_matched: NonNegativeInt
    parity_total: NonNegativeInt

    @model_validator(mode="after")
    def _parity_holds(self) -> RehashCandidate:
        if self.parity_matched != self.parity_total:
            raise ValueError(
                f"{self.name}: parity matched {self.parity_matched} of {self.parity_total}"
            )
        return self


class Rehash(BaseModel):
    """A change of ``rule``'s hash from ``old`` (its hash at commit ``base``) to ``new`` (at
    ``head``) that the replay of every candidate of the rule from ``start`` to ``end`` against
    ``database`` showed changes no bet: both replays had the sha256 ``replay_sha256``.

    A record the rehash command makes, not a proof: nothing re-runs the replay to check it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    rule: str = Field(min_length=1)
    old: str = Field(pattern=SHA256_PATTERN)
    new: str = Field(pattern=SHA256_PATTERN)
    base: str = Field(pattern=COMMIT_PATTERN)
    head: str = Field(pattern=COMMIT_PATTERN)
    start: date
    end: date
    database: str = Field(min_length=1)
    replay_sha256: str = Field(pattern=SHA256_PATTERN)
    link: str = Field(pattern=r"^https://github\.com/\S+$")
    candidates: list[RehashCandidate] = Field(min_length=1)

    @model_validator(mode="after")
    def _evidence(self) -> Rehash:
        if self.start > self.end:
            raise ValueError(f"replay range {self.start}..{self.end} is empty")
        if self.old == self.new:
            raise ValueError("old and new hash are the same")
        if not any(c.bets for c in self.candidates):
            raise ValueError("the replay placed no bet")
        return self


def load_rehashes(directory: Path) -> list[Rehash]:
    """The rehash ledger in ``directory``, one JSON object per line; empty without one."""
    path = directory / REHASH_LEDGER
    if not path.exists():
        return []
    return [
        Rehash.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()
    ]


def append_rehash(directory: Path, rehash: Rehash) -> None:
    with (directory / REHASH_LEDGER).open("a") as ledger:
        ledger.write(rehash.model_dump_json() + "\n")


def load_candidate_file(path: Path) -> Candidate:
    """Parse one candidate file; a draft may lack ``rule_sha256`` and have any file name."""
    try:
        return _candidate_adapter.validate_python(yaml.safe_load(path.read_text()))
    except (ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"{path}: {exc}") from exc


def load_candidates(directory: Path) -> list[Candidate]:
    if not directory.is_dir():
        return []

    candidates: list[Candidate] = []
    for path in sorted(directory.glob("*.yaml")):
        candidate = load_candidate_file(path)
        if candidate.name != path.stem:
            raise ValueError(f"{path}: name {candidate.name!r} must match the file name")
        if isinstance(candidate, MechanicalCandidate) and candidate.rule_sha256 is None:
            raise ValueError(f"{path}: a registered mechanical candidate must record rule_sha256")
        candidates.append(candidate)

    live_agents_per_sport = Counter(
        c.sport for c in candidates if c.kind == "agent" and c.status != CandidateStatus.KILLED
    )
    for sport, count in live_agents_per_sport.items():
        if count > 1:
            raise ValueError(
                f"{directory}: {count} active or promoted agent candidates for {sport}, at most one"
            )

    return sorted(candidates, key=lambda c: c.name)
