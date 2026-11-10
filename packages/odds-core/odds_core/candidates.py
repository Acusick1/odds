"""Betting candidate registry: one YAML file per candidate."""

from collections import Counter
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, JsonValue, PositiveInt, TypeAdapter

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
