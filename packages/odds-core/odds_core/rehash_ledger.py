"""The rehash ledger: hash changes recorded as leaving every bet the same.

Imports only the standard library and pydantic, so the ``candidate-versions`` hook, which has no
project install, validates rows with these models by loading this file.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt, model_validator

# The ledger, beside the candidate files; not a YAML file, so not a candidate.
REHASH_LEDGER = "rehashes.jsonl"
# The same patterns as the registry's, which this module cannot import without the project's dependencies.
NAME_PATTERN = r"^[a-z0-9]+(-[a-z0-9]+)*$"
SHA256_PATTERN = r"^[0-9a-f]{64}$"
COMMIT_PATTERN = r"^[0-9a-f]{40}$"
LINK_PATTERN = r"^https://github\.com/[^/\s]+/[^/\s]+/(pull|issues)/[0-9]+$"


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
    link: str = Field(pattern=LINK_PATTERN)
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
