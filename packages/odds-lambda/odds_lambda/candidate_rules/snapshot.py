"""The candidate registry as a process first read it, validated against the rule code that
process imported, so a checkout changed under a running process never pairs new candidate files
with old rule code.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import MappingProxyType

from odds_core.candidates import Candidate, RuleCandidate, candidates_dir, load_candidates

from odds_lambda.candidate_rules.pinning import validate_candidate_rules


@dataclass(frozen=True)
class RegistrySnapshot:
    candidates: tuple[Candidate, ...]
    # Each invalid rule candidate's name and its error, from validation at load.
    invalid: Mapping[str, ValueError]

    def get(self, name: str) -> Candidate | None:
        return next((c for c in self.candidates if c.name == name), None)


def load_snapshot(directory: Path) -> RegistrySnapshot:
    """Load the registry in ``directory`` and validate each rule candidate, hashing its rule now.

    ``ValueError`` when the registry itself cannot be read; an invalid candidate is recorded in
    ``invalid`` instead.
    """
    candidates = tuple(load_candidates(directory))
    invalid: dict[str, ValueError] = {}
    for candidate in candidates:
        if not isinstance(candidate, RuleCandidate):
            continue
        try:
            validate_candidate_rules([candidate])
        except ValueError as exc:
            invalid[candidate.name] = exc
    return RegistrySnapshot(candidates, MappingProxyType(invalid))


@cache
def registry_snapshot() -> RegistrySnapshot:
    """The registry this process loaded on its first call; ``ValueError`` while it cannot be
    read, which is not cached."""
    return load_snapshot(candidates_dir())
