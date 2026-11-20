"""The candidate registry as a process first read it, validated against the rule code that
process imported.

A long-running process (the scheduler, the MCP server) keeps running the rule code it imported
at startup, so it must also keep the registry it validated then: a checkout changed under it
(a branch switch, a pull without a restart) would otherwise pair new candidate files with old
rule code.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path

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
    return RegistrySnapshot(candidates, invalid)


@cache
def registry_snapshot() -> RegistrySnapshot:
    """The registry this process loaded on its first call; ``ValueError`` while it is invalid.

    ``odds scheduler start`` calls it before starting, so the scheduler fails to start on an
    invalid registry and every run in it sees the registry of its start.
    """
    return load_snapshot(candidates_dir())
