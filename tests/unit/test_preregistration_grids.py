"""Round 1's committed pre-registration files match the grid sections of their idea issues.

``tests/fixtures/preregistration_grids/<issue>.md`` holds each issue's "Pre-registered backtest
grid" section as it stood when the files were committed. The parser reads the parameter sets
the text states; the few facts that prose states too loosely to parse are asserted by hand
in ``HAND_ASSERTED``.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest
import yaml
from odds_analytics.holdout import Preregistration

ROOT = Path(__file__).resolve().parents[2]
FILES = sorted((ROOT / "candidates" / "preregistrations").glob("*.yaml"))
FIXTURES = ROOT / "tests" / "fixtures" / "preregistration_grids"

FOOTBALL = {
    "soccer_epl",
    "soccer_spain_la_liga",
    "soccer_italy_serie_a",
    "soccer_germany_bundesliga",
    "soccer_france_ligue_one",
    "soccer_efl_champ",
}
SELECTION_RULE = "https://github.com/Acusick1/odds/issues/821#issuecomment-6769611091"

# The grid key each label in an issue's `label ∈ {...}` sets stands for.
LABELS = {
    "m": "m",
    "g": "g",
    "r": "r",
    "o": "o",
    "t": "t",
    "N": "n",
    "L": "lookback_minutes",
    "lookback": "lookback_minutes",
    "p_min": "p_min",
    "ρ": "rho",
    "τ": "tau",
    "side": "side",
    "threshold": "threshold",
    "consensus-EV threshold": "threshold",
    "EV threshold": "threshold",
    "decision time": "decision_times",
    "decision times": "decision_times",
    "decision": "decision_times",
}

# Facts the prose states in a shape the parser does not read, by issue.
HAND_ASSERTED: dict[int, dict[str, list[object]]] = {
    # "The rule is `totals_implied` with `outcome: draw`"
    772: {"outcome": ["draw"]},
    # "a version in #772's family (`totals_implied`, with `outcome: favourite`)"
    839: {"outcome": ["favourite"]},
    # "Lines are fixed at {1.5, 3.5}. ... absent or within 2 pp of the model."
    837: {"lines": [[1.5, 3.5]], "exchange_tolerance": [0.02]},
    # "The draft's `max_input_age_minutes` is 60 ... A registered version uses 10"
    842: {"max_input_age_minutes": [10]},
}

SET = re.compile(r"^(?:- )?(?P<label>.+?) ∈ \{(?P<values>.*)\}")
FIXED_TIMES = re.compile(r"Decision times? (?:are fixed at )?(?P<value>\[[^\]]*\])")
AGE = re.compile(r"`max_input_age_minutes`(?: is|:) (?P<value>\d+)")
LOOKBACK_AGE = re.compile(r"and (?P<value>\d+) at the lookback")
CELLS = re.compile(r"(?P<count>\d+) cells")
PAIR = re.compile(r"^\[(?P<time>\d+)\] \(age (?P<age>\d+)\)$")


def split_top_level(text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    for i, char in enumerate(text):
        depth += char in "[("
        depth -= char in "])"
        if char == "," and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    parts.append(text[start:].strip())
    return parts


def number(text: str) -> int | float:
    text = text.replace("−", "-")
    return int(text) if re.fullmatch(r"-?\d+", text) else float(text)


def minutes(text: str) -> list[int]:
    return [int(x) for x in text.strip("[]").split(",")]


def parse_grid(text: str) -> tuple[dict[str, list[object]], int]:
    """The grid the issue text states, and its stated cell count."""
    grid: dict[str, list[object]] = {}
    pairs: list[object] = []
    for line in text.splitlines():
        for piece in line.split(" × "):
            match = SET.match(piece.strip())
            if match is None:
                continue
            label = match["label"].split(" (")[0]
            key = LABELS[label]
            items = split_top_level(match["values"])
            if all(PAIR.match(item) for item in items):
                pairs = [
                    {
                        "decision_times": [int(PAIR.match(item)["time"])],  # type: ignore[index]
                        "max_input_age_minutes": int(PAIR.match(item)["age"]),  # type: ignore[index]
                    }
                    for item in items
                ]
                grid["decision"] = pairs
            elif key == "decision_times":
                grid[key] = [minutes(item) for item in items]
            else:
                grid[key] = [item if item.isalpha() else number(item) for item in items]
    if "decision_times" not in grid and "decision" not in grid:
        fixed = FIXED_TIMES.search(text)
        assert fixed, "no decision times in the text"
        grid["decision_times"] = [minutes(fixed["value"])]
    if "decision" not in grid:
        age = AGE.search(text)
        assert age, "no input age in the text"
        grid["max_input_age_minutes"] = [int(age["value"])]
    lookback_age = LOOKBACK_AGE.search(text)
    if lookback_age:
        grid["lookback_max_input_age_minutes"] = [int(lookback_age["value"])]
    return grid, int(CELLS.search(text)["count"])  # type: ignore[index]


class TestPreregistrationFiles:
    def test_round_one_has_fifteen_files(self) -> None:
        assert len(FILES) == 15

    @pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
    def test_file_loads_and_is_named_for_its_identity(self, path: Path) -> None:
        prereg = Preregistration.model_validate(yaml.safe_load(path.read_text()))
        assert prereg.stem == path.stem
        assert prereg.version == 1
        assert set(prereg.leagues) == FOOTBALL
        assert prereg.selection_rule == SELECTION_RULE

    @pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
    def test_grid_matches_the_issue_text(self, path: Path) -> None:
        prereg = Preregistration.model_validate(yaml.safe_load(path.read_text()))
        parsed, cells = parse_grid((FIXTURES / f"{prereg.issue}.md").read_text())
        parsed.update(HAND_ASSERTED.get(prereg.issue, {}))
        assert prereg.grid == parsed
        assert math.prod(len(values) for values in prereg.grid.values()) == cells

    def test_every_fixture_has_a_file(self) -> None:
        issues = {int(p.stem) for p in FIXTURES.glob("*.md")}
        assert issues == {int(p.name.split("-")[0]) for p in FILES}

    def test_exchange_gap_early_has_one_grid_in_both_markets(self) -> None:
        one, other = (
            yaml.safe_load(
                (ROOT / "candidates" / "preregistrations" / f"842-{m}-v1.yaml").read_text()
            )
            for m in ("1x2", "totals")
        )
        assert one["grid"] == other["grid"]
