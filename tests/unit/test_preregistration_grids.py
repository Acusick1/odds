"""Round 1's committed pre-registration files match the grid sections of their idea issues and
are params their rules take.

``tests/fixtures/preregistration_grids/<issue>.md`` holds each issue's "Pre-registered backtest
grid" section as it stood when the files were committed. The parser reads the parameter sets
the text states; the few facts that prose states too loosely to parse are asserted by hand
in ``HAND_ASSERTED``.
"""

from __future__ import annotations

import itertools
import json
import re
from pathlib import Path

import pytest
import yaml
from odds_analytics.holdout import Preregistration
from odds_lambda.candidate_rules.registry import RULES
from odds_lambda.candidate_rules.rules import RuleParams

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

# The real param name each label in an issue's `label ∈ {...}` sets stands for, by issue. A rule
# that does not exist yet takes exactly these names in its params model.
DECISION = "minutes_before_kickoff"
COMMON = {"decision time": DECISION, "decision times": DECISION, "decision": DECISION}
LABELS: dict[int, dict[str, str]] = {
    771: {"m": "min_rise", "lookback": "lookbacks", "g": "gap"},
    772: {"ρ": "rho", "threshold": "threshold"},
    831: {"lookback": "lookbacks", "consensus-EV threshold": "threshold"},
    832: {"L": "lookbacks", "m": "min_rise", "g": "gap"},
    833: {"r": "min_ratio", "g": "gap"},
    834: {"threshold": "threshold", "o": "min_overround"},
    835: {"p_min": "min_probability", "threshold": "threshold"},
    836: {"side": "side"},
    837: {"threshold": "threshold"},
    838: {"lookback": "lookbacks", "EV threshold": "threshold"},
    839: {"τ": "min_over_probability", "threshold": "threshold"},
    840: {"N": "min_concurrent_kickoffs", "threshold": "threshold"},
    841: {"m": "min_margin"},
    842: {"t": "threshold"},
}

# Facts the prose states in a shape the parser does not read, by issue.
HAND_ASSERTED: dict[int, dict[str, list[object]]] = {
    # "The rule is `totals_implied` with `outcome: draw`"
    772: {"outcome": ["draw"]},
    # "a version in #772's family (`totals_implied`, with `outcome: favourite`)"
    839: {"outcome": ["favourite"]},
    # "Lines are fixed at {1.5, 3.5}. ... absent or within 2 pp of the model."
    837: {"lines": [[1.5, 3.5]], "exchange_tolerance": [0.02]},
    # "the draft's `max_input_age_minutes` is 60": the backtest and holdout reach the hourly band
    842: {"max_input_age_minutes": [60]},
}

# Round-1 rules not implemented yet.
PENDING_RULES = {
    "alternate_totals_left_behind",
    "crowded_gap",
    "exchange_implied_alternate_totals",
    "exchange_steam",
    "favourite_price",
    "fixed_totals_side",
    "last_result_overreaction",
    "team_news_lag",
    "totals_implied",
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


def parse_cells(issue: int, text: str) -> tuple[list[dict[str, object]], int]:
    """The cells the issue text states, in the real param names, and its stated cell count."""
    labels = {**COMMON, **LABELS[issue]}
    base_age = AGE.search(text)
    lookback_age = LOOKBACK_AGE.search(text)
    both_points = "at both points" in text
    params: dict[str, list[object]] = {}
    coupled: list[dict[str, object]] | None = None
    for line in text.splitlines():
        for piece in line.split(" × "):
            match = SET.match(piece.strip())
            if match is None:
                continue
            key = labels[match["label"].split(" (")[0]]
            items = split_top_level(match["values"])
            if all(PAIR.match(item) for item in items):
                coupled = [
                    {DECISION: [int(m["time"])], "max_input_age_minutes": int(m["age"])}
                    for item in items
                    if (m := PAIR.match(item))
                ]
            elif key == DECISION:
                params[key] = [minutes(item) for item in items]
            elif key == "lookbacks":
                age = int((lookback_age or base_age)["value"])  # type: ignore[index]
                assert lookback_age or both_points
                params[key] = [
                    [{"minutes_before_decision": number(item), "max_input_age_minutes": age}]
                    for item in items
                ]
            else:
                params[key] = [item if item.isalpha() else number(item) for item in items]
    if coupled is None:
        if DECISION not in params:
            fixed = FIXED_TIMES.search(text)
            assert fixed, "no decision times in the text"
            params[DECISION] = [minutes(fixed["value"])]
        assert base_age, "no input age in the text"
        params["max_input_age_minutes"] = [int(base_age["value"])]
    params.update(HAND_ASSERTED.get(issue, {}))
    cells = [
        dict(zip(params, values, strict=True)) for values in itertools.product(*params.values())
    ]
    if coupled is not None:
        cells = [{**cell, **pair} for cell in cells for pair in coupled]
    return cells, int(CELLS.search(text)["count"])  # type: ignore[index]


def by_content(cells: list[dict[str, object]]) -> list[str]:
    return sorted(json.dumps(cell, sort_keys=True) for cell in cells)


def load(path: Path) -> Preregistration:
    return Preregistration.model_validate(yaml.safe_load(path.read_text()))


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
    def test_cells_match_the_issue_text(self, path: Path) -> None:
        prereg = load(path)
        parsed, count = parse_cells(prereg.issue, (FIXTURES / f"{prereg.issue}.md").read_text())
        assert len(parsed) == count
        assert by_content(prereg.candidate_cells()) == by_content(parsed)

    @pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
    def test_every_cell_is_params_the_rule_takes(self, path: Path) -> None:
        prereg = load(path)
        spec = RULES.get(prereg.rule)
        for cell in prereg.candidate_cells():
            if spec is not None:
                spec.params_model.model_validate(cell)
                continue
            assert {DECISION, "max_input_age_minutes"} <= cell.keys()
            RuleParams.model_validate(
                {k: v for k, v in cell.items() if k in RuleParams.model_fields}
            )

    @pytest.mark.parametrize("path", FILES, ids=lambda p: p.stem)
    def test_names_an_existing_or_a_pending_rule(self, path: Path) -> None:
        assert load(path).rule in RULES.keys() | PENDING_RULES

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
