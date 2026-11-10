"""Unit tests for the import-closure code hash."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from odds_lambda.candidate_rules.code_hash import closure, code_sha256

from tests.unit.conftest import FIXTURE_PACKAGE

Write = Callable[[dict[str, str]], None]
ROOT = f"{FIXTURE_PACKAGE}.rule.decide"
P = FIXTURE_PACKAGE

RULE = """
from .helpers import LIMIT


def decide(x: float) -> bool:
    \"\"\"Whether x is over the limit.\"\"\"
    return x > LIMIT  # compare
"""
HELPERS = "LIMIT = 2\n\n\ndef unused() -> int:\n    return 1\n"
BOUNDS = "def low(): return 1\ndef high(): return 10\n"
REGISTRY = "REGISTRY={}\ndef register(fn):\n REGISTRY[fn.__name__]=fn\n return fn\n"
PLUGIN = "from .helpers import register\n@register\ndef bet(): return 1\n"
TRY_IMPORT = (
    "try:\n from .bounds import low as choice\nexcept ImportError:\n"
    " from .bounds import high as choice\ndef decide(): return choice()\n"
)
TYPE_CHECKING_IMPORT = (
    "from typing import TYPE_CHECKING\nfrom .bounds import low as choice\n"
    "if TYPE_CHECKING:\n from .bounds import high as choice\ndef decide(): return choice()\n"
)
PYDANTIC = (
    "from pydantic import BaseModel, field_validator\nLIMIT=1\nclass Params(BaseModel):\n"
    " stake:int=LIMIT\n @field_validator('stake')\n @classmethod\n"
    " def val(cls,v): return v+LIMIT\n"
)


def _helper_case(rule: str, helper: str) -> tuple[dict[str, str], dict[str, str]]:
    edited = helper.replace("=1", "=2").replace("return 1", "return 2")
    return {"rule": rule, "helpers": helper}, {"rule": rule, "helpers": edited}


# Each pair of package sources gives the root a different result; every one must change the
# hash. The first group were holes in a reachability hash; the second are common constructs.
BEHAVIOUR_CHANGES: dict[str, tuple[dict[str, str], dict[str, str]]] = {
    "import execution order": (
        {
            "bounds": BOUNDS,
            "rule": "from .bounds import low as choice\ndef decide(fn=choice): return fn()\n"
            "from .bounds import high as choice\n",
        },
        {
            "bounds": BOUNDS,
            "rule": "from .bounds import low as choice\nfrom .bounds import high as choice\n"
            "def decide(fn=choice): return fn()\n",
        },
    ),
    "conditional import": (
        {
            "bounds": BOUNDS,
            "rule": "if True:\n from .bounds import low as choice\nelse:\n"
            " from .bounds import high as choice\ndef decide(): return choice()\n",
        },
        {
            "bounds": BOUNDS,
            "rule": "if False:\n from .bounds import low as choice\nelse:\n"
            " from .bounds import high as choice\ndef decide(): return choice()\n",
        },
    ),
    "package init": (
        {
            "rule": "from .helpers import LIMIT\ndef decide(): return LIMIT\n",
            "helpers": "LIMIT=1\n",
            "__init__": "from . import helpers\nhelpers.LIMIT=2\n",
        },
        {
            "rule": "from .helpers import LIMIT\ndef decide(): return LIMIT\n",
            "helpers": "LIMIT=1\n",
            "__init__": "from . import helpers\nhelpers.LIMIT=3\n",
        },
    ),
    "cross-module decorator registry": (
        {
            "rule": 'from . import plugin\nfrom .helpers import REGISTRY\ndef decide(): return REGISTRY["bet"]()\n',
            "helpers": REGISTRY,
            "plugin": PLUGIN,
        },
        {
            "rule": 'from . import plugin\nfrom .helpers import REGISTRY\ndef decide(): return REGISTRY["bet"]()\n',
            "helpers": REGISTRY,
            "plugin": PLUGIN.replace("return 1", "return 2"),
        },
    ),
    "literal import_module": (
        {
            "rule": f'import importlib\ndef decide(): return importlib.import_module("{P}.helpers").bet()\n',
            "helpers": "def bet(): return 1\n",
        },
        {
            "rule": f'import importlib\ndef decide(): return importlib.import_module("{P}.helpers").bet()\n',
            "helpers": "def bet(): return 2\n",
        },
    ),
    "swapped module implementations": (
        {
            "rule": "from . import left, right\ndef decide(): return left.bet() - right.bet()\n",
            "left": "def bet(): return 1\n",
            "right": "def bet(): return 10\n",
        },
        {
            "rule": "from . import left, right\ndef decide(): return left.bet() - right.bet()\n",
            "left": "def bet(): return 10\n",
            "right": "def bet(): return 1\n",
        },
    ),
    "try primary only": (
        {"bounds": BOUNDS, "rule": TRY_IMPORT},
        {"bounds": BOUNDS.replace("return 1\n", "return 2\n"), "rule": TRY_IMPORT},
    ),
    "TYPE_CHECKING shadows runtime": (
        {"bounds": BOUNDS, "rule": TYPE_CHECKING_IMPORT},
        {"bounds": BOUNDS.replace("return 1\n", "return 2\n"), "rule": TYPE_CHECKING_IMPORT},
    ),
    "mutated import binding swap": (
        {
            "bounds": BOUNDS,
            "rule": "from .bounds import low as a, high as b\na.tag=1\nb.tag=1\n"
            "def decide(): return a()-b()\n",
        },
        {
            "bounds": BOUNDS,
            "rule": "from .bounds import high as a, low as b\na.tag=1\nb.tag=1\n"
            "def decide(): return a()-b()\n",
        },
    ),
    "default constant": _helper_case(
        "from .helpers import LIMIT\ndef decide(x=LIMIT): return x\n", "LIMIT=1\n"
    ),
    "inheritance": _helper_case(
        "from .helpers import Child\ndef decide(): return Child().bet()\n",
        "class Base:\n def bet(self): return 1\nclass Child(Base): pass\n",
    ),
    "partial": _helper_case(
        "from .helpers import BET\ndef decide(): return BET()\n",
        "from functools import partial\ndef bet(x): return x\nLIMIT=1\nBET=partial(bet,LIMIT)\n",
    ),
    "lambda": _helper_case(
        "from .helpers import BET\ndef decide(): return BET()\n", "LIMIT=1\nBET=lambda: LIMIT\n"
    ),
    "dict dispatch": _helper_case(
        'from .helpers import TABLE\ndef decide(): return TABLE["bet"]()\n',
        'def bet(): return 1\nTABLE={"bet":bet}\n',
    ),
    "dataclass": _helper_case(
        "from .helpers import Params\ndef decide(): return Params().stake\n",
        "from dataclasses import dataclass\nLIMIT=1\n@dataclass\nclass Params:\n stake:int=LIMIT\n",
    ),
    "enum": _helper_case(
        "from .helpers import Limit\ndef decide(): return Limit.BET.value\n",
        "from enum import Enum\nclass Limit(Enum):\n BET=1\n",
    ),
    "pydantic": _helper_case(
        "from .helpers import Params\ndef decide(): return Params().stake\n", PYDANTIC
    ),
    "pydantic validator": _helper_case(
        "from .helpers import Params\ndef decide(): return Params(stake=1).stake\n",
        PYDANTIC.replace("stake:int=LIMIT", "stake:int=0"),
    ),
    "class attribute": _helper_case(
        "from .helpers import Base\ndef decide(): return Base.LIMIT\n", "class Base:\n LIMIT=1\n"
    ),
}


def package(files: dict[str, str]) -> dict[str, str]:
    return {"__init__": "", **files}


class TestCodeSha256:
    @pytest.mark.parametrize("case", list(BEHAVIOUR_CHANGES))
    def test_a_change_in_behaviour_changes_it(self, fixture_package: Write, case: str) -> None:
        before, after = BEHAVIOUR_CHANGES[case]
        fixture_package(package(before))
        old = code_sha256([ROOT])
        fixture_package(package(after))
        assert code_sha256([ROOT]) != old

    @pytest.mark.parametrize(
        ("module", "old", "new"),
        [
            ("rule", "Whether x is over the limit.", "Whether x exceeds the limit."),
            ("rule", "# compare", "# compare with the limit"),
            ("rule", "return x > LIMIT", "return (\n        x > LIMIT\n    )"),
        ],
        ids=["docstring", "comment", "formatting"],
    )
    def test_a_docstring_comment_or_format_edit_leaves_it_unchanged(
        self, fixture_package: Write, module: str, old: str, new: str
    ) -> None:
        sources = {"rule": RULE, "helpers": HELPERS}
        fixture_package(package(sources))
        before = code_sha256([ROOT])
        fixture_package({module: sources[module].replace(old, new)})
        assert code_sha256([ROOT]) == before

    @pytest.mark.parametrize(
        ("module", "old", "new"),
        [
            ("helpers", "    return 1", "    return 5"),
            ("rule", "x > LIMIT", "x >= LIMIT"),
            ("__init__", "", "VERSION = 1\n"),
        ],
        ids=["unused function", "rule", "package init"],
    )
    def test_any_code_edit_in_a_closure_module_changes_it(
        self, fixture_package: Write, module: str, old: str, new: str
    ) -> None:
        sources = package({"rule": RULE, "helpers": HELPERS})
        fixture_package(sources)
        before = code_sha256([ROOT])
        fixture_package({module: sources[module].replace(old, new) if old else new})
        assert code_sha256([ROOT]) != before

    def test_modules_are_keyed_by_name(self, fixture_package: Write) -> None:
        fixture_package(package({"rule": RULE, "helpers": HELPERS}))
        before = code_sha256([ROOT])
        fixture_package({"helpers": "from .limits import LIMIT\n", "limits": HELPERS})
        moved = code_sha256([ROOT])
        fixture_package({"rule": RULE.replace(".helpers", ".limits")})
        assert len({before, moved, code_sha256([ROOT])}) == 3

    def test_extra_values_change_it(self, fixture_package: Write) -> None:
        fixture_package(package({"rule": RULE, "helpers": HELPERS}))
        assert code_sha256([ROOT], ["sharp_inputs=[]"]) != code_sha256([ROOT])


class TestClosure:
    def test_holds_the_roots_modules_their_imports_and_parent_packages(
        self, fixture_package: Write
    ) -> None:
        fixture_package(package({"rule": RULE, "helpers": HELPERS, "unrelated": HELPERS}))
        assert closure([ROOT]) == [P, f"{P}.helpers", f"{P}.rule"]

    def test_follows_conditional_and_function_local_imports(self, fixture_package: Write) -> None:
        rule = (
            "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n from . import a\n"
            "try:\n from . import b\nexcept ImportError:\n from . import c\n"
            "def decide():\n from . import d\n return d\n"
        )
        fixture_package(package({"rule": rule, **dict.fromkeys("abcd", "")}))
        assert closure([ROOT]) == [P, *(f"{P}.{m}" for m in "abcd"), f"{P}.rule"]

    def test_does_not_follow_third_party_imports(self, fixture_package: Write) -> None:
        fixture_package(package({"rule": "import pytest\ndef decide(): return pytest\n"}))
        assert closure([ROOT]) == [P, f"{P}.rule"]

    @pytest.mark.parametrize(
        "call",
        [
            "importlib.import_module(name)",
            f'importlib.import_module(".helpers", "{P}")',
            f'__import__("{P}.helpers")',
        ],
    )
    def test_a_dynamic_import_it_cannot_follow_fails(
        self, fixture_package: Write, call: str
    ) -> None:
        rule = f"import importlib\ndef decide(name):\n return {call}\n"
        fixture_package(package({"rule": rule, "helpers": HELPERS}))
        with pytest.raises(ValueError, match="dynamic import"):
            closure([ROOT])

    def test_a_root_without_a_source_fails(self) -> None:
        with pytest.raises(ValueError, match="no source"):
            closure([f"{P}_missing.rule.decide"])
