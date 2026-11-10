"""Unit tests for the import-closure code hash."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

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

    def test_an_empty_and_a_docstring_only_init_hash_alike(self, fixture_package: Write) -> None:
        fixture_package(package({"rule": RULE, "helpers": HELPERS}))
        before = code_sha256([ROOT])
        fixture_package({"__init__": '"""Package docs."""\n'})
        assert code_sha256([ROOT]) == before

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
            "def decide():\n from . import d\n return d.value\n"
        )
        fixture_package(package({"rule": rule, **dict.fromkeys("abcd", "")}))
        assert closure([ROOT]) == [P, *(f"{P}.{m}" for m in "abcd"), f"{P}.rule"]

    def test_does_not_follow_third_party_imports(self, fixture_package: Write) -> None:
        fixture_package(package({"rule": "import yaml\ndef decide(): return yaml.safe_load\n"}))
        assert closure([ROOT]) == [P, f"{P}.rule"]

    @pytest.mark.parametrize(
        "rule",
        [
            'import importlib\ndef decide(): return importlib.import_module("{P}.helpers")\n',
            'from importlib import import_module as load\ndef decide(): return load("{P}.helpers")\n',
            'from builtins import __import__ as load\ndef decide(): return load("{P}.helpers")\n',
            'def decide(): return __import__("{P}.helpers")\n',
            "from importlib.metadata import EntryPoint\n"
            'def decide(): return EntryPoint(name="b", value="{P}.helpers:bet", group="g").load()\n',
            "import importlib.util\ndef decide(): return importlib.util\n",
            'import runpy\ndef decide(): return runpy.run_module("{P}.helpers")\n',
            "import pkg_resources\ndef decide(): return pkg_resources\n",
            'import sys\ndef decide(): return sys.modules["{P}.helpers"]\n',
            'def decide(): return eval("1")\n',
            "from . import *\ndef decide(): return helpers.bet()\n",
            'import inspect\ndef decide(): return inspect.importlib.import_module("{P}.helpers")\n',
            'def decide():\n    "1"\n    return int(decide.__doc__)\n',
            "import sys as s\ndef decide(): return s.modules\n",
            "from sys import modules\ndef decide(): return modules\n",
            'import sys\ndef decide(): return getattr(sys, "modules")\n',
            'import sys\nsys.path.insert(0, "/tmp/other")\ndef decide(): return 1\n',
            "import gc\ndef decide(): return gc\n",
            "import ctypes\ndef decide(): return ctypes\n",
            "import types\ndef decide(): return types\n",
            "import site\ndef decide(): return site\n",
            "def decide(x, name): return getattr(x, name)\n",
            'def decide(x, v): setattr(x, "a" + "b", v)\n',
            "def decide(): return globals()\n",
            "def decide(): return vars()\n",
            "def decide(x): return x.__class__.__subclasses__()\n",
            "def decide(f): return f.__globals__\n",
            'def decide(x): return getattr(x, "__dict__")\n',
            "def decide(): return __loader__\n",
            "lookup = getattr\ndef decide(x): return lookup(x, 'a')\n",
            "def decide(): return __all__\n",
            'from os import sys as s\ndef decide(): return s.modules["importlib"]\n',
            "import os\ndef decide(): return os.sys\n",
            "from os import path\ndef decide(): return path\n",
            "from yaml import reader\ndef decide(): return reader\n",
            "import json\ndef decide(): return json\n",
            "from types import SimpleNamespace\ndef decide(): return SimpleNamespace\n",
            "from types import ModuleType as MappingProxyType\ndef decide(): return 1\n",
            "import math\nm = math\ndef decide(): return m.pi\n",
        ],
        ids=[
            "import_module",
            "aliased import_module",
            "aliased __import__",
            "__import__",
            "entry point",
            "importlib submodule",
            "runpy",
            "pkg_resources",
            "sys.modules",
            "eval",
            "star import",
            "inspect.importlib",
            "docstring as a value",
            "aliased sys",
            "from sys import modules",
            "getattr on sys",
            "sys.path",
            "gc",
            "ctypes",
            "types",
            "site",
            "getattr with a computed name",
            "setattr with a computed name",
            "globals",
            "vars",
            "__class__",
            "__globals__",
            "getattr of a dunder",
            "__loader__",
            "aliased getattr",
            "reading __all__",
            "from os import sys",
            "os.sys",
            "from os import a module not on the list",
            "from an allowed module, a module not on the list",
            "a module not on the list",
            "types extra symbol",
            "types other name aliased",
            "a module as a value",
        ],
    )
    def test_a_module_that_could_load_code_unseen_fails(
        self, fixture_package: Write, rule: str
    ) -> None:
        fixture_package(package({"rule": rule.replace("{P}", P), "helpers": HELPERS}))
        with pytest.raises(ValueError, match="cannot follow"):
            closure([ROOT])

    @pytest.mark.parametrize(
        "rule",
        [
            "from types import MappingProxyType\ndef decide(): return MappingProxyType({})\n",
            '__all__ = ["decide"]\ndef decide(): return 1\n',
            'class Row:\n    __tablename__ = "rows"\n    __table_args__ = ()\n'
            "def decide(): return Row\n",
            'def decide(x): return getattr(x, "a", None), hasattr(x, "b")\n',
            "class A:\n    def __init__(self): self.n = __name__\ndef decide(): return A()\n",
        ],
        ids=[
            "MappingProxyType",
            "__all__",
            "SQLAlchemy table",
            "literal getattr",
            "__init__ and __name__",
        ],
    )
    def test_the_allowances_pass(self, fixture_package: Write, rule: str) -> None:
        fixture_package(package({"rule": rule}))
        assert closure([ROOT]) == [P, f"{P}.rule"]

    def test_a_first_party_re_export_cannot_reach_a_refused_module(
        self, fixture_package: Write
    ) -> None:
        rule = "from .helpers import os as o\ndef decide(): return o.sys.modules\n"
        fixture_package(package({"rule": rule, "helpers": "import os\n"}))
        with pytest.raises(ValueError, match="sys"):
            closure([ROOT])

    def test_a_dynamic_import_in_any_closure_module_fails(self, fixture_package: Write) -> None:
        fixture_package(package({"rule": RULE, "helpers": HELPERS + "import importlib\n"}))
        with pytest.raises(ValueError, match="helpers"):
            closure([ROOT])

    def test_a_root_without_a_source_fails(self) -> None:
        with pytest.raises(ValueError, match="no source"):
            closure([f"{P}_missing.rule.decide"])


class TestOutsideTheHash:
    """What a rule reads at run time, other than its imports, is not hashed: rule code must not
    take behaviour from files, the environment or settings."""

    def test_a_data_file_is_not_hashed(self, fixture_package: Write, tmp_path: Path) -> None:
        config = tmp_path / "c.json"
        rule = f"from pathlib import Path\ndef decide(): return Path({str(config)!r}).read_text()\n"
        fixture_package(package({"rule": rule}))
        config.write_text('{"stake": 1}')
        before = code_sha256([ROOT])
        config.write_text('{"stake": 2}')
        assert code_sha256([ROOT]) == before

    def test_the_environment_is_not_hashed(
        self, fixture_package: Write, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rule = 'import os\ndef decide(): return int(os.environ["STAKE"])\n'
        fixture_package(package({"rule": rule}))
        monkeypatch.setenv("STAKE", "1")
        before = code_sha256([ROOT])
        monkeypatch.setenv("STAKE", "2")
        assert code_sha256([ROOT]) == before
