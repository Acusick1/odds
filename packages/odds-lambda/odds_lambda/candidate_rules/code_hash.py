"""A hash of the first-party modules some roots import, as normalised syntax trees.

The closure starts from the roots' modules and follows every import statement anywhere in a
module (conditional, ``TYPE_CHECKING`` and function-local ones included) and every parent
package's ``__init__``, through first-party (``odds_*``) modules only. A closure module that
could load a module the closure cannot see raises, so the hash fails rather than leaving code
out: one that imports a dynamic-import mechanism, names ``__import__``, or star-imports.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache
from pathlib import Path

FIRST_PARTY_PREFIX = "odds_"
# Packages that load modules by a name chosen at run time, however the loading call is spelt.
DYNAMIC_IMPORT_PACKAGES = frozenset({"importlib", "pkg_resources", "runpy", "builtins"})
# Names that load or run code chosen at run time.
DYNAMIC_IMPORT_NAMES = frozenset({"__import__", "__builtins__", "exec", "eval", "compile"})
# Attributes of ``sys`` that reach loaded modules or the import machinery.
SYS_IMPORT_ATTRIBUTES = frozenset({"modules", "meta_path", "path_hooks"})


@dataclass(frozen=True)
class _Module:
    name: str
    # The normalised syntax tree, without docstrings.
    text: str
    # Every module an import statement in it may load.
    imports: frozenset[str]


def is_first_party(module: str) -> bool:
    return module.split(".")[0].startswith(FIRST_PARTY_PREFIX)


def _strip_docstrings(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef | ast.Module):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                # An empty body, not ``pass``, so a docstring-only module equals an empty one.
                node.body = body[1:]


def _normalised(node: object) -> str:
    """A syntax tree as text, without positions or comments.

    Fields that are None or empty are left out, so a field a later Python version adds with
    such a default, such as ``type_params``, does not change the text; ``ast.dump`` itself
    changed which fields it prints in Python 3.13.
    """
    if isinstance(node, ast.AST):
        fields = ",".join(
            f"{name}={_normalised(value)}"
            for name, value in ast.iter_fields(node)
            if value is not None and value != []
        )
        return f"{type(node).__name__}({fields})"
    if isinstance(node, list):
        return f"[{','.join(_normalised(item) for item in node)}]"
    return repr(node)


def module_source(name: str) -> tuple[str, bool] | None:
    """The source of module ``name`` and whether it is a package; None without a source file."""
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    if spec is None or spec.origin is None or not spec.origin.endswith(".py"):
        return None
    return Path(spec.origin).read_text(), spec.submodule_search_locations is not None


def _relative(name: str, is_package: bool, level: int, target: str | None) -> str:
    if level == 0:
        return target or ""
    parts = name.split(".")
    base = parts if is_package else parts[:-1]
    base = base[: len(base) - (level - 1)]
    return ".".join([*base, target] if target else base)


def _dynamic(name: str, node: ast.AST) -> str | None:
    """Why ``node`` could load a module the closure cannot see, or None."""
    if isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name.split(".")[0] in DYNAMIC_IMPORT_PACKAGES:
                return f"imports {alias.name}"
    elif isinstance(node, ast.ImportFrom):
        source = (node.module or "").split(".")[0]
        if node.level == 0 and source in DYNAMIC_IMPORT_PACKAGES:
            return f"imports from {node.module}"
        if any(alias.name == "*" for alias in node.names):
            return "star-imports, which loads the submodules ``__all__`` names"
        if any(alias.name in DYNAMIC_IMPORT_NAMES for alias in node.names):
            return "imports a dynamic-import name"
    elif isinstance(node, ast.Name) and node.id in DYNAMIC_IMPORT_NAMES:
        return f"names {node.id}"
    elif isinstance(node, ast.Attribute) and node.attr == "__import__":
        return "names __import__"
    elif (
        isinstance(node, ast.Attribute)
        and node.attr in SYS_IMPORT_ATTRIBUTES
        and isinstance(node.value, ast.Name)
        and node.value.id == "sys"
    ):
        return f"names sys.{node.attr}"
    return None


def _imports(name: str, is_package: bool, tree: ast.Module) -> set[str]:
    """Each module an import in ``tree`` may load: ``import a.b`` loads ``a`` and ``a.b``, and
    ``from a import b`` loads ``a`` and, when it is one, the module ``a.b``.

    Raises ``ValueError`` for a module that could load one some other way."""
    found: set[str] = set()
    for node in ast.walk(tree):
        if (reason := _dynamic(name, node)) is not None:
            raise ValueError(
                f"{name}:{getattr(node, 'lineno', 0)}: {reason}, which the code hash cannot "
                "follow; import modules statically"
            )
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                found.update(".".join(parts[: i + 1]) for i in range(len(parts)))
        elif isinstance(node, ast.ImportFrom):
            source = _relative(name, is_package, node.level, node.module)
            found.add(source)
            found.update(f"{source}.{alias.name}" for alias in node.names)
    return found


@cache
def _parse(name: str) -> _Module | None:
    """The parsed module ``name``; None when it has no source."""
    if (found := module_source(name)) is None:
        return None
    source, is_package = found
    tree = ast.parse(source)
    imports = _imports(name, is_package, tree)
    _strip_docstrings(tree)
    return _Module(name, _normalised(tree), frozenset(imports))


def clear_cache() -> None:
    """Forget parsed modules, so the next hash rereads their source."""
    _parse.cache_clear()


def root_module(root: object | str) -> str:
    """The module of a function or class, or of a ``module.name`` path."""
    if isinstance(root, str):
        return root.rpartition(".")[0]
    module = inspect.getmodule(root)
    if module is None:
        raise ValueError(f"{root!r} has no module")
    return module.__name__


def _parents(name: str) -> list[str]:
    parts = name.split(".")
    return [".".join(parts[:i]) for i in range(1, len(parts))]


def closure(roots: Iterable[object | str]) -> list[str]:
    """The sorted names of the roots' modules and every first-party module they import, with
    each one's parent packages.

    A root's own module is included whatever its package, so a rule defined elsewhere, such as
    in a test, is hashed too. A root module without a source raises.
    """
    pending = [root_module(root) for root in roots]
    for name in pending:
        if _parse(name) is None:
            raise ValueError(f"root module {name} has no source")
    found: set[str] = set()
    while pending:
        name = pending.pop()
        if name in found:
            continue
        module = _parse(name)
        if module is None:
            # A name ``from a import b`` binds that is not a module.
            continue
        found.add(name)
        pending += [p for p in _parents(name) if is_first_party(p)]
        pending += [m for m in module.imports if is_first_party(m)]
    return sorted(found)


def code_sha256(roots: Iterable[object | str], extra: Iterable[str] = ()) -> str:
    """sha256 of each closure module's name and normalised syntax tree, in module-name order,
    then ``extra``."""
    digest = hashlib.sha256()
    for name in closure(roots):
        module = _parse(name)
        assert module is not None  # noqa: S101
        for text in (name, module.text):
            digest.update(text.encode())
            digest.update(b"\0")
    for text in extra:
        digest.update(text.encode())
        digest.update(b"\0")
    return digest.hexdigest()
