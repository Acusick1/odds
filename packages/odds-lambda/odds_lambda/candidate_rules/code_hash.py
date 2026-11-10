"""A hash of the first-party modules some roots import, as normalised syntax trees.

The closure starts from the roots' modules and follows every import statement anywhere in a
module (conditional, ``TYPE_CHECKING`` and function-local ones included) and every parent
package's ``__init__``, through first-party (``odds_*``) modules only. A closure module that
could load a module the closure cannot see raises, so the hash fails rather than leaving code
out: one that imports a module outside ``ALLOWED_MODULES``, reaches one through an imported
name or attribute, or has other reflective access (``REFLECTIVE_BUILTINS``, dunders outside
``ALLOWED_DUNDERS``, ``getattr`` and its kin with a computed name), or star-imports.
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
from types import ModuleType

FIRST_PARTY_PREFIX = "odds_"
# The modules outside the ``odds_*`` packages a closure module may import, exactly those the
# hashable rules' closures need. Any other import raises: an unlisted module, or a listed one's
# attribute that is an unlisted module, could reach the import machinery unseen.
ALLOWED_MODULES = frozenset(
    {
        "__future__",
        "collections",
        "collections.abc",
        "dataclasses",
        "datetime",
        "decimal",
        "enum",
        "functools",
        "math",
        # odds_core.database reads the environment.
        "os",
        "pathlib",
        "pydantic",
        "pydantic_settings",
        "sqlalchemy",
        "sqlalchemy.dialects.postgresql",
        "sqlalchemy.ext.asyncio",
        "sqlalchemy.pool",
        "sqlalchemy.types",
        "sqlmodel",
        "typing",
        "yaml",
        "zoneinfo",
    }
)
# Names a closure module may import from a module that is otherwise refused.
ALLOWED_FROM_IMPORTS: dict[str, frozenset[str]] = {"types": frozenset({"MappingProxyType"})}
# Dunder names and attributes closure code may use; any other can reach module objects, code,
# globals or docstrings at run time.
ALLOWED_DUNDERS = frozenset({"__name__", "__init__"})
# Builtins that run code or expose namespaces chosen at run time.
REFLECTIVE_BUILTINS = frozenset(
    {"__import__", "exec", "eval", "compile", "globals", "locals", "vars", "breakpoint"}
)
# Builtins safe only with a literal attribute name.
ATTRIBUTE_BUILTINS = frozenset({"getattr", "setattr", "delattr", "hasattr"})
# The narrow exceptions the current closures need.
# Declarations closure code may assign but never read: the export list (star imports are
# refused) and SQLAlchemy's table name and arguments.
DECLARED_DUNDERS = frozenset({"__all__", "__tablename__", "__table_args__"})
# Per module, dunder names it may read: the settings locate the project root from their file.
MODULE_DUNDERS: dict[str, frozenset[str]] = {"odds_core.config": frozenset({"__file__"})}


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


def _is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__") and name not in ALLOWED_DUNDERS


def _literal_attribute_calls(tree: ast.AST) -> set[int]:
    """The ids of ``getattr``-family names called with a literal, allowed attribute name."""
    allowed: set[int] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in ATTRIBUTE_BUILTINS
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
            and not _is_dunder(node.args[1].value)
        ):
            allowed.add(id(node.func))
    return allowed


def _allowed_module(module: str) -> bool:
    return is_first_party(module) or module in ALLOWED_MODULES


def _reflective(name: str, node: ast.AST, literal_calls: set[int]) -> str | None:
    """Why ``node`` in module ``name`` gives reflective access the closure cannot follow, or
    None."""
    if isinstance(node, ast.Import):
        for alias in node.names:
            if not _allowed_module(alias.name):
                return f"imports {alias.name}, not an allowed module,"
    elif isinstance(node, ast.ImportFrom):
        for alias in node.names:
            if alias.name == "*":
                return "star-imports, which loads the submodules ``__all__`` names"
            if _is_dunder(alias.name) or alias.name in REFLECTIVE_BUILTINS:
                return f"imports {alias.name}"
            if (
                node.level == 0
                and not _allowed_module(node.module or "")
                and alias.name not in ALLOWED_FROM_IMPORTS.get(node.module or "", frozenset())
            ):
                return f"imports {alias.name} from {node.module}, not an allowed module,"
    elif isinstance(node, ast.Name):
        declared = node.id in DECLARED_DUNDERS and isinstance(node.ctx, ast.Store)
        allowed = node.id in MODULE_DUNDERS.get(name, frozenset())
        if node.id in REFLECTIVE_BUILTINS or (_is_dunder(node.id) and not declared and not allowed):
            return f"names {node.id}"
        if node.id in ATTRIBUTE_BUILTINS and id(node) not in literal_calls:
            return f"uses {node.id} other than with a literal attribute name"
    elif isinstance(node, ast.Attribute) and _is_dunder(node.attr):
        return f"accesses {node.attr}"
    return None


def _import_object(module: str, attribute: str | None = None) -> object:
    """The real module, or its attribute; a submodule not yet loaded is imported."""
    loaded = importlib.import_module(module)
    if attribute is None:
        return loaded
    try:
        return getattr(loaded, attribute)
    except AttributeError:
        return importlib.import_module(f"{module}.{attribute}")


def _bindings(name: str, is_package: bool, tree: ast.Module) -> dict[str, object]:
    """The real object each name an import in ``tree`` binds refers to."""
    bound: dict[str, object] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                target = alias.name if alias.asname else alias.name.split(".")[0]
                bound[alias.asname or target] = _import_object(target)
        elif isinstance(node, ast.ImportFrom) and node.module != "__future__":
            source = _relative(name, is_package, node.level, node.module)
            for alias in node.names:
                bound[alias.asname or alias.name] = _import_object(source, alias.name)
    return bound


def _module_reached(value: object) -> str | None:
    """The name of ``value`` when it is a module closure code may not reach, or None."""
    if isinstance(value, ModuleType) and not _allowed_module(value.__name__):
        return value.__name__
    return None


def _check_bindings(name: str, is_package: bool, tree: ast.Module) -> None:
    """Raise when an imported name, or an attribute chain on one, reaches a module that is not
    allowed, or when a name bound to a module is used other than to take an attribute."""
    bound = _bindings(name, is_package, tree)
    for local, value in bound.items():
        if (module := _module_reached(value)) is not None:
            raise ValueError(
                f"{name}: {local} is the module {module}, not an allowed module, which the "
                "code hash cannot follow"
            )
    attribute_bases: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        chain: list[str] = []
        base: ast.expr = node
        while isinstance(base, ast.Attribute):
            chain.append(base.attr)
            base = base.value
        if isinstance(node.value, ast.Name):
            attribute_bases.add(id(node.value))
        if not isinstance(base, ast.Name) or base.id not in bound:
            continue
        value = bound[base.id]
        for attribute in reversed(chain):
            try:
                value = getattr(value, attribute)
            except Exception:  # noqa: BLE001 - an attribute only set at run time
                break
            if (module := _module_reached(value)) is not None:
                raise ValueError(
                    f"{name}:{node.lineno}: {base.id}.{attribute} reaches the module {module}, "
                    "not an allowed module, which the code hash cannot follow"
                )
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and isinstance(bound.get(node.id), ModuleType)
            and id(node) not in attribute_bases
        ):
            raise ValueError(
                f"{name}:{node.lineno}: the module {node.id} is used as a value, which the code "
                "hash cannot follow"
            )


def _imports(name: str, is_package: bool, tree: ast.Module) -> set[str]:
    """Each module an import in ``tree`` may load: ``import a.b`` loads ``a`` and ``a.b``, and
    ``from a import b`` loads ``a`` and, when it is one, the module ``a.b``.

    Raises ``ValueError`` for a module that could load one some other way."""
    found: set[str] = set()
    literal_calls = _literal_attribute_calls(tree)
    for node in ast.walk(tree):
        if (reason := _reflective(name, node, literal_calls)) is not None:
            raise ValueError(
                f"{name}:{getattr(node, 'lineno', 0)}: {reason}, which the code hash cannot follow"
            )
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                found.update(".".join(parts[: i + 1]) for i in range(len(parts)))
        elif isinstance(node, ast.ImportFrom):
            source = _relative(name, is_package, node.level, node.module)
            found.add(source)
            found.update(f"{source}.{alias.name}" for alias in node.names)
    _check_bindings(name, is_package, tree)
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
