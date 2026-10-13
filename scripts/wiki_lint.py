"""Lint the agent wiki and generate wiki/backlinks.json (spec: wiki/SCHEMA.md)."""

from __future__ import annotations

import argparse
import ast
import builtins
import json
import keyword
import logging
import re
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from pathlib import Path
from typing import NamedTuple, TypeVar

import griffe
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

REPO_ROOT = Path(__file__).resolve().parent.parent
PAGES_START = "<!-- pages:start -->"
PAGES_END = "<!-- pages:end -->"
PAGES_BLOCK_RE = re.compile(re.escape(PAGES_START) + r"(.*?)" + re.escape(PAGES_END), re.DOTALL)
DOCS_ALLOWED = ("docs/adr/", "docs/agents/")
DEFAULT_MODULE_DIRS = (
    "packages/odds-lambda/odds_lambda/jobs",
    "packages/odds-cli/odds_cli/commands",
)
DEFAULT_MCP_DIR = "packages/odds-mcp/odds_mcp"
DEFAULT_PACKAGES_DIR = "packages"
DEFAULT_SYMBOL_ROOT = DEFAULT_PACKAGES_DIR
ALLOWED_WORDS = frozenset({"main", "master"})
SKIP_DIRS = {"__pycache__", ".venv", "node_modules", ".git"}
COVERAGE_KEY = "coverage"

FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n?(.*)\Z", re.DOTALL)
FENCE_RE = re.compile(r"^(```|~~~).*?^\1[ \t]*$", re.DOTALL | re.MULTILINE)
LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)\)")
WIKILINK_RE = re.compile(r"\[\[[^\]]+\]\]")
INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
PATH_RE = re.compile(r"^([\w.\-/]+?)(?:#[\w\-]+)?$")
PATH_SYMBOL_RE = re.compile(r"^([\w.\-/]+?\.\w+)::([\w.]+)$")
DOTTED_RE = re.compile(r"^([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)(?:\(\))?$")
BARE_RE = re.compile(r"^([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?)(?:\(\))?$")
CONSTANT_RE = re.compile(r"^[A-Z][A-Z0-9_]+$")
HAS_EXT_RE = re.compile(r"\.[A-Za-z]\w*$")
HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", re.MULTILINE)

FunctionOrClass = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
M = TypeVar("M", bound=BaseModel)


class PageType(StrEnum):
    # Declaration order is the group order of the generated area index.
    REFERENCE = "reference"
    EXPLANATION = "explanation"
    HOWTO = "howto"

    @property
    def heading(self) -> str:
        return TYPE_HEADINGS[self]


TYPE_HEADINGS = {
    PageType.REFERENCE: "Reference",
    PageType.EXPLANATION: "Explanation",
    PageType.HOWTO: "How-to guides",
}


class PageStatus(StrEnum):
    DRAFT = "draft"
    STABLE = "stable"
    DEPRECATED = "deprecated"


class PageMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: PageType
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    sources: list[str]
    status: PageStatus
    superseded_by: str | None = None


class IndexMeta(BaseModel):
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)


class Kind(Enum):
    MISSING_FRONTMATTER = "missing frontmatter"
    INVALID_FRONTMATTER = "invalid frontmatter"
    MISSING_KEY = "missing key"
    OBSOLETE_KEY = "obsolete key"
    UNKNOWN_KEY = "unknown key"
    INVALID_TYPE = "invalid type"
    INVALID_STATUS = "invalid status"
    INVALID_FIELD = "invalid field"
    SUPERSEDED_MISSING = "superseded_by missing"
    SUPERSEDED_UNKNOWN = "superseded_by unknown"
    SUPERSEDED_NOT_DEPRECATED = "superseded_by on non-deprecated page"
    DUPLICATE_SLUG = "duplicate slug"
    SOURCE_NOT_CITATION = "source is not a citation"
    DOCS_CITATION = "docs citation"
    DOCS_LINK = "docs link"
    WIKILINK = "wikilink"
    PATH_MISSING = "path missing"
    PATH_DELETED = "path deleted"
    PATH_RENAMED = "path renamed"
    QUALIFY_PATH = "qualify path"
    PYTHON_PATH = "python cited by path"
    PATH_SYMBOL = "path::symbol syntax"
    SYMBOL_UNDEFINED = "symbol undefined"
    UNQUALIFIED_SYMBOL = "unqualified symbol"
    AMBIGUOUS_SYMBOL = "ambiguous symbol"
    ORPHAN = "orphan"
    INDEX_MISSING = "index missing"
    AREA_NOT_LINKED = "area not linked"
    INDEX_MARKERS = "index markers missing"
    INDEX_STALE = "index list stale"
    COVERAGE_MODULE = "uncovered module"
    COVERAGE_MCP_TOOL = "uncovered MCP tool"
    STALE = "stale"


COVERAGE_KINDS = {Kind.COVERAGE_MODULE, Kind.COVERAGE_MCP_TOOL}


@dataclass(frozen=True)
class Finding:
    slug: str
    kind: Kind
    message: str
    lines: tuple[int, ...] = ()


class Citation(NamedTuple):
    """A cited target: `label` is what the page wrote, `path`/`symbol` locate it in the repo."""

    label: str
    path: str
    symbol: str | None


@dataclass
class Page:
    slug: str
    area: str
    path: Path
    rel: str
    text: str
    body_offset: int
    meta: PageMeta | None
    citations: dict[Citation, set[int]] = field(default_factory=dict)

    @property
    def body(self) -> str:
        return body_from(self.text, self.body_offset)


def body_from(text: str, offset: int) -> str:
    return "".join(text.splitlines(keepends=True)[offset:])


def mask_fences(text: str) -> str:
    """Blank fenced code blocks, keeping line numbers intact."""
    return FENCE_RE.sub(lambda m: "\n" * m.group(0).count("\n"), text)


def docs_citation_forbidden(path: str) -> bool:
    return path.startswith("docs/") and not path.startswith(DOCS_ALLOWED)


def _is_tool_decorator(node: ast.expr) -> bool:
    target = node.func if isinstance(node, ast.Call) else node
    return isinstance(target, ast.Attribute) and target.attr == "tool"


def _target_names(target: ast.expr) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, ast.Tuple | ast.List):
        return [name for element in target.elts for name in _target_names(element)]
    return []


def bound_names(node: ast.stmt) -> list[str]:
    if isinstance(node, FunctionOrClass):
        return [node.name]
    if isinstance(node, ast.Assign):
        return [name for target in node.targets for name in _target_names(target)]
    if isinstance(node, ast.AnnAssign):
        return _target_names(node.target)
    return []


def resolve_symbol(tree: ast.Module, symbol: str) -> ast.stmt | None:
    """Resolve `A.B` as A at module level, then B in A's class body."""
    scope = tree.body
    parts = symbol.split(".")
    node: ast.stmt | None = None
    for index, part in enumerate(parts):
        node = next((n for n in scope if part in bound_names(n)), None)
        if node is None:
            return None
        if index < len(parts) - 1:
            if not isinstance(node, ast.ClassDef):
                return None
            scope = node.body
    return node


@dataclass(frozen=True)
class Span:
    start: int
    end: int

    def overlaps(self, start: int, count: int) -> bool:
        return count > 0 and start <= self.end and start + count - 1 >= self.start


@dataclass(frozen=True)
class WholeFile:
    """Extent of a symbol in a file that does not parse: any change counts."""

    def overlaps(self, start: int, count: int) -> bool:
        return count > 0


Extent = Span | WholeFile


def _span(node: ast.stmt) -> Span:
    decorators = node.decorator_list if isinstance(node, FunctionOrClass) else []
    start = min([node.lineno, *(d.lineno for d in decorators)])
    return Span(start, node.end_lineno or node.lineno)


@dataclass(frozen=True)
class Source:
    """One version of a Python file; tree is None when it does not parse."""

    tree: ast.Module | None

    @classmethod
    def parse(cls, text: str) -> Source:
        try:
            return cls(ast.parse(text))
        except SyntaxError:
            return cls(None)

    def extent(self, symbol: str) -> Extent | None:
        """None when the symbol is not defined."""
        if self.tree is None:
            return WholeFile()
        node = resolve_symbol(self.tree, symbol)
        return None if node is None else _span(node)


@dataclass
class FileChange:
    old: Source | None
    new: Source | None
    old_hunks: list[tuple[int, int]]
    new_hunks: list[tuple[int, int]]

    def touches(self, symbol: str | None) -> bool:
        if symbol is None:
            return bool(self.old_hunks)
        old = self.old.extent(symbol) if self.old else None
        new = self.new.extent(symbol) if self.new else None
        if old is None and new is None:
            return False
        if old is None or new is None:
            return True
        return any(old.overlaps(*h) for h in self.old_hunks) or any(
            new.overlaps(*h) for h in self.new_hunks
        )


@dataclass(frozen=True)
class Changes:
    commits: tuple[str, ...]
    working_tree: bool

    def __bool__(self) -> bool:
        return bool(self.commits) or self.working_tree

    def describe(self) -> str:
        return ", ".join([*self.commits, *(["working tree"] if self.working_tree else [])])


class GitHistory:
    """Git-backed page and citation history; every query is cached per commit."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._last_commits: dict[str, str | None] = {}
        self._tracked: dict[str, frozenset[str]] = {}
        self._renames: dict[str, dict[str, str]] = {}
        self._path_commits: dict[tuple[str, str], list[str]] = {}
        self._changes: dict[tuple[str, str | None], FileChange | None] = {}
        self._has_head: bool | None = None

    def _git(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(self.root), *args], capture_output=True, text=True, check=False
        )

    def is_modified(self, rel: str) -> bool:
        return bool(self._git("status", "--porcelain", "--", rel).stdout.strip())

    def last_commit(self, rel: str) -> str | None:
        if rel not in self._last_commits:
            result = self._git("log", "-1", "--format=%H", "--", rel)
            self._last_commits[rel] = result.stdout.strip() or None
        return self._last_commits[rel]

    def tracked_at(self, rev: str | None) -> frozenset[str]:
        if rev is None:
            return frozenset()
        if rev not in self._tracked:
            result = self._git("ls-tree", "-r", "--name-only", rev)
            self._tracked[rev] = frozenset(result.stdout.splitlines())
        return self._tracked[rev]

    def renamed_to(self, rev: str, path: str) -> str | None:
        if rev not in self._renames:
            result = self._git("diff", "-M", "--name-status", rev, "HEAD")
            renames: dict[str, str] = {}
            for line in result.stdout.splitlines():
                fields = line.split("\t")
                if fields[0].startswith("R") and len(fields) == 3:
                    renames[fields[1]] = fields[2]
            self._renames[rev] = renames
        return self._renames[rev].get(path)

    def changes_since(self, base: str | None, path: str, symbol: str | None) -> Changes:
        """Commits after `base` (and uncommitted edits) that touch the symbol, or the whole file."""
        commits = [] if base is None else self._commits_after(base, path)
        touching = tuple(
            commit[:7]
            for commit in commits
            if symbol is None or self._touches(path, symbol, f"{commit}^1", commit)
        )
        return Changes(touching, self._head_exists() and self._touches(path, symbol, "HEAD", None))

    def _head_exists(self) -> bool:
        if self._has_head is None:
            self._has_head = self._git("rev-parse", "--verify", "--quiet", "HEAD").returncode == 0
        return self._has_head

    def _commits_after(self, base: str, path: str) -> list[str]:
        key = (base, path)
        if key not in self._path_commits:
            result = self._git("log", "--format=%H", f"{base}..HEAD", "--", path)
            self._path_commits[key] = result.stdout.split() if result.returncode == 0 else []
        return self._path_commits[key]

    def _show(self, rev: str, path: str) -> Source | None:
        result = self._git("show", f"{rev}:{path}")
        return Source.parse(result.stdout) if result.returncode == 0 else None

    def _touches(self, path: str, symbol: str | None, old_rev: str, new_rev: str | None) -> bool:
        change = self._file_change(path, old_rev, new_rev)
        return change is not None and change.touches(symbol)

    def _file_change(self, path: str, old_rev: str, new_rev: str | None) -> FileChange | None:
        """new_rev None compares against the working tree."""
        key = (path, new_rev)
        if key not in self._changes:
            self._changes[key] = self._compute_change(path, old_rev, new_rev)
        return self._changes[key]

    def _compute_change(self, path: str, old_rev: str, new_rev: str | None) -> FileChange | None:
        old = self._show(old_rev, path)
        if new_rev is None:
            if old is None:
                return None
            target = self.root / path
            new = Source.parse(target.read_text(errors="replace")) if target.is_file() else None
            diff = self._git("diff", "-U0", "--no-color", old_rev, "--", path).stdout
        else:
            new = self._show(new_rev, path)
            diff = self._git("diff", "-U0", "--no-color", old_rev, new_rev, "--", path).stdout
        old_hunks: list[tuple[int, int]] = []
        new_hunks: list[tuple[int, int]] = []
        for m in HUNK_RE.finditer(diff):
            old_hunks.append((int(m.group(1)), int(m.group(2) or 1)))
            new_hunks.append((int(m.group(3)), int(m.group(4) or 1)))
        return FileChange(old, new, old_hunks, new_hunks)


@dataclass
class SymbolIndex:
    """Top-level names and `Class.member` names across the packages, with dotted paths."""

    names: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    members: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))

    def lookup(self, token: str) -> list[str]:
        return self.members.get(token) or self.names.get(token) or []


class Target(NamedTuple):
    """Where a dotted citation lives: a repo file, and a symbol path within it (None: whole file)."""

    path: str
    symbol: str | None


@dataclass
class Packages:
    """Installed Python packages, loaded statically with griffe (nothing is imported)."""

    root: Path
    search_roots: tuple[Path, ...]
    names: dict[str, Path] = field(default_factory=dict)
    _collection: griffe.ModulesCollection | None = None
    _index: SymbolIndex | None = None

    @classmethod
    def discover(cls, root: Path, package_roots: Sequence[str] | None) -> Packages:
        """Package roots hold importable packages: dirs with an `__init__.py` directly inside."""
        if package_roots is None:
            base = root / DEFAULT_PACKAGES_DIR
            search = tuple(sorted(d for d in base.glob("*") if d.is_dir())) if base.is_dir() else ()
        else:
            search = tuple(root / r for r in package_roots)
        packages = cls(root, search)
        for search_root in search:
            for init in sorted(search_root.glob("*/__init__.py")):
                if not SKIP_DIRS & set(init.parts):
                    packages.names.setdefault(init.parent.name, search_root)
        return packages

    def module_name(self, rel: str) -> str | None:
        """Dotted module of a repo-relative `.py` path inside a package, else None."""
        path = self.root / rel
        for name, search_root in self.names.items():
            package_dir = search_root / name
            if path.suffix == ".py" and path.is_relative_to(package_dir):
                parts = [name, *path.relative_to(package_dir).with_suffix("").parts]
                return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        return None

    def _modules(self) -> griffe.ModulesCollection:
        if self._collection is None:
            logging.getLogger("griffe").setLevel(logging.CRITICAL)
            loader = griffe.GriffeLoader(
                search_paths=list(self.search_roots), allow_inspection=False
            )
            for name in self.names:
                loader.load(name)
            loader.resolve_aliases(implicit=False, external=False)
            self._collection = loader.modules_collection
        return self._collection

    def _final(self, obj: griffe.Object | griffe.Alias) -> griffe.Object | None:
        if not isinstance(obj, griffe.Alias):
            return obj
        try:
            return obj.final_target
        except griffe.AliasResolutionError:
            return None

    def resolve(self, dotted: str) -> Target | None:
        """Aliases (re-exports) resolve to their definition; unresolved imports do not resolve."""
        if dotted.split(".")[0] not in self.names:
            return None
        try:
            found = self._modules()[dotted]
        except KeyError:
            return None
        obj = self._final(found)
        if obj is None or obj.filepath is None or isinstance(obj.filepath, list):
            return None
        path = Path(obj.filepath).resolve().relative_to(self.root).as_posix()
        if obj.is_module:
            return Target(path, None)
        return Target(path, obj.path[len(obj.module.path) + 1 :])

    def symbol_index(self) -> SymbolIndex:
        if self._index is None:
            index = SymbolIndex()
            for name in self.names:
                self._index_module(index, self._modules()[name])
            self._index = index
        return self._index

    def _index_module(self, index: SymbolIndex, module: griffe.Module) -> None:
        for member in module.members.values():
            if isinstance(member, griffe.Alias):
                continue
            if isinstance(member, griffe.Module):
                self._index_module(index, member)
            elif member.is_function or member.is_class or CONSTANT_RE.match(member.name):
                index.names[member.name].append(member.path)
            if isinstance(member, griffe.Class):
                for child in member.members.values():
                    if not isinstance(child, griffe.Alias) and (
                        child.is_function or child.is_attribute
                    ):
                        index.members[f"{member.name}.{child.name}"].append(child.path)


def generate_page_list(area: str, pages: Iterable[Page]) -> str:
    """Generated area-index content: pages grouped by type, sorted by title."""
    groups: dict[PageType, list[Page]] = {t: [] for t in PageType}
    for page in pages:
        if page.area == area and page.meta is not None:
            groups[page.meta.type].append(page)
    sections: list[str] = []
    for page_type, members in groups.items():
        entries = [(p.meta.title, p) for p in members if p.meta is not None]
        if not entries:
            continue
        lines = [
            f"- [{title}]({p.slug}.md) — {p.meta.description if p.meta else ''}"
            for title, p in sorted(entries, key=lambda e: (e[0].casefold(), e[1].slug))
        ]
        sections.append(f"### {page_type.heading}\n\n" + "\n".join(lines))
    return "\n\n".join(sections)


class WikiLinter:
    def __init__(
        self,
        wiki_dir: Path,
        root: Path,
        fix: bool = False,
        module_dirs: tuple[str, ...] = DEFAULT_MODULE_DIRS,
        mcp_dir: str = DEFAULT_MCP_DIR,
        symbol_root: str = DEFAULT_SYMBOL_ROOT,
        package_roots: Sequence[str] | None = None,
    ) -> None:
        self.wiki_dir = wiki_dir.resolve()
        self.root = root.resolve()
        self.fix = fix
        self.module_dirs = module_dirs
        self.mcp_dir = mcp_dir
        self.symbol_root = symbol_root
        self.history = GitHistory(self.root)
        self.findings: list[Finding] = []
        self.pages: dict[str, Page] = {}
        self.inbound: dict[str, set[str]] = {}
        self.cites: dict[str, set[str]] = defaultdict(set)
        self.packages = Packages.discover(self.root, package_roots)
        self._basename_index: dict[str, list[str]] | None = None
        self._sources: dict[str, Source] = {}

    @property
    def errors(self) -> list[Finding]:
        return [
            f for f in self.findings if f.kind is not Kind.STALE and f.kind not in COVERAGE_KINDS
        ]

    @property
    def stale(self) -> list[Finding]:
        return [f for f in self.findings if f.kind is Kind.STALE]

    @property
    def coverage(self) -> list[Finding]:
        return [f for f in self.findings if f.kind in COVERAGE_KINDS]

    def _add(self, slug: str, kind: Kind, message: str, lines: Iterable[int] = ()) -> None:
        self.findings.append(Finding(slug, kind, message, tuple(sorted(lines))))

    def run(self) -> None:
        areas = self._areas()
        for area in areas:
            self._load_area(area)
        self.inbound = {slug: set() for slug in self.pages}
        by_path = {page.path.resolve(): slug for slug, page in self.pages.items()}

        if self.fix:
            for area in areas:
                self._fix_area_index(area)

        self._check_root_index(areas)
        for area in areas:
            self._check_area_index(area)

        for slug, page in self.pages.items():
            self._validate_meta(page)
            for target in self._check_links(slug, page.path, page.body):
                if target in by_path and by_path[target] != slug:
                    self.inbound[by_path[target]].add(slug)
            self._check_citations(page)
        for page in self.pages.values():
            self._check_staleness(page)
        self._check_coverage()

        for slug in self.pages:
            if not self.inbound[slug]:
                self._add(
                    slug,
                    Kind.ORPHAN,
                    "orphan: no inbound link from a page other than its area index",
                )

    def _areas(self) -> list[str]:
        return sorted(
            d.name
            for d in self.wiki_dir.iterdir()
            if d.is_dir() and not d.name.startswith(("_", "."))
        )

    def _load_area(self, area: str) -> None:
        for path in sorted((self.wiki_dir / area).glob("*.md")):
            slug = path.stem
            if path.name == "index.md" or slug.startswith("_"):
                continue
            if slug in self.pages:
                other = self.pages[slug].path.relative_to(self.wiki_dir)
                self._add(slug, Kind.DUPLICATE_SLUG, f"duplicate slug: also at {other}")
                continue
            text = path.read_text()
            meta, offset = self._parse_frontmatter(slug, text, PageMeta)
            self.pages[slug] = Page(slug, area, path, self._rel(path), text, offset, meta)

    def _parse_frontmatter(self, key: str, text: str, model: type[M]) -> tuple[M | None, int]:
        """Parse and validate frontmatter; returns the model and the body's line offset."""
        match = FRONTMATTER_RE.match(text)
        if not match:
            self._add(key, Kind.MISSING_FRONTMATTER, "missing YAML frontmatter")
            return None, 0
        raw, body = match.groups()
        offset = text[: len(text) - len(body)].count("\n")
        try:
            data = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            self._add(key, Kind.INVALID_FRONTMATTER, f"invalid frontmatter YAML: {exc}")
            return None, offset
        if not isinstance(data, dict):
            self._add(key, Kind.INVALID_FRONTMATTER, "frontmatter is not a mapping")
            return None, offset
        try:
            return model.model_validate(data), offset
        except ValidationError as exc:
            self._report_validation(key, exc)
            return None, offset

    def _report_validation(self, key: str, exc: ValidationError) -> None:
        for err in exc.errors():
            name = str(err["loc"][0]) if err["loc"] else ""
            value = err["input"]
            if err["type"] == "missing" or value is None or value == "":
                self._add(key, Kind.MISSING_KEY, f"frontmatter missing key: {name}")
            elif err["type"] == "extra_forbidden" and name == "verified":
                self._add(key, Kind.OBSOLETE_KEY, "verified is obsolete; staleness comes from git")
            elif err["type"] == "extra_forbidden":
                self._add(key, Kind.UNKNOWN_KEY, f"unknown frontmatter key: {name}")
            elif name == "type":
                self._add(key, Kind.INVALID_TYPE, f"invalid type: {value!r}")
            elif name == "status":
                self._add(key, Kind.INVALID_STATUS, f"invalid status: {value!r}")
            else:
                self._add(key, Kind.INVALID_FIELD, f"invalid {name}: {err['msg']}")

    def _validate_meta(self, page: Page) -> None:
        meta = page.meta
        if meta is None:
            return
        if meta.status is PageStatus.DEPRECATED:
            if meta.superseded_by is None:
                self._add(
                    page.slug, Kind.SUPERSEDED_MISSING, "deprecated page missing superseded_by"
                )
            elif meta.superseded_by not in self.pages:
                self._add(
                    page.slug,
                    Kind.SUPERSEDED_UNKNOWN,
                    f"superseded_by names unknown slug: {meta.superseded_by}",
                )
        elif meta.superseded_by is not None:
            self._add(
                page.slug,
                Kind.SUPERSEDED_NOT_DEPRECATED,
                "superseded_by is only allowed when status is deprecated",
            )
        for source in meta.sources:
            self._check_source(page, source)

    def _fix_area_index(self, area: str) -> None:
        path = self.wiki_dir / area / "index.md"
        if not path.is_file():
            return
        text = path.read_text()
        match = PAGES_BLOCK_RE.search(text)
        if not match:
            return
        generated = generate_page_list(area, self.pages.values())
        block = (
            f"{PAGES_START}\n{generated}\n{PAGES_END}"
            if generated
            else f"{PAGES_START}\n{PAGES_END}"
        )
        path.write_text(text[: match.start()] + block + text[match.end() :])

    def _check_root_index(self, areas: list[str]) -> None:
        path = self.wiki_dir / "index.md"
        if not path.is_file():
            self._add("index", Kind.INDEX_MISSING, "index.md is missing")
            return
        linked = set(self._check_links("index", path, path.read_text()))
        for area in areas:
            if (self.wiki_dir / area / "index.md").resolve() not in linked:
                self._add("index", Kind.AREA_NOT_LINKED, f"area index not linked: {area}/index.md")

    def _check_area_index(self, area: str) -> None:
        key = f"{area}/index"
        path = self.wiki_dir / area / "index.md"
        if not path.is_file():
            self._add(key, Kind.INDEX_MISSING, "index.md is missing")
            return
        text = path.read_text()
        _, offset = self._parse_frontmatter(key, text, IndexMeta)
        body = body_from(text, offset)
        match = PAGES_BLOCK_RE.search(body)
        if not match:
            self._add(key, Kind.INDEX_MARKERS, f"missing {PAGES_START} / {PAGES_END} block")
        elif match.group(1).strip() != generate_page_list(area, self.pages.values()).strip():
            self._add(key, Kind.INDEX_STALE, "generated page list is stale: run --fix")
        self._check_links(key, path, body)

    def _rel(self, path: Path) -> str:
        return path.resolve().relative_to(self.root).as_posix()

    def _check_staleness(self, page: Page) -> None:
        # An uncommitted edit to the page (or a new page) counts as re-checked.
        if self.history.is_modified(page.rel):
            return
        base = self.history.last_commit(page.rel)
        for citation, lines in sorted(page.citations.items(), key=lambda kv: kv[0].label):
            if not (self.root / citation.path).is_file():
                continue
            changes = self.history.changes_since(base, citation.path, citation.symbol)
            if changes:
                self._add(
                    page.slug,
                    Kind.STALE,
                    f"{citation.label} changed in {changes.describe()}",
                    lines,
                )

    def _basenames(self) -> dict[str, list[str]]:
        if self._basename_index is None:
            index: dict[str, list[str]] = defaultdict(list)
            for file in sorted((self.root / self.symbol_root).rglob("*")):
                if file.is_file() and not SKIP_DIRS & set(file.parts):
                    index[file.name].append(self._rel(file))
            self._basename_index = index
        return self._basename_index

    def _source(self, path: str) -> Source:
        if path not in self._sources:
            self._sources[path] = Source.parse((self.root / path).read_text(errors="replace"))
        return self._sources[path]

    def _dotted(self, rel: str) -> str:
        return self.packages.module_name(rel) or rel

    def _check_coverage(self) -> None:
        cited_symbols = {
            (c.path, c.symbol) for page in self.pages.values() for c in page.citations if c.symbol
        }
        for directory in self.module_dirs:
            for file in sorted((self.root / directory).glob("*.py")):
                rel = self._rel(file)
                if file.name != "__init__.py" and rel not in self.cites:
                    self._add(
                        COVERAGE_KEY,
                        Kind.COVERAGE_MODULE,
                        f"module not cited by any page: {self._dotted(rel)}",
                    )
        for file in sorted((self.root / self.mcp_dir).rglob("*.py")):
            if SKIP_DIRS & set(file.parts):
                continue
            rel = self._rel(file)
            tree = self._source(rel).tree
            if tree is None:
                continue
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
                    and any(_is_tool_decorator(d) for d in node.decorator_list)
                    and (rel, node.name) not in cited_symbols
                ):
                    self._add(
                        COVERAGE_KEY,
                        Kind.COVERAGE_MCP_TOOL,
                        f"MCP tool not cited by any page: {self._dotted(rel)}.{node.name}",
                    )

    def _check_source(self, page: Page, source: str) -> None:
        if source.startswith(("http://", "https://")):
            return
        frontmatter = page.text.splitlines()[: page.body_offset]
        lines = {i for i, line in enumerate(frontmatter, 1) if source in line}
        self._check_citation(page, source, lines, is_source=True)

    def _check_citations(self, page: Page) -> None:
        masked = mask_fences(page.body)
        for match in INLINE_CODE_RE.finditer(masked):
            raw = match.group(1)
            line = page.body_offset + masked.count("\n", 0, match.start()) + 1
            if self._flag_unqualified(page, raw.strip(), line):
                continue
            self._check_citation(page, raw, {line})

    def _flag_unqualified(self, page: Page, token: str, line: int) -> bool:
        ident = BARE_RE.match(token)
        if not ident or (self.root / token).is_file():
            return False
        name = ident.group(1)
        if (
            name in ALLOWED_WORDS
            or name.split(".")[0] in self.packages.names
            or keyword.iskeyword(name)
            or hasattr(builtins, name)
        ):
            return False
        defined = sorted(self.packages.symbol_index().lookup(name))
        if len(defined) == 1:
            self._add(
                page.slug,
                Kind.UNQUALIFIED_SYMBOL,
                f"unqualified symbol {name}: cite as {defined[0]}",
                [line],
            )
        elif defined:
            self._add(
                page.slug,
                Kind.AMBIGUOUS_SYMBOL,
                f"ambiguous symbol {name}: cite as one of {', '.join(defined)}",
                [line],
            )
        return bool(defined)

    def _check_citation(
        self, page: Page, raw: str, lines: set[int], is_source: bool = False
    ) -> None:
        text = raw.strip()
        while text.startswith("./"):
            text = text[2:]
        dotted = DOTTED_RE.match(text)
        if dotted and dotted.group(1).split(".")[0] in self.packages.names:
            self._check_dotted(page, dotted.group(1), raw, lines)
            return
        symbol_syntax = PATH_SYMBOL_RE.match(text)
        if symbol_syntax:
            path, symbol = symbol_syntax.groups()
            module = self.packages.module_name(path)
            hint = f"{module}.{symbol}" if module else "<dotted path>"
            self._add(
                page.slug,
                Kind.PATH_SYMBOL,
                f"use dotted path, not path::symbol: {raw} -> {hint}",
                lines,
            )
            return
        match = PATH_RE.match(text)
        if not match:
            if is_source:
                self._add(
                    page.slug,
                    Kind.SOURCE_NOT_CITATION,
                    f"source is not a code citation: {raw}",
                    lines,
                )
            return
        path = match.group(1)
        if docs_citation_forbidden(path):
            self._add(page.slug, Kind.DOCS_CITATION, f"pages do not cite docs/: {raw}", lines)
            return
        target = (self.root / path).resolve()
        if target.is_file() and target.is_relative_to(self.root):
            module = self.packages.module_name(path)
            if module:
                self._add(
                    page.slug,
                    Kind.PYTHON_PATH,
                    f"cite as dotted path {module}: {raw}",
                    lines,
                )
                return
            self._register(page, Citation(path, path, None), lines)
            return
        base = self.history.last_commit(page.rel)
        if path in self.history.tracked_at(base):
            renamed = self.history.renamed_to(base, path) if base else None
            if renamed:
                self._add(
                    page.slug, Kind.PATH_RENAMED, f"cited path renamed to {renamed}: {raw}", lines
                )
            else:
                self._add(page.slug, Kind.PATH_DELETED, f"cited path was deleted: {raw}", lines)
        elif HAS_EXT_RE.search(path) and (is_source or self._is_repo_path(path)):
            self._add(page.slug, Kind.PATH_MISSING, f"cited path does not exist: {raw}", lines)
        elif HAS_EXT_RE.search(path):
            matches = self._basenames().get(path, [])
            if len(matches) == 1:
                hint = self.packages.module_name(matches[0]) or matches[0]
                self._add(page.slug, Kind.QUALIFY_PATH, f"qualify path: {raw} is {hint}", lines)
        elif is_source:
            self._add(page.slug, Kind.PATH_MISSING, f"cited path does not exist: {raw}", lines)

    def _check_dotted(self, page: Page, dotted: str, raw: str, lines: set[int]) -> None:
        target = self.packages.resolve(dotted)
        if target is None:
            self._add(page.slug, Kind.SYMBOL_UNDEFINED, f"cited symbol not defined: {raw}", lines)
            return
        self._register(page, Citation(dotted, target.path, target.symbol), lines)

    def _is_repo_path(self, path: str) -> bool:
        # Absolute paths and object-store keys (`latest/model.pkl`) are prose, not citations.
        head = path.split("/", 1)[0]
        return "/" in path and bool(head) and (self.root / head).exists()

    def _register(self, page: Page, citation: Citation, lines: set[int]) -> None:
        self.cites[citation.path].add(page.slug)
        page.citations.setdefault(citation, set()).update(lines)

    def _check_links(self, slug: str, path: Path, text: str) -> list[Path]:
        """Report wikilinks and docs/ links; return the existing files linked to.

        Broken links and anchors are lychee's job (see .pre-commit-config.yaml).
        """
        text = mask_fences(text)
        for match in WIKILINK_RE.finditer(text):
            self._add(slug, Kind.WIKILINK, f"wikilink syntax not allowed: {match.group(0)}")
        text = INLINE_CODE_RE.sub("", text)
        targets: list[Path] = []
        for match in LINK_RE.finditer(text):
            href = match.group(1)
            if re.match(r"^[a-z][a-z0-9+.\-]*:", href, re.IGNORECASE) or href.startswith("#"):
                continue
            target = (path.parent / href.partition("#")[0]).resolve()
            if not target.is_file():
                continue
            if target.is_relative_to(self.root) and docs_citation_forbidden(self._rel(target)):
                self._add(slug, Kind.DOCS_LINK, f"pages do not link into docs/: {href}")
                continue
            targets.append(target)
        return targets

    def backlinks(self) -> dict[str, dict[str, list[str]]]:
        return {
            "pages": {slug: sorted(src) for slug, src in sorted(self.inbound.items())},
            "cites": {path: sorted(src) for path, src in sorted(self.cites.items())},
        }


def render(findings: list[Finding]) -> list[str]:
    out: list[str] = []
    by_slug: dict[str, list[Finding]] = defaultdict(list)
    for f in findings:
        if f.kind is not Kind.STALE and f.kind not in COVERAGE_KINDS:
            by_slug[f.slug].append(f)
    for slug in sorted(by_slug):
        out.append(f"{slug}:")
        out.extend(f"  {f.message}" for f in by_slug[slug])
    coverage = [f for f in findings if f.kind in COVERAGE_KINDS]
    if coverage:
        out.append("Coverage:")
        out.extend(f"  {f.message}" for f in coverage)
    stale = [f for f in findings if f.kind is Kind.STALE]
    if stale:
        out.append("Stale pages:")
        for slug in sorted({f.slug for f in stale}):
            out.append(f"{slug}:")
            for f in (f for f in stale if f.slug == slug):
                where = ", ".join(str(n) for n in f.lines)
                out.append(f"  stale: {f.message} (page lines {where})")
    return out


def lint(
    wiki_dir: Path,
    root: Path = REPO_ROOT,
    write: bool = True,
    fix: bool = False,
    fail_stale: bool = False,
    package_roots: Sequence[str] | None = None,
) -> int:
    linter = WikiLinter(wiki_dir, root, fix=fix, package_roots=package_roots)
    linter.run()
    for line in render(linter.findings):
        print(line)
    if write:
        out = wiki_dir / "backlinks.json"
        out.write_text(json.dumps(linter.backlinks(), indent=2, sort_keys=True) + "\n")
    failed = bool(linter.errors or linter.coverage)
    return 1 if failed or (fail_stale and linter.stale) else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wiki-dir", type=Path, default=Path("wiki"))
    parser.add_argument("--no-write", action="store_true", help="skip writing backlinks.json")
    parser.add_argument("--fix", action="store_true", help="regenerate area index page lists")
    parser.add_argument("--fail-stale", action="store_true", help="exit 1 when pages are stale")
    args = parser.parse_args(argv)
    return lint(args.wiki_dir, write=not args.no_write, fix=args.fix, fail_stale=args.fail_stale)


if __name__ == "__main__":
    sys.exit(main())
