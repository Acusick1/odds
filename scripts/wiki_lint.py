"""Lint the agent wiki and generate wiki/backlinks.json (spec: wiki/SCHEMA.md)."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
# Fixed order of the generated area-index groups, with their plural headings.
TYPE_HEADINGS = {
    "concept": "Concepts",
    "process": "Processes",
    "component": "Components",
    "decision": "Decisions",
    "experiment": "Experiments",
    "howto": "How-tos",
}
PAGE_STATUSES = {"draft", "stable", "deprecated"}
REQUIRED_KEYS = ("type", "title", "description", "sources", "status")
INDEX_REQUIRED_KEYS = ("title", "description")
PAGES_START = "<!-- pages:start -->"
PAGES_END = "<!-- pages:end -->"
PAGES_BLOCK_RE = re.compile(re.escape(PAGES_START) + r"(.*?)" + re.escape(PAGES_END), re.DOTALL)
DOCS_ALLOWED = ("docs/adr/", "docs/agents/")
DEFAULT_MODULE_DIRS = (
    "packages/odds-lambda/odds_lambda/jobs",
    "packages/odds-cli/odds_cli/commands",
)
DEFAULT_MCP_DIR = "packages/odds-mcp/odds_mcp"
DEFAULT_SYMBOL_ROOT = "packages"
SKIP_DIRS = {"__pycache__", ".venv", "node_modules", ".git"}
UNBOUNDED = 10**9

FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n?(.*)\Z", re.DOTALL)
FENCE_RE = re.compile(r"^(```|~~~).*?^\1[ \t]*$", re.DOTALL | re.MULTILINE)
LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)\)")
WIKILINK_RE = re.compile(r"\[\[[^\]]+\]\]")
INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
CITE_RE = re.compile(r"^([\w.\-/]+?)(?:::([\w.]+))?(?:#([\w\-]+))?$")
IDENT_RE = re.compile(r"^([A-Za-z_]\w*)(?:\(\))?$")
HAS_EXT_RE = re.compile(r"\.[A-Za-z]\w*$")
HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", re.MULTILINE)
HEADING_RE = re.compile(r"^#{1,6}\s+(.*?)\s*#*\s*$", re.MULTILINE)


def github_slug(heading: str) -> str:
    text = re.sub(r"[`*]", "", heading.strip().lower())
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def heading_slugs(text: str) -> set[str]:
    seen: dict[str, int] = {}
    slugs: set[str] = set()
    for match in HEADING_RE.finditer(FENCE_RE.sub("", text)):
        base = github_slug(match.group(1))
        count = seen.get(base, 0)
        seen[base] = count + 1
        slugs.add(base if count == 0 else f"{base}-{count}")
    return slugs


def symbol_defined(text: str, symbol: str) -> bool:
    name = re.escape(symbol)
    pattern = rf"^(?:\s*(?:async\s+def|def|class)\s+{name}\b|{name}\s*[:=])"
    return re.search(pattern, text, re.MULTILINE) is not None


def mask_fences(text: str) -> str:
    """Blank fenced code blocks, keeping line numbers intact."""
    return FENCE_RE.sub(lambda m: "\n" * m.group(0).count("\n"), text)


def docs_citation_forbidden(path: str) -> bool:
    return path.startswith("docs/") and not path.startswith(DOCS_ALLOWED)


def _span(node: ast.AST) -> tuple[int, int]:
    decorators = getattr(node, "decorator_list", [])
    start = min([node.lineno, *(d.lineno for d in decorators)])  # type: ignore[attr-defined]
    return start, node.end_lineno  # type: ignore[attr-defined, return-value]


def _find_def(scope: list[ast.stmt], name: str) -> ast.stmt | None:
    for node in scope:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            if node.name == name:
                return node
        elif isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                return node
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        ):
            return node
    return None


def symbol_range(tree: ast.Module, symbol: str) -> tuple[int, int] | None:
    """Line range of a (dotted) symbol; falls back to its last segment defined anywhere."""
    parts = symbol.split(".")
    scope: list[ast.stmt] = tree.body
    node: ast.stmt | None = None
    for part in parts:
        node = _find_def(scope, part)
        if node is None:
            break
        scope = getattr(node, "body", [])
    if node is not None:
        return _span(node)
    for candidate in ast.walk(tree):
        if (
            isinstance(candidate, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
            and candidate.name == parts[-1]
        ):
            return _span(candidate)
    return None


def hunk_overlaps(start: int, count: int, span: tuple[int, int]) -> bool:
    return count > 0 and start <= span[1] and start + count - 1 >= span[0]


@dataclass
class _Source:
    """One version of a file; tree is None when it does not parse."""

    tree: ast.Module | None

    @classmethod
    def parse(cls, text: str) -> _Source:
        try:
            return cls(ast.parse(text))
        except SyntaxError:
            return cls(None)

    def range(self, symbol: str) -> tuple[int, int] | None:
        if self.tree is None:
            return (1, UNBOUNDED)
        return symbol_range(self.tree, symbol)


@dataclass
class _FileChange:
    old: _Source | None
    new: _Source | None
    old_hunks: list[tuple[int, int]]
    new_hunks: list[tuple[int, int]]


def generate_page_list(
    area: str, meta: dict[str, dict[str, object]], page_area: dict[str, str]
) -> str:
    """Return the generated area-index content: pages grouped by type, sorted by title."""
    groups: dict[str, list[tuple[str, str, str]]] = {t: [] for t in TYPE_HEADINGS}
    for slug, page_meta in meta.items():
        page_type = page_meta.get("type")
        if page_area[slug] != area or page_type not in groups:
            continue
        title = str(page_meta.get("title") or slug)
        groups[str(page_type)].append((title, slug, str(page_meta.get("description") or "")))
    sections: list[str] = []
    for page_type, entries in groups.items():
        if not entries:
            continue
        lines = [
            f"- [{title}]({slug}.md) — {description}"
            for title, slug, description in sorted(entries, key=lambda e: (e[0].casefold(), e[1]))
        ]
        sections.append(f"### {TYPE_HEADINGS[page_type]}\n\n" + "\n".join(lines))
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
    ) -> None:
        self.wiki_dir = wiki_dir.resolve()
        self.root = root.resolve()
        self.fix = fix
        self.module_dirs = module_dirs
        self.mcp_dir = mcp_dir
        self.symbol_root = symbol_root
        self.errors: dict[str, list[str]] = defaultdict(list)
        self.stale: dict[str, list[str]] = defaultdict(list)
        self.coverage: list[str] = []
        self.inbound: dict[str, set[str]] = {}
        self.cites: dict[str, set[str]] = defaultdict(set)
        self._cite_symbols: set[tuple[str, str]] = set()
        # slug -> (path, symbol) -> page line numbers where the citation appears
        self._citations: dict[str, dict[tuple[str, str | None], set[int]]] = defaultdict(
            lambda: defaultdict(set)
        )
        self._page_files: dict[str, Path] = {}
        self._texts: dict[str, str] = {}
        self._offsets: dict[str, int] = {}
        self._symbol_index: dict[str, list[str]] | None = None
        self._basename_index: dict[str, list[str]] | None = None
        self._page_commits: dict[str, str | None] = {}
        self._renames: dict[str, dict[str, str]] = {}
        self._path_commits: dict[tuple[str, str], list[str]] = {}
        self._changes: dict[tuple[str, str], _FileChange | None] = {}

    def run(self) -> None:
        areas = self._areas()
        pages: dict[str, Path] = {}
        meta: dict[str, dict[str, object]] = {}
        bodies: dict[str, str] = {}
        page_area: dict[str, str] = {}
        for area in areas:
            for path in sorted((self.wiki_dir / area).glob("*.md")):
                slug = path.stem
                if path.name == "index.md" or slug.startswith("_"):
                    continue
                if slug in pages:
                    other = pages[slug].relative_to(self.wiki_dir)
                    self.errors[slug].append(f"duplicate slug: also at {other}")
                    continue
                pages[slug] = path
                page_area[slug] = area
                text = path.read_text()
                self._page_files[slug] = path
                self._texts[slug] = text
                page_meta, bodies[slug] = self._parse_frontmatter(slug, text)
                self._offsets[slug] = text[: len(text) - len(bodies[slug])].count("\n")
                if page_meta is not None:
                    meta[slug] = page_meta
        self.inbound = {slug: set() for slug in pages}
        by_path = {path.resolve(): slug for slug, path in pages.items()}

        if self.fix:
            for area in areas:
                self._fix_area_index(area, meta, page_area)

        self._check_root_index(areas)
        for area in areas:
            self._check_area_index(area, meta, page_area)

        for slug, path in pages.items():
            self._validate_meta(slug, meta.get(slug), pages)
            for target in self._check_links(slug, path, bodies[slug]):
                if target in by_path and by_path[target] != slug:
                    self.inbound[by_path[target]].add(slug)
            self._check_citations(slug, bodies[slug], self._offsets[slug])
        for slug in pages:
            self._check_staleness(slug)
        self._check_coverage()

        for slug in pages:
            if not self.inbound[slug]:
                self.errors[slug].append(
                    "orphan: no inbound link from a page other than its area index"
                )

    def _areas(self) -> list[str]:
        return sorted(
            d.name
            for d in self.wiki_dir.iterdir()
            if d.is_dir() and not d.name.startswith(("_", "."))
        )

    def _parse_frontmatter(self, key: str, text: str) -> tuple[dict[str, object] | None, str]:
        match = FRONTMATTER_RE.match(text)
        if not match:
            self.errors[key].append("missing YAML frontmatter")
            return None, text
        raw, body = match.groups()
        try:
            meta = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            self.errors[key].append(f"invalid frontmatter YAML: {exc}")
            return None, body
        if not isinstance(meta, dict):
            self.errors[key].append("frontmatter is not a mapping")
            return None, body
        return meta, body

    def _missing_keys(self, key: str, meta: dict[str, object], required: tuple[str, ...]) -> None:
        for name in required:
            if meta.get(name) in (None, "") and not (name == "sources" and meta.get(name) == []):
                self.errors[key].append(f"frontmatter missing key: {name}")

    def _validate_meta(
        self, slug: str, meta: dict[str, object] | None, pages: dict[str, Path]
    ) -> None:
        if meta is None:
            return
        self._missing_keys(slug, meta, REQUIRED_KEYS)
        if "verified" in meta:
            self.errors[slug].append("verified is obsolete; staleness comes from git")
        if meta.get("type") not in (None, "") and meta["type"] not in TYPE_HEADINGS:
            self.errors[slug].append(f"invalid type: {meta['type']!r}")
        status = meta.get("status")
        if status not in (None, "") and status not in PAGE_STATUSES:
            self.errors[slug].append(f"invalid status: {status!r}")
        superseded = meta.get("superseded_by")
        if status == "deprecated":
            if superseded in (None, ""):
                self.errors[slug].append("deprecated page missing superseded_by")
            elif superseded not in pages:
                self.errors[slug].append(f"superseded_by names unknown slug: {superseded}")
        elif superseded not in (None, ""):
            self.errors[slug].append("superseded_by is only allowed when status is deprecated")
        sources = meta.get("sources")
        if sources is not None:
            if not isinstance(sources, list):
                self.errors[slug].append("sources must be a list")
            else:
                for source in sources:
                    self._check_source(slug, str(source))

    def _fix_area_index(
        self, area: str, meta: dict[str, dict[str, object]], page_area: dict[str, str]
    ) -> None:
        path = self.wiki_dir / area / "index.md"
        if not path.is_file():
            return
        text = path.read_text()
        match = PAGES_BLOCK_RE.search(text)
        if not match:
            return
        generated = generate_page_list(area, meta, page_area)
        block = (
            f"{PAGES_START}\n{generated}\n{PAGES_END}"
            if generated
            else f"{PAGES_START}\n{PAGES_END}"
        )
        path.write_text(text[: match.start()] + block + text[match.end() :])

    def _check_root_index(self, areas: list[str]) -> None:
        path = self.wiki_dir / "index.md"
        if not path.is_file():
            self.errors["index"].append("index.md is missing")
            return
        linked = set(self._check_links("index", path, path.read_text()))
        for area in areas:
            if (self.wiki_dir / area / "index.md").resolve() not in linked:
                self.errors["index"].append(f"area index not linked: {area}/index.md")

    def _check_area_index(
        self,
        area: str,
        meta: dict[str, dict[str, object]],
        page_area: dict[str, str],
    ) -> None:
        key = f"{area}/index"
        path = self.wiki_dir / area / "index.md"
        if not path.is_file():
            self.errors[key].append("index.md is missing")
            return
        index_meta, body = self._parse_frontmatter(key, path.read_text())
        if index_meta is not None:
            self._missing_keys(key, index_meta, INDEX_REQUIRED_KEYS)
        match = PAGES_BLOCK_RE.search(body)
        if not match:
            self.errors[key].append(f"missing {PAGES_START} / {PAGES_END} block")
        elif match.group(1).strip() != generate_page_list(area, meta, page_area).strip():
            self.errors[key].append("generated page list is stale: run --fix")
        self._check_links(key, path, body)

    def _git(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(self.root), *args], capture_output=True, text=True, check=False
        )

    def _rel(self, path: Path) -> str:
        return path.resolve().relative_to(self.root).as_posix()

    def _page_commit(self, slug: str) -> str | None:
        rel = self._rel(self._page_files[slug])
        if rel not in self._page_commits:
            result = self._git("log", "-1", "--format=%H", "--", rel)
            self._page_commits[rel] = result.stdout.strip() or None
        return self._page_commits[rel]

    def _renamed_to(self, slug: str, path: str) -> str | None:
        base = self._page_commit(slug)
        if base is None:
            return None
        if base not in self._renames:
            result = self._git("diff", "-M", "--name-status", base, "HEAD")
            renames: dict[str, str] = {}
            for line in result.stdout.splitlines():
                fields = line.split("\t")
                if fields[0].startswith("R") and len(fields) == 3:
                    renames[fields[1]] = fields[2]
            self._renames[base] = renames
        return self._renames[base].get(path)

    def _commits_after(self, base: str, path: str) -> list[str]:
        key = (base, path)
        if key not in self._path_commits:
            result = self._git("log", "--no-merges", "--format=%H", f"{base}..HEAD", "--", path)
            self._path_commits[key] = result.stdout.split() if result.returncode == 0 else []
        return self._path_commits[key]

    def _show(self, rev: str, path: str) -> _Source | None:
        result = self._git("show", f"{rev}:{path}")
        return _Source.parse(result.stdout) if result.returncode == 0 else None

    def _file_change(self, path: str, old_rev: str, new_rev: str | None) -> _FileChange | None:
        """Parsed old and new versions plus -U0 hunks; new_rev None means the working tree."""
        key = (path, new_rev or "working tree")
        if key not in self._changes:
            old = self._show(old_rev, path)
            if new_rev is None:
                target = self.root / path
                new = (
                    _Source.parse(target.read_text(errors="replace")) if target.is_file() else None
                )
                diff_args = ["diff", "-U0", "--no-color", old_rev, "--", path]
            else:
                new = self._show(new_rev, path)
                diff_args = ["diff", "-U0", "--no-color", old_rev, new_rev, "--", path]
            if new_rev is None and old is None:
                self._changes[key] = None
            else:
                diff = self._git(*diff_args).stdout
                old_hunks: list[tuple[int, int]] = []
                new_hunks: list[tuple[int, int]] = []
                for m in HUNK_RE.finditer(diff):
                    old_hunks.append((int(m.group(1)), int(m.group(2) or 1)))
                    new_hunks.append((int(m.group(3)), int(m.group(4) or 1)))
                self._changes[key] = _FileChange(old, new, old_hunks, new_hunks)
        return self._changes[key]

    def _symbol_changed(
        self, path: str, symbol: str | None, old_rev: str, new_rev: str | None
    ) -> bool:
        change = self._file_change(path, old_rev, new_rev)
        if change is None:
            return False
        if symbol is None:
            return bool(change.old_hunks)
        old_range = change.old.range(symbol) if change.old else None
        new_range = change.new.range(symbol) if change.new else None
        if (old_range is None) != (new_range is None):
            return True
        if old_range is None or new_range is None:
            return False
        return any(hunk_overlaps(*h, old_range) for h in change.old_hunks) or any(
            hunk_overlaps(*h, new_range) for h in change.new_hunks
        )

    def _check_staleness(self, slug: str) -> None:
        rel = self._rel(self._page_files[slug])
        # Any uncommitted change to the page (including a new page) counts as re-checked.
        if self._git("status", "--porcelain", "--", rel).stdout.strip():
            return
        base = self._page_commit(slug)
        head_ok = self._git("rev-parse", "--verify", "--quiet", "HEAD").returncode == 0
        for (path, symbol), lines in sorted(
            self._citations[slug].items(), key=lambda kv: (kv[0][0], kv[0][1] or "")
        ):
            if not (self.root / path).is_file():
                continue
            # Whole-file semantics for non-Python files and citations without a symbol.
            tracked = symbol if path.endswith(".py") else None
            changed: list[str] = []
            if base:
                for commit in self._commits_after(base, path):
                    if tracked is None or self._symbol_changed(path, tracked, f"{commit}^", commit):
                        changed.append(commit[:7])
            if head_ok and self._symbol_changed(path, tracked, "HEAD", None):
                changed.append("working tree")
            if changed:
                label = f"{path}::{symbol}" if symbol else path
                where = ", ".join(str(n) for n in sorted(lines))
                self.stale[slug].append(
                    f"{label} changed in {', '.join(changed)} (page lines {where})"
                )

    def _symbols(self) -> dict[str, list[str]]:
        """Names of top-level functions and classes under symbol_root, with defining paths."""
        if self._symbol_index is None:
            index: dict[str, list[str]] = defaultdict(list)
            for file in sorted((self.root / self.symbol_root).rglob("*.py")):
                if SKIP_DIRS & set(file.parts):
                    continue
                try:
                    tree = ast.parse(file.read_text(errors="replace"))
                except SyntaxError:
                    continue
                for node in tree.body:
                    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                        index[node.name].append(self._rel(file))
            self._symbol_index = index
        return self._symbol_index

    def _basenames(self) -> dict[str, list[str]]:
        if self._basename_index is None:
            index: dict[str, list[str]] = defaultdict(list)
            for file in sorted((self.root / self.symbol_root).rglob("*")):
                if file.is_file() and not SKIP_DIRS & set(file.parts):
                    index[file.name].append(self._rel(file))
            self._basename_index = index
        return self._basename_index

    def _check_coverage(self) -> None:
        for directory in self.module_dirs:
            for file in sorted((self.root / directory).glob("*.py")):
                rel = self._rel(file)
                if file.name != "__init__.py" and rel not in self.cites:
                    self.coverage.append(f"module not cited by any page: {rel}")
        for file in sorted((self.root / self.mcp_dir).rglob("*.py")):
            if SKIP_DIRS & set(file.parts):
                continue
            rel = self._rel(file)
            try:
                tree = ast.parse(file.read_text(errors="replace"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and any(
                    isinstance(d.func if isinstance(d, ast.Call) else d, ast.Attribute)
                    and (d.func if isinstance(d, ast.Call) else d).attr == "tool"  # type: ignore[union-attr]
                    for d in node.decorator_list
                ):
                    if not any(
                        p == rel and sym.split(".")[-1] == node.name
                        for p, sym in self._cite_symbols
                    ):
                        self.coverage.append(f"MCP tool not cited by any page: {rel}::{node.name}")

    def _check_source(self, slug: str, source: str) -> None:
        if source.startswith(("http://", "https://")):
            return
        frontmatter = self._texts[slug].splitlines()[: self._offsets[slug]]
        lines = {i for i, line in enumerate(frontmatter, 1) if source in line}
        self._check_citation(slug, source, lines, source=True)

    def _check_citations(self, slug: str, body: str, offset: int) -> None:
        masked = mask_fences(body)
        for match in INLINE_CODE_RE.finditer(masked):
            raw = match.group(1)
            line = offset + masked.count("\n", 0, match.start()) + 1
            ident = IDENT_RE.match(raw.strip())
            if ident and not (self.root / ident.group(1)).is_file():
                defined = self._symbols().get(ident.group(1), [])
                if len(defined) == 1:
                    name = ident.group(1)
                    self.errors[slug].append(
                        f"unqualified symbol {name}: cite as {defined[0]}::{name}"
                    )
                continue
            self._check_citation(slug, raw, {line})

    def _check_citation(self, slug: str, raw: str, lines: set[int], source: bool = False) -> None:
        text = raw.strip()
        while text.startswith("./"):
            text = text[2:]
        match = CITE_RE.match(text)
        if not match:
            if source:
                self.errors[slug].append(f"source is not a code citation: {raw}")
            return
        path, symbol, anchor = match.groups()
        if docs_citation_forbidden(path):
            self.errors[slug].append(f"pages do not cite docs/: {raw}")
            return
        target = (self.root / path).resolve()
        if target.is_file() and target.is_relative_to(self.root):
            self._register(slug, path, symbol, anchor, raw, target, lines)
        elif HAS_EXT_RE.search(path) and ("/" in path or source):
            renamed = self._renamed_to(slug, path)
            if renamed:
                self.errors[slug].append(f"cited path renamed to {renamed}: {raw}")
            else:
                self.errors[slug].append(f"cited path does not exist: {raw}")
        elif HAS_EXT_RE.search(path):
            matches = self._basenames().get(path, [])
            if len(matches) == 1:
                self.errors[slug].append(f"qualify path: {raw} is {matches[0]}")
        elif source:
            self.errors[slug].append(f"cited path does not exist: {raw}")

    def _register(
        self,
        slug: str,
        path: str,
        symbol: str | None,
        anchor: str | None,
        raw: str,
        target: Path,
        lines: set[int],
    ) -> None:
        self.cites[path].add(slug)
        self._citations[slug][(path, symbol)] |= lines
        if symbol:
            self._cite_symbols.add((path, symbol))
        text = target.read_text(errors="replace")
        if symbol and path.endswith(".py") and not symbol_defined(text, symbol.split(".")[-1]):
            self.errors[slug].append(f"cited symbol not defined: {raw}")
        if anchor and path.endswith(".md") and anchor not in heading_slugs(text):
            self.errors[slug].append(f"cited anchor not found: {raw}")

    def _check_links(self, slug: str, path: Path, text: str) -> list[Path]:
        """Validate links in text; return resolved paths of the files linked to."""
        text = mask_fences(text)
        for match in WIKILINK_RE.finditer(text):
            self.errors[slug].append(f"wikilink syntax not allowed: {match.group(0)}")
        text = INLINE_CODE_RE.sub("", text)
        targets: list[Path] = []
        for match in LINK_RE.finditer(text):
            href = match.group(1)
            if re.match(r"^[a-z][a-z0-9+.\-]*:", href, re.IGNORECASE) or href.startswith("#"):
                continue
            file_part, _, anchor = href.partition("#")
            target = (path.parent / file_part).resolve()
            if target.is_file() and target.is_relative_to(self.root):
                if docs_citation_forbidden(self._rel(target)):
                    self.errors[slug].append(f"pages do not link into docs/: {href}")
                    continue
            if not target.is_file():
                self.errors[slug].append(f"broken link (file not found): {href}")
                continue
            if anchor and target.suffix == ".md":
                if anchor not in heading_slugs(target.read_text()):
                    self.errors[slug].append(f"broken link (anchor not found): {href}")
                    continue
            targets.append(target)
        return targets

    def backlinks(self) -> dict[str, dict[str, list[str]]]:
        return {
            "pages": {slug: sorted(src) for slug, src in sorted(self.inbound.items())},
            "cites": {path: sorted(src) for path, src in sorted(self.cites.items())},
        }


def lint(
    wiki_dir: Path,
    root: Path = REPO_ROOT,
    write: bool = True,
    fix: bool = False,
    fail_stale: bool = False,
) -> int:
    linter = WikiLinter(wiki_dir, root, fix=fix)
    linter.run()
    for slug in sorted(linter.errors):
        print(f"{slug}:")
        for error in linter.errors[slug]:
            print(f"  {error}")
    if linter.coverage:
        print("Coverage:")
        for line in linter.coverage:
            print(f"  {line}")
    if linter.stale:
        print("Stale pages:")
        for slug in sorted(linter.stale):
            print(f"{slug}:")
            for line in linter.stale[slug]:
                print(f"  stale: {line}")
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
