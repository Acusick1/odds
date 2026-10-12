"""Lint the agent wiki and generate wiki/backlinks.json (spec: wiki/SCHEMA.md)."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
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
REQUIRED_KEYS = ("type", "title", "description", "sources", "verified", "status")
INDEX_REQUIRED_KEYS = ("title", "description")
PAGES_START = "<!-- pages:start -->"
PAGES_END = "<!-- pages:end -->"
SHA_RE = re.compile(r"\A[0-9a-f]{4,40}\Z")
PAGES_BLOCK_RE = re.compile(re.escape(PAGES_START) + r"(.*?)" + re.escape(PAGES_END), re.DOTALL)
CITE_EXTENSIONS = (".py", ".md", ".yaml", ".yml", ".toml", ".tf", ".json", ".sh")

FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---\n?(.*)\Z", re.DOTALL)
FENCE_RE = re.compile(r"^(```|~~~).*?^\1[ \t]*$", re.DOTALL | re.MULTILINE)
LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)\)")
WIKILINK_RE = re.compile(r"\[\[[^\]]+\]\]")
INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
CITE_RE = re.compile(r"^([\w.\-/]+?)(?:::([\w.]+))?(?:#([\w\-]+))?$")
HEADING_RE = re.compile(r"^#{1,6}\s+(.*?)\s*#*\s*$", re.MULTILINE)


class _StringLoader(yaml.SafeLoader):
    """SafeLoader that keeps numeric-looking scalars (``verified: 7887039``) as strings."""


_StringLoader.yaml_implicit_resolvers = {
    ch: [
        (tag, rx)
        for tag, rx in resolvers
        if tag not in ("tag:yaml.org,2002:int", "tag:yaml.org,2002:float")
    ]
    for ch, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}


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


def parse_citation(raw: str, root_files: bool = False) -> tuple[str, str | None, str | None] | None:
    """Return (path, symbol, anchor) if raw is a code citation, else None.

    Bare root-level names (``AGENTS.md``) count only where root_files is set, since
    prose often names files by basename.
    """
    match = CITE_RE.match(raw.strip())
    if not match:
        return None
    path, symbol, anchor = match.groups()
    if ("/" not in path and not root_files) or not path.endswith(CITE_EXTENSIONS):
        return None
    return path, symbol, anchor


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
    def __init__(self, wiki_dir: Path, root: Path, fix: bool = False) -> None:
        self.wiki_dir = wiki_dir.resolve()
        self.root = root.resolve()
        self.fix = fix
        self.errors: dict[str, list[str]] = defaultdict(list)
        self.stale: dict[str, list[str]] = defaultdict(list)
        self.inbound: dict[str, set[str]] = {}
        self.cites: dict[str, set[str]] = defaultdict(set)
        self._page_paths: dict[str, set[str]] = defaultdict(set)
        self._commit_ok: dict[str, bool] = {}
        self._changes: dict[tuple[str, str], list[str]] = {}

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
                page_meta, bodies[slug] = self._parse_frontmatter(slug, path.read_text())
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
            self._check_citations(slug, bodies[slug])
            self._check_staleness(slug, meta.get(slug))

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
            meta = yaml.load(raw, Loader=_StringLoader)  # noqa: S506 - SafeLoader subclass
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

    def _check_staleness(self, slug: str, meta: dict[str, object] | None) -> None:
        if meta is None or meta.get("verified") in (None, ""):
            return
        verified = str(meta["verified"])
        if not self._valid_commit(verified):
            self.errors[slug].append(f"verified is not a valid commit: {verified}")
            return
        for path in sorted(self._page_paths[slug]):
            shas = self._changes_since(verified, path)
            if shas:
                self.stale[slug].append(
                    f"stale: {path} changed in {', '.join(shas)} since {verified}"
                )

    def _git(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(self.root), *args], capture_output=True, text=True, check=False
        )

    def _valid_commit(self, sha: str) -> bool:
        if sha not in self._commit_ok:
            self._commit_ok[sha] = bool(SHA_RE.match(sha)) and (
                self._git("rev-parse", "--verify", "--quiet", f"{sha}^{{commit}}").returncode == 0
            )
        return self._commit_ok[sha]

    def _changes_since(self, verified: str, path: str) -> list[str]:
        cache_key = (verified, path)
        if cache_key not in self._changes:
            result = self._git("log", "--format=%h", f"{verified}..HEAD", "--", path)
            self._changes[cache_key] = result.stdout.split() if result.returncode == 0 else []
        return self._changes[cache_key]

    def _check_source(self, slug: str, source: str) -> None:
        if source.startswith(("http://", "https://")):
            return
        citation = parse_citation(source, root_files=True)
        if citation is None:
            self.errors[slug].append(f"source is not a code citation: {source}")
            return
        self._check_citation(slug, source, citation)

    def _check_citations(self, slug: str, body: str) -> None:
        for match in INLINE_CODE_RE.finditer(FENCE_RE.sub("", body)):
            raw = match.group(1)
            citation = parse_citation(raw)
            if citation is not None:
                self._check_citation(slug, raw, citation)

    def _check_citation(
        self, slug: str, raw: str, citation: tuple[str, str | None, str | None]
    ) -> None:
        path, symbol, anchor = citation
        target = self.root / path
        if not target.is_file():
            self.errors[slug].append(f"cited path does not exist: {raw}")
            return
        self.cites[path].add(slug)
        self._page_paths[slug].add(path)
        text = target.read_text(errors="replace")
        if symbol and not symbol_defined(text, symbol.split(".")[-1]):
            self.errors[slug].append(f"cited symbol not defined: {raw}")
        if anchor and path.endswith(".md") and anchor not in heading_slugs(text):
            self.errors[slug].append(f"cited anchor not found: {raw}")

    def _check_links(self, slug: str, path: Path, text: str) -> list[Path]:
        """Validate links in text; return resolved paths of the files linked to."""
        text = FENCE_RE.sub("", text)
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
    if linter.stale:
        print("Stale pages:")
        for slug in sorted(linter.stale):
            print(f"{slug}:")
            for line in linter.stale[slug]:
                print(f"  {line}")
    if write:
        out = wiki_dir / "backlinks.json"
        out.write_text(json.dumps(linter.backlinks(), indent=2, sort_keys=True) + "\n")
    return 1 if linter.errors or (fail_stale and linter.stale) else 0


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
