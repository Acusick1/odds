"""Unit tests for scripts/wiki_lint.py."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "wiki_lint", Path(__file__).resolve().parents[2] / "scripts" / "wiki_lint.py"
)
assert _SPEC is not None and _SPEC.loader is not None
wiki_lint = importlib.util.module_from_spec(_SPEC)
sys.modules["wiki_lint"] = wiki_lint
_SPEC.loader.exec_module(wiki_lint)

PAGES_START = "<!-- pages:start -->"
PAGES_END = "<!-- pages:end -->"
# Commit the `repo` fixture created; the default `verified` for page().
HEAD_SHA = ""


def git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def page(
    body: str = "Body.",
    sources: str = "  - pkg/mod.py::calculate",
    verified: str | None = None,
    **meta: str,
) -> str:
    fields = {
        "type": "concept",
        "title": "T",
        "description": "D",
        "verified": verified or HEAD_SHA,
        "status": "stable",
        **meta,
    }
    head = "\n".join(f"{k}: {v}" for k, v in fields.items())
    return f"---\n{head}\nsources:\n{sources}\n---\n\n# T\n\n{body}\n"


def area_index(block: str = "") -> str:
    return f"---\ntitle: Area\ndescription: An area.\n---\n\nIntro.\n\n{PAGES_START}\n{block}\n{PAGES_END}\n"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text(
        "LIMIT = 3\n\n\ndef calculate() -> int:\n    return 1\n\n\n"
        "class Box:\n    async def run(self) -> None: ...\n"
    )
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "ARCH.md").write_text("# Arch\n\n## Fetch tiers\n")
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", "pkg", "docs")
    git(tmp_path, "commit", "-q", "-m", "init")
    sha = git(tmp_path, "rev-parse", "--short", "HEAD")
    global HEAD_SHA
    HEAD_SHA = sha

    wiki = tmp_path / "wiki"
    (wiki / "alpha").mkdir(parents=True)
    (wiki / "beta").mkdir()
    (wiki / "SCHEMA.md").write_text("# Schema\n")
    (wiki / "log.md").write_text("")
    (wiki / "index.md").write_text("- [Alpha](alpha/index.md)\n- [Beta](beta/index.md)\n")
    (wiki / "alpha" / "a.md").write_text(
        page("See [B](../beta/b.md#t) and `pkg/mod.py::Box.run`.", verified=sha, title="A")
    )
    (wiki / "beta" / "b.md").write_text(
        page(
            "See [A](../alpha/a.md), [arch](../../docs/ARCH.md#fetch-tiers).",
            verified=sha,
            title="B",
        )
    )
    (wiki / "alpha" / "index.md").write_text(area_index())
    (wiki / "beta" / "index.md").write_text(area_index())
    wiki_lint.lint(wiki, tmp_path, write=False, fix=True)
    return tmp_path


def run(repo: Path, fix: bool = False) -> tuple[int, dict[str, list[str]]]:
    linter = wiki_lint.WikiLinter(repo / "wiki", repo, fix=fix)
    linter.run()
    return (1 if linter.errors else 0), dict(linter.errors)


def write_page(repo: Path, area: str, slug: str, text: str) -> None:
    (repo / "wiki" / area / f"{slug}.md").write_text(text)


class TestWikiLint:
    def test_valid_wiki_passes(self, repo: Path) -> None:
        code, errors = run(repo)
        assert errors == {}
        assert code == 0

    def test_broken_link(self, repo: Path) -> None:
        write_page(repo, "alpha", "a", page("See [B](../beta/b.md#t) [X](nope.md).", title="A"))
        _, errors = run(repo)
        assert any("nope.md" in e for e in errors["a"])

    def test_broken_anchor(self, repo: Path) -> None:
        write_page(repo, "beta", "b", page("See [A](../alpha/a.md#missing).", title="B"))
        _, errors = run(repo)
        assert any("anchor" in e and "a.md#missing" in e for e in errors["b"])

    def test_missing_cited_symbol(self, repo: Path) -> None:
        write_page(repo, "alpha", "a", page("`pkg/mod.py::gone` [B](../beta/b.md).", title="A"))
        _, errors = run(repo)
        assert any("symbol" in e and "gone" in e for e in errors["a"])

    def test_missing_cited_path_in_sources(self, repo: Path) -> None:
        write_page(repo, "alpha", "a", page("[B](../beta/b.md)", sources="  - pkg/none.py::f"))
        _, errors = run(repo)
        assert any("path does not exist" in e for e in errors["a"])

    def test_wikilink_syntax(self, repo: Path) -> None:
        write_page(repo, "alpha", "a", page("See [[b]] and [B](../beta/b.md).", title="A"))
        _, errors = run(repo)
        assert any("wikilink" in e for e in errors["a"])

    def test_invalid_type_and_missing_key(self, repo: Path) -> None:
        write_page(
            repo,
            "alpha",
            "a",
            "---\ntype: bogus\ntitle: T\nsources: []\n---\n[B](../beta/b.md)\n",
        )
        _, errors = run(repo)
        assert any("invalid type" in e for e in errors["a"])
        assert any("missing key: description" in e for e in errors["a"])
        assert any("missing key: verified" in e for e in errors["a"])
        assert any("missing key: status" in e for e in errors["a"])

    def test_invalid_status(self, repo: Path) -> None:
        write_page(repo, "alpha", "a", page("[B](../beta/b.md)", status="weird"))
        _, errors = run(repo)
        assert any("invalid status" in e for e in errors["a"])

    def test_ignores_underscore_files_and_dirs(self, repo: Path) -> None:
        write_page(repo, "alpha", "_draft", "no frontmatter")
        (repo / "wiki" / "_templates").mkdir()
        (repo / "wiki" / "_templates" / "x.md").write_text("junk")
        _, errors = run(repo)
        assert errors == {}

    def test_backlinks_json(self, repo: Path) -> None:
        code = wiki_lint.lint(repo / "wiki", repo, write=True)
        assert code == 0
        data = json.loads((repo / "wiki" / "backlinks.json").read_text())
        assert data["pages"] == {"a": ["b"], "b": ["a"]}
        assert data["cites"]["pkg/mod.py"] == ["a", "b"] or data["cites"]["pkg/mod.py"] == ["a"]

    def test_no_write_skips_backlinks(self, repo: Path) -> None:
        wiki_lint.lint(repo / "wiki", repo, write=False)
        assert not (repo / "wiki" / "backlinks.json").exists()


class TestLayout:
    def test_duplicate_slug_across_areas(self, repo: Path) -> None:
        write_page(repo, "beta", "a", page("[A](../alpha/a.md)", title="Dup"))
        _, errors = run(repo)
        assert any("duplicate slug" in e for e in errors["a"])

    def test_root_index_must_link_area_indexes(self, repo: Path) -> None:
        (repo / "wiki" / "index.md").write_text("- [Alpha](alpha/index.md)\n")
        _, errors = run(repo)
        assert errors == {"index": ["area index not linked: beta/index.md"]}

    def test_area_index_requires_frontmatter_keys(self, repo: Path) -> None:
        (repo / "wiki" / "beta" / "index.md").write_text(
            f"---\ntitle: Beta\n---\n{PAGES_START}\n{PAGES_END}\n"
        )
        _, errors = run(repo)
        assert any("missing key: description" in e for e in errors["beta/index"])

    def test_area_index_requires_markers(self, repo: Path) -> None:
        (repo / "wiki" / "beta" / "index.md").write_text("---\ntitle: B\ndescription: D\n---\n")
        _, errors = run(repo)
        assert any("pages:start" in e for e in errors["beta/index"])

    def test_orphan_when_only_area_index_links(self, repo: Path) -> None:
        write_page(repo, "alpha", "c", page("Alone.", title="C"))
        wiki_lint.lint(repo / "wiki", repo, write=False, fix=True)
        _, errors = run(repo)
        assert errors == {"c": [errors["c"][0]]}
        assert "orphan" in errors["c"][0]

    def test_links_resolve_relative_to_own_directory(self, repo: Path) -> None:
        write_page(repo, "alpha", "a", page("[B](b.md)", title="A"))
        _, errors = run(repo)
        assert any("broken link" in e and "b.md" in e for e in errors["a"])


class TestFrontmatter:
    def test_deprecated_requires_superseded_by(self, repo: Path) -> None:
        write_page(repo, "alpha", "a", page("[B](../beta/b.md)", status="deprecated", title="A"))
        _, errors = run(repo)
        assert any("superseded_by" in e for e in errors["a"])

    def test_superseded_by_must_exist(self, repo: Path) -> None:
        write_page(
            repo,
            "alpha",
            "a",
            page("[B](../beta/b.md)", status="deprecated", superseded_by="nope", title="A"),
        )
        _, errors = run(repo)
        assert any("unknown slug: nope" in e for e in errors["a"])

    def test_superseded_by_valid(self, repo: Path) -> None:
        write_page(
            repo,
            "alpha",
            "a",
            page("[B](../beta/b.md)", status="deprecated", superseded_by="b", title="A"),
        )
        _, errors = run(repo)
        assert errors == {}

    def test_superseded_by_requires_deprecated(self, repo: Path) -> None:
        write_page(repo, "alpha", "a", page("[B](../beta/b.md)", superseded_by="b", title="A"))
        _, errors = run(repo)
        assert any("only allowed when status is deprecated" in e for e in errors["a"])

    def test_numeric_looking_verified_is_kept_as_string(self, repo: Path) -> None:
        write_page(repo, "alpha", "a", page("[B](../beta/b.md)", verified="0123456", title="A"))
        _, errors = run(repo)
        assert any("not a valid commit: 0123456" in e for e in errors["a"])


class TestAreaIndex:
    def test_stale_block_reported(self, repo: Path) -> None:
        (repo / "wiki" / "alpha" / "index.md").write_text(area_index("- old"))
        _, errors = run(repo)
        assert any("--fix" in e for e in errors["alpha/index"])

    def test_fix_generates_grouped_sorted_block(self, repo: Path) -> None:
        sha = git(repo, "rev-parse", "--short", "HEAD")
        write_page(
            repo,
            "alpha",
            "z",
            page("[A](a.md)", verified=sha, title="Zed", type="howto", description="Do z"),
        )
        write_page(
            repo,
            "alpha",
            "y",
            page("[A](a.md)", verified=sha, title="apple", type="concept", description="Y"),
        )
        write_page(
            repo,
            "alpha",
            "x",
            page("[A](a.md)", verified=sha, title="Banana", type="concept", description="X"),
        )
        write_page(
            repo,
            "alpha",
            "w",
            page("[A](a.md)", verified=sha, title="W", type="process", description="W"),
        )
        write_page(
            repo,
            "alpha",
            "a",
            page(
                "[B](../beta/b.md) [Z](z.md) [W](w.md) [X](x.md) [Y](y.md)",
                verified=sha,
                title="A",
                type="component",
                description="D",
            ),
        )
        _, errors = run(repo, fix=True)
        assert errors == {}
        text = (repo / "wiki" / "alpha" / "index.md").read_text()
        block = text.split(PAGES_START)[1].split(PAGES_END)[0]
        assert block.strip() == (
            "### Concepts\n\n"
            "- [apple](y.md) — Y\n"
            "- [Banana](x.md) — X\n\n"
            "### Processes\n\n"
            "- [W](w.md) — W\n\n"
            "### Components\n\n"
            "- [A](a.md) — D\n\n"
            "### How-tos\n\n"
            "- [Zed](z.md) — Do z"
        )
        assert text.startswith("---\ntitle: Area")
        assert "Intro." in text
        _, errors = run(repo)
        assert errors == {}


class TestStaleness:
    def modify_cited(self, repo: Path) -> str:
        (repo / "pkg" / "mod.py").write_text((repo / "pkg" / "mod.py").read_text() + "\nX = 1\n")
        git(repo, "add", "pkg/mod.py")
        git(repo, "commit", "-q", "-m", "change")
        return git(repo, "rev-parse", "--short", "HEAD")

    def test_fresh_pages_not_stale(self, repo: Path) -> None:
        linter = wiki_lint.WikiLinter(repo / "wiki", repo)
        linter.run()
        assert dict(linter.stale) == {}

    def test_changed_cited_file_is_stale(self, repo: Path) -> None:
        new = self.modify_cited(repo)
        old = git(repo, "rev-parse", "--short", "HEAD~1")
        linter = wiki_lint.WikiLinter(repo / "wiki", repo)
        linter.run()
        assert dict(linter.errors) == {}
        assert linter.stale["a"] == [f"stale: pkg/mod.py changed in {new} since {old}"]
        assert linter.stale["b"] == linter.stale["a"]

    def test_stale_does_not_fail_unless_requested(self, repo: Path) -> None:
        self.modify_cited(repo)
        assert wiki_lint.lint(repo / "wiki", repo, write=False) == 0
        assert wiki_lint.lint(repo / "wiki", repo, write=False, fail_stale=True) == 1

    def test_stale_section_printed(self, repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
        self.modify_cited(repo)
        wiki_lint.lint(repo / "wiki", repo, write=False)
        assert "Stale pages:" in capsys.readouterr().out

    def test_invalid_verified_commit_is_error(self, repo: Path) -> None:
        write_page(repo, "alpha", "a", page("[B](../beta/b.md)", verified="deadbee", title="A"))
        _, errors = run(repo)
        assert any("not a valid commit: deadbee" in e for e in errors["a"])

    def test_git_calls_cached_per_verified_path(self, repo: Path) -> None:
        sha = git(repo, "rev-parse", "--short", "HEAD")
        write_page(
            repo,
            "beta",
            "b",
            page("[A](../alpha/a.md) `pkg/mod.py::Box.run`", verified=sha, title="B"),
        )
        linter = wiki_lint.WikiLinter(repo / "wiki", repo)
        linter.run()
        assert list(linter._changes) == [(sha, "pkg/mod.py")]
