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
    **meta: str,
) -> str:
    fields = {
        "type": "concept",
        "title": "T",
        "description": "D",
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
        "def other() -> int:\n    return 2\n\n\n"
        "class Box:\n    async def run(self) -> None: ...\n\n    def stop(self) -> None: ...\n"
    )
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    (tmp_path / "docs" / "adr").mkdir(parents=True)
    (tmp_path / "docs" / "agents").mkdir()
    (tmp_path / "docs" / "ARCH.md").write_text("# Arch\n\n## Fetch tiers\n")
    (tmp_path / "docs" / "adr" / "0001-x.md").write_text("# Decision\n")
    (tmp_path / "docs" / "agents" / "domain.md").write_text("# Domain\n")
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", "pkg", "docs", "pyproject.toml")
    git(tmp_path, "commit", "-q", "-m", "init")

    wiki = tmp_path / "wiki"
    (wiki / "alpha").mkdir(parents=True)
    (wiki / "beta").mkdir()
    (wiki / "SCHEMA.md").write_text("# Schema\n")
    (wiki / "index.md").write_text("- [Alpha](alpha/index.md)\n- [Beta](beta/index.md)\n")
    (wiki / "alpha" / "a.md").write_text(
        page("See [B](../beta/b.md#t) and `pkg/mod.py::Box.run`.", title="A")
    )
    (wiki / "beta" / "b.md").write_text(
        page(
            "See [A](../alpha/a.md), [adr](../../docs/adr/0001-x.md#decision).",
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

    def test_verified_is_obsolete(self, repo: Path) -> None:
        write_page(repo, "alpha", "a", page("[B](../beta/b.md)", verified="abc1234", title="A"))
        _, errors = run(repo)
        assert errors["a"] == ["verified is obsolete; staleness comes from git"]


class TestAreaIndex:
    def test_stale_block_reported(self, repo: Path) -> None:
        (repo / "wiki" / "alpha" / "index.md").write_text(area_index("- old"))
        _, errors = run(repo)
        assert any("--fix" in e for e in errors["alpha/index"])

    def test_fix_generates_grouped_sorted_block(self, repo: Path) -> None:
        write_page(
            repo,
            "alpha",
            "z",
            page("[A](a.md)", title="Zed", type="howto", description="Do z"),
        )
        write_page(
            repo,
            "alpha",
            "y",
            page("[A](a.md)", title="apple", type="concept", description="Y"),
        )
        write_page(
            repo,
            "alpha",
            "x",
            page("[A](a.md)", title="Banana", type="concept", description="X"),
        )
        write_page(
            repo,
            "alpha",
            "w",
            page("[A](a.md)", title="W", type="process", description="W"),
        )
        write_page(
            repo,
            "alpha",
            "a",
            page(
                "[B](../beta/b.md) [Z](z.md) [W](w.md) [X](x.md) [Y](y.md)",
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


LINK_B = " [B](../beta/b.md)"


def put_a(repo: Path, body: str, sources: str = "  - pkg/mod.py::calculate") -> None:
    write_page(repo, "alpha", "a", page(body + LINK_B, sources=sources, title="A"))


def write_file(repo: Path, rel: str, text: str) -> None:
    (repo / rel).parent.mkdir(parents=True, exist_ok=True)
    (repo / rel).write_text(text)


class TestCitations:
    def test_leading_dot_slash_is_normalised(self, repo: Path) -> None:
        put_a(repo, "`./pkg/mod.py::calculate`", sources="  - ./pkg/mod.py::calculate")
        linter = wiki_lint.WikiLinter(repo / "wiki", repo)
        linter.run()
        assert dict(linter.errors) == {}
        assert "pkg/mod.py" in linter.cites

    def test_root_and_nested_non_python_files_are_citations(self, repo: Path) -> None:
        write_file(repo, "deployment/aws/Dockerfile.scraper", "FROM x\n")
        put_a(repo, "`pyproject.toml` and `deployment/aws/Dockerfile.scraper`")
        linter = wiki_lint.WikiLinter(repo / "wiki", repo)
        linter.run()
        assert dict(linter.errors) == {}
        assert linter.cites["pyproject.toml"] == {"a"}
        assert linter.cites["deployment/aws/Dockerfile.scraper"] == {"a"}

    def test_bad_path_is_error(self, repo: Path) -> None:
        put_a(repo, "`pkg/missing.py` and `pkg/missing.toml::x`")
        _, errors = run(repo)
        assert len([e for e in errors["a"] if "does not exist" in e]) == 2

    def test_prose_without_slash_or_extension_is_ignored(self, repo: Path) -> None:
        put_a(repo, "`os.path` and `a/b` and `v1.2` and `x.y`")
        _, errors = run(repo)
        assert errors == {}

    def test_basename_matching_one_packages_file_must_be_qualified(self, repo: Path) -> None:
        write_file(repo, "packages/core/decision.py", "def decide() -> None: ...\n")
        put_a(repo, "See `decision.py`.")
        _, errors = run(repo)
        assert errors["a"] == ["qualify path: decision.py is packages/core/decision.py"]

    def test_ambiguous_basename_is_ignored(self, repo: Path) -> None:
        write_file(repo, "packages/x/util.py", "")
        write_file(repo, "packages/y/util.py", "")
        put_a(repo, "See `util.py`.")
        _, errors = run(repo)
        assert errors == {}

    def test_unqualified_symbol_is_error(self, repo: Path) -> None:
        write_file(repo, "packages/core/decision.py", "def decide_forward() -> None: ...\n")
        put_a(repo, "`decide_forward` and `decide_forward()`")
        _, errors = run(repo)
        expected = (
            "unqualified symbol decide_forward: cite as packages/core/decision.py::decide_forward"
        )
        assert errors["a"] == [expected, expected]

    def test_qualified_symbol_and_fenced_block_are_fine(self, repo: Path) -> None:
        write_file(repo, "packages/core/decision.py", "class Decider: ...\n")
        put_a(repo, "`packages/core/decision.py::Decider`\n\n```\n`Decider`\n```\n")
        _, errors = run(repo)
        assert errors == {}

    def test_symbol_defined_twice_is_not_flagged(self, repo: Path) -> None:
        write_file(repo, "packages/x/m.py", "def main() -> None: ...\n")
        write_file(repo, "packages/y/m.py", "def main() -> None: ...\n")
        put_a(repo, "`main`")
        _, errors = run(repo)
        assert errors == {}

    def test_docs_citations_are_errors(self, repo: Path) -> None:
        put_a(
            repo, "`docs/ARCH.md#fetch-tiers` [x](../../docs/ARCH.md)", sources="  - docs/ARCH.md"
        )
        _, errors = run(repo)
        assert sum("do not cite docs/" in e for e in errors["a"]) == 2
        assert sum("do not link into docs/" in e for e in errors["a"]) == 1

    def test_adr_and_agents_docs_are_allowed(self, repo: Path) -> None:
        put_a(
            repo,
            "`docs/adr/0001-x.md#decision` `docs/agents/domain.md` [d](../../docs/agents/domain.md)",
            sources="  - docs/adr/0001-x.md",
        )
        _, errors = run(repo)
        assert errors == {}

    def test_missing_symbol_in_python_file(self, repo: Path) -> None:
        put_a(repo, "`pkg/mod.py::Box.nope` `pkg/mod.py::LIMIT`")
        _, errors = run(repo)
        assert errors["a"] == ["cited symbol not defined: pkg/mod.py::Box.nope"]


def commit(repo: Path, *paths: str, message: str = "change") -> str:
    git(repo, "add", *paths)
    git(repo, "commit", "-q", "-m", message)
    return git(repo, "rev-parse", "--short=7", "HEAD")


def edit_mod(repo: Path, old: str, new: str) -> None:
    mod = repo / "pkg" / "mod.py"
    mod.write_text(mod.read_text().replace(old, new))


def stale(repo: Path) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    linter = wiki_lint.WikiLinter(repo / "wiki", repo)
    linter.run()
    return dict(linter.errors), dict(linter.stale)


@pytest.fixture
def committed(repo: Path) -> Path:
    """The `repo` fixture with its wiki committed, so pages have a last commit."""
    put_a(repo, "Runs `pkg/mod.py::Box.run`.")
    commit(repo, "wiki", message="wiki")
    return repo


class TestStaleness:
    def test_fresh_pages_not_stale(self, committed: Path) -> None:
        assert stale(committed) == ({}, {})

    def test_uncommitted_pages_are_not_stale_from_history(self, repo: Path) -> None:
        edit_mod(repo, "return 1", "return 10")
        commit(repo, "pkg")
        assert stale(repo) == ({}, {})

    def test_edit_inside_symbol_is_stale_with_sha_and_page_lines(self, committed: Path) -> None:
        edit_mod(committed, "return 1", "return 10")
        sha = commit(committed, "pkg")
        errors, found = stale(committed)
        assert errors == {}
        source_line = (committed / "wiki/alpha/a.md").read_text().splitlines().index(
            "  - pkg/mod.py::calculate"
        ) + 1
        assert found["a"] == [f"pkg/mod.py::calculate changed in {sha} (page lines {source_line})"]
        assert "b" in found

    def test_edit_outside_symbol_is_not_stale(self, committed: Path) -> None:
        edit_mod(committed, "return 2", "return 20")
        commit(committed, "pkg")
        edit_mod(committed, "LIMIT = 3", "LIMIT = 4")
        commit(committed, "pkg")
        assert stale(committed) == ({}, {})

    def test_only_commits_touching_the_symbol_are_reported(self, committed: Path) -> None:
        edit_mod(committed, "return 2", "return 20")
        commit(committed, "pkg")
        edit_mod(committed, "return 1", "return 10")
        hit = commit(committed, "pkg")
        edit_mod(committed, "return 20", "return 200")
        commit(committed, "pkg")
        _, found = stale(committed)
        assert found["a"][0].startswith(f"pkg/mod.py::calculate changed in {hit} ")

    def test_line_shift_above_symbol_is_not_stale(self, committed: Path) -> None:
        edit_mod(committed, "LIMIT = 3\n", "LIMIT = 3\nOTHER = 4\nMORE = 5\n")
        commit(committed, "pkg")
        assert stale(committed) == ({}, {})

    def test_method_range_resolves_the_method_only(self, committed: Path) -> None:
        edit_mod(committed, "def stop(self) -> None: ...", "def stop(self) -> int: ...")
        commit(committed, "pkg")
        assert stale(committed) == ({}, {})
        edit_mod(committed, "async def run(self) -> None: ...", "async def run(self) -> int: ...")
        sha = commit(committed, "pkg")
        _, found = stale(committed)
        body_line = (committed / "wiki/alpha/a.md").read_text().splitlines().index(
            "Runs `pkg/mod.py::Box.run`. [B](../beta/b.md)"
        ) + 1
        assert found["a"] == [f"pkg/mod.py::Box.run changed in {sha} (page lines {body_line})"]

    def test_added_or_removed_symbol_is_stale(self, committed: Path) -> None:
        edit_mod(committed, "def other() -> int:\n    return 2\n", "")
        put_a(committed, "`pkg/mod.py::other`")
        commit(committed, "wiki", message="cite other")
        edit_mod(committed, "LIMIT = 3\n", "LIMIT = 3\n\n\ndef other() -> int:\n    return 2\n")
        sha = commit(committed, "pkg")
        _, found = stale(committed)
        assert any(f"pkg/mod.py::other changed in {sha}" in line for line in found["a"])

    def test_whole_file_citation_is_stale_on_any_change(self, committed: Path) -> None:
        put_a(committed, "Config in `pyproject.toml`.")
        commit(committed, "wiki", message="cite toml")
        (committed / "pyproject.toml").write_text("[project]\nname = 'x'\n")
        sha = commit(committed, "pyproject.toml")
        _, found = stale(committed)
        assert [line for line in found["a"] if line.startswith("pyproject.toml")] == [
            f"pyproject.toml changed in {sha} (page lines {[i for i, t in enumerate((committed / 'wiki/alpha/a.md').read_text().splitlines(), 1) if 'Config in' in t][0]})"
        ]

    def test_working_tree_change_is_stale_until_page_is_edited(self, committed: Path) -> None:
        edit_mod(committed, "return 1", "return 10")
        _, found = stale(committed)
        assert "pkg/mod.py::calculate changed in working tree" in found["a"][0]
        put_a(committed, "Runs `pkg/mod.py::Box.run`. Rechecked.")
        _, found = stale(committed)
        assert set(found) == {"b"}

    def test_working_tree_change_outside_symbol_is_not_stale(self, committed: Path) -> None:
        edit_mod(committed, "return 2", "return 20")
        assert stale(committed) == ({}, {})

    def test_working_tree_whole_file_change(self, committed: Path) -> None:
        put_a(committed, "Config in `pyproject.toml`.")
        commit(committed, "wiki", message="cite toml")
        (committed / "pyproject.toml").write_text("[project]\nname = 'x'\n")
        _, found = stale(committed)
        assert any("pyproject.toml changed in working tree" in line for line in found["a"])

    def test_renamed_cited_file_is_error(self, committed: Path) -> None:
        git(committed, "mv", "pkg/mod.py", "pkg/renamed.py")
        commit(committed, "pkg")
        errors, _ = stale(committed)
        assert "cited path renamed to pkg/renamed.py: pkg/mod.py::Box.run" in errors["a"]
        assert not any("does not exist" in e for e in errors["a"])

    def test_deleted_cited_file_is_error(self, committed: Path) -> None:
        git(committed, "rm", "-q", "pkg/mod.py")
        git(committed, "commit", "-q", "-m", "delete")
        errors, _ = stale(committed)
        assert any("does not exist" in e for e in errors["a"])

    def test_diffs_are_cached_per_path_and_commit(self, committed: Path) -> None:
        edit_mod(committed, "return 1", "return 10")
        sha = commit(committed, "pkg")
        write_page(
            committed,
            "beta",
            "b",
            page("[A](../alpha/a.md) `pkg/mod.py::Box.run`", title="B"),
        )
        linter = wiki_lint.WikiLinter(committed / "wiki", committed)
        linter.run()
        assert [k for k in linter._changes if k[1] != "working tree"] == [
            ("pkg/mod.py", git(committed, "rev-parse", "HEAD"))
        ]
        assert sha

    def test_stale_does_not_fail_unless_requested(self, committed: Path) -> None:
        edit_mod(committed, "return 1", "return 10")
        commit(committed, "pkg")
        assert wiki_lint.lint(committed / "wiki", committed, write=False) == 0
        assert wiki_lint.lint(committed / "wiki", committed, write=False, fail_stale=True) == 1

    def test_stale_section_printed(
        self, committed: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        edit_mod(committed, "return 1", "return 10")
        commit(committed, "pkg")
        wiki_lint.lint(committed / "wiki", committed, write=False)
        out = capsys.readouterr().out
        assert "Stale pages:" in out
        assert "pkg/mod.py::calculate changed in" in out


class TestCoverage:
    @pytest.fixture
    def layout(self, repo: Path) -> Path:
        write_file(repo, "pkg/jobs/__init__.py", "")
        write_file(repo, "pkg/jobs/fetch.py", "async def main() -> None: ...\n")
        write_file(repo, "pkg/cli/scrape.py", "def scrape() -> None: ...\n")
        write_file(
            repo,
            "pkg/mcp/server.py",
            "from x import mcp\n\n\n@mcp.tool()\nasync def get_odds() -> None: ...\n\n\n"
            "@other.tool\ndef get_scores() -> None: ...\n\n\ndef helper() -> None: ...\n",
        )
        return repo

    def coverage(self, repo: Path) -> list[str]:
        linter = wiki_lint.WikiLinter(
            repo / "wiki",
            repo,
            module_dirs=("pkg/jobs", "pkg/cli"),
            mcp_dir="pkg/mcp",
        )
        linter.run()
        return linter.coverage

    def test_uncited_modules_and_tools_are_reported(self, layout: Path) -> None:
        assert self.coverage(layout) == [
            "module not cited by any page: pkg/jobs/fetch.py",
            "module not cited by any page: pkg/cli/scrape.py",
            "MCP tool not cited by any page: pkg/mcp/server.py::get_odds",
            "MCP tool not cited by any page: pkg/mcp/server.py::get_scores",
        ]

    def test_cited_modules_and_tool_functions_are_covered(self, layout: Path) -> None:
        put_a(
            layout,
            "`pkg/jobs/fetch.py::main` `pkg/cli/scrape.py` `pkg/mcp/server.py::get_odds`"
            " `pkg/mcp/server.py::get_scores`",
        )
        assert self.coverage(layout) == []

    def test_citing_the_tool_file_alone_does_not_cover_tools(self, layout: Path) -> None:
        put_a(layout, "`pkg/jobs/fetch.py` `pkg/cli/scrape.py` `pkg/mcp/server.py`")
        assert self.coverage(layout) == [
            "MCP tool not cited by any page: pkg/mcp/server.py::get_odds",
            "MCP tool not cited by any page: pkg/mcp/server.py::get_scores",
        ]

    def test_default_roots_cover_real_layout_and_fail_the_lint(self, repo: Path) -> None:
        write_file(repo, "packages/odds-lambda/odds_lambda/jobs/fetch.py", "")
        assert wiki_lint.lint(repo / "wiki", repo, write=False) == 1
