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

PAGES_START = wiki_lint.PAGES_START
PAGES_END = wiki_lint.PAGES_END
Kind = wiki_lint.Kind
Finding = wiki_lint.Finding


PKG = "packages/lib/pkg"
MOD = f"{PKG}/mod.py"


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
    sources: str = "  - pkg.mod.calculate",
    **meta: str,
) -> str:
    fields = {
        "type": "reference",
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
    (tmp_path / PKG).mkdir(parents=True)
    (tmp_path / PKG / "__init__.py").write_text("from pkg.mod import Box as Crate\n")
    (tmp_path / PKG / "mod.py").write_text(
        "LIMIT = 3\n\n\ndef calculate() -> int:\n    return 1\n\n\n"
        "def other() -> int:\n    return 2\n\n\n"
        "class Box:\n    limit: int = 5\n    x = 1\n\n    async def run(self) -> None: ...\n\n    def stop(self) -> None: ...\n\n\n"
        "TIMEOUT: float = 2.0\n"
    )
    (tmp_path / "pyproject.toml").write_text("[project]\n")
    (tmp_path / "docs" / "adr").mkdir(parents=True)
    (tmp_path / "docs" / "agents").mkdir()
    (tmp_path / "docs" / "ARCH.md").write_text("# Arch\n\n## Fetch tiers\n")
    (tmp_path / "docs" / "adr" / "0001-x.md").write_text("# Decision\n")
    (tmp_path / "docs" / "agents" / "domain.md").write_text("# Domain\n")
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", "packages", "docs", "pyproject.toml")
    git(tmp_path, "commit", "-q", "-m", "init")

    wiki = tmp_path / "wiki"
    (wiki / "alpha").mkdir(parents=True)
    (wiki / "beta").mkdir()
    (wiki / "SCHEMA.md").write_text("# Schema\n")
    (wiki / "index.md").write_text("- [Alpha](alpha/index.md)\n- [Beta](beta/index.md)\n")
    (wiki / "alpha" / "a.md").write_text(
        page("See [B](../beta/b.md#t) and `pkg.mod.Box.run`.", title="A")
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


def run(repo: Path, fix: bool = False) -> list[Finding]:
    linter = wiki_lint.WikiLinter(repo / "wiki", repo, fix=fix)
    linter.run()
    return linter.errors


def kinds(findings: list[Finding], slug: str) -> list[Kind]:
    return [f.kind for f in findings if f.slug == slug]


def messages(findings: list[Finding], slug: str) -> list[str]:
    return [f.message for f in findings if f.slug == slug]


def write_page(repo: Path, area: str, slug: str, text: str) -> None:
    (repo / "wiki" / area / f"{slug}.md").write_text(text)


class TestWikiLint:
    def test_valid_wiki_passes(self, repo: Path) -> None:
        assert run(repo) == []

    def test_wikilink_syntax(self, repo: Path) -> None:
        write_page(repo, "alpha", "a", page("See [[b]] and [B](../beta/b.md).", title="A"))
        assert kinds(run(repo), "a") == [Kind.WIKILINK]

    def test_broken_links_and_anchors_are_left_to_lychee(self, repo: Path) -> None:
        write_page(
            repo,
            "alpha",
            "a",
            page("[B](../beta/b.md#nope) [X](nope.md) `docs/ARCH.md`", title="A"),
        )
        assert kinds(run(repo), "a") == [Kind.DOCS_CITATION]

    def test_missing_cited_symbol(self, repo: Path) -> None:
        write_page(repo, "alpha", "a", page("`pkg.mod.gone` [B](../beta/b.md).", title="A"))
        findings = run(repo)
        assert kinds(findings, "a") == [Kind.SYMBOL_UNDEFINED]
        assert "gone" in messages(findings, "a")[0]

    def test_missing_cited_path_in_sources(self, repo: Path) -> None:
        write_page(repo, "alpha", "a", page("[B](../beta/b.md)", sources="  - scripts/none.py"))
        assert kinds(run(repo), "a") == [Kind.PATH_MISSING]

    def test_invalid_type_and_missing_keys(self, repo: Path) -> None:
        write_page(
            repo,
            "alpha",
            "a",
            "---\ntype: bogus\ntitle: T\nsources: []\n---\n[B](../beta/b.md)\n",
        )
        findings = run(repo)
        assert kinds(findings, "a") == [Kind.INVALID_TYPE, Kind.MISSING_KEY, Kind.MISSING_KEY]
        assert messages(findings, "a")[1:] == [
            "frontmatter missing key: description",
            "frontmatter missing key: status",
        ]

    def test_blank_value_counts_as_missing(self, repo: Path) -> None:
        write_page(repo, "alpha", "a", page("[B](../beta/b.md)", description="''", title="A"))
        assert messages(run(repo), "a") == ["frontmatter missing key: description"]

    @pytest.mark.parametrize("old", ["concept", "process", "component", "decision", "experiment"])
    def test_retired_types_are_invalid(self, repo: Path, old: str) -> None:
        write_page(repo, "alpha", "a", page("[B](../beta/b.md)", type=old, title="A"))
        assert kinds(run(repo), "a") == [Kind.INVALID_TYPE]

    def test_invalid_status(self, repo: Path) -> None:
        write_page(repo, "alpha", "a", page("[B](../beta/b.md)", status="weird"))
        assert kinds(run(repo), "a") == [Kind.INVALID_STATUS]

    def test_unknown_frontmatter_key(self, repo: Path) -> None:
        write_page(repo, "alpha", "a", page("[B](../beta/b.md)", owner="me", title="A"))
        assert kinds(run(repo), "a") == [Kind.UNKNOWN_KEY]

    def test_frontmatter_problems_are_findings(self, repo: Path) -> None:
        write_page(repo, "alpha", "a", "no frontmatter [B](../beta/b.md)")
        write_page(repo, "beta", "b", "---\n: [\n---\n[A](../alpha/a.md)\n")
        findings = run(repo)
        assert kinds(findings, "a") == [Kind.MISSING_FRONTMATTER]
        assert kinds(findings, "b") == [Kind.INVALID_FRONTMATTER]

    def test_ignores_underscore_files_and_dirs(self, repo: Path) -> None:
        write_page(repo, "alpha", "_draft", "no frontmatter")
        (repo / "wiki" / "_templates").mkdir()
        (repo / "wiki" / "_templates" / "x.md").write_text("junk")
        assert run(repo) == []

    def test_backlinks_json(self, repo: Path) -> None:
        code = wiki_lint.lint(repo / "wiki", repo, write=True)
        assert code == 0
        data = json.loads((repo / "wiki" / "backlinks.json").read_text())
        assert data["pages"] == {"a": ["b"], "b": ["a"]}
        assert data["cites"][MOD] == ["a", "b"]

    def test_no_write_skips_backlinks(self, repo: Path) -> None:
        wiki_lint.lint(repo / "wiki", repo, write=False)
        assert not (repo / "wiki" / "backlinks.json").exists()

    def test_findings_are_rendered_per_slug(
        self, repo: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        write_page(repo, "alpha", "a", page("`pkg.mod.gone` [B](../beta/b.md)", title="A"))
        assert wiki_lint.lint(repo / "wiki", repo, write=False) == 1
        assert capsys.readouterr().out == "a:\n  cited symbol not defined: pkg.mod.gone\n"


class TestLayout:
    def test_duplicate_slug_across_areas(self, repo: Path) -> None:
        write_page(repo, "beta", "a", page("[A](../alpha/a.md)", title="Dup"))
        assert kinds(run(repo), "a") == [Kind.DUPLICATE_SLUG]

    def test_root_index_must_link_area_indexes(self, repo: Path) -> None:
        (repo / "wiki" / "index.md").write_text("- [Alpha](alpha/index.md)\n")
        assert run(repo) == [
            Finding("index", Kind.AREA_NOT_LINKED, "area index not linked: beta/index.md")
        ]

    def test_area_index_requires_frontmatter_keys(self, repo: Path) -> None:
        (repo / "wiki" / "beta" / "index.md").write_text(
            f"---\ntitle: Beta\n---\n{PAGES_START}\n{PAGES_END}\n"
        )
        findings = run(repo)
        assert messages(findings, "beta/index")[0] == "frontmatter missing key: description"

    def test_area_index_requires_markers(self, repo: Path) -> None:
        (repo / "wiki" / "beta" / "index.md").write_text("---\ntitle: B\ndescription: D\n---\n")
        assert kinds(run(repo), "beta/index") == [Kind.INDEX_MARKERS]

    def test_orphan_when_only_area_index_links(self, repo: Path) -> None:
        write_page(repo, "alpha", "c", page("Alone.", title="C"))
        wiki_lint.lint(repo / "wiki", repo, write=False, fix=True)
        assert kinds(run(repo), "c") == [Kind.ORPHAN]

    def test_links_resolve_relative_to_own_directory(self, repo: Path) -> None:
        write_page(repo, "alpha", "a", page("[B](b.md)", title="A"))
        assert kinds(run(repo), "b") == [Kind.ORPHAN]


class TestFrontmatter:
    def test_deprecated_requires_superseded_by(self, repo: Path) -> None:
        write_page(repo, "alpha", "a", page("[B](../beta/b.md)", status="deprecated", title="A"))
        assert kinds(run(repo), "a") == [Kind.SUPERSEDED_MISSING]

    def test_superseded_by_must_exist(self, repo: Path) -> None:
        write_page(
            repo,
            "alpha",
            "a",
            page("[B](../beta/b.md)", status="deprecated", superseded_by="nope", title="A"),
        )
        findings = run(repo)
        assert kinds(findings, "a") == [Kind.SUPERSEDED_UNKNOWN]
        assert "unknown slug: nope" in messages(findings, "a")[0]

    def test_superseded_by_valid(self, repo: Path) -> None:
        write_page(
            repo,
            "alpha",
            "a",
            page("[B](../beta/b.md)", status="deprecated", superseded_by="b", title="A"),
        )
        assert run(repo) == []

    def test_superseded_by_requires_deprecated(self, repo: Path) -> None:
        write_page(repo, "alpha", "a", page("[B](../beta/b.md)", superseded_by="b", title="A"))
        assert kinds(run(repo), "a") == [Kind.SUPERSEDED_NOT_DEPRECATED]

    def test_verified_is_obsolete(self, repo: Path) -> None:
        write_page(repo, "alpha", "a", page("[B](../beta/b.md)", verified="abc1234", title="A"))
        assert [f for f in run(repo) if f.slug == "a"] == [
            Finding("a", Kind.OBSOLETE_KEY, "verified is obsolete; staleness comes from git")
        ]


class TestAreaIndex:
    def test_stale_block_reported(self, repo: Path) -> None:
        (repo / "wiki" / "alpha" / "index.md").write_text(area_index("- old"))
        assert kinds(run(repo), "alpha/index") == [Kind.INDEX_STALE]

    def test_fix_generates_grouped_sorted_block(self, repo: Path) -> None:
        for slug, title, page_type, description in [
            ("z", "Zed", "howto", "Do z"),
            ("y", "apple", "explanation", "Y"),
            ("x", "Banana", "explanation", "X"),
            ("w", "W", "howto", "W"),
        ]:
            write_page(
                repo,
                "alpha",
                slug,
                page("[A](a.md)", title=title, type=page_type, description=description),
            )
        write_page(
            repo,
            "alpha",
            "a",
            page(
                "[B](../beta/b.md) [Z](z.md) [W](w.md) [X](x.md) [Y](y.md)",
                title="A",
                type="reference",
                description="D",
            ),
        )
        assert run(repo, fix=True) == []
        text = (repo / "wiki" / "alpha" / "index.md").read_text()
        block = text.split(PAGES_START)[1].split(PAGES_END)[0]
        assert block.strip() == (
            "### Reference\n\n"
            "- [A](a.md) — D\n\n"
            "### Explanation\n\n"
            "- [apple](y.md) — Y\n"
            "- [Banana](x.md) — X\n\n"
            "### How-to guides\n\n"
            "- [W](w.md) — W\n"
            "- [Zed](z.md) — Do z"
        )
        assert text.startswith("---\ntitle: Area")
        assert "Intro." in text
        assert run(repo) == []


LINK_B = " [B](../beta/b.md)"


def put_a(repo: Path, body: str, sources: str = "  - pkg.mod.calculate") -> None:
    write_page(repo, "alpha", "a", page(body + LINK_B, sources=sources, title="A"))


def write_file(repo: Path, rel: str, text: str) -> None:
    (repo / rel).parent.mkdir(parents=True, exist_ok=True)
    (repo / rel).write_text(text)


def lint_repo(repo: Path) -> wiki_lint.WikiLinter:
    linter = wiki_lint.WikiLinter(repo / "wiki", repo)
    linter.run()
    return linter


class TestCitations:
    def test_leading_dot_slash_is_normalised(self, repo: Path) -> None:
        write_file(repo, "scripts/tool.py", "x = 1\n")
        put_a(repo, "`./scripts/tool.py`", sources="  - ./scripts/tool.py")
        linter = lint_repo(repo)
        assert linter.findings == []
        assert "scripts/tool.py" in linter.cites

    def test_root_and_nested_non_python_files_are_citations(self, repo: Path) -> None:
        write_file(repo, "deployment/aws/Dockerfile.scraper", "FROM x\n")
        put_a(repo, "`pyproject.toml` and `deployment/aws/Dockerfile.scraper`")
        linter = lint_repo(repo)
        assert linter.findings == []
        assert linter.cites["pyproject.toml"] == {"a"}
        assert linter.cites["deployment/aws/Dockerfile.scraper"] == {"a"}

    def test_bad_path_is_error(self, repo: Path) -> None:
        put_a(repo, "`packages/lib/missing.py` and `packages/missing.toml`")
        assert kinds(run(repo), "a") == [Kind.PATH_MISSING, Kind.PATH_MISSING]

    def test_prose_without_slash_or_extension_is_ignored(self, repo: Path) -> None:
        put_a(repo, "`os.path` and `a/b` and `v1.2` and `x.y`")
        assert run(repo) == []

    def test_basename_matching_one_packages_file_must_be_qualified(self, repo: Path) -> None:
        write_file(repo, "packages/core/decision.py", "def decide() -> None: ...\n")
        put_a(repo, "See `decision.py`.")
        assert run(repo) == [
            Finding(
                "a",
                Kind.QUALIFY_PATH,
                "qualify path: decision.py is packages/core/decision.py",
                (line_of(repo, "decision.py"),),
            )
        ]

    def test_ambiguous_basename_is_ignored(self, repo: Path) -> None:
        write_file(repo, "packages/x/util.py", "")
        write_file(repo, "packages/y/util.py", "")
        put_a(repo, "See `util.py`.")
        assert run(repo) == []

    def test_docs_citations_are_errors(self, repo: Path) -> None:
        put_a(
            repo, "`docs/ARCH.md#fetch-tiers` [x](../../docs/ARCH.md)", sources="  - docs/ARCH.md"
        )
        assert sorted(f.kind.name for f in run(repo)) == [
            "DOCS_CITATION",
            "DOCS_CITATION",
            "DOCS_LINK",
        ]

    def test_adr_and_agents_docs_are_allowed(self, repo: Path) -> None:
        put_a(
            repo,
            "`docs/adr/0001-x.md#decision` `docs/agents/domain.md` [d](../../docs/agents/domain.md)",
            sources="  - docs/adr/0001-x.md",
        )
        assert run(repo) == []


class TestSymbolResolution:
    @pytest.mark.parametrize(
        "symbol",
        [
            "LIMIT",
            "TIMEOUT",
            "calculate",
            "Box",
            "Box.run",
            "Box.stop",
            "Box.limit",
            "Box.x",
            "Box.run()",
        ],
    )
    def test_defined_symbols_resolve(self, repo: Path, symbol: str) -> None:
        put_a(repo, f"`pkg.mod.{symbol}`")
        assert run(repo) == []

    def test_module_and_package_citations_resolve_to_the_whole_file(self, repo: Path) -> None:
        put_a(repo, "`pkg.mod` and `pkg`")
        linter = lint_repo(repo)
        assert linter.findings == []
        assert linter.cites[MOD] == {"a", "b"}
        assert {c for c in linter.pages["a"].citations if c.label == "pkg.mod"} == {
            wiki_lint.Citation("pkg.mod", MOD, None)
        }

    def test_reexport_alias_resolves_to_its_definition(self, repo: Path) -> None:
        put_a(repo, "`pkg.Crate` and `pkg.Crate.run`")
        linter = lint_repo(repo)
        assert linter.findings == []
        assert set(linter.pages["a"].citations) >= {
            wiki_lint.Citation("pkg.Crate", MOD, "Box"),
            wiki_lint.Citation("pkg.Crate.run", MOD, "Box.run"),
        }

    @pytest.mark.parametrize(
        "symbol",
        ["Box.nope", "Other.run", "run", "limit", "calculate.inner", "Box.run.inner", "LIMIT.x"],
    )
    def test_undefined_symbols_are_errors(self, repo: Path, symbol: str) -> None:
        put_a(repo, f"`pkg.mod.{symbol}`")
        assert run(repo) == [
            Finding(
                "a",
                Kind.SYMBOL_UNDEFINED,
                f"cited symbol not defined: pkg.mod.{symbol}",
                (line_of(repo, f"pkg.mod.{symbol}`"),),
            )
        ]

    def test_unknown_module_and_unresolved_import_are_errors(self, repo: Path) -> None:
        put_a(repo, "`pkg.nomod.f` and `pkg.mod.datetime`")
        write_file(repo, f"{PKG}/imports.py", "from datetime import datetime\n")
        put_a(repo, "`pkg.nomod.f` and `pkg.imports.datetime`")
        assert kinds(run(repo), "a") == [Kind.SYMBOL_UNDEFINED] * 2

    def test_method_name_defined_elsewhere_does_not_satisfy_a_wrong_class(self, repo: Path) -> None:
        write_file(
            repo, f"{PKG}/two.py", "class A:\n    def go(self) -> None: ...\n\n\nclass B: ...\n"
        )
        put_a(repo, "`pkg.two.B.go`")
        assert kinds(run(repo), "a") == [Kind.SYMBOL_UNDEFINED]

    def test_package_python_file_cited_by_path_must_be_dotted(self, repo: Path) -> None:
        put_a(repo, f"`{MOD}` and `{PKG}/__init__.py`")
        findings = run(repo)
        assert kinds(findings, "a") == [Kind.PYTHON_PATH] * 2
        assert messages(findings, "a") == [
            f"cite as dotted path pkg.mod: {MOD}",
            f"cite as dotted path pkg: {PKG}/__init__.py",
        ]

    def test_non_package_python_file_cited_by_path_is_fine(self, repo: Path) -> None:
        write_file(repo, "scripts/tool.py", "x = 1\n")
        put_a(repo, "`scripts/tool.py`")
        assert run(repo) == []

    def test_double_colon_syntax_is_an_error(self, repo: Path) -> None:
        write_file(repo, "scripts/tool.py", "x = 1\n")
        put_a(repo, f"`{MOD}::Box.run` and `scripts/tool.py::x`", sources="  - pkg.mod.Box")
        findings = run(repo)
        assert kinds(findings, "a") == [Kind.PATH_SYMBOL] * 2
        assert messages(findings, "a")[0] == (
            f"use dotted path, not path::symbol: {MOD}::Box.run -> pkg.mod.Box.run"
        )

    def test_package_roots_are_injectable(self, repo: Path) -> None:
        write_file(repo, "elsewhere/other/__init__.py", "")
        write_file(repo, "elsewhere/other/m.py", "def f() -> None: ...\n")
        src = "  - other.m.f"
        put_a(repo, "`other.m.f`", sources=src)
        write_page(repo, "beta", "b", page("[A](../alpha/a.md)", sources=src, title="B"))
        linter = wiki_lint.WikiLinter(repo / "wiki", repo, package_roots=("elsewhere",))
        linter.run()
        assert linter.errors == []
        assert linter.cites["elsewhere/other/m.py"] == {"a", "b"}

    def test_unparseable_module_cannot_be_cited(self, repo: Path) -> None:
        write_file(repo, f"{PKG}/broken.py", "def (\n")
        put_a(repo, "`pkg.broken.anything`")
        assert kinds(run(repo), "a") == [Kind.SYMBOL_UNDEFINED]


class TestUnqualifiedIdentifiers:
    @pytest.fixture
    def packages(self, repo: Path) -> Path:
        write_file(
            repo,
            f"{PKG}/decision.py",
            (
                "MAX_TIER = 3\nlogger = object()\n\n\ndef decide_forward(limit: int = 1) -> None: ...\n\n\n"
                "class Decider:\n    rate: int = 1\n\n    def pick(self, choice: str) -> None: ...\n"
            ),
        )
        return repo

    def test_function_is_flagged_with_and_without_call_parens(self, packages: Path) -> None:
        put_a(packages, "`decide_forward` and `decide_forward()`")
        expected = "unqualified symbol decide_forward: cite as pkg.decision.decide_forward"
        findings = run(packages)
        assert messages(findings, "a") == [expected]
        assert findings[0].lines == (line_of(packages, "`decide_forward`"),)

    def test_class_member_and_constant_are_flagged(self, packages: Path) -> None:
        put_a(packages, "`Decider.pick()` `Decider.rate` `MAX_TIER` `Box`")
        assert messages(run(packages), "a") == [
            "unqualified symbol Decider.pick: cite as pkg.decision.Decider.pick",
            "unqualified symbol Decider.rate: cite as pkg.decision.Decider.rate",
            "unqualified symbol MAX_TIER: cite as pkg.decision.MAX_TIER",
            "unqualified symbol Box: cite as pkg.mod.Box",
        ]

    def test_name_defined_in_several_modules_is_ambiguous(self, repo: Path) -> None:
        write_file(repo, f"{PKG}/x.py", "def entry() -> None: ...\n")
        write_file(repo, f"{PKG}/y.py", "def entry() -> None: ...\n")
        put_a(repo, "`entry`")
        assert run(repo) == [
            Finding(
                "a",
                Kind.AMBIGUOUS_SYMBOL,
                "ambiguous symbol entry: cite as one of pkg.x.entry, pkg.y.entry",
                (line_of(repo, "`entry`"),),
            )
        ]

    def test_parameters_and_class_fields_are_not_flagged(self, packages: Path) -> None:
        put_a(packages, "`limit` `choice` `rate` `self`")
        assert run(packages) == []

    def test_bare_method_name_is_flagged_with_its_dotted_path(self, packages: Path) -> None:
        put_a(packages, "`pick` and `pick()`")
        assert messages(run(packages), "a") == [
            "unqualified symbol pick: cite as pkg.decision.Decider.pick"
        ]

    def test_method_name_in_several_classes_is_ambiguous(self, packages: Path) -> None:
        put_a(packages, "`run`")
        write_file(packages, f"{PKG}/two.py", "class A:\n    def run(self) -> None: ...\n")
        findings = run(packages)
        assert kinds(findings, "a") == [Kind.AMBIGUOUS_SYMBOL]
        assert "pkg.mod.Box.run, pkg.two.A.run" in messages(findings, "a")[0]

    def test_lowercase_module_variable_is_flagged(self, packages: Path) -> None:
        write_file(packages, f"{PKG}/db.py", "async_session_maker = object()\n")
        put_a(packages, "`async_session_maker`")
        assert messages(run(packages), "a") == [
            "unqualified symbol async_session_maker: cite as pkg.db.async_session_maker"
        ]

    def test_variable_in_conditional_block_is_flagged(self, packages: Path) -> None:
        write_file(
            packages, f"{PKG}/cond.py", "try:\n    engine_pool = 1\nexcept Exception:\n    pass\n"
        )
        put_a(packages, "`engine_pool`")
        assert kinds(run(packages), "a") == [Kind.UNQUALIFIED_SYMBOL]

    def test_generic_words_matching_only_variables_are_not_flagged(self, packages: Path) -> None:
        write_file(packages, f"{PKG}/gen.py", "status = 1\nname = 2\nlogger = 3\n")
        put_a(packages, "`status` `name` `logger`")
        assert run(packages) == []

    def test_generic_word_naming_a_function_is_still_flagged(self, packages: Path) -> None:
        write_file(packages, f"{PKG}/gen.py", "def status() -> None: ...\n")
        put_a(packages, "`status`")
        assert kinds(run(packages), "a") == [Kind.UNQUALIFIED_SYMBOL]

    def test_single_character_and_dunder_names_are_not_flagged(self, packages: Path) -> None:
        write_file(
            packages, f"{PKG}/tiny.py", "x = 1\n\n\nclass K:\n    def __call__(self) -> None: ...\n"
        )
        put_a(packages, "`x` `__call__` `__all__`")
        assert run(packages) == []

    def test_identical_findings_merge_line_numbers(self, packages: Path) -> None:
        put_a(packages, "`decide_forward`\n\n`decide_forward()`\n\n`pkg.mod.nope` `pkg.mod.nope()`")
        findings = run(packages)
        assert [(f.kind, len(f.lines)) for f in findings if f.slug == "a"] == [
            (Kind.UNQUALIFIED_SYMBOL, 2),
            (Kind.SYMBOL_UNDEFINED, 1),
        ]

    def test_builtins_keywords_and_branch_names_are_not_flagged(self, repo: Path) -> None:
        write_file(
            repo, f"{PKG}/shadow.py", "def list() -> None: ...\n\n\ndef main() -> None: ...\n"
        )
        put_a(repo, "`list` `None` `lambda` `main` `master`")
        assert run(repo) == []

    def test_dotted_citation_and_fenced_block_are_fine(self, packages: Path) -> None:
        put_a(
            packages,
            "`pkg.decision.Decider.pick`\n\n```\n`Decider` `decide_forward`\n```\n",
        )
        assert run(packages) == []


def commit(repo: Path, *paths: str, message: str = "change") -> str:
    git(repo, "add", *paths)
    git(repo, "commit", "-q", "-m", message)
    return git(repo, "rev-parse", "--short=7", "HEAD")


def edit_mod(repo: Path, old: str, new: str) -> None:
    mod = repo / MOD
    mod.write_text(mod.read_text().replace(old, new))


def stale(repo: Path) -> tuple[list[Finding], list[Finding]]:
    linter = lint_repo(repo)
    return linter.errors, linter.stale


def line_of(repo: Path, needle: str) -> int:
    lines = (repo / "wiki/alpha/a.md").read_text().splitlines()
    return next(i for i, text in enumerate(lines, 1) if needle in text)


@pytest.fixture
def committed(repo: Path) -> Path:
    """The `repo` fixture with its wiki committed, so pages have a last commit."""
    put_a(repo, "Runs `pkg.mod.Box.run`.")
    commit(repo, "wiki", message="wiki")
    return repo


class TestStaleness:
    def test_fresh_pages_not_stale(self, committed: Path) -> None:
        assert stale(committed) == ([], [])

    def test_uncommitted_pages_are_not_stale_from_history(self, repo: Path) -> None:
        edit_mod(repo, "return 1", "return 10")
        commit(repo, "packages")
        assert stale(repo) == ([], [])

    def test_edit_inside_symbol_is_stale_with_sha_and_page_lines(self, committed: Path) -> None:
        edit_mod(committed, "return 1", "return 10")
        sha = commit(committed, "packages")
        errors, found = stale(committed)
        assert errors == []
        assert found[0] == Finding(
            "a",
            Kind.STALE,
            f"pkg.mod.calculate changed in {sha}",
            (line_of(committed, "  - pkg.mod.calculate"),),
        )
        assert {f.slug for f in found} == {"a", "b"}

    def test_edit_outside_symbol_is_not_stale(self, committed: Path) -> None:
        edit_mod(committed, "return 2", "return 20")
        commit(committed, "packages")
        edit_mod(committed, "LIMIT = 3", "LIMIT = 4")
        commit(committed, "packages")
        assert stale(committed) == ([], [])

    def test_only_commits_touching_the_symbol_are_reported(self, committed: Path) -> None:
        edit_mod(committed, "return 2", "return 20")
        commit(committed, "packages")
        edit_mod(committed, "return 1", "return 10")
        hit = commit(committed, "packages")
        edit_mod(committed, "return 20", "return 200")
        commit(committed, "packages")
        _, found = stale(committed)
        assert found[0].message == f"pkg.mod.calculate changed in {hit}"

    def test_line_shift_above_symbol_is_not_stale(self, committed: Path) -> None:
        edit_mod(committed, "LIMIT = 3\n", "LIMIT = 3\nOTHER = 4\nMORE = 5\n")
        commit(committed, "packages")
        assert stale(committed) == ([], [])

    def test_method_range_resolves_the_method_only(self, committed: Path) -> None:
        edit_mod(committed, "def stop(self) -> None: ...", "def stop(self) -> int: ...")
        commit(committed, "packages")
        assert stale(committed) == ([], [])
        edit_mod(committed, "async def run(self) -> None: ...", "async def run(self) -> int: ...")
        sha = commit(committed, "packages")
        _, found = stale(committed)
        assert found == [
            Finding(
                "a",
                Kind.STALE,
                f"pkg.mod.Box.run changed in {sha}",
                (line_of(committed, "Runs `pkg.mod.Box.run`"),),
            )
        ]

    def test_alias_citation_tracks_the_definition(self, committed: Path) -> None:
        put_a(committed, "`pkg.Crate.run`")
        commit(committed, "wiki", message="cite alias")
        edit_mod(committed, "async def run(self) -> None: ...", "async def run(self) -> int: ...")
        sha = commit(committed, "packages")
        _, found = stale(committed)
        assert [f.message for f in found if f.slug == "a"] == [f"pkg.Crate.run changed in {sha}"]

    def test_class_attribute_edit_is_stale_for_that_attribute_only(self, committed: Path) -> None:
        put_a(committed, "`pkg.mod.Box.limit`")
        commit(committed, "wiki", message="cite attr")
        edit_mod(committed, "x = 1", "x = 2")
        commit(committed, "packages")
        assert stale(committed) == ([], [])
        edit_mod(committed, "limit: int = 5", "limit: int = 6")
        sha = commit(committed, "packages")
        _, found = stale(committed)
        assert [f.message for f in found if f.slug == "a"] == [
            f"pkg.mod.Box.limit changed in {sha}"
        ]

    def test_symbols_in_conditional_blocks_and_init_track_their_ranges(
        self, committed: Path
    ) -> None:
        write_file(
            committed,
            f"{PKG}/cond.py",
            "import sys\n\nif sys.platform == 'x':\n    def f() -> int:\n        return 1\nelse:\n"
            "    def g() -> int:\n        return 2\n\n\nclass K:\n    def __init__(self) -> None:\n"
            "        self.a = 1\n        self.b = 2\n",
        )
        commit(committed, "packages")
        put_a(committed, "`pkg.cond.f` `pkg.cond.g` `pkg.cond.K.a` `pkg.cond.K.b`")
        commit(committed, "wiki", message="cite cond")
        assert stale(committed) == ([], [])
        cond = committed / PKG / "cond.py"
        cond.write_text(
            cond.read_text().replace("return 2", "return 3").replace("self.a = 1", "self.a = 5")
        )
        sha = commit(committed, "packages")
        _, found = stale(committed)
        assert sorted(messages(found, "a")) == [
            f"pkg.cond.K.a changed in {sha}",
            f"pkg.cond.g changed in {sha}",
        ]

    def test_added_or_removed_symbol_is_stale(self, committed: Path) -> None:
        edit_mod(committed, "def other() -> int:\n    return 2\n", "")
        put_a(committed, "`pkg.mod.other`")
        commit(committed, "wiki", message="cite other")
        edit_mod(committed, "LIMIT = 3\n", "LIMIT = 3\n\n\ndef other() -> int:\n    return 2\n")
        sha = commit(committed, "packages")
        _, found = stale(committed)
        assert f"pkg.mod.other changed in {sha}" in messages(found, "a")

    def test_whole_file_citation_is_stale_on_any_change(self, committed: Path) -> None:
        put_a(committed, "Config in `pyproject.toml`.")
        commit(committed, "wiki", message="cite toml")
        (committed / "pyproject.toml").write_text("[project]\nname = 'x'\n")
        sha = commit(committed, "pyproject.toml")
        _, found = stale(committed)
        toml = [f for f in found if f.message.startswith("pyproject.toml")]
        assert toml == [
            Finding(
                "a",
                Kind.STALE,
                f"pyproject.toml changed in {sha}",
                (line_of(committed, "Config in"),),
            )
        ]

    def test_working_tree_change_is_stale_until_page_is_edited(self, committed: Path) -> None:
        edit_mod(committed, "return 1", "return 10")
        _, found = stale(committed)
        assert found[0].message == "pkg.mod.calculate changed in working tree"
        put_a(committed, "Runs `pkg.mod.Box.run`. Rechecked.")
        _, found = stale(committed)
        assert {f.slug for f in found} == {"b"}

    def test_working_tree_change_outside_symbol_is_not_stale(self, committed: Path) -> None:
        edit_mod(committed, "return 2", "return 20")
        assert stale(committed) == ([], [])

    def test_working_tree_whole_file_change(self, committed: Path) -> None:
        put_a(committed, "Config in `pyproject.toml`.")
        commit(committed, "wiki", message="cite toml")
        (committed / "pyproject.toml").write_text("[project]\nname = 'x'\n")
        _, found = stale(committed)
        assert "pyproject.toml changed in working tree" in messages(found, "a")

    def test_commit_and_working_tree_changes_are_both_listed(self, committed: Path) -> None:
        edit_mod(committed, "return 1", "return 10")
        sha = commit(committed, "packages")
        edit_mod(committed, "return 10", "return 11")
        _, found = stale(committed)
        assert found[0].message == f"pkg.mod.calculate changed in {sha}, working tree"

    def test_unparseable_file_version_compares_whole_file(self, committed: Path) -> None:
        edit_mod(committed, "return 2", "return (")
        broke = commit(committed, "packages")
        edit_mod(committed, "return (", "return 2")
        fixed = commit(committed, "packages")
        _, found = stale(committed)
        assert f"pkg.mod.calculate changed in {fixed}, {broke}" in messages(found, "a")

    def test_stale_does_not_fail_unless_requested(self, committed: Path) -> None:
        edit_mod(committed, "return 1", "return 10")
        commit(committed, "packages")
        assert wiki_lint.lint(committed / "wiki", committed, write=False) == 0
        assert wiki_lint.lint(committed / "wiki", committed, write=False, fail_stale=True) == 1

    def test_stale_section_printed(
        self, committed: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        edit_mod(committed, "return 1", "return 10")
        sha = commit(committed, "packages")
        wiki_lint.lint(committed / "wiki", committed, write=False)
        line = line_of(committed, "  - pkg.mod.calculate")
        out = capsys.readouterr().out
        assert out.startswith("Stale pages:\na:\n")
        assert f"  stale: pkg.mod.calculate changed in {sha} (page lines {line})\n" in out

    def test_history_is_queried_once_per_path_and_commit(
        self, committed: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        edit_mod(committed, "return 1", "return 10")
        commit(committed, "packages")
        head = git(committed, "rev-parse", "HEAD")
        write_page(
            committed,
            "beta",
            "b",
            page("[A](../alpha/a.md) `pkg.mod.Box.run`", title="B"),
        )
        calls: list[list[str]] = []
        real_run = subprocess.run

        def spy(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            calls.append(args)
            return real_run(args, **kwargs)  # type: ignore[call-overload,no-any-return]

        monkeypatch.setattr(wiki_lint.subprocess, "run", spy)
        lint_repo(committed)
        commit_diffs = [c for c in calls if "diff" in c and head in c]
        assert len(commit_diffs) == 1


class TestMergeCommits:
    def test_evil_merge_that_edits_the_symbol_is_stale(self, committed: Path) -> None:
        trunk = git(committed, "branch", "--show-current")
        git(committed, "checkout", "-q", "-b", "side")
        write_file(committed, "side.txt", "side\n")
        commit(committed, "side.txt", message="side")
        git(committed, "checkout", "-q", trunk)
        write_file(committed, "trunk.txt", "trunk\n")
        commit(committed, "trunk.txt", message="trunk")
        git(committed, "merge", "-q", "--no-ff", "--no-commit", "side")
        edit_mod(committed, "return 1", "return 10")
        merge = commit(committed, "packages", message="evil merge")
        assert len(git(committed, "rev-list", "--parents", "-n", "1", "HEAD").split()) == 3
        _, found = stale(committed)
        assert found[0].message == f"pkg.mod.calculate changed in {merge}"

    def test_clean_merge_of_unrelated_files_is_not_stale(self, committed: Path) -> None:
        trunk = git(committed, "branch", "--show-current")
        git(committed, "checkout", "-q", "-b", "side")
        write_file(committed, "side.txt", "side\n")
        commit(committed, "side.txt", message="side")
        git(committed, "checkout", "-q", trunk)
        write_file(committed, "trunk.txt", "trunk\n")
        commit(committed, "trunk.txt", message="trunk")
        git(committed, "merge", "-q", "--no-ff", "side", "-m", "merge")
        assert stale(committed) == ([], [])


class TestMissingFiles:
    @pytest.fixture
    def cited_toml(self, committed: Path) -> Path:
        write_file(committed, "notes.toml", "a = 1\n")
        commit(committed, "notes.toml")
        put_a(committed, "Config in `notes.toml` and `Makefile`.")
        write_file(committed, "Makefile", "all:\n")
        commit(committed, "Makefile")
        commit(committed, "wiki", message="cite")
        return committed

    @pytest.fixture
    def cited_script(self, committed: Path) -> Path:
        write_file(committed, "scripts/tool.py", "x = 1\n")
        commit(committed, "scripts")
        put_a(committed, "Runs `scripts/tool.py`.")
        commit(committed, "wiki", message="cite script")
        return committed

    def test_renamed_cited_file_is_error(self, cited_script: Path) -> None:
        git(cited_script, "mv", "scripts/tool.py", "scripts/renamed.py")
        commit(cited_script, "scripts")
        findings, _ = stale(cited_script)
        assert (
            Finding(
                "a",
                Kind.PATH_RENAMED,
                "cited path renamed to scripts/renamed.py: scripts/tool.py",
                (line_of(cited_script, "Runs"),),
            )
            in findings
        )
        assert Kind.PATH_MISSING not in kinds(findings, "a")

    def test_deleted_cited_file_is_error(self, cited_script: Path) -> None:
        git(cited_script, "rm", "-q", "scripts/tool.py")
        git(cited_script, "commit", "-q", "-m", "delete")
        findings, _ = stale(cited_script)
        assert Kind.PATH_DELETED in kinds(findings, "a")

    def test_renamed_module_names_its_new_dotted_path(self, committed: Path) -> None:
        git(committed, "mv", MOD, f"{PKG}/renamed.py")
        commit(committed, "packages")
        findings, _ = stale(committed)
        assert "cited module renamed: pkg.mod.Box.run -> pkg.renamed.Box.run" in messages(
            findings, "a"
        )
        assert Kind.SYMBOL_UNDEFINED not in kinds(findings, "a")

    def test_deleted_module_makes_the_dotted_citation_undefined(self, committed: Path) -> None:
        git(committed, "rm", "-q", MOD)
        git(committed, "commit", "-q", "-m", "delete")
        findings, _ = stale(committed)
        assert Kind.SYMBOL_UNDEFINED in kinds(findings, "a")
        assert Kind.PATH_RENAMED not in kinds(findings, "a")

    def test_deleted_top_level_files_are_errors_without_a_slash(self, cited_toml: Path) -> None:
        git(cited_toml, "rm", "-q", "notes.toml", "Makefile")
        git(cited_toml, "commit", "-q", "-m", "delete")
        findings, _ = stale(cited_toml)
        assert messages(findings, "a") == [
            "cited path was deleted: notes.toml",
            "cited path was deleted: Makefile",
        ]

    def test_renamed_top_level_file_names_its_new_path(self, cited_toml: Path) -> None:
        git(cited_toml, "mv", "notes.toml", "settings.toml")
        git(cited_toml, "commit", "-q", "-m", "rename")
        findings, _ = stale(cited_toml)
        assert "cited path renamed to settings.toml: notes.toml" in messages(findings, "a")

    def test_prose_token_never_tracked_is_ignored(self, committed: Path) -> None:
        put_a(committed, "`ghost.toml` and `Dockerfile`")
        commit(committed, "wiki", message="prose")
        assert stale(committed) == ([], [])


class TestCoverage:
    @pytest.fixture
    def layout(self, repo: Path) -> Path:
        write_file(repo, f"{PKG}/jobs/__init__.py", "")
        write_file(repo, f"{PKG}/jobs/fetch.py", "async def main() -> None: ...\n")
        write_file(repo, f"{PKG}/cli/__init__.py", "")
        write_file(repo, f"{PKG}/cli/scrape.py", "def scrape() -> None: ...\n")
        write_file(
            repo,
            f"{PKG}/mcp/server.py",
            "from x import mcp\n\n\n@mcp.tool()\nasync def get_odds() -> None: ...\n\n\n"
            "@other.tool\ndef get_scores() -> None: ...\n\n\ndef helper() -> None: ...\n",
        )
        write_file(repo, f"{PKG}/mcp/__init__.py", "")
        return repo

    def coverage(self, repo: Path) -> list[Finding]:
        linter = wiki_lint.WikiLinter(
            repo / "wiki",
            repo,
            module_dirs=(f"{PKG}/jobs", f"{PKG}/cli"),
            mcp_dir=f"{PKG}/mcp",
        )
        linter.run()
        return linter.coverage

    def test_uncited_modules_and_tools_are_reported_by_dotted_path(self, layout: Path) -> None:
        found = self.coverage(layout)
        assert [f.kind for f in found] == [
            Kind.COVERAGE_MODULE,
            Kind.COVERAGE_MODULE,
            Kind.COVERAGE_MCP_TOOL,
            Kind.COVERAGE_MCP_TOOL,
        ]
        assert [f.message for f in found] == [
            "module not cited by any page: pkg.jobs.fetch",
            "module not cited by any page: pkg.cli.scrape",
            "MCP tool not cited by any page: pkg.mcp.server.get_odds",
            "MCP tool not cited by any page: pkg.mcp.server.get_scores",
        ]

    def test_cited_modules_and_tool_functions_are_covered(self, layout: Path) -> None:
        put_a(
            layout,
            "`pkg.jobs.fetch.main` `pkg.cli.scrape` `pkg.mcp.server.get_odds`"
            " `pkg.mcp.server.get_scores`",
        )
        assert self.coverage(layout) == []

    def test_citing_the_tool_module_alone_does_not_cover_tools(self, layout: Path) -> None:
        put_a(layout, "`pkg.jobs.fetch` `pkg.cli.scrape` `pkg.mcp.server`")
        assert [f.kind for f in self.coverage(layout)] == [Kind.COVERAGE_MCP_TOOL] * 2

    def test_default_roots_cover_real_layout_and_fail_the_lint(self, repo: Path) -> None:
        write_file(repo, "packages/odds-lambda/odds_lambda/jobs/fetch.py", "")
        assert wiki_lint.lint(repo / "wiki", repo, write=False) == 1
