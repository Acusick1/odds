---
status: accepted
---

# A code-citing wiki replaces docs/

`wiki/` replaces `docs/` and the reference content in `README.md` and `AGENTS.md`. The long docs drifted out of step with the code, and nothing showed which parts were still true: a 2026-06 comprehension audit and the build of this wiki together found more than 80 doc claims the code contradicts (#434–#454). The wiki follows Diátaxis (reference, explanation and how-to pages, each on one subject), and its pages cite the code they describe by dotted Python path, the convention Sphinx, mkdocstrings and griffe share. `scripts/wiki_lint.py` runs in pre-commit and CI and derives freshness from git: a page is stale when a symbol it cites changes after the page was last edited, and a new job module, CLI command module or MCP tool fails the lint until a page cites it. No existing tool covers these checks: docrot (alpha) resolves symbols only, and Swimm is commercial and requires its own document format. ADRs stay in `docs/adr/`, and agent tooling conventions stay in `docs/agents/`.

## Considered options

- **Keep `docs/` and add a code-verification pass.** This would have fixed today's drift but given no signal when the code changed next. The page sizes also prevented symbol-level staleness checks.
- **A graph index such as Neo4j `ki`.** At about 60 pages, generated area indexes plus `backlinks.json` provide navigation and impact lookup without running a database.

## Consequences

- Evaluation numbers come from one run each, with a single grader, so they are a small sample:
  - Questions with code access: the wiki scored 0.93 against 0.86 for the docs and 0.70 for code alone.
  - Five change tasks: rubric 0.94 against 0.90, and quality 4.6 against 4.2.
  - Token cost per question was the same in both conditions.
- The main measured gain came from having verified documentation at all. The wiki's own advantage was small.
- In a replay over 10 PRs, about 3 pages per PR needed edits.
- A change to a cited symbol blocks the commit until the citing page is edited, and that cost is accepted.

## Evidence

- Prior `docs/` text: `git show origin/main:docs/` (replaced in 9b3ae09).
- Comprehension audit findings: #434-#454.
- docrot: <https://pypi.org/project/docrot/>, <https://github.com/dager23/docrot>.
- Swimm: <https://docs.swimm.io/continuous-integration/github-app>, <https://swimm.io/pricing>. Both tools researched 2026-10-13.
