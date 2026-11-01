# Wiki schema

The wiki is this repo's documentation. It follows [Diátaxis](https://diataxis.fr/): every page is reference, explanation or a how-to. Code is authoritative. Pages describe the code and cite it, and never act as a second copy of values or defaults. Decisions are recorded as ADRs in `docs/adr/`. `README.md` is the entry point for humans.

Readers start at `index.md`, open the area index, then the page, then the code it cites.

## Layout

```
wiki/
├── index.md          # root: what the repo is, one line per area
├── SCHEMA.md
└── <area>/
    ├── index.md      # area intro (hand-written) + page list (generated)
    └── <slug>.md
```

- **Areas** follow the code's subsystems (collection, storage, scheduling, …). Diátaxis structures reference "according to the structure of the machinery itself", and the same applies here.
- **Nesting:** none below area level.
- **Slugs** are lowercase, hyphenated, and unique across the whole wiki.
- **History** lives in git. Pages describe the current state.

### Root index

1. Two sentences on what the repo is.
2. One line per area: `- [Area](area/index.md) — read when …`.
3. A pointer to the hard constraints in `AGENTS.md`, which agents load on every turn.

### Area index

The frontmatter has a `title` and a `description`, followed by a short hand-written intro. The page list goes between `<!-- pages:start -->` and `<!-- pages:end -->`. `scripts/wiki_lint.py --fix` generates that list from page frontmatter, grouped by type, so never edit it by hand.

## Page types

| Type | Diátaxis purpose | Contains | Excludes |
|------|------------------|----------|----------|
| `reference` | Describe the machinery: what it is, what it does, its inputs, outputs, defaults and behaviour | Facts, mirroring one module, job, table or command group | Rationale, instructions, opinion. Link to explanation instead. |
| `explanation` | Discuss a subject: concepts, how parts fit together, why things are the way they are, experiment findings | Context, design reasoning, trade-offs, results | Step-by-step instructions, exhaustive facts. Link to reference instead. |
| `howto` | Get a competent user through one task | Ordered steps, each with a short reason | Background. Link to explanation instead. |

Decisions are not pages. A hard-to-reverse engineering choice with alternatives and consequences is an ADR in `docs/adr/` (Nygard: context, decision, status, consequences). Explanation pages link to it. Strategy (what to try and how candidates are judged) is not an ADR: it lives on [Goal and strategy](foundations/goal-and-strategy.md) and is rewritten whenever it changes.

There are no tutorials. This is a single-developer project, and the how-tos cover onboarding.

One subject per page: one module, job, table, concept or task. Split a page along the code's structure when it covers parts that change independently.

## Frontmatter

```yaml
---
type: reference
title: Fetch-tier classification
description: How hours-to-kickoff maps to a fetch tier and which code paths assign tiers.
sources:
  - odds_lambda.tier_utils.calculate_tier
mentions:
  - odds_lambda.storage.readers.OddsReader
status: stable
---
```

- `description` is one sentence saying which questions the page answers. Agents see it in the area index.
- `sources` lists the code the page as a whole describes.
- `mentions` (optional) lists dotted functions, classes or attributes that the page only names, such as "the job runs inside `odds_core.alerts.job_alert_context`". See [Citations](#citations) and [Staleness](#staleness).
- `status` is `draft`, `stable` or `deprecated`. A deprecated page carries `superseded_by: <slug>` and a single line pointing to its replacement. Delete it once nothing links to it.

## Warnings and known defects

- State behaviour plainly, in prose, where it belongs. For example: "`OddsValidator` never blocks a write, so invalid odds are always stored."
- When the behaviour is a known defect, link its issue (`#438`). The issue tracks the fix. The page states only what the code does today, never a plan to change it.
- Reserve an alert for real danger: irreversible actions, data loss, the production database, quota spend. Use `> [!WARNING]`, and never two in a row.

## Citations

- **Python in the installed packages** (`odds_core`, `odds_lambda`, `odds_analytics`, `odds_mcp`, `odds_cli`): cite by dotted import path in backticks, the convention Sphinx, mkdocstrings and griffe share. Examples: `` `odds_lambda.storage.writers.OddsWriter.store_envelope` ``, and `` `odds_lambda.jobs.fetch_odds` `` for a whole module.
- **Everything else:** cite by repo-relative path. That covers scripts, experiments, Terraform, workflows, `pyproject.toml`, and agent prompts under `packages/odds-mcp/agents/`.
- **Bare identifiers:** an identifier in backticks that names a definition in the packages must be written as its dotted path. Short forms hide the claim from the staleness check.
- **Coverage:** every non-obvious factual claim carries an inline citation or is covered by `sources`. Cite values, defaults and flags rather than restating them bare.
- **Mentions:** a page that only names a symbol, or restates no more than its signature and docstring, lists it under `mentions` and cites it in the body as usual. Any claim about what the symbol does beyond that makes the page describe it, so leave it out of `mentions`.
- **Explanation evidence:** explanation cites its evidence, which can be an ADR, a PR (`#123`), a commit (`abc1234`), or a results file under `experiments/results/`. Reasoning with no evidence is labelled `(inferred)`.
- **Deployed AWS state:** it cannot be verified from the repo. Pages that mention it say so and give the command that checks it.
- **`docs/`:** pages do not cite it, apart from `docs/adr/` and `docs/agents/`.

## Links

Use relative markdown links: `[Text](slug.md)` or `[Text](../area/slug.md#heading)`. The first mention of another page's subject links to it. Every page is listed in its area index and has at least one other inbound link.

## Staleness

A page is stale when code it cites has changed since the page was last changed. Git supplies both sides:

- **Dotted citations:** stale if any commit after the page's last commit touches the symbol's line range, or adds or removes the symbol. A citation of a re-export also tracks each import statement that binds it, so retargeting the alias counts.
- **Mentions:** every citation of a symbol listed in `mentions` tracks only its interface instead of its whole range: a function's decorators, signature and docstring, or a class's decorators, header line and docstring plus the same parts of its `__init__`. A class without `__init__`, such as a dataclass or pydantic model, tracks its annotated fields instead. Edits inside the body, or to any other method, leave the page fresh. A mention whose symbol disappears is an error, like any citation.
- **Squash merges:** PRs are squash-merged, so a page edit counts only if it survives the squash. A page whose content is unchanged since the branch's merge base with `origin/main` dates from its last commit at or before that base, so a page edited and then reverted on a branch is judged as never edited (`--squash-onto`). A page with a net change dates from its last commit as usual, so code changed later on the same branch still makes it stale. On `main` there is no branch to judge. A missing `origin/main` fails the lint in CI and prints a warning elsewhere.
- **Merges:** a merge commit counts only for what it changes relative to every parent, such as a conflict resolution. Changes it brings in from a branch count through that branch's own commits.
- **Path citations:** stale if any later commit changes the file.
- **Uncommitted changes:** these count too, unless the page itself is edited.
- **Renamed or deleted cited files:** these are errors, not staleness.

The lint prints each stale page with the changed citations, whether each is a mention or describes the symbol, and the page lines that cite them. A stale mention means the symbol's interface changed: check that the page still names it correctly.

## Coverage

Every scheduled job module (`odds_lambda.jobs.*`), every CLI command module (`odds_cli.commands.*`) and every MCP tool function must be cited by at least one page.

## Operations

**Query.** Read `index.md`, then the area index, then the page. Skip straight to a page whose slug you already know. When you need certainty or the page is silent, read the cited code: `uv run python scripts/wiki_lint.py --locate <slug or dotted path>` prints each citation's file and line range.

**Update, in the same change as the code.**

1. Run `uv run python scripts/wiki_lint.py`.
2. Edit each stale page so it reads as current. Editing the page clears the flag.
3. Add pages for anything the coverage check reports.
4. Run `uv run python scripts/wiki_lint.py --fix` to regenerate the area indexes.

**Lint.** `uv run python scripts/wiki_lint.py` runs in pre-commit and CI. It checks:

- frontmatter, area index membership, orphan pages, and the generated index blocks;
- that citations resolve (dotted paths via griffe, file paths on disk) and that no bare identifiers remain;
- that no pages cite `docs/` or name the production machine;
- coverage and staleness.

lychee checks links and anchors, and vale checks prose.
