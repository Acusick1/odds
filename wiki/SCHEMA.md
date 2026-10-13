# Wiki schema

The wiki is this repo's documentation. Its pages explain what each subsystem does, why it does it, and where its hazards are, and they cite the code that does it. Code is authoritative: the wiki points into the code and never restates it as a second source. Readers start at `index.md`, open one area index, then the page, then the cited code. `README.md` is the entry point for humans, and decisions live in `docs/adr/`.

## Layout

```
wiki/
├── index.md          # root: hand-written, ≤500 words
├── SCHEMA.md
└── <area>/
    ├── index.md      # area intro (hand-written) + page list (generated)
    └── <slug>.md
```

- One directory per area. Areas follow the code's subsystems (for example ingestion, scheduling, training). Decisions, experiments and how-tos live in the area they concern.
- No nesting below area level.
- Slugs are lowercase, hyphenated and unique across the whole wiki.
- History is in git. Pages describe the current state and never narrate changes.

### Root index

The root `index.md` has three parts:

1. Two sentences on what the repo is.
2. One line per area: `- [Area](area/index.md) — read when …`.
3. The handful of constraints an agent must not violate (timezones, production database, API cost), each linking to its page.

### Area index

Each area `index.md` has YAML frontmatter with `title` and `description`, a short hand-written intro, and the page list between `<!-- pages:start -->` and `<!-- pages:end -->`. `scripts/wiki_lint.py --fix` generates that list from page frontmatter as `- [Title](slug.md) — description`, grouped by type. Never edit the generated block by hand.

## Pages

### Granularity

A page covers one subject a developer changes or reasons about as a unit: a job, a pipeline, a table and its writer, a model-training stage, a decision. Someone asking about that subject should find the answer on that one page, including its hazards and the conventions that touch it, without needing a sibling page. Target 400–1200 words. Split a page only when it passes ~1500 words or covers things that change independently. Repeat a one-line fact rather than force a hop, and link for depth.

### Types

| Type | Covers | Examples |
|------|--------|----------|
| `component` | a code unit and its behaviour | a job, the decision module, the odds writer and its tables |
| `concept` | a term used across areas, with its single definition | fetch tier, CLV, sharp benchmark |
| `process` | a flow spanning components | odds ingestion, self-scheduling, matchday wake-up |
| `decision` | a choice, its rationale and evidence | bet365 target, walk-forward CV |
| `experiment` | an experiment and its result | feature-group ablation |
| `howto` | an operator task | deploy, add a scheduled job |

### Frontmatter

```yaml
---
type: component
title: Fetch-tier classification
description: How hours-to-kickoff maps to a fetch tier, which code paths assign tiers, and where they disagree.
sources:
  - packages/odds-lambda/odds_lambda/tier_utils.py::calculate_tier
status: stable
---
```

- `description` is one sentence saying what questions the page answers. Agents see it in the area index, so write it to help them choose whether to open the page.
- `sources` lists the code the page as a whole describes. Inline citations cover individual claims.
- `status` is `draft`, `stable` or `deprecated`. A deprecated page carries `superseded_by: <slug>` and a one-line body pointing to its replacement. Delete it once nothing links to it.

### Body

Lead with what the subject is and where it lives. Follow with its behaviour, then `## Hazards` (divergences, silent failures and traps, each cited), then `## Related` links.

## Citations

- Cite Python as `` `path/from/repo/root.py::symbol` `` (dotted for methods: `Class.method`). Cite other files by path: Terraform, workflows, shell scripts, `pyproject.toml`. Never cite by line number.
- An identifier in backticks that names a definition in `packages/` must carry its path. A bare `` `decide_forward` `` hides the claim from the staleness check.
- A hazard that spans two files cites both.
- A claim about CLI behaviour cites the command's function.
- Every non-obvious factual claim carries an inline citation or is covered by `sources`. Values, defaults and flags are cited, never restated bare.
- Rationale cites its evidence: an ADR, a PR (`#123`), a commit (`abc1234`, for evidence that only survives in history), an experiment page, or a results file under `experiments/results/`. Rationale with no evidence is labelled `(inferred)`.
- Deployed AWS state cannot be verified from the repo. Pages that mention it say so and give the command that checks it.
- Pages do not cite `docs/`. The only exceptions are `docs/adr/` and `docs/agents/`.

## Links

- Use relative markdown links (`[Text](slug.md)`, `[Text](../area/slug.md#heading)`). No wikilink syntax.
- The first mention of another page's subject in a body links to that page.
- Every page is listed in its area index and has at least one other inbound link.

## Staleness

A page is stale when code it cites has changed since the page was last changed. Git supplies both sides, so pages carry no verification stamp:

- For a `path::symbol` citation, the check looks for any commit after the page's last commit that touches the symbol's line range or adds or removes the symbol.
- For a whole-file citation, any change to the file after the page's last commit counts.
- Uncommitted changes count too. A cited symbol edited in the working tree marks the page stale unless the page is edited as well, so the check works as a pre-commit hook.
- A renamed or deleted cited file is an error, not staleness.

The lint prints each stale page with the changed symbols and the page lines that cite them.

## Coverage

Every scheduled job module (`packages/odds-lambda/odds_lambda/jobs/*.py`), every CLI command module (`packages/odds-cli/odds_cli/commands/*.py`) and every MCP tool function must be cited by at least one page, so a new job, command or tool fails the lint until it is documented.

## Operations

**Query.** Read `index.md`, then the matching area index, then the page, then the cited code when you need certainty or the page is silent.

**Update (in the change that touches code).**

1. Run `uv run python scripts/wiki_lint.py`.
2. For each stale page, re-check the listed claims against the new code and edit the page so it reads as current. Editing the page clears the flag.
3. Add pages for anything the coverage check reports.
4. Run `uv run python scripts/wiki_lint.py --fix` to regenerate the area indexes.

**Lint.** `uv run python scripts/wiki_lint.py` runs in pre-commit and CI. It checks:

- frontmatter;
- links and anchors;
- that cited paths exist and cited symbols are defined;
- that no identifiers are unqualified;
- that no pages cite `docs/`;
- area index membership and orphan pages;
- that the generated index blocks are current;
- coverage;
- staleness.

It also writes `backlinks.json` (gitignored), which maps each page to its inbound pages and each code path to the pages that cite it.
