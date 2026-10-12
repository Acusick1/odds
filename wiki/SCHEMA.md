# Wiki schema

The wiki is the agent-facing map of this repo: pages that explain what each subsystem does, why, and where its hazards are, and cite the code that does it. Code is authoritative; the wiki points into it and never replaces it. Agents read `index.md`, open one area index, then the page, then the cited code.

## Layout

```
wiki/
├── index.md          # root: hand-written, ≤500 words
├── log.md            # append-only change log
├── SCHEMA.md
└── <area>/
    ├── index.md      # area intro (hand-written) + page list (generated)
    └── <slug>.md
```

- One directory per area. Areas follow the code's subsystems (for example ingestion, scheduling, training), not the old docs' boundaries. Decisions, experiments and how-tos live in the area they concern.
- No nesting below area level.
- Slugs are lowercase, hyphenated and unique across the whole wiki.

### Root index

The root `index.md` has three parts:

1. Two sentences on what the repo is.
2. One line per area: `- [Area](area/index.md) — read when …`.
3. The handful of constraints an agent must not violate (timezones, production database, API cost), each linking to its page.

### Area index

Each area `index.md` has YAML frontmatter with `title` and `description`, a short hand-written intro, and the page list between `<!-- pages:start -->` and `<!-- pages:end -->`. `scripts/wiki_lint.py --fix` generates that list from page frontmatter as `- [Title](slug.md) — description`, grouped by type. Never edit the generated block by hand.

## Pages

### Granularity

A page covers one subject a developer changes or reasons about as a unit: a job, a pipeline, a table and its writer, a model-training stage, a decision. Someone asking about that subject should find the answer on that one page, including its hazards and the conventions that touch it, without needing a sibling page. Target 400–1200 words. Split only when a page passes ~1500 words or covers things that change independently. Repeat a one-line fact rather than force a hop; link for depth.

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
  - docs/ARCHITECTURE.md#fetch-tiers
verified: 7887039
status: stable
---
```

- `description` is one sentence saying what questions the page answers. It is what agents see in the area index, so write it for choosing whether to open the page.
- `verified` is the short commit SHA at which every claim was last checked against the code.
- `status` is `draft`, `stable` or `deprecated`. A deprecated page carries `superseded_by: <slug>` and a one-line body pointing to it. Delete it once nothing links to it.

### Body

Lead with what it is and where it lives. Follow with behaviour, then `## Hazards` (divergences, silent failures and traps, each cited), then `## Related` links.

## Citations

- Cite code as `` `path/from/repo/root.py::symbol` ``. Use the symbol, never a line number. A bare path is allowed for whole files and for non-Python files.
- Every non-obvious factual claim carries a citation inline or is covered by `sources`. Values, defaults and flags are cited, never restated bare.
- Rationale needs a source (a PR, an ADR, an experiment page or a doc). Rationale without one is labelled `(inferred)`.
- Deployed AWS state cannot be verified from the repo. Pages that mention it say so and give the command that checks it.
- When docs and code disagree, the page states what the code does and records the disagreement under `## Hazards`.

## Links

- Use relative markdown links (`[Text](slug.md)`, `[Text](../area/slug.md#heading)`). No wikilink syntax.
- The first mention of another page's subject in a body links to that page.
- Every page is listed in its area index and has at least one other inbound link.

## Operations

**Query.** Read `index.md`, then the matching area index, then the page, then the cited code when you need certainty or the page is silent.

**Update (in the PR that changes code).** `scripts/wiki_lint.py` flags every page whose cited files changed since its `verified` commit.

1. For each flagged page, re-check the claims that cite the changed files.
2. Edit the page so it reads as current.
3. Set `verified` to the PR's base commit or later.
4. Create pages for new subjects.
5. Run `scripts/wiki_lint.py --fix` to regenerate the area indexes.
6. Append a line to `log.md`.

**Lint** (`uv run python scripts/wiki_lint.py`). It checks:

- frontmatter fields and values;
- links and anchors;
- that cited paths exist and cited symbols are defined;
- that every page is listed in its area index and none are orphans;
- that the generated index blocks are current;
- staleness (cited paths changed since `verified`).

It also writes `backlinks.json`, which maps each page to its inbound pages and each code path to the pages that cite it.
