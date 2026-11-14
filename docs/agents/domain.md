# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

This is a single-context repo.

- **[`wiki/foundations/glossary.md`](../../wiki/foundations/glossary.md)**: the domain vocabulary, one line per term, each linking to the page that defines it.
- **`wiki/index.md`**: the area index for the code you're about to touch, and its concept pages.
- **`docs/adr/`**: read ADRs that touch the area you're about to work in.

The glossary takes the place of the `GLOSSARY.md` the `/domain-modeling` skill describes. When a term is resolved, add or edit its line there: the term, a one- or two-sentence definition on the same line ending with a link to the page that defines it, and an `_Avoid_` line of synonyms. Code strings such as market keys may appear as examples; the definition itself stays free of implementation detail. Never create a root `GLOSSARY.md`.

## Use the wiki's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name), use the glossary's term. Don't use the words it lists under `_Avoid_`.

If the concept you need is not in the glossary, that's a signal: either you're inventing language the project doesn't use (reconsider) or there's a real gap (add the term).

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-0007 (event-sourced orders), but worth reopening because…_
