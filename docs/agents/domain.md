# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

This is a single-context repo.

- **[`wiki/foundations/glossary.md`](../../wiki/foundations/glossary.md)**: the definition of every domain term and the words to avoid for it.
- **`wiki/index.md`**: the area index for the code you're about to touch, and its pages.
- **`docs/adr/`**: read ADRs that touch the area you're about to work in.

The glossary takes the place of the `GLOSSARY.md` the `/domain-modeling` skill describes, and it is the only place a term is defined: other pages link to its `#term` anchor instead. When a term is resolved, add or edit it there: a `###` heading, a definition of one or two sentences ending with a link to the page describing the machinery behind it, and an `_Avoid_` line of the synonyms to drop. Never create a root `GLOSSARY.md`.

## Use the wiki's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name), use the glossary's term. Don't use the words it lists under `_Avoid_` in prose; code identifiers keep their own names.

If the concept you need is not in the glossary, that's a signal: either you're inventing language the project doesn't use (reconsider) or there's a real gap (add the term).

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-0007 (event-sourced orders), but worth reopening because…_
