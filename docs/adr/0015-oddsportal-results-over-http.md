---
status: accepted
---

# Read OddsPortal results over HTTP and stop storing its closing snapshots

The results job reads finished matches from OddsPortal's HTTP results feed, through the client of [ADR-0009](0009-oddsportal-feed-client.md), instead of driving a browser. It no longer stores an OddsPortal closing snapshot.

## Context

ADR-0009 moved the live odds job to the feed and left the results job on OddsHarvester. The results page was rebuilt in the same redesign, so the harvester found no rows and the job settled nothing while reporting success. Production's last MLB result write is 2026-07-13 and its last `op_live_` EPL one 2026-05-25 (#536, #546). The results page names a paginated archive feed that carries team names, kick-off times, a status and the scores.

## Decision

- `fetch-oddsportal-results` settles events from the archive feed through `odds_lambda.oddsportal_client.OddsPortalClient.fetch_results`. Lookback, matching rule and failure guard are on [results-jobs](../../wiki/collection/results-jobs.md#fetch-oddsportal-results).
- The job stores no closing odds. Neither database holds an `oddsportal_closing` snapshot, and the retail-EV close ([ADR-0012](0012-retail-ev-against-betfair-exchange-close.md)) comes from Betfair and FDUK.

## Consequences

- No scheduled job needs a browser or OddsHarvester.
- The archive feed is as undocumented as the odds feed and can change without notice. A change must fail loudly, not look like a quiet day.
- A results page lists the current season only, and cancelled or postponed matches stay pending.

## Evidence

- Issues #536 and #546; the recorded feed page `tests/fixtures/oddsportal/mlb_results_feed.json`.
