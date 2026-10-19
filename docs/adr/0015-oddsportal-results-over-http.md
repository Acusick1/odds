---
status: accepted
---

# Read OddsPortal results over HTTP and settle events from them

The results job reads finished matches from OddsPortal's results page over plain HTTP, through the client of [ADR-0009](0009-oddsportal-feed-client.md), and no longer drives a browser.

## Context

ADR-0009 moved the live odds job to the feed and left the results job on OddsHarvester. The results page was rebuilt in the same redesign. The harvester found no `eventRow` elements, so from about July 2026 the job returned zero matches, exited normally and kept its heartbeat green. Production's last MLB result write is 2026-07-13 and its last `op_live_` EPL one is 2026-05-25, so every `op_live_` event since then stays SCHEDULED or LIVE (#536, #546). The Odds API `/scores` job covers only EPL events that carry an Odds API id.

A read-only check on 2026-10-19 found that the results page's RSC payload names a paginated archive feed, `/proxy/ajax-sport-country-tournament-archive_/{sportId}/{tournamentId}/{hash}/1/0/page/N/`. It is encrypted like the odds feed. Each page holds 50 rows, newest first, with `total` and `onePage`, team names, a kick-off timestamp, a status id (3 finished, 5 cancelled) and `homeResult` / `awayResult`.

## Decision

- `odds_lambda.oddsportal_client.OddsPortalClient.fetch_results` pages through the archive feed back to the oldest pending event.
- `fetch-oddsportal-results` settles pending events (past kick-off and SCHEDULED, LIVE, or FINAL without scores) with the finished matches it finds. A match applies to an event through the shared read-only finder `odds_lambda.storage.event_merge.find_match_id` (normalised team names, within two hours), and only when it is the nearest result of those teams to the event.
- The scheduled job looks back 14 days. `odds fetch results` takes a longer lookback, or none, for backfills.
- The job raises when rows come back, none parse as finished and events older than 24 hours are pending, so a changed page cannot again look like a quiet day.
- The job stores no closing odds. Neither database holds an `oddsportal_closing` snapshot, and the retail-EV close ([ADR-0012](0012-retail-ev-against-betfair-exchange-close.md)) comes from Betfair and FDUK.

## Consequences

- No Chromium and no OddsHarvester in any scheduled job. The dependency, the scraper Dockerfile and `LeagueSpec.sport` can go (tracked in an issue).
- The archive feed is as undocumented as the odds feed and can change without notice. A change fails loudly: an unreadable page raises `OddsPortalError`, and the guard above catches a parser that reads nothing.
- A results page lists the current season only. An event from an earlier season stays pending until another source, such as the FDUK loader, settles it.
- Cancelled and postponed matches are not settled and stay pending.

## Evidence

- Issues #536 and #546; ADR-0009; the recorded feed page in `tests/fixtures/oddsportal/mlb_results_feed.json`.
