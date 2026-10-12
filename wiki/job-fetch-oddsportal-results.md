---
type: component
title: fetch-oddsportal-results job
summary: A daily ~08:00 UTC scrape of results and closing odds that marks events FINAL and reschedules itself in `finally`.
sources:
  - docs/ARCHITECTURE.md#oddsportal-results-job
  - packages/odds-lambda/odds_lambda/jobs/fetch_oddsportal_results.py::main
  - packages/odds-lambda/odds_lambda/jobs/fetch_oddsportal_results.py::process_results
verified_at: 7887039
---

# fetch-oddsportal-results job

`packages/odds-lambda/odds_lambda/jobs/fetch_oddsportal_results.py::main(ctx)` scrapes OddsPortal's historic results page (one page, the spec's primary market) for past matches, sets events FINAL with scores and stores one closing odds snapshot per event. It runs on the scraper Lambda and in the default local `bootstrap_jobs`, once per sport ([deployment matrix](job-deployment-matrix.md)). League config comes from `LeagueSpec` in [fetch-oddsportal](job-fetch-oddsportal.md); the scraper is [OddsPortal scraper](oddsportal-scraper.md).

## `process_results` phases

Three session scopes, so no connection is held across the browser run (Neon closes idle connections):

1. **Read**: `get_pending_events` returns plain `PendingEvent` dataclasses for SCHEDULED or LIVE events of the sport with `commence_time` in the past. None means exit.
2. **Scrape**: `run_harvester_historic(spec)` with no open session.
3. **Write**: for each record, `_match_record_to_event` matches on exact home and away team strings and a commence-time delta of at most 24 h; `update_event_status(FINAL, scores)`; `build_raw_data(... use_opening=False)` and `store_odds_snapshot` tagged `api_request_id="oddsportal_closing"`; commit per record with rollback on error. Records with no digit scores are skipped.

Rerunning is idempotent because finished events leave the pending query ([event lifecycle](event-lifecycle.md)). `_db_market_for_spec` maps `home_away` to `h2h` and `1x2` to `1x2`, and raises for other primary markets.

## Scheduling

`main` runs `process_results` inside `job_alert_context`, then in `finally` books tomorrow 08:00 UTC through `self_schedule`, ignoring overnight windows. A scheduling failure sends `send_critical` (itself wrapped so it cannot mask the original exception). See [self-scheduling](self-scheduling.md).

## Hazards

- There is no pre-schedule: if the process is killed before `finally` (Lambda timeout, 900 s), the chain breaks until something re-invokes the job.
- `ctx.sport=None` processes `DEFAULT_SPORT_KEY` (`soccer_epl`) but reschedules under the bare job name, which matches no per-sport rule.
- Team matching is exact string equality; an unmatched scrape increments `events_not_matched` and nothing alerts ([event matching](event-matching.md)).
- Heartbeat expectation is 26 h per sport (`heartbeat:fetch-oddsportal-results-<suffix>`), see [health monitor](health-monitor.md).
- The OddsHarvester call here does not take the Playwright semaphore that the upcoming scrape takes.

## Related

- [fetch-scores job](job-fetch-scores.md), [job registry](job-registry.md).
