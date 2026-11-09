-- Grants for the read-only `odds_web` role (the web UI, packages/odds-web).
--
-- Run as a superuser after `CREATE ROLE odds_web WITH LOGIN PASSWORD '...'`, in the database
-- the role reads. Safe to re-run. Only SELECT on the tables the UI's readers use: no
-- pg_read_all_data, and no default privileges, so a table added later stays unreadable until
-- it is listed here. The APScheduler tables exist only after the scheduler has booted once.

-- Role-level defaults, beside the app's own per-transaction read-only and statement timeout.
ALTER ROLE odds_web SET default_transaction_read_only = on;

GRANT USAGE ON SCHEMA public TO odds_web;

GRANT SELECT ON
    events,
    odds_snapshots,
    prices,
    paper_trades,
    candidate_decisions,
    fetch_logs,
    alert_history,
    schedules,
    jobs,
    job_results,
    tasks
TO odds_web;

-- The status strip's freshness check takes the MLB fixture calendar from the MLB Stats API
-- schedule (odds_lambda.collection_calendar.fixture_calendar).
GRANT SELECT ON mlb_probable_pitchers TO odds_web;
