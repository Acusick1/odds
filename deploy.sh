#!/usr/bin/env bash
#
# Deploy-with-confidence for the always-on local scheduler and the web UI.
#
# Pulls the latest code, syncs dependencies, applies migrations, smoke-tests
# the bootstrapped jobs, applies the odds_web role's grants and checks them as that role,
# and only then restarts the systemd-managed scheduler and then odds-web. If migration,
# smoke, the grants or the role check fails, the script aborts BEFORE the restarts;
# the running services are untouched on the old code unless a migration was pending,
# in which case both were stopped first and stay down.
#
# When the database is not at the migration head, both services are stopped before
# the migration runs, because a migration may drop or rewrite what the running code
# reads, and the database is dumped (scripts/backup-prod-db.sh) before migrating.
# Every failure after that stop says which services are down.
#
# After each restart it checks that the service is active, started after this
# deploy began, and is not crash-looping, and exits non-zero otherwise. The scheduler
# is restarted and verified first; a web failure fails the deploy but never stops or
# restarts the scheduler, and the failure message reports both services' states.
#
# Run from the repo root on the host running odds-scheduler.service and odds-web.service:
#   ./deploy.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# A non-interactive shell (`ssh host ./deploy.sh`) does not read the rc file
# that puts the uv installer's directory on PATH. Resolve every tool before the
# pull so a missing one cannot leave the checkout ahead of the running scheduler.
export PATH="$HOME/.local/bin:$PATH"
for tool in git uv sudo systemctl; do
    if ! command -v "$tool" >/dev/null; then
        echo "deploy.sh: $tool not found on PATH ($PATH); nothing was changed" >&2
        exit 1
    fi
done

SERVICE=odds-scheduler
WEB_SERVICE=odds-web
SYSTEMCTL="$(command -v systemctl)"

# The web unit is installed by hand (docs: wiki/operations/howto-set-up-odds-web-host.md).
if ! "$SYSTEMCTL" cat "$WEB_SERVICE.service" >/dev/null 2>&1; then
    echo "deploy.sh: $WEB_SERVICE.service is not installed; nothing was changed" >&2
    exit 1
fi

# The restart makes new code serve, and the stop precedes a pending migration.
# Without a terminal sudo cannot prompt, so require every NOPASSWD rule before
# changing anything.
if [[ ! -t 0 ]]; then
    # One allowed command per line, so a rule must equal the whole command: a rule for
    # 'stop odds-web-other' does not allow 'stop odds-web'.
    allowed_commands="$(sudo -n -l | grep 'NOPASSWD:' \
        | tr ',' '\n' \
        | sed -E 's/^[[:space:]]*(\([^)]*\)[[:space:]]*)?(NOPASSWD:[[:space:]]*)?//; s/[[:space:]]+$//')"
    for service in "$SERVICE" "$WEB_SERVICE"; do
        for action in restart stop; do
            if ! grep -qxF "$SYSTEMCTL $action $service" <<<"$allowed_commands"; then
                echo "deploy.sh: no NOPASSWD sudo rule for '$SYSTEMCTL $action $service' and no terminal to prompt; nothing was changed" >&2
                exit 1
            fi
        done
    done
fi

scheduler_down=0
web_down=0
migration_committed=0
grants_applied=0
previous_head=""
previous_revision=""
report_services_down() {
    local status=$? stopped=()
    (( scheduler_down == 1 )) && stopped+=("$SERVICE")
    (( web_down == 1 )) && stopped+=("$WEB_SERVICE")
    if (( status != 0 && scheduler_down == 0 && web_down == 1 )); then
        # Only reachable after the scheduler restarted and was verified on the new code
        # and schema: downgrading or checking out old code would break the running scheduler.
        {
            echo "deploy.sh: FAILED with $WEB_SERVICE STOPPED."
            echo "$SERVICE is running on the new code and schema: leave it alone, do not downgrade the database or check out old code."
            echo "Fix the web service: journalctl -u $WEB_SERVICE, then sudo $SYSTEMCTL restart $WEB_SERVICE"
            echo "Or rerun ./deploy.sh."
        } >&2
    elif (( status != 0 && ${#stopped[@]} > 0 )); then
        {
            echo "deploy.sh: FAILED with ${stopped[*]} STOPPED."
            echo "To bring the OLD code back (do not just start the services: the checkout holds the new code):"
            if (( migration_committed == 1 )); then
                echo "  uv run alembic downgrade $previous_revision   # run first: the downgrade lives in the new code"
                echo "  (rows the migration deleted do not come back; restore the pre-deploy dump in \${ODDS_BACKUP_DIR:-~/backups} for those)"
            fi
            if (( grants_applied == 1 )); then
                echo "  git checkout $previous_head && uv sync"
                echo "  uv run python -m odds_web.grants   # on the old code: re-grants the tables it reads (skip if it predates odds_web.grants)"
                echo "  sudo $SYSTEMCTL start $SERVICE $WEB_SERVICE"
            else
                echo "  git checkout $previous_head && uv sync && sudo $SYSTEMCTL start $SERVICE $WEB_SERVICE"
            fi
            echo "Or fix the failure and rerun ./deploy.sh."
        } >&2
    fi
}
trap report_services_down EXIT

previous_head="$(git rev-parse HEAD)"
deploy_started_at="$(date +%s)"

echo "==> git pull"
git pull --ff-only

echo "==> uv sync"
uv sync

# Migration ordering: a pending migration runs with the scheduler stopped, so the
# old code never reads a schema it does not know. The hand-run agent's MCP server
# is not managed here: do not deploy during an agent session.
if ! current_output="$(uv run alembic current)"; then
    echo "deploy.sh: 'alembic current' failed, so whether a migration is pending is unknown; $SERVICE was not stopped" >&2
    exit 1
fi
if ! grep -q "(head)" <<<"$current_output"; then
    previous_revision="$(grep -oE '^[0-9a-z]+' <<<"$current_output" | head -1)"
    echo "==> migration pending (database at '${previous_revision:-no revision}'): stopping $SERVICE and $WEB_SERVICE"
    sudo "$SYSTEMCTL" stop "$SERVICE"
    scheduler_down=1
    sudo "$SYSTEMCTL" stop "$WEB_SERVICE"
    web_down=1
    # Dump with the scheduler stopped, so the archive is the exact state the
    # migration starts from. A failed dump aborts before the migration.
    echo "==> backing up the database"
    ./scripts/backup-prod-db.sh "pre-deploy-$(git rev-parse --short HEAD)"
fi
echo "==> alembic upgrade head"
uv run alembic upgrade head
migration_committed=1

# Smoke runs with SCHEDULER_DRY_RUN=true so self-scheduling and alerting are
# no-ops: jobs do their real work (gated by their cadence) but write nothing to
# the live schedule store and post nothing to Discord. The running scheduler
# keeps serving old code until smoke passes. Jobs that spend Odds API quota
# (fetch-odds, fetch-scores) are skipped by default so a deploy costs no units.
echo "==> scheduler smoke (SCHEDULER_DRY_RUN=true)"
SCHEDULER_DRY_RUN=true uv run odds scheduler smoke

# The odds_web role reads only the tables in odds_web.role_check.READ_TABLES. Grant them over
# DATABASE_URL (the superuser), then connect as odds_web over WEB_DATABASE_URL, so a migration
# that adds a table the UI reads cannot reach a restart with the role unable to read it.
echo "==> odds_web grants"
uv run python -m odds_web.grants
grants_applied=1
echo "==> odds_web role check"
uv run python -m odds_web.role_check

# Only reached when migrate, smoke, the grants and the role check all succeed (set -e aborts otherwise).
service_states() {
    echo "  $SERVICE: $(systemctl is-active "$SERVICE" 2>/dev/null || true)" >&2
    echo "  $WEB_SERVICE: $(systemctl is-active "$WEB_SERVICE" 2>/dev/null || true)" >&2
}
fail() {
    echo "deploy.sh: $1" >&2
    systemctl status "$2" --no-pager >&2 || true
    echo "deploy.sh: service states:" >&2
    service_states
    exit 1
}

# restart_and_verify <service>: restart, then check it is active, started after this deploy
# began, and not crash-looping.
restart_and_verify() {
    local service="$1" restarts_before restarts_after active_since
    echo "==> restarting $service"
    sudo "$SYSTEMCTL" restart "$service" || fail "systemctl restart $service failed" "$service"
    if [[ "$service" == "$SERVICE" ]]; then scheduler_down=0; else web_down=0; fi

    echo "==> verifying $service restart"
    restarts_before="$(systemctl show "$service" -p NRestarts --value)"
    sleep 15
    systemctl is-active --quiet "$service" || fail "$service is not active after the restart" "$service"
    active_since="$(systemctl show "$service" --timestamp=unix -p ActiveEnterTimestamp --value)"
    active_since="${active_since#@}"
    [[ "$active_since" =~ ^[0-9]+$ ]] || fail "could not read $service's ActiveEnterTimestamp ('$active_since')" "$service"
    (( active_since >= deploy_started_at )) \
        || fail "$service has been active since $active_since (unix), before this deploy began at $deploy_started_at; it is still on the old code" "$service"
    restarts_after="$(systemctl show "$service" -p NRestarts --value)"
    (( restarts_after == restarts_before )) \
        || fail "$service restarted $((restarts_after - restarts_before)) time(s) within 15 s of the deploy; it is crash-looping" "$service"
    systemctl status "$service" --no-pager || true
}

restart_and_verify "$SERVICE"

# The scheduler is serving the new code; a web failure from here on fails the deploy but
# leaves the scheduler alone.
restart_and_verify "$WEB_SERVICE"

echo "==> scheduled jobs"
uv run odds scheduler list-jobs

echo "==> deploy complete"
