#!/usr/bin/env bash
# Restores a dump into a throwaway Postgres container and compares its row
# counts with production's.
#
# Usage (repo root, on the production host):
#   ./scripts/restore-check.sh <file>.dump
#
# The scratch container runs the compose `postgres` service's image, publishes
# no port and is removed on exit; production is only read. Production keeps
# collecting after the dump is taken, so its counts can be higher than the
# dump's by the rows written since.

set -euo pipefail

if [[ $# -ne 1 || ! -f "$1" ]]; then
    echo "usage: $0 <file>.dump" >&2
    exit 2
fi

dump="$1"
cd "$(dirname "${BASH_SOURCE[0]}")/.."

tables=(events odds_snapshots prices candidate_decisions paper_trades)
query=""
for table in "${tables[@]}"; do
    query+="${query:+ union all }select '$table', count(*) from $table"
done

image="$(docker compose config --images postgres)"
container="odds-restore-check-$$"
trap 'docker rm -f -v "$container" >/dev/null 2>&1 || true' EXIT

docker run -d --name "$container" -e POSTGRES_PASSWORD=scratch -e POSTGRES_DB=odds "$image" >/dev/null
# The entrypoint runs a temporary server while it initialises, so wait for the final one.
deadline=$((SECONDS + 120))
until docker logs "$container" 2>&1 | grep -q 'PostgreSQL init process complete' &&
    docker exec "$container" pg_isready -U postgres -d odds >/dev/null 2>&1; do
    if [[ "$(docker inspect -f '{{.State.Running}}' "$container")" != true || $SECONDS -ge $deadline ]]; then
        echo "restore-check.sh: scratch Postgres did not start" >&2
        docker logs --tail 20 "$container" >&2
        exit 1
    fi
    sleep 1
done

docker exec -i "$container" pg_restore -U postgres -d odds --no-owner --no-acl --exit-on-error <"$dump"

restored="$(docker exec "$container" psql -U postgres -d odds -AtF' ' -c "$query")"
production="$(docker compose exec -T postgres psql -U postgres -d odds -AtF' ' -c "$query")"

printf '%-20s %12s %12s\n' table restored production
join <(sort <<<"$restored") <(sort <<<"$production") | while read -r table r p; do
    printf '%-20s %12s %12s\n' "$table" "$r" "$p"
done
