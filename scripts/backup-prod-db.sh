#!/usr/bin/env bash
# Dumps the production database before anything writes to it.
#
# Usage (repo root, on the production host):
#   ./scripts/backup-prod-db.sh <label>
#
# Writes a pg_dump custom-format archive of the compose `postgres` service's
# `odds` database to $ODDS_BACKUP_DIR (default ~/backups) as
# odds-<label>-<UTC timestamp>.dump, checks that pg_restore can read it, and
# prints its path. The label names what is about to write, e.g. pre-712 for a
# PR's migration or pre-deploy-<sha>.
#
# deploy.sh runs this before every pending migration. Run it by hand before any
# other write to production (data fixes, backfills, manual SQL).

set -euo pipefail

if [[ $# -ne 1 || -z "$1" ]]; then
    echo "usage: $0 <label>" >&2
    exit 2
fi

cd "$(dirname "${BASH_SOURCE[0]}")/.."

backup_dir="${ODDS_BACKUP_DIR:-$HOME/backups}"
mkdir -p "$backup_dir"
target="$backup_dir/odds-$1-$(date -u +%Y%m%dT%H%MZ).dump"
partial="$target.partial"

trap 'rm -f "$partial"' EXIT

docker compose exec -T postgres pg_dump -U postgres -Fc odds >"$partial"
docker compose exec -T postgres pg_restore --list <"$partial" >/dev/null
mv "$partial" "$target"

echo "backup-prod-db.sh: wrote $target ($(du -h "$target" | cut -f1))"
