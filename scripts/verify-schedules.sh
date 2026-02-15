#!/usr/bin/env bash
# Verifies that all self-scheduling EventBridge rules are active.
#
# Usage:
#   ./scripts/verify-schedules.sh --rule-prefix <prefix> [--region <region>]
#
# Checks for each rule:
#   - Rule exists
#   - State is ENABLED
#   - Schedule is a cron expression (not a placeholder rate)
#   - Cron year (if present) is not in the past

set -euo pipefail

RULE_PREFIX=""
REGION="${AWS_REGION:-eu-west-1}"

JOBS=("fetch-odds" "fetch-scores" "update-status" "check-health" "fetch-polymarket")

usage() {
  echo "Usage: $0 --rule-prefix <prefix> [--region <region>]"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rule-prefix) RULE_PREFIX="$2"; shift 2 ;;
    --region)      REGION="$2"; shift 2 ;;
    *) usage ;;
  esac
done

[[ -z "$RULE_PREFIX" ]] && usage

echo "==> Verifying dynamic rule states (prefix=$RULE_PREFIX, region=$REGION)..."
FAILED=0

for job in "${JOBS[@]}"; do
  rule_name="${RULE_PREFIX}-${job}"
  echo -n "    $rule_name: "

  rule_json=$(aws events describe-rule \
    --name "$rule_name" \
    --region "$REGION" \
    --output json 2>&1) || {
    echo "FAIL (rule not found)"
    FAILED=$((FAILED + 1))
    continue
  }

  state=$(echo "$rule_json" | python3 -c "import json,sys; print(json.load(sys.stdin)['State'])")
  schedule=$(echo "$rule_json" | python3 -c "import json,sys; print(json.load(sys.stdin)['ScheduleExpression'])")

  if [[ "$state" != "ENABLED" ]]; then
    echo "FAIL (State=$state, expected ENABLED)"
    FAILED=$((FAILED + 1))
    continue
  fi

  if [[ "$schedule" == rate* ]]; then
    echo "FAIL (Schedule='$schedule' — rule not self-scheduled)"
    FAILED=$((FAILED + 1))
    continue
  fi

  if [[ "$schedule" =~ cron\(.*([0-9]{4})\) ]]; then
    cron_year="${BASH_REMATCH[1]}"
    current_year=$(date -u +%Y)
    if [[ "$cron_year" -lt "$current_year" ]]; then
      echo "FAIL (Schedule='$schedule' — cron year $cron_year is in the past)"
      FAILED=$((FAILED + 1))
      continue
    fi
  fi

  echo "OK (State=$state, Schedule=$schedule)"
done

if [[ "$FAILED" -gt 0 ]]; then
  echo ""
  echo "ERROR: $FAILED rule(s) are stale or inactive."
  exit 1
fi

echo ""
echo "All ${#JOBS[@]} dynamic rules verified successfully."
