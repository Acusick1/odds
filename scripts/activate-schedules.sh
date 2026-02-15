#!/usr/bin/env bash
# Invokes Lambda for each self-scheduling job and polls until rules activate.
#
# Usage:
#   ./scripts/activate-schedules.sh --lambda-name <name> --rule-prefix <prefix> \
#     [--jobs fetch-odds,fetch-scores,...] [--region <region>]

set -euo pipefail

LAMBDA_NAME=""
RULE_PREFIX=""
REGION="${AWS_REGION:-eu-west-1}"
JOBS_CSV=""
POLL_INTERVAL=10
POLL_TIMEOUT=120

usage() {
  echo "Usage: $0 --lambda-name <name> --rule-prefix <prefix> --jobs <csv> [--region <region>]"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lambda-name) LAMBDA_NAME="$2"; shift 2 ;;
    --rule-prefix) RULE_PREFIX="$2"; shift 2 ;;
    --jobs)        JOBS_CSV="$2"; shift 2 ;;
    --region)      REGION="$2"; shift 2 ;;
    *) usage ;;
  esac
done

[[ -z "$LAMBDA_NAME" || -z "$RULE_PREFIX" || -z "$JOBS_CSV" ]] && usage

IFS=',' read -ra JOBS <<< "$JOBS_CSV"

echo "==> Invoking Lambda for each self-scheduling job..."
for job in "${JOBS[@]}"; do
  echo -n "    $job: "
  response_file="/tmp/lambda-response-${job}.json"

  func_error=$(aws lambda invoke \
    --function-name "$LAMBDA_NAME" \
    --region "$REGION" \
    --payload "{\"job\":\"$job\"}" \
    --cli-binary-format raw-in-base64-out \
    "$response_file" \
    --query 'FunctionError' \
    --output text)

  if [[ "$func_error" != "None" ]]; then
    echo "FAIL"
    echo "    ERROR: Lambda returned FunctionError=$func_error for job '$job'"
    echo "    Response payload:"
    cat "$response_file"
    echo ""
    exit 1
  fi
  echo "OK"
done

echo "==> Polling rules (every ${POLL_INTERVAL}s, timeout ${POLL_TIMEOUT}s)..."
declare -A pending
for job in "${JOBS[@]}"; do pending[$job]=1; done

elapsed=0
while [[ ${#pending[@]} -gt 0 && $elapsed -lt $POLL_TIMEOUT ]]; do
  for job in "${!pending[@]}"; do
    rule_name="${RULE_PREFIX}-${job}"
    rule_json=$(aws events describe-rule \
      --name "$rule_name" \
      --region "$REGION" \
      --output json 2>/dev/null) || continue

    state=$(echo "$rule_json" | python3 -c "import json,sys; print(json.load(sys.stdin)['State'])")
    schedule=$(echo "$rule_json" | python3 -c "import json,sys; print(json.load(sys.stdin)['ScheduleExpression'])")

    if [[ "$state" == "ENABLED" && "$schedule" != rate* ]]; then
      echo "    $rule_name: active ($schedule)"
      unset "pending[$job]"
    fi
  done
  if [[ ${#pending[@]} -gt 0 ]]; then
    sleep "$POLL_INTERVAL"
    elapsed=$((elapsed + POLL_INTERVAL))
  fi
done

if [[ ${#pending[@]} -gt 0 ]]; then
  echo ""
  echo "ERROR: ${#pending[@]} rule(s) failed to activate within ${POLL_TIMEOUT}s:"
  for job in "${!pending[@]}"; do
    rule_name="${RULE_PREFIX}-${job}"
    rule_json=$(aws events describe-rule \
      --name "$rule_name" \
      --region "$REGION" \
      --output json 2>/dev/null) || { echo "    $rule_name: not found"; continue; }
    state=$(echo "$rule_json" | python3 -c "import json,sys; print(json.load(sys.stdin)['State'])")
    schedule=$(echo "$rule_json" | python3 -c "import json,sys; print(json.load(sys.stdin)['ScheduleExpression'])")
    echo "    $rule_name: State=$state, Schedule=$schedule"
  done
  exit 1
fi

echo ""
echo "All ${#JOBS[@]} dynamic rules activated."
