#!/usr/bin/env bash
# Verifies that all self-scheduling EventBridge rules are active after a deployment.
#
# Usage:
#   ./scripts/verify-schedules.sh --lambda-name <name> --rule-prefix <prefix> [--region <region>]
#
# Steps:
#   1. Invokes Lambda once per self-scheduling job to trigger self-scheduling
#   2. Waits for Lambda executions to complete
#   3. Verifies each dynamic rule is ENABLED with a specific cron schedule

set -euo pipefail

LAMBDA_NAME=""
RULE_PREFIX=""
REGION="${AWS_REGION:-eu-west-1}"
WAIT_SECONDS=60

JOBS=("fetch-odds" "fetch-scores" "update-status" "check-health" "fetch-polymarket")

usage() {
  echo "Usage: $0 --lambda-name <name> --rule-prefix <prefix> [--region <region>]"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --lambda-name) LAMBDA_NAME="$2"; shift 2 ;;
    --rule-prefix) RULE_PREFIX="$2"; shift 2 ;;
    --region)      REGION="$2"; shift 2 ;;
    *) usage ;;
  esac
done

[[ -z "$LAMBDA_NAME" || -z "$RULE_PREFIX" ]] && usage

echo "==> Invoking Lambda for each self-scheduling job..."
for job in "${JOBS[@]}"; do
  echo "    Invoking: $job"
  aws lambda invoke \
    --function-name "$LAMBDA_NAME" \
    --region "$REGION" \
    --payload "{\"job\":\"$job\"}" \
    --cli-binary-format raw-in-base64-out \
    /tmp/lambda-response-"$job".json \
    --query 'FunctionError' \
    --output text > /dev/null
done

echo "==> Waiting ${WAIT_SECONDS}s for Lambda executions to complete..."
sleep "$WAIT_SECONDS"

echo "==> Verifying dynamic rule states..."
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
    echo "FAIL (ScheduleExpression='$schedule', expected cron — Lambda may not have self-scheduled)"
    FAILED=$((FAILED + 1))
    continue
  fi

  echo "OK (State=$state, Schedule=$schedule)"
done

if [[ "$FAILED" -gt 0 ]]; then
  echo ""
  echo "ERROR: $FAILED rule(s) failed verification."
  exit 1
fi

echo ""
echo "All $((${#JOBS[@]})) dynamic rules verified successfully."
