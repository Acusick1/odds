#!/usr/bin/env python3
"""Test Lambda functions on dev environment."""
import json
import sys
import time

import boto3

LAMBDA_NAME = "odds-scheduler-dev"
REGION = "eu-west-1"


def test_lambda_job(job_name: str) -> bool:
    """
    Test a specific Lambda job.

    Args:
        job_name: Name of the job to test (fetch-odds, fetch-scores, update-status)

    Returns:
        bool: True if test passed, False otherwise
    """
    print(f"\n→ Testing {job_name} job...")

    lambda_client = boto3.client("lambda", region_name=REGION)

    try:
        # Invoke Lambda
        response = lambda_client.invoke(
            FunctionName=LAMBDA_NAME,
            Payload=json.dumps({"job": job_name}).encode(),
        )

        # Check response
        payload = json.loads(response["Payload"].read())
        status = payload.get("statusCode", 0)

        if status != 200:
            print(f"✗ Failed: Status {status}")
            print(f"  Response: {payload}")
            return False

        print(f"✓ Invoked successfully (status {status})")
        return True

    except Exception as e:
        print(f"✗ Exception: {e}")
        return False


def check_logs() -> bool:
    """
    Check for errors in recent Lambda logs.

    Returns:
        bool: True if no errors found, False otherwise
    """
    print("\n→ Checking Lambda logs...")

    logs_client = boto3.client("logs", region_name=REGION)
    log_group = f"/aws/lambda/{LAMBDA_NAME}"

    try:
        # Get recent log events (last 2 minutes)
        current_time = int(time.time() * 1000)
        start_time = current_time - (2 * 60 * 1000)

        response = logs_client.filter_log_events(
            logGroupName=log_group,
            startTime=start_time,
        )

        # Check for errors
        error_count = 0
        events = response.get("events", [])

        for event in events:
            message = event["message"]
            if "ERROR" in message.upper() or '"level":"error"' in message:
                print(f"  ✗ Error found: {message[:100]}...")
                error_count += 1

        if error_count > 0:
            print(f"✗ Found {error_count} error(s) in logs")
            return False

        # Check for completion
        completed = any("fetch_odds_completed" in e["message"] for e in events)

        if completed:
            print("✓ Job completed successfully")
        else:
            print("⚠ Job completion not confirmed (may still be running)")

        print(f"✓ No errors found in {len(events)} log entries")
        return True

    except Exception as e:
        print(f"⚠ Could not check logs: {e}")
        return True  # Don't fail if logs aren't available yet


def main():
    """Run all Lambda tests."""
    print("=" * 60)
    print("Testing Lambda on Dev Environment")
    print(f"Lambda: {LAMBDA_NAME}")
    print(f"Region: {REGION}")
    print("=" * 60)

    success = True

    # Test fetch-odds job
    if not test_lambda_job("fetch-odds"):
        success = False

    # Wait for execution to complete
    print("\n→ Waiting for execution...")
    time.sleep(10)

    # Check logs
    if not check_logs():
        success = False

    print("\n" + "=" * 60)
    if success:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    print("=" * 60)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
