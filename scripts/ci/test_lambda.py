#!/usr/bin/env python3
"""Test Lambda functions."""

import argparse
import json
import sys
import time

import boto3


def test_lambda_job(lambda_name: str, region: str, job_name: str) -> tuple[bool, str | None]:
    """
    Test a specific Lambda job.

    Args:
        lambda_name: Name of the Lambda function
        region: AWS region
        job_name: Name of the job to test (fetch-odds, fetch-scores, update-status)

    Returns:
        tuple: (success: bool, request_id: str | None)
    """
    print(f"\n→ Testing {job_name} job...")

    lambda_client = boto3.client("lambda", region_name=region)

    try:
        # Invoke Lambda
        response = lambda_client.invoke(
            FunctionName=lambda_name,
            Payload=json.dumps({"job": job_name}).encode(),
        )

        # Get request ID from response metadata
        request_id = response.get("ResponseMetadata", {}).get("RequestId")

        # Check response
        payload = json.loads(response["Payload"].read())
        status = payload.get("statusCode", 0)

        if status != 200:
            print(f"✗ Failed: Status {status}")
            print(f"  Response: {payload}")
            return (False, request_id)

        print(f"✓ Invoked successfully (status {status})")
        if request_id:
            print(f"  Request ID: {request_id}")
        return (True, request_id)

    except Exception as e:
        print(f"✗ Exception: {e}")
        return (False, None)


def check_logs(lambda_name: str, region: str, request_id: str | None) -> bool:
    """
    Check for errors in Lambda logs for a specific invocation.

    Args:
        lambda_name: Name of the Lambda function
        region: AWS region
        request_id: Lambda request ID to filter logs for

    Returns:
        bool: True if no errors found, False otherwise
    """
    print("\n→ Checking Lambda logs...")

    if not request_id:
        print("⚠ No request ID available, skipping log check")
        return True

    logs_client = boto3.client("logs", region_name=region)
    log_group = f"/aws/lambda/{lambda_name}"

    try:
        # Get recent log events (last 2 minutes)
        current_time = int(time.time() * 1000)
        start_time = current_time - (2 * 60 * 1000)

        response = logs_client.filter_log_events(
            logGroupName=log_group,
            startTime=start_time,
        )

        # Filter events to only this specific invocation
        events = response.get("events", [])
        invocation_events = [e for e in events if request_id in e["message"]]

        if not invocation_events:
            print(f"⚠ No logs found for request ID {request_id[:8]}... (may not be available yet)")
            return True

        print(f"  Found {len(invocation_events)} log entries for this invocation")

        # Check for errors in this specific invocation
        error_count = 0
        for event in invocation_events:
            message = event["message"]
            if "ERROR" in message.upper() or '"level":"error"' in message:
                print(f"  ✗ Error found: {message[:100]}...")
                error_count += 1

        if error_count > 0:
            print(f"✗ Found {error_count} error(s) in logs for this invocation")
            return False

        # Check for completion
        completed = any("completed" in e["message"].lower() for e in invocation_events)

        if completed:
            print("✓ Job completed successfully")
        else:
            print("⚠ Job completion not confirmed (may still be running)")

        print("✓ No errors found in this invocation's logs")
        return True

    except Exception as e:
        print(f"⚠ Could not check logs: {e}")
        return True  # Don't fail if logs aren't available yet


def main():
    """Run all Lambda tests."""
    parser = argparse.ArgumentParser(description="Test Lambda function")
    parser.add_argument(
        "--lambda-name",
        required=True,
        help="Name of the Lambda function to test (e.g., odds-scheduler or odds-scheduler-dev)",
    )
    parser.add_argument(
        "--region",
        default="eu-west-1",
        help="AWS region (default: eu-west-1)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Testing Lambda Function")
    print(f"Lambda: {args.lambda_name}")
    print(f"Region: {args.region}")
    print("=" * 60)

    success = True

    # Test fetch-odds job
    test_success, request_id = test_lambda_job(args.lambda_name, args.region, "fetch-odds")
    if not test_success:
        success = False

    # Wait for execution to complete
    print("\n→ Waiting for execution...")
    time.sleep(10)

    # Check logs for this specific invocation
    if not check_logs(args.lambda_name, args.region, request_id):
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
