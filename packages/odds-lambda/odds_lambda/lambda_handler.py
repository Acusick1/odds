"""
Universal AWS Lambda handler for all scheduled jobs.

This handler routes EventBridge events to the appropriate job module.
The event payload specifies which job to execute.

Event payload format:
{
    "job": "fetch-odds" | "fetch-scores" | "update-status" | "check-health" | "fetch-polymarket" | "backfill-polymarket"
}

Environment variables required:
- DATABASE_URL: PostgreSQL connection string
- ODDS_API_KEY: The Odds API key
- SCHEDULER_BACKEND: Should be 'aws'
- AWS_REGION: AWS region (automatically provided by Lambda)
- AWS_LAMBDA_ARN: This Lambda function's ARN
"""

import asyncio
import json

import structlog
from odds_core.config import get_settings
from odds_core.logging_setup import configure_logging

# Configure structured logging for Lambda (JSON output for CloudWatch)
configure_logging(get_settings(), json_output=True)

logger = structlog.get_logger()


async def _run_job_async(job_name: str):
    """Run the job module's main function asynchronously."""
    if job_name == "fetch-odds":
        from odds_lambda.jobs import fetch_odds

        await fetch_odds.main()

    elif job_name == "fetch-scores":
        from odds_lambda.jobs import fetch_scores

        await fetch_scores.main()

    elif job_name == "update-status":
        from odds_lambda.jobs import update_status

        await update_status.main()

    elif job_name == "check-health":
        from odds_lambda.jobs import check_health

        await check_health.main()

    elif job_name == "fetch-polymarket":
        from odds_lambda.jobs import fetch_polymarket

        await fetch_polymarket.main()

    elif job_name == "backfill-polymarket":
        from odds_lambda.jobs import backfill_polymarket

        await backfill_polymarket.main()

    else:
        raise ValueError(f"Unknown job: {job_name}")


def lambda_handler(event, context):
    """
    AWS Lambda entry point.

    Args:
        event: EventBridge event payload with 'job' key
        context: Lambda context object

    Returns:
        dict: Response with statusCode and body

    Raises:
        Exception: If job execution fails
    """
    try:
        job_name = event.get("job")

        logger.info(
            "lambda_invoked",
            job=job_name,
            request_id=context.aws_request_id,
            function_name=context.function_name,
            memory_limit=context.memory_limit_in_mb,
        )

        # Validate job name
        if not job_name:
            raise ValueError("Missing 'job' in event payload")

        # Run async job
        # Use asyncio.run() to ensure clean event loop per invocation
        asyncio.run(_run_job_async(job_name))

        logger.info(
            "lambda_completed",
            job=job_name,
            request_id=context.aws_request_id,
        )

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "status": "success",
                    "job": job_name,
                    "request_id": context.aws_request_id,
                }
            ),
        }

    except Exception as e:
        logger.error(
            "lambda_failed",
            error=str(e),
            error_type=type(e).__name__,
            request_id=context.aws_request_id if context else None,
            exc_info=True,
        )

        # Send critical alert
        from odds_core.config import get_settings

        app_settings = get_settings()
        if app_settings.alerts.alert_enabled:
            from odds_cli.alerts.base import send_critical

            asyncio.run(
                send_critical(
                    f"🚨 Lambda handler failed: {type(e).__name__}: {str(e)} "
                    f"(job: {event.get('job', 'unknown')})"
                )
            )

        return {
            "statusCode": 500,
            "body": json.dumps(
                {
                    "status": "error",
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
            ),
        }
