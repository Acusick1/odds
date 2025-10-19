"""
Universal AWS Lambda handler for all scheduled jobs.

This handler routes EventBridge events to the appropriate job module.
The event payload specifies which job to execute.

Event payload format:
{
    "job": "fetch-odds" | "fetch-scores" | "update-status"
}

Environment variables required:
- DATABASE_URL: PostgreSQL connection string
- ODDS_API_KEY: The Odds API key
- SCHEDULER_BACKEND: Should be 'aws'
- AWS_REGION: AWS region
- LAMBDA_ARN: This Lambda function's ARN
"""

import asyncio
import json
import sys
from pathlib import Path

import structlog

# Configure structured logging for Lambda
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger()

# Add application root to Python path
app_root = Path(__file__).parent
if str(app_root) not in sys.path:
    sys.path.insert(0, str(app_root))


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
            request_id=context.request_id,
            function_name=context.function_name,
            memory_limit=context.memory_limit_in_mb,
        )

        # Validate job name
        if not job_name:
            raise ValueError("Missing 'job' in event payload")

        # Route to appropriate job module
        if job_name == "fetch-odds":
            from jobs import fetch_odds

            asyncio.run(fetch_odds.main())

        elif job_name == "fetch-scores":
            from jobs import fetch_scores

            asyncio.run(fetch_scores.main())

        elif job_name == "update-status":
            from jobs import update_status

            asyncio.run(update_status.main())

        else:
            raise ValueError(f"Unknown job: {job_name}")

        logger.info(
            "lambda_completed",
            job=job_name,
            request_id=context.request_id,
        )

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "status": "success",
                    "job": job_name,
                    "request_id": context.request_id,
                }
            ),
        }

    except Exception as e:
        logger.error(
            "lambda_failed",
            error=str(e),
            error_type=type(e).__name__,
            request_id=context.request_id if context else None,
            exc_info=True,
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
