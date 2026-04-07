"""
Universal AWS Lambda handler for all scheduled jobs.

This handler routes EventBridge events to the appropriate job module.
The event payload specifies which job to execute, optionally with a sport.

Event payload format:
{
    "job": "fetch-odds" | "fetch-odds-epl" | "fetch-scores" | ...,
    "sport": "soccer_epl"  (optional, extracted from compound job name if absent)
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


async def _run_job_async(job_name: str, **kwargs: object) -> None:
    """Run the job module's main function asynchronously.

    Extra kwargs from the event payload are passed to the job function.
    Jobs that accept parameters should declare **kwargs to receive them;
    jobs that don't will ignore extra fields (we inspect the signature).
    """
    import inspect

    from odds_lambda.scheduling.jobs import get_job_function

    job_fn = get_job_function(job_name)
    sig = inspect.signature(job_fn)

    # Determine which kwargs the function can accept
    accepted_params = set(sig.parameters.keys())
    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

    if kwargs:
        if has_var_keyword:
            await job_fn(**kwargs)
        else:
            # Pass only kwargs that match named parameters
            filtered = {k: v for k, v in kwargs.items() if k in accepted_params}
            if filtered:
                await job_fn(**filtered)
            else:
                await job_fn()
    else:
        await job_fn()


def lambda_handler(event: dict, context: object) -> dict:
    """
    AWS Lambda entry point.

    Resolves compound job names (e.g. "fetch-odds-epl") into a base job name
    and sport parameter. The sport is added to the structlog context for
    CloudWatch filtering and passed to the job function.

    Args:
        event: EventBridge event payload with 'job' key and optional 'sport'
        context: Lambda context object

    Returns:
        dict with statusCode and body
    """
    try:
        from odds_lambda.scheduling.jobs import resolve_job_name

        raw_job_name = event.get("job")

        # Validate job name
        if not raw_job_name:
            raise ValueError("Missing 'job' in event payload")

        # Resolve compound name: "fetch-odds-epl" -> ("fetch-odds", "soccer_epl")
        base_job_name, resolved_sport = resolve_job_name(raw_job_name)

        # Explicit sport in payload takes precedence over suffix-derived sport
        sport = event.get("sport") or resolved_sport

        # Bind sport to structlog context for all downstream log entries
        if sport:
            structlog.contextvars.bind_contextvars(sport=sport)

        logger.info(
            "lambda_invoked",
            job=base_job_name,
            sport=sport,
            raw_job=raw_job_name,
            request_id=context.aws_request_id,
            function_name=context.function_name,
            memory_limit=context.memory_limit_in_mb,
        )

        # Build job params: everything except "job", plus sport if resolved
        job_params: dict[str, object] = {k: v for k, v in event.items() if k != "job"}
        if sport and "sport" not in job_params:
            job_params["sport"] = sport

        # Run async job
        asyncio.run(_run_job_async(base_job_name, **job_params))

        logger.info(
            "lambda_completed",
            job=base_job_name,
            request_id=context.aws_request_id,
        )

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "status": "success",
                    "job": base_job_name,
                    "sport": sport,
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
    finally:
        structlog.contextvars.unbind_contextvars("sport")
