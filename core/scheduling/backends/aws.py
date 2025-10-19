"""AWS EventBridge + Lambda scheduler backend."""

import json
import os
from datetime import datetime

import boto3
import structlog

from core.scheduling.backends.base import SchedulerBackend

logger = structlog.get_logger()


class AWSEventBridgeBackend(SchedulerBackend):
    """
    AWS Lambda + EventBridge scheduler implementation.

    Each job execution dynamically schedules its next run by updating
    EventBridge rules with one-time schedule expressions.

    Features:
    - Dynamic scheduling based on game proximity
    - One-time schedule expressions (at(...) syntax)
    - Self-scheduling jobs
    - Zero cost when no games scheduled

    Requirements:
    - AWS_REGION environment variable
    - LAMBDA_ARN environment variable
    - IAM permissions: events:PutRule, events:PutTargets, events:DisableRule
    """

    def __init__(self):
        """Initialize AWS EventBridge client."""
        self.aws_region = os.getenv("AWS_REGION")
        self.lambda_arn = os.getenv("LAMBDA_ARN")

        if not self.aws_region:
            raise ValueError("AWS_REGION environment variable is required for AWS backend")

        if not self.lambda_arn:
            raise ValueError("LAMBDA_ARN environment variable is required for AWS backend")

        self.events_client = boto3.client("events", region_name=self.aws_region)
        self.lambda_client = boto3.client("lambda", region_name=self.aws_region)

        logger.info("aws_backend_initialized", region=self.aws_region, lambda_arn=self.lambda_arn)

    async def schedule_next_execution(self, job_name: str, next_time: datetime) -> None:
        """
        Update EventBridge rule to trigger at specific time.

        Creates or updates a one-time schedule expression for the job.
        The rule targets the Lambda function with job name in payload.

        Args:
            job_name: Job identifier (e.g., 'fetch-odds')
            next_time: UTC datetime for next execution

        Raises:
            Exception: If AWS API calls fail
        """
        rule_name = f"odds-{job_name}"

        try:
            # Create/update one-time schedule
            # Format: at(YYYY-MM-DDTHH:MM:SS)
            schedule_expression = f"at({next_time.strftime('%Y-%m-%dT%H:%M:%S')})"

            self.events_client.put_rule(
                Name=rule_name,
                ScheduleExpression=schedule_expression,
                State="ENABLED",
                Description=f"Dynamic schedule for {job_name}",
            )

            logger.info(
                "eventbridge_rule_updated",
                rule=rule_name,
                schedule=schedule_expression,
                next_time=next_time.isoformat(),
            )

            # Ensure Lambda target is attached
            self.events_client.put_targets(
                Rule=rule_name,
                Targets=[
                    {
                        "Id": "1",
                        "Arn": self.lambda_arn,
                        "Input": json.dumps({"job": job_name}),
                    }
                ],
            )

            logger.info("eventbridge_target_updated", rule=rule_name, target=self.lambda_arn)

        except Exception as e:
            logger.error(
                "eventbridge_scheduling_failed",
                job=job_name,
                next_time=next_time.isoformat(),
                error=str(e),
                exc_info=True,
            )
            raise

    async def cancel_scheduled_execution(self, job_name: str) -> None:
        """
        Disable EventBridge rule to cancel scheduled execution.

        Args:
            job_name: Job identifier to cancel

        Raises:
            Exception: If AWS API call fails
        """
        rule_name = f"odds-{job_name}"

        try:
            self.events_client.disable_rule(Name=rule_name)
            logger.info("eventbridge_rule_disabled", rule=rule_name)

        except self.events_client.exceptions.ResourceNotFoundException:
            logger.warning("eventbridge_rule_not_found", rule=rule_name)

        except Exception as e:
            logger.error(
                "eventbridge_cancel_failed",
                job=job_name,
                error=str(e),
                exc_info=True,
            )
            raise

    def get_backend_name(self) -> str:
        """Return backend identifier."""
        return "aws_eventbridge"
