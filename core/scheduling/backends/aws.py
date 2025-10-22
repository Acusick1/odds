"""AWS EventBridge + Lambda scheduler backend."""

import json
import os
import re
from datetime import UTC, datetime, timedelta

import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from core.scheduling.backends.base import (
    BackendHealth,
    JobStatus,
    RetryConfig,
    ScheduledJob,
    SchedulerBackend,
    ValidationResult,
)
from core.scheduling.exceptions import (
    BackendUnavailableError,
    CancellationFailedError,
    ConfigurationError,
    JobNotFoundError,
    SchedulingFailedError,
)

logger = structlog.get_logger()

# AWS EventBridge schedule format
AWS_SCHEDULE_FORMAT = "at({timestamp})"


def _format_aws_schedule(dt: datetime) -> str:
    """Format datetime for AWS EventBridge at() expression."""
    return f"at({dt.strftime('%Y-%m-%dT%H:%M:%S')})"


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
    - Health monitoring and status queries
    - Dry-run mode for testing
    - Automatic retry with exponential backoff

    Requirements:
    - AWS_REGION environment variable
    - LAMBDA_ARN environment variable
    - IAM permissions: events:PutRule, events:PutTargets, events:DisableRule,
      events:DescribeRule, events:ListRules
    """

    def __init__(
        self,
        dry_run: bool = False,
        retry_config: RetryConfig | None = None,
        aws_region: str | None = None,
        lambda_arn: str | None = None,
    ):
        """
        Initialize AWS EventBridge client.

        Args:
            dry_run: If True, log operations without executing them
            retry_config: Retry configuration (uses defaults if None)
            aws_region: AWS region override (uses AWS_REGION env var if None)
            lambda_arn: Lambda ARN override (uses LAMBDA_ARN env var if None)

        Raises:
            ConfigurationError: If required configuration is missing
        """
        super().__init__(dry_run=dry_run, retry_config=retry_config)

        self.aws_region = aws_region or os.getenv("AWS_REGION")
        self.lambda_arn = lambda_arn or os.getenv("LAMBDA_ARN")

        # Health check caching (5-minute TTL)
        self._health_check_cache: tuple[BackendHealth, datetime] | None = None
        self._health_check_cache_ttl = timedelta(minutes=5)

        # Validate configuration first
        validation = self.validate_configuration()
        if not validation.is_valid:
            raise ConfigurationError(
                f"AWS backend configuration invalid: {', '.join(validation.errors)}"
            )

        # Initialize boto3 clients
        try:
            import boto3
        except ImportError as e:
            raise ConfigurationError(
                "boto3 is required for AWS backend. Install with: pip install boto3"
            ) from e

        try:
            self.events_client = boto3.client("events", region_name=self.aws_region)
            self.lambda_client = boto3.client("lambda", region_name=self.aws_region)
        except Exception as e:
            raise BackendUnavailableError(f"Failed to initialize AWS clients: {e}") from e

        logger.info(
            "aws_backend_initialized",
            region=self.aws_region,
            lambda_arn=self.lambda_arn,
            dry_run=self.dry_run,
        )

    def validate_configuration(self) -> ValidationResult:
        """Validate AWS backend configuration."""
        from core.scheduling.health_check import ValidationBuilder

        builder = ValidationBuilder()

        # Check required configuration
        builder.check_required(
            self.aws_region, "AWS_REGION", "AWS_REGION environment variable is required"
        )
        builder.check_required(
            self.lambda_arn, "LAMBDA_ARN", "LAMBDA_ARN environment variable is required"
        )

        # Check boto3 availability
        try:
            import boto3  # noqa: F401
        except ImportError:
            builder.add_error("boto3 package not installed (pip install boto3)")

        # Check if AWS credentials are configured (warning only)
        if builder._errors:  # Only check if no errors so far
            return builder.build()

        try:
            import boto3

            session = boto3.Session()
            credentials = session.get_credentials()
            if not credentials:
                builder.add_warning("AWS credentials not found in environment")
        except Exception as e:
            builder.add_warning(f"Could not verify AWS credentials: {e}")

        return builder.build()

    async def health_check(self) -> BackendHealth:
        """
        Perform comprehensive health check of AWS backend.

        Results are cached for 5 minutes to avoid excessive API calls.
        """
        # Return cached result if fresh
        if self._health_check_cache:
            result, timestamp = self._health_check_cache
            age = datetime.now(UTC) - timestamp
            if age < self._health_check_cache_ttl:
                logger.debug(
                    "health_check_cache_hit",
                    age_seconds=age.total_seconds(),
                    ttl_seconds=self._health_check_cache_ttl.total_seconds(),
                )
                return result

        # Perform actual health check
        result = await self._perform_health_check()

        # Cache result
        self._health_check_cache = (result, datetime.now(UTC))

        return result

    async def _perform_health_check(self) -> BackendHealth:
        """Perform actual health check (called when cache is stale)."""
        from core.scheduling.health_check import HealthCheckBuilder

        builder = HealthCheckBuilder(self.get_backend_name())

        # Check configuration
        validation = self.validate_configuration()
        builder.check_condition(
            validation.is_valid,
            "Configuration valid",
            f"Configuration invalid: {', '.join(validation.errors)}",
        )

        # Check EventBridge connectivity
        try:
            self.events_client.list_rules(Limit=1)
            builder.pass_check("EventBridge API accessible")
        except Exception as e:
            builder.fail_check(f"EventBridge API failed: {e}")
            builder.add_detail("eventbridge_error", str(e))

        # Check Lambda connectivity
        try:
            # Attempt to get Lambda function info
            function_name = self.lambda_arn.split(":")[-1] if self.lambda_arn else None
            if function_name:
                self.lambda_client.get_function(FunctionName=function_name)
                builder.pass_check("Lambda function accessible")
            else:
                builder.fail_check("Lambda ARN format invalid")
        except self.lambda_client.exceptions.ResourceNotFoundException:
            builder.fail_check("Lambda function not found")
        except Exception as e:
            builder.fail_check(f"Lambda API failed: {e}")
            builder.add_detail("lambda_error", str(e))

        # Check IAM permissions by attempting to describe a rule
        try:
            # Try to describe a non-existent rule to test permissions
            try:
                self.events_client.describe_rule(Name="odds-permission-test")
            except self.events_client.exceptions.ResourceNotFoundException:
                # Expected - we just want to verify we have describe permissions
                builder.pass_check("IAM permissions verified")
        except Exception as e:
            builder.fail_check(f"IAM permissions check failed: {e}")
            builder.add_detail("iam_error", str(e))

        return builder.build()

    async def get_scheduled_jobs(self) -> list[ScheduledJob]:
        """Get list of all currently scheduled jobs."""
        if self.dry_run:
            logger.info("dry_run_get_scheduled_jobs")
            return []

        try:
            # List all EventBridge rules with odds- prefix
            response = self.events_client.list_rules(NamePrefix="odds-")

            jobs = []
            for rule in response.get("Rules", []):
                job_name = rule["Name"].replace("odds-", "", 1)

                # Parse schedule expression to get next run time
                schedule_expr = rule.get("ScheduleExpression", "")
                next_run = self._parse_schedule_expression(schedule_expr)

                # Get rule state
                state = rule.get("State", "UNKNOWN")
                status = JobStatus.SCHEDULED if state == "ENABLED" else JobStatus.CANCELLED

                jobs.append(
                    ScheduledJob(
                        job_name=job_name,
                        next_run_time=next_run,
                        status=status,
                    )
                )

            return jobs

        except Exception as e:
            raise BackendUnavailableError(f"Failed to list scheduled jobs: {e}") from e

    async def get_job_status(self, job_name: str) -> ScheduledJob | None:
        """Get status of a specific job."""
        if self.dry_run:
            logger.info("dry_run_get_job_status", job=job_name)
            return None

        rule_name = f"odds-{job_name}"

        try:
            response = self.events_client.describe_rule(Name=rule_name)

            schedule_expr = response.get("ScheduleExpression", "")
            next_run = self._parse_schedule_expression(schedule_expr)

            state = response.get("State", "UNKNOWN")
            status = JobStatus.SCHEDULED if state == "ENABLED" else JobStatus.CANCELLED

            return ScheduledJob(
                job_name=job_name,
                next_run_time=next_run,
                status=status,
            )
        except self.events_client.exceptions.ResourceNotFoundException:
            return None
        except Exception as e:
            raise BackendUnavailableError(f"Failed to get job status: {e}") from e

    async def schedule_next_execution(self, job_name: str, next_time: datetime) -> None:
        """
        Update EventBridge rule to trigger at specific time with retry logic.

        Creates or updates a one-time schedule expression for the job.
        The rule targets the Lambda function with job name in payload.

        Uses tenacity for automatic retry with exponential backoff.

        Args:
            job_name: Job identifier (e.g., 'fetch-odds')
            next_time: UTC datetime for next execution

        Raises:
            SchedulingFailedError: If scheduling fails after all retries
        """
        if self.dry_run:
            logger.info(
                "dry_run_schedule",
                job=job_name,
                next_time=next_time.isoformat(),
                backend="aws_eventbridge",
            )
            return

        # Delegate to retrying implementation
        try:
            await self._schedule_with_retry(job_name, next_time)
        except Exception as e:
            raise SchedulingFailedError(
                f"Failed to schedule {job_name} after {self.retry_config.max_attempts} attempts: {e}"
            ) from e

    async def _schedule_with_retry(self, job_name: str, next_time: datetime) -> None:
        """
        Internal method with tenacity retry decorator.

        Separated to allow tenacity to wrap with retry logic.
        """
        # Create retry decorator dynamically based on config
        retry_decorator = retry(
            stop=stop_after_attempt(self.retry_config.max_attempts),
            wait=wait_exponential(
                multiplier=self.retry_config.initial_delay_seconds,
                max=self.retry_config.max_delay_seconds,
            ),
            retry=retry_if_exception_type(Exception),
            reraise=True,
            before_sleep=lambda retry_state: logger.warning(
                "eventbridge_scheduling_retry",
                job=job_name,
                attempt=retry_state.attempt_number,
                max_attempts=self.retry_config.max_attempts,
                error=str(retry_state.outcome.exception()) if retry_state.outcome else "unknown",
            ),
        )

        # Apply retry decorator and execute
        @retry_decorator
        async def _execute():
            rule_name = f"odds-{job_name}"
            schedule_expression = _format_aws_schedule(next_time)

            # Create/update one-time schedule
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

        await _execute()

    async def cancel_scheduled_execution(self, job_name: str) -> None:
        """
        Disable EventBridge rule to cancel scheduled execution.

        Args:
            job_name: Job identifier to cancel

        Raises:
            CancellationFailedError: If cancellation fails
            JobNotFoundError: If job doesn't exist
        """
        if self.dry_run:
            logger.info("dry_run_cancel", job=job_name, backend="aws_eventbridge")
            return

        rule_name = f"odds-{job_name}"

        try:
            self.events_client.disable_rule(Name=rule_name)
            logger.info("eventbridge_rule_disabled", rule=rule_name)

        except self.events_client.exceptions.ResourceNotFoundException as e:
            raise JobNotFoundError(f"Job {job_name} not found in EventBridge") from e

        except Exception as e:
            logger.error(
                "eventbridge_cancel_failed",
                job=job_name,
                error=str(e),
                exc_info=True,
            )
            raise CancellationFailedError(f"Failed to cancel {job_name}: {e}") from e

    def get_backend_name(self) -> str:
        """Return backend identifier."""
        return "aws_eventbridge"

    @staticmethod
    def _parse_schedule_expression(expression: str) -> datetime | None:
        """
        Parse EventBridge schedule expression to datetime.

        Args:
            expression: Schedule expression like "at(2025-10-20T14:30:00)"

        Returns:
            Datetime if parseable, None otherwise
        """
        if not expression:
            return None

        # Use regex for robust parsing
        match = re.match(r"at\((\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\)", expression)
        if not match:
            logger.warning(
                "invalid_schedule_expression",
                expression=expression,
                expected_format="at(YYYY-MM-DDTHH:MM:SS)",
            )
            return None

        try:
            return datetime.strptime(match.group(1), "%Y-%m-%dT%H:%M:%S")
        except ValueError as e:
            logger.warning(
                "schedule_parse_failed",
                expression=expression,
                timestamp=match.group(1),
                error=str(e),
            )
            return None
