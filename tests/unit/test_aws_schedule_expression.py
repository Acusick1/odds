"""Tests for parsing EventBridge one-time cron expressions."""

from datetime import UTC, datetime

from odds_lambda.scheduling.backends.aws import AWSEventBridgeBackend


class TestParseScheduleExpression:
    def test_returns_a_utc_instant(self) -> None:
        parsed = AWSEventBridgeBackend._parse_schedule_expression("cron(30 14 24 10 ? 2025)")
        assert parsed == datetime(2025, 10, 24, 14, 30, tzinfo=UTC)
