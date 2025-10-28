"""Health check builder for scheduler backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.scheduling.backends.base import BackendHealth, ValidationResult


@dataclass
class HealthCheckBuilder:
    """
    Builder pattern for constructing BackendHealth results.

    Provides a cleaner, more maintainable way to build health check results
    compared to manually managing lists.

    Example:
        builder = HealthCheckBuilder("local_apscheduler")
        builder.pass_check("Configuration valid")
        builder.fail_check("Database unreachable")
        builder.add_detail("scheduler_state", "running")
        return builder.build()
    """

    backend_name: str
    _checks_passed: list[str] = field(default_factory=list)
    _checks_failed: list[str] = field(default_factory=list)
    _details: dict[str, Any] = field(default_factory=dict)

    def pass_check(self, message: str) -> HealthCheckBuilder:
        """
        Record a passed health check.

        Args:
            message: Description of what passed

        Returns:
            Self for method chaining
        """
        self._checks_passed.append(message)
        return self

    def fail_check(self, message: str) -> HealthCheckBuilder:
        """
        Record a failed health check.

        Args:
            message: Description of what failed

        Returns:
            Self for method chaining
        """
        self._checks_failed.append(message)
        return self

    def add_detail(self, key: str, value: Any) -> HealthCheckBuilder:
        """
        Add detail information to health check.

        Args:
            key: Detail key
            value: Detail value

        Returns:
            Self for method chaining
        """
        self._details[key] = value
        return self

    def check_condition(self, condition: bool, pass_msg: str, fail_msg: str) -> HealthCheckBuilder:
        """
        Conditionally pass or fail a check based on boolean condition.

        Args:
            condition: If True, check passes; if False, check fails
            pass_msg: Message when condition is True
            fail_msg: Message when condition is False

        Returns:
            Self for method chaining
        """
        if condition:
            self.pass_check(pass_msg)
        else:
            self.fail_check(fail_msg)
        return self

    def build(self) -> BackendHealth:
        """
        Build final BackendHealth result.

        Returns:
            BackendHealth with all checks and details
        """
        return BackendHealth(
            is_healthy=len(self._checks_failed) == 0,
            backend_name=self.backend_name,
            checks_passed=tuple(self._checks_passed),
            checks_failed=tuple(self._checks_failed),
            details=self._details if self._details else None,
        )


@dataclass
class ValidationBuilder:
    """
    Builder pattern for constructing ValidationResult.

    Provides a cleaner way to build validation results with errors and warnings.

    Example:
        builder = ValidationBuilder()
        builder.add_error("Missing API key")
        builder.add_warning("Using default timeout")
        return builder.build()
    """

    _errors: list[str] = field(default_factory=list)
    _warnings: list[str] = field(default_factory=list)

    def add_error(self, message: str) -> ValidationBuilder:
        """
        Add validation error.

        Args:
            message: Error description

        Returns:
            Self for method chaining
        """
        self._errors.append(message)
        return self

    def add_warning(self, message: str) -> ValidationBuilder:
        """
        Add validation warning.

        Args:
            message: Warning description

        Returns:
            Self for method chaining
        """
        self._warnings.append(message)
        return self

    def check_required(
        self, value: Any, name: str, error_msg: str | None = None
    ) -> ValidationBuilder:
        """
        Check if required value is present.

        Args:
            value: Value to check
            name: Name of the value
            error_msg: Optional custom error message

        Returns:
            Self for method chaining
        """
        if not value:
            msg = error_msg or f"Missing required value: {name}"
            self.add_error(msg)
        return self

    def build(self) -> ValidationResult:
        """
        Build final ValidationResult.

        Returns:
            ValidationResult with all errors and warnings
        """
        return ValidationResult(
            is_valid=len(self._errors) == 0,
            errors=tuple(self._errors),
            warnings=tuple(self._warnings),
        )
