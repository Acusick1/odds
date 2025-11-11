"""Exceptions for scheduler backend operations."""


class SchedulerBackendError(Exception):
    """Base exception for all scheduler backend errors."""


class SchedulingFailedError(SchedulerBackendError):
    """Failed to schedule next execution."""


class CancellationFailedError(SchedulerBackendError):
    """Failed to cancel scheduled execution."""


class BackendUnavailableError(SchedulerBackendError):
    """Backend service is unavailable or not properly configured."""


class ConfigurationError(SchedulerBackendError):
    """Backend configuration is invalid or incomplete."""


class JobNotFoundError(SchedulerBackendError):
    """Requested job does not exist in scheduler."""
