"""Centralized logging configuration for CLI and Lambda environments."""

import logging
import logging.handlers
from pathlib import Path

import structlog

from odds_core.config import Settings


def configure_logging(settings: Settings, json_output: bool = False) -> None:
    """
    Configure structlog and stdlib logging based on settings.

    Args:
        settings: Application settings containing logging configuration
        json_output: If True, use JSON rendering (for Lambda). If False, use human-readable
                    console output (for CLI)

    Note:
        This function is idempotent - safe to call multiple times.
        Creates log directory if it doesn't exist.
    """
    # Create log directory if needed
    log_path = Path(settings.logging.file)
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as e:
        # If we can't create the directory, fall back to console-only logging
        print(f"Warning: Could not create log directory {log_path.parent}: {e}")
        print("Falling back to console-only logging")
        _configure_console_only(settings, json_output)
        return

    # Parse log level (default to INFO if invalid)
    log_level_str = settings.logging.level.upper()
    try:
        log_level = getattr(logging, log_level_str)
    except AttributeError:
        print(f"Warning: Invalid log level '{log_level_str}', defaulting to INFO")
        log_level = logging.INFO

    # Shared processors for structlog (before final rendering)
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]

    # Configure stdlib logging handlers with different formatters
    try:
        # File handler - plain text without colors
        file_handler = logging.handlers.RotatingFileHandler(
            settings.logging.file,
            maxBytes=10_485_760,  # 10MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)

        # Use ProcessorFormatter for plain text output to file
        if json_output:
            file_formatter = structlog.stdlib.ProcessorFormatter(
                processors=shared_processors
                + [
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    structlog.processors.JSONRenderer(),
                ],
                foreign_pre_chain=shared_processors,
            )
        else:
            file_formatter = structlog.stdlib.ProcessorFormatter(
                processors=shared_processors
                + [
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    structlog.dev.ConsoleRenderer(colors=False),  # No colors for file
                ],
                foreign_pre_chain=shared_processors,
            )
        file_handler.setFormatter(file_formatter)

        # Console handler - colored output
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)

        # Use ProcessorFormatter for colored output to console
        if json_output:
            console_formatter = structlog.stdlib.ProcessorFormatter(
                processors=shared_processors
                + [
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    structlog.processors.JSONRenderer(),
                ],
                foreign_pre_chain=shared_processors,
            )
        else:
            console_formatter = structlog.stdlib.ProcessorFormatter(
                processors=shared_processors
                + [
                    structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                    structlog.dev.ConsoleRenderer(colors=True),  # Colors for console
                ],
                foreign_pre_chain=shared_processors,
            )
        console_handler.setFormatter(console_formatter)

        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format="%(message)s",
            handlers=[file_handler, console_handler],
            force=True,  # Allow reconfiguration
        )
    except (OSError, PermissionError) as e:
        print(f"Warning: Could not create log file {settings.logging.file}: {e}")
        print("Falling back to console-only logging")
        _configure_console_only(settings, json_output)
        return

    # Configure structlog to use stdlib logging
    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def _configure_console_only(settings: Settings, json_output: bool) -> None:
    """
    Fallback configuration for console-only logging when file logging fails.

    Args:
        settings: Application settings containing logging configuration
        json_output: If True, use JSON rendering. If False, use human-readable output
    """
    # Parse log level
    log_level_str = settings.logging.level.upper()
    try:
        log_level = getattr(logging, log_level_str)
    except AttributeError:
        log_level = logging.INFO

    # Shared processors for structlog
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]

    # Console handler with appropriate formatter
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    if json_output:
        console_formatter = structlog.stdlib.ProcessorFormatter(
            processors=shared_processors
            + [
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(),
            ],
            foreign_pre_chain=shared_processors,
        )
    else:
        console_formatter = structlog.stdlib.ProcessorFormatter(
            processors=shared_processors
            + [
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.dev.ConsoleRenderer(colors=True),
            ],
            foreign_pre_chain=shared_processors,
        )
    console_handler.setFormatter(console_formatter)

    # Configure stdlib logging with console only
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        handlers=[console_handler],
        force=True,
    )

    # Configure structlog to use stdlib logging
    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
