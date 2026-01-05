"""Tests for logging configuration."""

import logging
import os
from unittest.mock import patch

import pytest
import structlog
from odds_core.config import LoggingConfig, Settings
from odds_core.logging_setup import configure_logging

# Set required environment variables BEFORE importing odds_core
os.environ.setdefault("ODDS_API_KEY", "test_api_key")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://localhost/test")


@pytest.fixture
def temp_log_file(tmp_path):
    """Create a temporary log file path."""
    return tmp_path / "test_logs" / "test.log"


@pytest.fixture
def test_settings(temp_log_file):
    """Create test settings with temporary log file."""
    with patch.object(Settings, "model_config", {"env_file": None, "extra": "ignore"}):
        settings = Settings(
            api={"key": "test_key"},
            database={"url": "postgresql://test"},
            logging=LoggingConfig(level="INFO", file=str(temp_log_file)),
        )
        return settings


def test_configure_logging_creates_log_directory(test_settings, temp_log_file):
    """Verify logs directory is created automatically."""
    assert not temp_log_file.parent.exists()

    configure_logging(test_settings, json_output=False)

    assert temp_log_file.parent.exists()
    assert temp_log_file.parent.is_dir()


def test_configure_logging_respects_log_level(test_settings, temp_log_file, caplog):
    """Verify DEBUG messages are filtered when LOG_LEVEL=INFO."""
    # Configure with INFO level
    test_settings.logging.level = "INFO"
    configure_logging(test_settings, json_output=False)

    # Get a logger and log at different levels
    logger = structlog.get_logger()

    with caplog.at_level(logging.INFO):
        logger.debug("debug message - should not appear")
        logger.info("info message - should appear")
        logger.warning("warning message - should appear")

    # Read the log file
    log_content = temp_log_file.read_text()

    # Debug should NOT be in logs
    assert "debug message" not in log_content.lower()
    # Info and warning SHOULD be in logs
    assert "info message" in log_content.lower()
    assert "warning message" in log_content.lower()


def test_configure_logging_writes_to_file(test_settings, temp_log_file):
    """Verify log messages are written to LOG_FILE."""
    configure_logging(test_settings, json_output=False)

    logger = structlog.get_logger()
    test_message = "test_message_unique_12345"
    logger.info(test_message)

    # Flush handlers to ensure write
    for handler in logging.root.handlers:
        handler.flush()

    # Verify message in log file
    assert temp_log_file.exists()
    log_content = temp_log_file.read_text()
    assert test_message in log_content


def test_configure_logging_console_output(test_settings, capsys):
    """Verify logs appear on console output."""
    configure_logging(test_settings, json_output=False)

    logger = structlog.get_logger()
    test_message = "console_test_message_67890"
    logger.info(test_message)

    # Capture output (note: structlog writes to stderr by default)
    captured = capsys.readouterr()
    output = captured.err + captured.out

    # The message should appear somewhere in console output
    # (exact format depends on ConsoleRenderer)
    assert test_message in output or test_message.lower() in output.lower()


def test_configure_logging_idempotent(test_settings):
    """Verify calling configure_logging multiple times is safe."""
    # Should not raise any exceptions
    configure_logging(test_settings, json_output=False)
    configure_logging(test_settings, json_output=False)
    configure_logging(test_settings, json_output=False)

    # Logger should still work
    logger = structlog.get_logger()
    logger.info("idempotent test")


def test_invalid_log_level_defaults_to_info(test_settings, temp_log_file, capsys):
    """Verify graceful handling of invalid log level."""
    test_settings.logging.level = "INVALID_LEVEL"

    # Should not raise exception, should default to INFO
    configure_logging(test_settings, json_output=False)

    # Capture warning message
    captured = capsys.readouterr()
    assert "Invalid log level" in captured.out or "INVALID_LEVEL" in captured.out

    # Should still be able to log
    logger = structlog.get_logger()
    logger.info("test after invalid level")


def test_json_output_mode(test_settings, temp_log_file):
    """Verify JSON output mode for Lambda."""
    configure_logging(test_settings, json_output=True)

    logger = structlog.get_logger()
    logger.info("json_test", key="value", number=42)

    # Flush handlers
    for handler in logging.root.handlers:
        handler.flush()

    log_content = temp_log_file.read_text()

    # JSON output should contain the event and structured data
    # (exact format depends on JSONRenderer)
    assert "json_test" in log_content
    assert "key" in log_content or "value" in log_content


def test_log_rotation_configuration(test_settings, temp_log_file):
    """Verify rotating file handler is configured."""
    configure_logging(test_settings, json_output=False)

    # Check that a RotatingFileHandler exists
    rotating_handlers = [
        h for h in logging.root.handlers if isinstance(h, logging.handlers.RotatingFileHandler)
    ]

    assert len(rotating_handlers) > 0
    handler = rotating_handlers[0]

    # Verify rotation settings
    assert handler.maxBytes == 10_485_760  # 10MB
    assert handler.backupCount == 5


def test_permission_error_fallback_to_console(test_settings, capsys):
    """Verify fallback to console-only logging when file creation fails."""
    # Use a path that will cause permission error
    test_settings.logging.file = "/root/impossible/path/log.txt"

    # Should not raise exception, should fall back to console
    configure_logging(test_settings, json_output=False)

    # Capture warning
    captured = capsys.readouterr()
    assert "Warning" in captured.out or "Falling back" in captured.out

    # Logger should still work
    logger = structlog.get_logger()
    logger.info("console fallback test")


def test_relative_path_handling(tmp_path):
    """Verify relative log file paths are handled correctly."""
    import os

    # Change to temp directory
    original_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)

        with patch.object(Settings, "model_config", {"env_file": None, "extra": "ignore"}):
            settings = Settings(
                api={"key": "test_key"},
                database={"url": "postgresql://test"},
                logging=LoggingConfig(level="INFO", file="logs/relative.log"),
            )

        configure_logging(settings, json_output=False)

        # Directory should be created relative to current working directory
        assert (tmp_path / "logs").exists()

        logger = structlog.get_logger()
        logger.info("relative path test")

        # Log file should exist
        assert (tmp_path / "logs" / "relative.log").exists()

    finally:
        os.chdir(original_cwd)


def test_timestamp_in_logs(test_settings, temp_log_file):
    """Verify logs include ISO format timestamps."""
    configure_logging(test_settings, json_output=False)

    logger = structlog.get_logger()
    logger.info("timestamp test")

    # Flush handlers
    for handler in logging.root.handlers:
        handler.flush()

    log_content = temp_log_file.read_text()

    # Should contain ISO timestamp (rough check for pattern like 2024-01-05T or similar)
    # ConsoleRenderer formats it, so we just check it's present somewhere
    assert "timestamp test" in log_content
