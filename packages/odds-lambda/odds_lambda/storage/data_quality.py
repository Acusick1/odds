"""Record a data-quality issue both in the log stream and in ``data_quality_logs``."""

from __future__ import annotations

from typing import Any

import structlog
from odds_core.models import DataQualityLog
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger()


def record_issue(
    session: AsyncSession,
    event_id: str,
    issue_type: str,
    description: str,
    context: dict[str, Any],
) -> None:
    """Log a warning named ``issue_type`` and add a matching ``DataQualityLog`` row; no flush."""
    logger.warning(issue_type, event_id=event_id, description=description, **context)
    session.add(
        DataQualityLog(
            event_id=event_id,
            severity="warning",
            issue_type=issue_type,
            description=description,
            raw_data=context,
        )
    )
