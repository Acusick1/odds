"""Unit tests for event id conventions."""

from __future__ import annotations

import pytest
from odds_core.event_ids import LIVE_EVENT_ID_PREFIX, is_odds_api_event_id


class TestIsOddsApiEventId:
    @pytest.mark.parametrize(
        "event_id",
        [
            "op_2025-2026_LIV_BOU_2025-08-15",
            "fduk_2025_1",
            f"{LIVE_EVENT_ID_PREFIX}ars_che_2026-01-01",
        ],
    )
    def test_scraped_ids(self, event_id: str) -> None:
        assert not is_odds_api_event_id(event_id)

    def test_hex_id(self) -> None:
        assert is_odds_api_event_id("625c096fabb6047cbeb082dab9affce4")
