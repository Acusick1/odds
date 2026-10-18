import pytest
from odds_core.event_source import EventSource, canonical_event_id


class TestEventSource:
    @pytest.mark.parametrize(
        ("event_id", "source"),
        [
            ("abc123", EventSource.ODDS_API),
            ("op_2025-2026_ARS_CHE_2026-04-15", EventSource.ODDSPORTAL_HISTORICAL),
            ("op_live_ARS_CHE_2026-04-15T1500", EventSource.ODDSPORTAL_LIVE),
            ("fduk_2025-2026_ARS_CHE_2026-04-15", EventSource.FOOTBALL_DATA_UK),
        ],
    )
    def test_of(self, event_id: str, source: EventSource) -> None:
        assert EventSource.of(event_id) is source

    def test_canonical_prefers_api_then_source_order_then_id(self) -> None:
        assert canonical_event_id(["fduk_x", "op_live_x", "zzz"]) == "zzz"
        assert canonical_event_id(["fduk_x", "op_live_x", "op_2025_x"]) == "op_2025_x"
        assert canonical_event_id(["op_live_b", "op_live_a"]) == "op_live_a"
