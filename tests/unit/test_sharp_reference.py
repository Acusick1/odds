"""The agent's sharp reference is one definition shared by the MCP server and paper trading."""

from odds_core.sharp_reference import AGENT_SHARP_BOOKMAKERS


class TestAgentSharpBookmakers:
    def test_priority_order(self) -> None:
        assert list(AGENT_SHARP_BOOKMAKERS) == ["pinnacle", "betfair_exchange"]

    def test_mcp_server_uses_shared_definition(self) -> None:
        import odds_mcp.server as server

        assert server.AGENT_SHARP_BOOKMAKERS is AGENT_SHARP_BOOKMAKERS

    def test_model_default_stays_pinnacle_only(self) -> None:
        from odds_analytics.feature_extraction import DEFAULT_SHARP_BOOKMAKERS

        assert DEFAULT_SHARP_BOOKMAKERS == ["pinnacle"]
