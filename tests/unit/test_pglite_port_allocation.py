import socket
from collections.abc import Callable
from typing import Any

import pytest
from py_pglite.config import PGliteConfig
from py_pglite.manager import PGliteManager


class TestPglitePortAllocation:
    def test_occupied_port_is_retried(
        self, start_pglite_manager: Callable[[PGliteManager], None], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original_socket = socket.socket
        with original_socket() as occupied:
            occupied.bind(("127.0.0.1", 0))
            occupied.listen()
            occupied_port = occupied.getsockname()[1]
            calls = 0

            class FirstPortSocket(original_socket):
                def getsockname(self) -> tuple[str, int]:
                    return ("127.0.0.1", occupied_port)

            def choose_socket(*args: Any, **kwargs: Any) -> socket.socket:
                nonlocal calls
                calls += 1
                return (
                    FirstPortSocket(*args, **kwargs)
                    if calls == 1
                    else original_socket(*args, **kwargs)
                )

            monkeypatch.setattr(socket, "socket", choose_socket)
            manager = PGliteManager(PGliteConfig(use_tcp=True, timeout=30))
            try:
                start_pglite_manager(manager)
                assert manager.config.tcp_port != occupied_port
                assert manager.is_running()
            finally:
                manager.stop()
