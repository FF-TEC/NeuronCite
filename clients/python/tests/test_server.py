"""Tests for ``neuroncite.server.NeuronCiteServer``.

Tests verify the subprocess manager's URL construction, context manager
protocol, and error class attributes. Subprocess spawning is not tested here
because it requires a running binary; these tests focus on the Python logic.
"""

from __future__ import annotations

import pytest

from neuroncite.server import NeuronCiteServer, ServerStartError


class TestServerProperties:
    """Tests for ``NeuronCiteServer`` property methods and initialization."""

    def test_url_default(self) -> None:
        """The default URL uses ``127.0.0.1:3030``."""
        server = NeuronCiteServer(binary_path="/usr/bin/neuroncite")
        assert server.url == "http://127.0.0.1:3030"

    def test_url_custom_port(self) -> None:
        """A custom port is reflected in the URL."""
        server = NeuronCiteServer(binary_path="/usr/bin/neuroncite", port=8080)
        assert server.url == "http://127.0.0.1:8080"

    def test_url_custom_bind_address(self) -> None:
        """A custom bind address is reflected in the URL."""
        server = NeuronCiteServer(
            binary_path="/usr/bin/neuroncite",
            bind_address="0.0.0.0",
            port=3030,
        )
        assert server.url == "http://0.0.0.0:3030"

    def test_double_start_raises(self) -> None:
        """Calling ``start()`` twice without ``stop()`` raises ``RuntimeError``.

        This test simulates the double-start condition by setting the internal
        ``_process`` attribute to a non-None value.
        """
        server = NeuronCiteServer(binary_path="/usr/bin/neuroncite")
        server._process = object()  # Simulate a running process
        with pytest.raises(RuntimeError, match="already running"):
            server.start()
        server._process = None  # Clean up

    def test_stop_when_not_running(self) -> None:
        """Calling ``stop()`` on a non-running server is a no-op."""
        server = NeuronCiteServer(binary_path="/usr/bin/neuroncite")
        server.stop()  # Should not raise


class TestServerStartError:
    """Tests for the ``ServerStartError`` exception class."""

    def test_attributes(self) -> None:
        """The exception stores ``returncode`` and ``stderr`` attributes."""
        err = ServerStartError(
            "Server exited with code 1",
            returncode=1,
            stderr="error: port already in use",
        )
        assert err.returncode == 1
        assert err.stderr == "error: port already in use"
        assert "Server exited with code 1" in str(err)

    def test_defaults(self) -> None:
        """Default values for optional attributes."""
        err = ServerStartError("timeout")
        assert err.returncode is None
        assert err.stderr == ""
