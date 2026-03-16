"""Subprocess manager for launching and stopping a NeuronCite server.

``NeuronCiteServer`` starts the ``neuroncite serve`` binary as a child process,
polls the health endpoint until the server is ready, and terminates the process
on shutdown.  It implements the context manager protocol so the server lifetime
can be scoped to a ``with`` block.
"""

from __future__ import annotations

import signal
import subprocess
import sys
import time

import requests


class ServerStartError(Exception):
    """Raised when the NeuronCite server process fails to become healthy within
    the configured timeout.

    Attributes:
        returncode: The exit code of the server process, or ``None`` if the
                    process was still running when the timeout expired.
        stderr:     Captured stderr output from the server process (may be
                    truncated).
    """

    def __init__(
        self,
        message: str,
        returncode: int | None = None,
        stderr: str = "",
    ) -> None:
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(message)


class NeuronCiteServer:
    """Manages the lifecycle of a NeuronCite server subprocess.

    The server is started with ``start()`` and stopped with ``stop()``.  The
    context manager protocol delegates to these two methods so the server
    process is cleaned up even if an exception occurs.

    Args:
        binary_path:  Filesystem path to the ``neuroncite`` executable.
        port:         TCP port for the HTTP server.
        bind_address: IP address to bind to ("127.0.0.1" for localhost only,
                      "0.0.0.0" for LAN access).
        log_level:    Server log verbosity ("error", "warn", "info", "debug",
                      "trace").
    """

    def __init__(
        self,
        binary_path: str,
        port: int = 3030,
        bind_address: str = "127.0.0.1",
        log_level: str = "info",
    ) -> None:
        self._binary_path = binary_path
        self._port = port
        self._bind_address = bind_address
        self._log_level = log_level
        self._process: subprocess.Popen[bytes] | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def url(self) -> str:
        """Base URL of the running server (e.g. ``http://127.0.0.1:3030``)."""
        return f"http://{self._bind_address}:{self._port}"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, timeout: float = 30) -> None:
        """Launch the server subprocess and block until it is healthy.

        Spawns ``neuroncite serve`` with the configured port, bind address, and
        log level, then polls ``GET /api/v1/health`` in a loop until the server
        responds with HTTP 200 or the timeout expires.

        Args:
            timeout: Maximum seconds to wait for the health endpoint to respond.

        Raises:
            ServerStartError: If the server exits prematurely or does not become
                              healthy within the timeout.
            RuntimeError:     If the server is already running (``start()`` was
                              called twice without an intervening ``stop()``).
        """
        if self._process is not None:
            raise RuntimeError("Server is already running")

        cmd = [
            self._binary_path,
            "serve",
            "--port", str(self._port),
            "--bind", self._bind_address,
            "--log-level", self._log_level,
        ]
        creation_flags = 0
        if sys.platform == "win32":
            # CREATE_NEW_PROCESS_GROUP is required for CTRL_BREAK_EVENT to
            # target the child process specifically. Without it, the signal
            # is sent to the parent's console group and may affect unrelated
            # processes or fail to reach the child.
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=creation_flags,
        )

        health_url = f"{self.url}/api/v1/health"
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            # Check whether the process exited before becoming healthy
            retcode = self._process.poll()
            if retcode is not None:
                stderr = self._process.stderr.read().decode("utf-8", errors="replace") if self._process.stderr else ""
                self._process = None
                raise ServerStartError(
                    f"Server process exited with code {retcode} before becoming healthy",
                    returncode=retcode,
                    stderr=stderr,
                )
            try:
                resp = requests.get(health_url, timeout=2)
                if resp.status_code == 200:
                    return
            except requests.ConnectionError:
                pass
            time.sleep(0.25)

        # Timeout reached -- kill the process and report
        self._kill_process()
        raise ServerStartError(
            f"Server did not become healthy within {timeout} seconds"
        )

    def stop(self, timeout: float = 15) -> None:
        """Terminate the server subprocess gracefully.

        Sends SIGINT (Unix) or CTRL_BREAK_EVENT (Windows) to request graceful
        shutdown, then waits up to *timeout* seconds for the process to exit.
        If the process does not exit in time, it is killed forcefully.

        Args:
            timeout: Maximum seconds to wait for graceful shutdown before
                     sending SIGKILL / TerminateProcess.
        """
        if self._process is None:
            return

        try:
            if sys.platform == "win32":
                # On Windows, SIGINT does not propagate to child processes
                # reliably.  CTRL_BREAK_EVENT is the closest equivalent for
                # console applications.
                self._process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                self._process.send_signal(signal.SIGINT)
        except OSError:
            # Process already terminated between the None check and the signal
            self._process = None
            return

        try:
            self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=5)
        finally:
            self._process = None

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> NeuronCiteServer:
        """Start the server and return this instance for use in a ``with`` block."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Stop the server when exiting the ``with`` block."""
        self.stop()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _kill_process(self) -> None:
        """Forcefully terminate the server process and reset internal state."""
        if self._process is not None:
            try:
                self._process.kill()
                self._process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                pass
            self._process = None
