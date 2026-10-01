"""A local port that comes out on the CoreIO — `ssh -L`, without a second terminal.

Dockge's UI listens on CoreIO port 5001, which Spot does not forward off the payload, so it
used to take ``ssh -p 20022 -L 5001:127.0.0.1:5001 spot@192.168.80.3`` in another window.
`start.py` now opens the same forward itself: ``http://localhost:8081`` is Dockge while the
console is up. Plain TCP over a `direct-tcpip` channel per connection, so Dockge's websocket
works exactly as through ``ssh -L``.

Never fatal. No password, an unreachable CoreIO or a busy local port is one warning and the
console carries on; a dropped SSH session is reopened on the next connection.

    DOCKGE_LOCAL_PORT=8081   # 0 turns it off
    DOCKGE_REMOTE_PORT=5001
"""
import os
import select
import socket
import threading

import paramiko
from loguru import logger

_CHUNK = 65536


class PortForward:
    """Listen on ``bind:local_port`` and carry each connection to CoreIO ``127.0.0.1:remote_port``."""

    def __init__(self, local_port: int, remote_port: int, label: str, bind: str = "127.0.0.1") -> None:
        self.local_port = local_port
        self.remote_port = remote_port
        self.label = label
        self.bind = bind
        self.host = os.getenv("COREIO_HOST", "192.168.80.3")
        self.port = int(os.getenv("COREIO_PORT", "20022"))
        self.username = os.getenv("COREIO_USER", "spot")
        self.password = os.getenv("COREIO_PASSWORD", "")
        self._client: paramiko.SSHClient | None = None
        self._lock = threading.Lock()

    def start(self) -> bool:
        """Start listening in the background. False (with a warning) if it cannot."""
        if not self.password:
            logger.warning(f"{self.label}: COREIO_PASSWORD is not set — no forward on :{self.local_port}")
            return False
        try:
            self._transport()
        except Exception as exc:
            logger.warning(f"{self.label}: cannot SSH to {self.username}@{self.host}:{self.port} ({exc}) "
                           f"— will retry when :{self.local_port} is opened")
        try:
            server = socket.create_server((self.bind, self.local_port), reuse_port=False)
        except OSError as exc:
            logger.warning(f"{self.label}: local port {self.local_port} is unavailable ({exc})")
            return False
        threading.Thread(target=self._accept, args=(server,), daemon=True, name=f"forward-{self.local_port}").start()
        logger.success(f"{self.label} at http://localhost:{self.local_port} (CoreIO :{self.remote_port})")
        return True

    def _transport(self) -> paramiko.Transport:
        """The SSH transport, reopened if it has dropped."""
        with self._lock:
            transport = self._client.get_transport() if self._client is not None else None
            if transport is not None and transport.is_active():
                return transport
            if self._client is not None:
                self._client.close()
            client = paramiko.SSHClient()
            client.load_system_host_keys()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect(hostname=self.host, port=self.port, username=self.username, password=self.password,
                           timeout=10.0, allow_agent=False, look_for_keys=False)
            transport = client.get_transport()
            transport.set_keepalive(30)
            self._client = client
            return transport

    def _accept(self, server: socket.socket) -> None:
        while True:
            conn, peer = server.accept()
            threading.Thread(target=self._serve, args=(conn, peer), daemon=True).start()

    def _serve(self, conn: socket.socket, peer: tuple) -> None:
        try:
            channel = self._transport().open_channel(
                "direct-tcpip", ("127.0.0.1", self.remote_port), peer, timeout=10.0
            )
        except Exception as exc:
            logger.debug(f"{self.label}: could not open a channel to CoreIO :{self.remote_port} ({exc})")
            conn.close()
            return
        try:
            while True:
                readable, _, _ = select.select([conn, channel], [], [])
                if conn in readable:
                    data = conn.recv(_CHUNK)
                    if not data:
                        break
                    channel.sendall(data)
                if channel in readable:
                    data = channel.recv(_CHUNK)
                    if not data:
                        break
                    conn.sendall(data)
        except OSError:
            pass
        finally:
            channel.close()
            conn.close()


def start_dockge_forward() -> PortForward | None:
    """Dockge's UI on ``DOCKGE_LOCAL_PORT`` (8081), unless that is 0."""
    local = int(os.getenv("DOCKGE_LOCAL_PORT", "8081"))
    if local <= 0:
        return None
    forward = PortForward(local, int(os.getenv("DOCKGE_REMOTE_PORT", "5001")), "Dockge")
    return forward if forward.start() else None


__all__ = ["PortForward", "start_dockge_forward"]
