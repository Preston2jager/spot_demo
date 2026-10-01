"""One HTTP connection to a service running on Spot's CoreIO.

Spot forwards only ports 20022 (SSH) and 21443 (nginx) off the CoreIO, so a service that
binds CoreIO-localhost — which is all of them, deliberately — cannot be dialled from the
laptop at all. Every request rides a `direct-tcpip` channel on an SSH session instead,
opened with the same `COREIO_*` credentials the speaker uses. Nothing new is exposed on
the robot, and a channel per request costs no handshake while keeping a wedged call from
poisoning the next one.

The tunnel is only needed from *outside* the robot. Code running on the CoreIO itself —
the `demo_web` container is on the host network — reaches the same service as a plain
localhost socket, which is what `direct=True` is: no SSH login into the machine you are
already running on. Only `_open_stream` knows which of the two it is.

Two clients ride this: `spot_detector_coreio.CoreIODetector` and
`spot_realsense.RealSenseClient`. They ask for different things — one posts JPEGs and
reads JSON, the other reads pictures — but the transport under them is the same, and
`label` is what makes a failure say which service could not be reached.
"""
import json
import socket

import paramiko
from loguru import logger

_HEADER_END = b"\r\n\r\n"


class CoreIOUnavailable(RuntimeError):
    """A CoreIO service could not be reached, or answered with a fault.

    Deliberately distinct from an unwelcome *answer*: an empty detection list or a depth
    frame with no valid pixels is a real reply from a healthy service, and treating either
    as unavailability is what sends callers down a fallback path they did not need.
    """


class CoreIOLink:
    """An SSH session to the CoreIO, and one-shot HTTP requests over it."""

    def __init__(
        self,
        service_port: int,
        label: str,
        host: str,
        port: int,
        username: str,
        password: str,
        timeout: float = 10.0,
        direct: bool = False,
        service_host: str = "127.0.0.1",
    ) -> None:
        self.service_port = service_port
        self.label = label
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.timeout = timeout
        self.direct = direct
        self.service_host = service_host
        self._client: paramiko.SSHClient | None = None

    # ── connection ────────────────────────────────────────────────────────────

    def connect(self, timeout: float = 15.0) -> None:
        """Open the SSH session that carries the channels.

        A no-op in direct mode, where each request dials the service itself — but still
        worth calling there, because callers follow it with a health check and that is
        what proves the service is up either way.
        """
        if self.direct:
            return
        if not self.password:
            raise CoreIOUnavailable("COREIO_PASSWORD is not set — add it to .env")
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(
                hostname=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                timeout=timeout,
                allow_agent=False,
                look_for_keys=False,
            )
        except Exception as exc:
            raise CoreIOUnavailable(
                f"cannot SSH to {self.username}@{self.host}:{self.port} — {exc}"
            ) from exc
        self._client = client
        logger.debug(f"{self.label}: connected to {self.username}@{self.host}:{self.port}")

    def close(self) -> None:
        """Close the SSH session if one is open. Direct mode holds nothing to close."""
        if self._client is not None:
            self._client.close()
            self._client = None

    @property
    def connected(self) -> bool:
        """Whether a request could be made right now without opening a session first."""
        if self.direct:
            return True
        if self._client is None:
            return False
        transport = self._client.get_transport()
        return transport is not None and transport.is_active()

    # ── requests ──────────────────────────────────────────────────────────────

    def request(
        self,
        method: str,
        path: str,
        body: bytes = b"",
        content_type: str | None = None,
    ) -> tuple[int, bytes]:
        """Send one request down a fresh connection and return (status, body).

        `Connection: close` keeps this to reading until EOF — no keep-alive or chunked
        decoding to get wrong. The connection is an SSH channel from off the robot and a
        plain socket from on it; both are read the same way.
        """
        channel = self._open_stream()
        channel.settimeout(self.timeout)
        try:
            head = (
                f"{method} {path} HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{self.service_port}\r\n"
                f"Connection: close\r\n"
                f"Content-Length: {len(body)}\r\n"
            )
            if content_type:
                head += f"Content-Type: {content_type}\r\n"
            channel.sendall(head.encode("utf-8") + b"\r\n" + body)
            raw = self._read_until_eof(channel)
        except CoreIOUnavailable:
            raise
        except Exception as exc:
            raise CoreIOUnavailable(f"{self.label} request failed: {exc}") from exc
        finally:
            channel.close()
        return self._split(raw)

    def json(
        self,
        method: str,
        path: str,
        body: bytes = b"",
        content_type: str | None = None,
    ) -> dict:
        """A request whose reply is JSON, with any non-200 raised as unavailability."""
        status, payload = self.request(method, path, body, content_type)
        try:
            decoded = json.loads(payload.decode("utf-8"))
        except ValueError as exc:
            raise CoreIOUnavailable(f"{self.label} returned non-JSON (HTTP {status})") from exc
        if status != 200:
            raise CoreIOUnavailable(f"{self.label} returned HTTP {status}: {decoded.get('error')}")
        return decoded

    def binary(self, path: str) -> bytes:
        """A GET whose reply is a picture. A non-200 carries a JSON reason; use it."""
        status, payload = self.request("GET", path)
        if status != 200:
            reason = payload[:200].decode("utf-8", "replace")
            try:
                reason = json.loads(payload).get("error", reason)
            except ValueError:
                pass
            raise CoreIOUnavailable(f"{self.label} returned HTTP {status}: {reason}")
        return payload

    # ── plumbing ──────────────────────────────────────────────────────────────

    def _open_stream(self):
        """Open one connection to the service — a socket in direct mode, else a tunnel."""
        if self.direct:
            try:
                return socket.create_connection(
                    (self.service_host, self.service_port), timeout=self.timeout
                )
            except OSError as exc:
                raise CoreIOUnavailable(
                    f"nothing answering on {self.service_host}:{self.service_port} — "
                    f"is the service running? ({exc})"
                ) from exc

        if self._client is None:
            self.connect()
        transport = self._client.get_transport()  # type: ignore[union-attr]
        if transport is None or not transport.is_active():
            raise CoreIOUnavailable("SSH session to the CoreIO has dropped")
        try:
            return transport.open_channel(
                "direct-tcpip", ("127.0.0.1", self.service_port), ("127.0.0.1", 0)
            )
        except Exception as exc:
            raise CoreIOUnavailable(
                f"CoreIO refused a tunnel to 127.0.0.1:{self.service_port} — "
                f"is the service running? ({exc})"
            ) from exc

    def _read_until_eof(self, channel: object) -> bytes:
        """Drain the connection until the server closes it."""
        chunks: list[bytes] = []
        while True:
            chunk = channel.recv(65536)  # type: ignore[attr-defined]
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)

    def _split(self, raw: bytes) -> tuple[int, bytes]:
        """Status code and body out of a raw HTTP response."""
        if not raw:
            raise CoreIOUnavailable(f"{self.label} closed the connection without answering")
        head, _, body = raw.partition(_HEADER_END)
        status_line = head.split(b"\r\n", 1)[0].decode("latin-1", "replace")
        try:
            status = int(status_line.split()[1])
        except (IndexError, ValueError) as exc:
            raise CoreIOUnavailable(
                f"unparseable reply from {self.label}: {status_line!r}"
            ) from exc
        return status, body
