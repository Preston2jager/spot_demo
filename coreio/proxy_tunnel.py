"""Lend the CoreIO this laptop's internet, for as long as a deploy takes.

The CoreIO has no default route and no DNS — `getent hosts pypi.org` fails — so nothing
on it can apt-get, pip install, or fetch a tarball. `core_io_setup.md` solves that with a
reverse proxy tunnel; this is that tunnel, with no squid to install and nothing left
running afterwards.

An SSH **remote** forward makes the robot's `127.0.0.1:3128` come out here, and each
connection that arrives is served as an HTTP proxy: `CONNECT` for TLS, absolute-form
requests for plain HTTP. The laptop is on two networks at once — Spot's wifi and a real
one — which is the whole trick.

    with ReverseProxy(ssh_client) as proxy:
        run(client, f"http_proxy={proxy.url} wget -qO- https://pypi.org")

`docker build --network host` is what carries it into a build: the RUN steps then share
the CoreIO's loopback, so `http://127.0.0.1:3128` means this laptop there too. Without
`--network host` the container gets its own loopback and the proxy is invisible.
"""
import socket
import threading
from typing import Self

from loguru import logger

_CHUNK = 65536
_HEADER_LIMIT = 65536
# Long enough for a slow package index, short enough that a wedged connection during a
# build does not hang the whole thing.
_TIMEOUT = 120.0


class ReverseProxy:
    """An HTTP proxy on the robot's localhost, served from this machine."""

    def __init__(self, client: object, port: int = 3128) -> None:
        self.client = client
        self.port = port
        self._forwarded = False

    @property
    def url(self) -> str:
        """What to set http_proxy to, from the robot's point of view."""
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> Self:
        transport = self.client.get_transport()
        transport.request_port_forward("127.0.0.1", self.port, handler=self._on_connection)
        self._forwarded = True
        logger.info(f"[proxy] the CoreIO's {self.url} now comes out of this laptop")
        return self

    def __exit__(self, *_exc: object) -> None:
        if not self._forwarded:
            return
        try:
            self.client.get_transport().cancel_port_forward("127.0.0.1", self.port)
        except Exception as exc:
            logger.debug(f"[proxy] could not cancel the forward: {exc}")
        self._forwarded = False

    # ── one connection ────────────────────────────────────────────────────────

    def _on_connection(self, channel: object, _origin: object, _server: object) -> None:
        """Hand the connection to a thread of its own.

        Paramiko calls this from the transport's own loop, so anything slow here would
        stall every other channel on the session — including the one streaming the build.
        """
        threading.Thread(target=self._serve, args=(channel,), daemon=True).start()

    def _serve(self, channel: object) -> None:
        """Speak HTTP-proxy on one channel until it is done."""
        upstream = None
        try:
            head, rest = _read_headers(channel)
            if head is None:
                return
            request_line = head.split(b"\r\n", 1)[0].decode("latin-1")
            method, target, _version = request_line.split(" ", 2)

            if method.upper() == "CONNECT":
                host, _, port = target.rpartition(":")
                upstream = socket.create_connection((host, int(port)), timeout=30)
                channel.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
            else:
                host, port, path = _split_absolute(target)
                upstream = socket.create_connection((host, port), timeout=30)
                upstream.sendall(_to_origin_form(head, path, host) + rest)
                rest = b""

            if rest:
                upstream.sendall(rest)
            _pump(channel, upstream)
        except Exception as exc:
            logger.debug(f"[proxy] connection failed: {type(exc).__name__}: {exc}")
            try:
                channel.sendall(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
            except Exception:
                pass
        finally:
            for closeable in (channel, upstream):
                try:
                    if closeable is not None:
                        closeable.close()
                except Exception:
                    pass


# ── helpers ───────────────────────────────────────────────────────────────────


def _read_headers(channel: object) -> tuple[bytes | None, bytes]:
    """Read up to the blank line, returning the head and whatever body came with it."""
    buffer = b""
    while b"\r\n\r\n" not in buffer:
        chunk = channel.recv(_CHUNK)
        if not chunk:
            return (None, b"")
        buffer += chunk
        if len(buffer) > _HEADER_LIMIT:
            raise ValueError("request head too large")
    head, _, rest = buffer.partition(b"\r\n\r\n")
    return head, rest


def _split_absolute(target: str) -> tuple[str, int, str]:
    """Pull host, port and path out of the absolute-form URI a proxy request carries."""
    if not target.startswith("http://"):
        raise ValueError(f"cannot proxy {target!r}")
    authority, _, path = target[len("http://"):].partition("/")
    host, _, port = authority.partition(":")
    return host, int(port or 80), "/" + path


def _to_origin_form(head: bytes, path: str, host: str) -> bytes:
    """Rewrite the head for the origin server: origin-form path, no proxy hop headers."""
    lines = head.split(b"\r\n")
    method, _, version = lines[0].decode("latin-1").split(" ", 2)
    rewritten = [f"{method} {path} {version}".encode("latin-1")]
    seen_host = False
    for line in lines[1:]:
        name = line.split(b":", 1)[0].strip().lower()
        if name in (b"proxy-connection", b"proxy-authorization"):
            continue
        if name == b"host":
            seen_host = True
        rewritten.append(line)
    if not seen_host:
        rewritten.append(f"Host: {host}".encode("latin-1"))
    return b"\r\n".join(rewritten) + b"\r\n\r\n"


def _pump(channel: object, upstream: socket.socket) -> None:
    """Copy bytes both ways until either side stops talking."""
    channel.settimeout(_TIMEOUT)
    upstream.settimeout(_TIMEOUT)

    def forward(read: object, write: object) -> None:
        try:
            while True:
                data = read.recv(_CHUNK)
                if not data:
                    break
                write.sendall(data)
        except Exception:
            pass
        finally:
            try:
                write.shutdown(socket.SHUT_WR)
            except Exception:
                pass

    outbound = threading.Thread(target=forward, args=(channel, upstream), daemon=True)
    outbound.start()
    forward(upstream, channel)
    outbound.join(timeout=_TIMEOUT)
