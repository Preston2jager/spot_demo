"""The RealSense on Spot's CoreIO, as seen from the console.

The other half is `coreio/service/realsense_server.py`, which subscribes to the topics
`realsense2_camera` already publishes and serves them as pictures — the camera itself
belongs to that ROS 2 node and cannot be opened twice. Deploy it with
`python coreio/deploy_realsense.py`.

Everything rides the same `CoreIOLink` as the detector: an SSH channel per request from
the laptop, a plain socket when the console is running on the CoreIO itself. Set
`COREIO_RS_ENABLED=0` to leave the panel out entirely — a robot with no RealSense fitted
should not be asked about one several times a second.

Two kinds of answer, and the difference matters to the caller:

* `color_jpeg()` / `depth_jpeg()` are *pictures*, for the video panel.
* `depth_at(x, y)` is a *measurement*, in metres, from the aligned depth frame — so the
  pixel the operator clicked in the colour image is the pixel that is measured.
* `rgbd()` is raw data for the 3D reconstruction: colour, depth in metres and intrinsics.
"""
import json
import os
import struct
import threading
from typing import Self

import cv2
import numpy as np
from loguru import logger

from .coreio_link import CoreIOLink, CoreIOUnavailable

__all__ = ["CoreIOUnavailable", "RealSenseClient", "build_realsense", "realsense_enabled"]


def _flag(name: str, default: bool) -> bool:
    """Read a boolean from the environment, treating an unset variable as the default."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def realsense_enabled() -> bool:
    """Whether the RealSense panel should be offered at all."""
    return _flag("COREIO_RS_ENABLED", True)


class RealSenseClient:
    """Colour, depth, and distances from the CoreIO's RealSense.

    One SSH session, serialised: the console's capture loop pulls frames while a browser
    click can ask for a distance at the same moment, and paramiko would rather open those
    channels one at a time than have two threads racing on the same transport. The lock
    costs nothing at the two or three requests a second this actually sees.
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        username: str | None = None,
        password: str | None = None,
        service_port: int | None = None,
        timeout: float | None = None,
        direct: bool | None = None,
    ) -> None:
        self.host = host or os.getenv("COREIO_HOST", "192.168.80.3")
        self.port = int(port or os.getenv("COREIO_PORT", "20022"))
        self.username = username or os.getenv("COREIO_USER", "spot")
        self.password = password or os.getenv("COREIO_PASSWORD", "")
        self.service_port = int(service_port or os.getenv("COREIO_RS_PORT", "8782"))
        self.timeout = float(timeout if timeout is not None else os.getenv("COREIO_RS_TIMEOUT", "8"))
        # True only when this process is already on the CoreIO — the demo_web container on
        # the host network reaches the bridge as a localhost port, no SSH needed.
        self.direct = _flag("COREIO_RS_DIRECT", False) if direct is None else direct
        self._lock = threading.Lock()
        self._link = CoreIOLink(
            service_port=self.service_port,
            label="CoreIO RealSense",
            host=self.host,
            port=self.port,
            username=self.username,
            password=self.password,
            timeout=self.timeout,
            direct=self.direct,
            service_host=os.getenv("COREIO_RS_HOST", "127.0.0.1"),
        )

    def __enter__(self) -> Self:
        self.connect()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def connect(self, timeout: float = 15.0) -> None:
        """Open the SSH session the frames come down."""
        with self._lock:
            self._link.connect(timeout)

    def close(self) -> None:
        with self._lock:
            self._link.close()

    # ── what the console asks for ─────────────────────────────────────────────

    def health(self) -> dict:
        """What is arriving and how old it is — the cheapest proof the bridge is alive."""
        with self._lock:
            return self._link.json("GET", "/health")

    def color_jpeg(self) -> bytes:
        """The latest colour frame, already JPEG — the console forwards it untouched."""
        with self._lock:
            return self._link.binary("/color.jpg")

    def depth_jpeg(self) -> bytes:
        """The latest depth frame, colour-mapped on the robot.

        Mapped there rather than here because the alternative is shipping a 16-bit frame
        over Spot's wifi every time somebody looks at it, to render it exactly the same
        way. Pixels with no reading come back black; that is not a colour in the ramp.
        """
        with self._lock:
            return self._link.binary("/depth.jpg")

    def depth_at(self, u: float, v: float, window: int = 9) -> dict:
        """The distance in metres where the operator pointed, as a fraction of the frame.

        Fractions rather than pixels because the browser is looking at a scaled copy: the
        page knows where in the picture the click landed, and the bridge knows how big the
        sensor's frame is. Medianing over a small window is what makes the number readable
        — single depth pixels drop out constantly on edges and dark surfaces, and a
        readout flickering between 1.4 m and nothing says less than no readout at all.
        """
        u = min(max(float(u), 0.0), 1.0)
        v = min(max(float(v), 0.0), 1.0)
        with self._lock:
            return self._link.json("GET", f"/depth/at?u={u:.6f}&v={v:.6f}&window={int(window)}")

    def rgbd(self, step: int = 2) -> dict:
        """Colour (BGR), depth (metres, 0 = no reading) and intrinsics of one moment.

        Needs the bridge's ``/rgbd`` — redeploy with `python coreio/deploy_realsense.py` if
        an older bridge answers 404.
        """
        with self._lock:
            raw = self._link.binary(f"/rgbd?step={int(step)}")
        if raw[:4] != b"RGBD":
            raise CoreIOUnavailable("CoreIO RealSense sent something that is not an RGBD frame")
        header_len, jpg_len, png_len = struct.unpack(">III", raw[4:16])
        offset = 16
        header = json.loads(raw[offset: offset + header_len])
        offset += header_len
        colour = cv2.imdecode(np.frombuffer(raw[offset: offset + jpg_len], np.uint8), cv2.IMREAD_COLOR)
        offset += jpg_len
        depth = cv2.imdecode(np.frombuffer(raw[offset: offset + png_len], np.uint8), cv2.IMREAD_UNCHANGED)
        if colour is None or depth is None:
            raise CoreIOUnavailable("CoreIO RealSense RGBD frame did not decode")
        header["color"] = colour
        header["depth"] = depth.astype(np.float32) * 0.001
        return header


def build_realsense(quiet: bool = False) -> RealSenseClient | None:
    """Connect to the bridge, or return None with one line saying why not.

    None rather than an exception: a missing RealSense is a panel the console does not
    offer, not a console that fails to start. The check is done once, at startup, for the
    same reason the detector caches its failures — a camera that is not fitted will not
    appear because something asked again.

    ``quiet`` turns the failure lines down to debug, for the console's slow background
    retry: a bridge deployed after the console started should light the tabs up, but a
    bridge that is simply not there should not say so twice a minute.
    """
    warn = logger.debug if quiet else logger.warning
    if not realsense_enabled():
        logger.debug("RealSense: disabled by COREIO_RS_ENABLED=0")
        return None
    client = RealSenseClient()
    try:
        client.connect()
        health = client.health()
    except CoreIOUnavailable as exc:
        warn(f"RealSense: {exc} — the depth panel will be unavailable")
        client.close()
        return None
    except Exception as exc:
        warn(f"RealSense: unexpected failure ({exc}) — the depth panel will be unavailable")
        client.close()
        return None

    colour = health.get("color", {})
    depth = health.get("depth", {})
    logger.info(
        "RealSense bridge up — colour {}, depth {}".format(
            "x".join(str(v) for v in colour.get("size") or []) or "waiting",
            "x".join(str(v) for v in depth.get("size") or []) or "waiting",
        )
    )
    return client
