"""The live map: Spot's depth cameras, stitched into the graph's frame on the CoreIO GPU.

The recorded map in the Map tab is a *memory* — the point clouds stored in the waypoint
snapshots, from whenever the map was walked. This is the other half: what the robot can
see right now, in the same seed frame, so the room the operator is watching on the video
panel and the room the robot is navigating are visibly the same room.

**There is no lidar on this robot.** What there is, is five depth cameras — front-left,
front-right, left, right and back — which between them cover all the way around the body,
and which are the same sensors those recorded clouds came from. A sweep is one frame from
each; stitching is putting all five into the seed frame and merging them with everything
seen so far.

The split of work is the point:

* here — fetch the five depth frames and work out where each camera was, because this is
  the process that holds Spot's credentials and its GraphNav localisation;
* on the CoreIO — unproject two and a half million candidate points, filter, transform
  and voxel-merge them, because that is a GPU's afternoon and a laptop's whole budget;
* in the browser — draw the voxels that were *new*, which is all that ever crosses wifi
  after the first second.

`coreio/service/cloud_server.py` is the other end, deployed with
`python coreio/deploy_cloud.py`. Reached exactly like the detector and the RealSense
bridge: an SSH channel per request, or a plain socket when the console is itself running
on the CoreIO.

Localisation is required, and deliberately not arranged here. Points are placed in the
seed frame the *graph* is anchored in, and without a localisation there is no such frame
— only odometry, which would draw a perfectly detailed room floating somewhere near the
map. So an unlocalised robot is reported as unlocalised, and the console offers the
operator the same `ensure_navigation()` every GraphNav mission calls.
"""
import json
import math
import os
import struct
import threading
import time
import zlib
from typing import Self

from bosdyn.client.frame_helpers import BODY_FRAME_NAME, ODOM_FRAME_NAME, get_a_tform_b
from bosdyn.client.math_helpers import SE3Pose
from loguru import logger

from .coreio_link import CoreIOLink, CoreIOUnavailable

__all__ = ["CoreIOUnavailable", "LiveCloud", "build_live_cloud", "live_cloud_enabled"]

# The five body cameras, which is the whole way around. `*_depth` rather than
# `*_depth_in_visual_frame`: the plain sources are the sensor's own 424x240 frames with
# the intrinsics that go with them, while the visual-frame variants are resampled to
# 640x480 to line up with the fisheye pictures — three times the bytes over wifi to
# describe the same geometry.
_SOURCES = [
    name.strip()
    for name in os.getenv(
        "SPOT_DEPTH_SOURCES",
        "frontleft_depth,frontright_depth,left_depth,right_depth,back_depth",
    ).split(",")
    if name.strip()
]


def _flag(name: str, default: bool) -> bool:
    """Read a boolean from the environment, treating an unset variable as the default."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def live_cloud_enabled() -> bool:
    """Whether the live map should be offered at all."""
    return _flag("COREIO_CLOUD_ENABLED", True)


class LiveCloud:
    """Sweeps of Spot's depth cameras, stitched on the CoreIO and cached for the page.

    Owns one background thread, which sweeps only while somebody is watching: the console
    calls `want()` on every status poll, and the loop stands down a few seconds after the
    polls stop. A Live tab nobody has open must not keep the robot's GPU — or its image
    service — busy for the rest of the demo.
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
        self.service_port = int(service_port or os.getenv("COREIO_CLOUD_PORT", "8783"))
        self.timeout = float(
            timeout if timeout is not None else os.getenv("COREIO_CLOUD_TIMEOUT", "20")
        )
        self.direct = _flag("COREIO_CLOUD_DIRECT", False) if direct is None else direct
        # How often to sweep. Two a second is about what Spot's image service gives for
        # five depth frames anyway, and the wire cost is the *new* voxels, not the frames.
        self.hz = float(os.getenv("COREIO_CLOUD_HZ", "2"))
        self.near_m = float(os.getenv("COREIO_CLOUD_NEAR_M", "0.35"))
        # Shorter than the recorded map's 8 m on purpose: this is drawn while the robot is
        # moving, and stereo error grows with the square of range, so a live sweep's far
        # returns are the speckle that makes a room unreadable rather than the room.
        self.far_m = float(os.getenv("COREIO_CLOUD_FAR_M", "6.0"))

        self._lock = threading.Lock()
        self._link = CoreIOLink(
            service_port=self.service_port,
            label="CoreIO cloud",
            host=self.host,
            port=self.port,
            username=self.username,
            password=self.password,
            timeout=self.timeout,
            direct=self.direct,
            service_host=os.getenv("COREIO_CLOUD_HOST", "127.0.0.1"),
        )
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._wanted_at = 0.0
        self._agent: object | None = None
        self._state = "idle"
        self._error: str | None = None
        self._stats: dict = {}
        self._robot: dict | None = None
        self._rate = 0.0

    def __enter__(self) -> Self:
        self.connect()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ── connection ────────────────────────────────────────────────────────────

    def connect(self, timeout: float = 15.0) -> None:
        with self._lock:
            self._link.connect(timeout)

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
        with self._lock:
            self._link.close()

    def health(self) -> dict:
        with self._lock:
            return self._link.json("GET", "/health")

    # ── what the console asks ─────────────────────────────────────────────────

    def want(self, agent: object) -> None:
        """Say that a browser is looking at the live map, and start sweeping if needed.

        Called on every status poll rather than once on open: a page that is closed, hidden
        or reloaded stops polling without telling anybody, and this way the loop notices.
        """
        self._agent = agent
        self._wanted_at = time.time()
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="live-cloud")
        self._thread.start()

    def status(self) -> dict:
        """Everything the page needs to draw the panel, and nothing it has to ask twice."""
        stats = dict(self._stats)
        return {
            "state": self._state,
            "error": self._error,
            "running": bool(self._thread is not None and self._thread.is_alive()),
            "seq": int(stats.get("seq", 0)),
            "total": int(stats.get("total", 0)),
            "added": int(stats.get("added", 0)),
            "device": stats.get("device"),
            "voxel": stats.get("voxel"),
            "sweep_ms": stats.get("last_ms"),
            "rate": round(self._rate, 2),
            "robot": self._robot,
            "cameras": len(_SOURCES),
        }

    def cloud_since(self, seq: int) -> tuple[dict, bytes]:
        """The voxels accumulated after `seq` — the whole cloud when asked for seq 0."""
        with self._lock:
            raw = self._link.binary(f"/cloud?since={int(seq)}")
        return _unframe(raw)

    def clear(self) -> dict:
        """Forget the accumulation, on the robot and therefore everywhere."""
        with self._lock:
            reply = self._link.json("POST", "/clear")
        self._stats = reply
        return reply

    # ── the loop ──────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        """Sweep while somebody is watching, then stand down."""
        logger.debug("Live cloud: sweeping")
        period = 1.0 / max(self.hz, 0.1)
        previous = 0.0
        while not self._stop.is_set():
            if time.time() - self._wanted_at > 6.0:
                break
            started = time.time()
            try:
                self._sweep()
            except CoreIOUnavailable as exc:
                self._fail(str(exc))
            except Exception as exc:
                self._fail(f"sweep failed: {exc}")
            # Sweeps per second as actually achieved, measured start to start — the time
            # inside one sweep would report the rate the wire *could* sustain, which is
            # not what an operator watching the scan build is asking about.
            if previous:
                self._rate = 1.0 / max(started - previous, 1e-3)
            previous = started
            self._stop.wait(max(0.0, period - (time.time() - started)))
        self._state = "idle" if not self._error else self._state
        logger.debug("Live cloud: stopped sweeping")

    def _fail(self, message: str) -> None:
        """One line per new failure, not one per sweep at two a second."""
        if self._error != message:
            logger.warning(f"Live cloud: {message}")
        self._state = "error"
        self._error = message
        # A failed sweep is worth pausing on: a robot that cannot be reached will not be
        # reachable 400 ms later either, and the timeout is paid on every attempt.
        self._stop.wait(2.0)

    def _sweep(self) -> None:
        """One frame from every depth camera, posted to the CoreIO and merged there."""
        agent = self._agent
        if agent is None or getattr(agent, "img_client", None) is None:
            self._state, self._error = "error", "no robot connection"
            return

        pose = _seed_tform_body(agent)
        if pose is None:
            self._state = "unlocalised"
            self._error = None
            self._robot = None
            self._stop.wait(1.0)
            return

        seed_tform_body, waypoint_id = pose
        images = agent.img_client.get_image_from_sources(_SOURCES)  # type: ignore[union-attr]
        header, blobs = _pack(images, seed_tform_body, self.near_m, self.far_m)
        if not blobs:
            self._state, self._error = "error", "no usable depth frame in this sweep"
            return

        body = _frame(header, blobs)
        with self._lock:
            status, raw = self._link.request(
                "POST", "/sweep", body, content_type="application/octet-stream"
            )
        if status != 200:
            raise CoreIOUnavailable(f"stitcher returned HTTP {status}: {raw[:160]!r}")
        self._stats, _ = _unframe(raw)
        self._state, self._error = "sweeping", None
        self._robot = _pose_dict(seed_tform_body, waypoint_id)


# ── the packing ───────────────────────────────────────────────────────────────


def _seed_tform_body(agent: object) -> tuple[SE3Pose, str] | None:
    """Where the robot is in the graph's seed frame, or None if it is not localised."""
    client = getattr(agent, "graph_nav_client", None)
    if client is None:
        return None
    try:
        state = client.get_localization_state()
    except Exception as exc:
        logger.debug(f"Live cloud: localisation unavailable ({exc})")
        return None
    localization = state.localization
    if not localization.waypoint_id:
        return None
    return SE3Pose.from_proto(localization.seed_tform_body), localization.waypoint_id


def _pack(
    images: list, seed_tform_body: SE3Pose, near_m: float, far_m: float
) -> tuple[dict, list[bytes]]:
    """Describe every depth frame well enough that the GPU needs nothing else.

    The transform sent is *seed from sensor*, composed here rather than there: the sensor's
    pose against the body is fixed kinematics and comes with the image, and the body's pose
    in the seed frame comes from GraphNav. Composing them this way — instead of chaining
    the image's odometry to the localisation's — keeps odometry drift between the two
    timestamps out of the answer.
    """
    from bosdyn.api import image_pb2

    cameras: list[dict] = []
    blobs: list[bytes] = []
    for response in images:
        shot = response.shot
        image = shot.image
        if image.pixel_format != image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
            logger.debug(f"Live cloud: {response.source.name} is not 16-bit depth — skipped")
            continue
        if image.format != image_pb2.Image.FORMAT_RAW:
            logger.debug(f"Live cloud: {response.source.name} arrived encoded — skipped")
            continue

        odom_tform_sensor = get_a_tform_b(
            shot.transforms_snapshot, ODOM_FRAME_NAME, shot.frame_name_image_sensor
        )
        odom_tform_body = get_a_tform_b(
            shot.transforms_snapshot, ODOM_FRAME_NAME, BODY_FRAME_NAME
        )
        if odom_tform_sensor is None or odom_tform_body is None:
            logger.debug(f"Live cloud: {response.source.name} carries no sensor transform — skipped")
            continue
        seed_tform_sensor = seed_tform_body * odom_tform_body.inverse() * odom_tform_sensor

        intrinsics = response.source.pinhole.intrinsics
        cameras.append({
            "name": response.source.name,
            "width": image.cols,
            "height": image.rows,
            "fx": intrinsics.focal_length.x,
            "fy": intrinsics.focal_length.y,
            "cx": intrinsics.principal_point.x,
            "cy": intrinsics.principal_point.y,
            # Spot reports millimetres (scale 1000); read it rather than assume it, because
            # a wrong scale is a room drawn a thousand times too big and no error anywhere.
            "depth_scale": response.source.depth_scale or 1000.0,
            "seed_tform_sensor": [float(v) for v in seed_tform_sensor.to_matrix().reshape(-1)],
            "compressed": True,
        })
        # Level 1: depth frames are mostly runs of equal values and zero, so the cheap
        # setting already halves them, and the expensive ones cost more laptop than they
        # save wifi.
        blobs.append(zlib.compress(image.data, 1))

    return {"near_m": near_m, "far_m": far_m, "cameras": cameras}, blobs


def _frame(header: dict, blobs: list[bytes]) -> bytes:
    """A framed request: header length, header, then each blob with its own length."""
    head = json.dumps(header).encode("utf-8")
    parts = [struct.pack(">I", len(head)), head]
    for blob in blobs:
        parts.append(struct.pack(">I", len(blob)))
        parts.append(blob)
    return b"".join(parts)


def _unframe(raw: bytes) -> tuple[dict, bytes]:
    """Split a framed reply into its JSON header and its packed float32 points."""
    if len(raw) < 4:
        raise CoreIOUnavailable("stitcher sent a truncated reply")
    (header_len,) = struct.unpack_from(">I", raw, 0)
    try:
        header = json.loads(raw[4:4 + header_len].decode("utf-8"))
    except ValueError as exc:
        raise CoreIOUnavailable("stitcher sent an undecodable header") from exc
    return header, raw[4 + header_len:]


def _pose_dict(seed_tform_body: SE3Pose, waypoint_id: str) -> dict:
    """The robot marker: where it is in the seed frame, and which way it is facing."""
    quaternion = seed_tform_body.rotation
    yaw = math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y ** 2 + quaternion.z ** 2),
    )
    return {
        "x": round(float(seed_tform_body.x), 3),
        "y": round(float(seed_tform_body.y), 3),
        "z": round(float(seed_tform_body.z), 3),
        "yaw": round(float(yaw), 4),
        "waypoint": waypoint_id,
    }


def build_live_cloud() -> LiveCloud | None:
    """Connect to the stitcher, or return None with one line saying why not.

    None rather than an exception, for the same reason the RealSense bridge does it: a
    robot without the service deployed is a tab the console does not offer, not a console
    that fails to start.
    """
    if not live_cloud_enabled():
        logger.debug("Live cloud: disabled by COREIO_CLOUD_ENABLED=0")
        return None
    client = LiveCloud()
    try:
        client.connect()
        health = client.health()
    except CoreIOUnavailable as exc:
        logger.warning(f"Live cloud: {exc} — the live map will be unavailable")
        client.close()
        return None
    except Exception as exc:
        logger.warning(f"Live cloud: unexpected failure ({exc}) — the live map will be unavailable")
        client.close()
        return None

    logger.info(
        "Live cloud stitcher up on {} — {} m voxels, {:,} point budget".format(
            health.get("device", "?"), health.get("voxel", "?"), int(health.get("budget", 0))
        )
    )
    return client
