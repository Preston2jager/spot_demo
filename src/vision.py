"""What the robot sees: objects segmented, classified and graded for grip, from the CoreIO.

The other half is `coreio/vision/service/vision_server.py`, running as the `spot-vision`
Dockge stack. It takes frames and returns every object in them — each with an outline,
a group (`teddy_bear`, `bottle`, `other`), a confidence, an ``uncertain`` flag and a grip
material. This module is the laptop's side of that:

* `VisionClient` — the transport, the same `CoreIOLink` SSH channel the RealSense uses.
* `Vision` — the shared service object: it captures frames off Spot's cameras, asks the
  CoreIO about them, and hands every result to its listeners (the video panel) as a `Scene`.
* camera geometry — turning an object's pixel into a point in the vision frame, using the
  depth camera behind the same lens, and back again.

One request can carry any mix of cameras; its result is split by view (`scan_views`): the
**body** cameras — all five ("surround"), or only the front pair when the operator has
switched surround vision off — and the **hand**, the gripper camera. A grasp survey asks for
both at once, with the arm held out in front and the gripper open, so the hand camera looks
where the robot is about to work. Every frame is turned the right way up before it is
classified (`_upright`) — the front pair lie on their sides, the right camera is upside down,
and the hand rolls with the wrist — and the outlines are turned back into the camera's own
pixels, so nothing downstream knows it happened.

There is no local fallback any more. The old YOLO path could drop to OpenVINO on the
laptop; a two-stage segmentation pipeline cannot usefully run there, and a sorting demo
that quietly switches models mid-run is harder to reason about than one that says the
CoreIO is not answering.
"""
import json
import math
import os
import struct
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from bosdyn.api import image_pb2
from bosdyn.client.frame_helpers import (
    BODY_FRAME_NAME,
    GROUND_PLANE_FRAME_NAME,
    HAND_FRAME_NAME,
    VISION_FRAME_NAME,
    get_a_tform_b,
)
from bosdyn.client.image import build_image_request
from loguru import logger

from .coreio_link import CoreIOLink, CoreIOUnavailable

__all__ = [
    "FRONT",
    "GROUPS",
    "HAND",
    "SURROUND",
    "CameraFrame",
    "CoreIOUnavailable",
    "Scene",
    "SceneObject",
    "Vision",
    "build_vision",
    "survey_sources",
]

HAND = "hand_color_image"
# Left to right as the stitched view shows them: the robot's left, then its front — whose
# two cameras are cross-eyed, so the *right* one sees the left half — then right, then back.
SURROUND = (
    "left_fisheye_image",
    "frontright_fisheye_image",
    "frontleft_fisheye_image",
    "right_fisheye_image",
    "back_fisheye_image",
)
# The front pair alone — what a grasp surveys with when surround vision is switched off.
FRONT = ("frontright_fisheye_image", "frontleft_fisheye_image")
DEPTH_OF = {
    "left_fisheye_image": "left_depth_in_visual_frame",
    "frontright_fisheye_image": "frontright_depth_in_visual_frame",
    "frontleft_fisheye_image": "frontleft_depth_in_visual_frame",
    "right_fisheye_image": "right_depth_in_visual_frame",
    "back_fisheye_image": "back_depth_in_visual_frame",
    HAND: "hand_depth_in_hand_color_frame",
}

# The three bins, in the order the page lists them. Colours are BGR for OpenCV and hex for
# the page; the same three everywhere so a bottle is the same blue on both screens.
GROUPS = {
    "teddy_bear": {"label": "Teddy bear", "bgr": (60, 170, 255), "hex": "#ffaa3c"},
    "bottle": {"label": "Bottle", "bgr": (255, 190, 50), "hex": "#32beff"},
    "other": {"label": "Other", "bgr": (140, 220, 90), "hex": "#5adc8c"},
}
UNCERTAIN_BGR = (40, 200, 250)


def survey_sources(surround: bool) -> tuple[str, ...]:
    """What a grasp survey scans: every body camera or only the front pair, and the hand."""
    return (SURROUND if surround else FRONT) + (HAND,)


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


@dataclass
class SceneObject:
    """One object the CoreIO found, in the pixel frame of the camera it was found in."""

    id: int
    camera: str
    group: str
    label: str
    seg_label: str
    conf: float
    uncertain: bool
    material: str
    xyxy: list[int]
    polygon: list[list[int]]
    cx: int
    cy: int
    area: float
    scores: dict = field(default_factory=dict)
    # How this outline came about: "detected" in this scan, or — for a registered object
    # drawn while choosing — "remembered" from an earlier scan, or "projected" from its position.
    source: str = "detected"

    @classmethod
    def from_reply(cls, index: int, reply: dict) -> "SceneObject":
        group = reply.get("group", "other")
        return cls(
            id=index,
            camera=reply["camera"],
            group=group if group in GROUPS else "other",
            label=reply.get("label", "object"),
            seg_label=reply.get("seg_label", ""),
            conf=float(reply.get("conf", 0.0)),
            uncertain=bool(reply.get("uncertain", False)),
            material=reply.get("material", "medium"),
            xyxy=list(reply.get("xyxy", [0, 0, 0, 0])),
            polygon=[list(p) for p in reply.get("polygon", [])],
            cx=int(reply.get("cx", 0)),
            cy=int(reply.get("cy", 0)),
            area=float(reply.get("area", 0.0)),
            scores=dict(reply.get("scores", {})),
        )

    def summary(self) -> str:
        mark = "?" if self.uncertain else ""
        return f"#{self.id} '{self.label}'{mark} → {GROUPS[self.group]['label']} bin, {self.conf:.2f} ({self.material})"


@dataclass
class CameraFrame:
    """One camera's picture at scan time, with what is needed to act on a pixel of it."""

    source: str
    image: np.ndarray
    response: object
    depth: np.ndarray | None = None  # metres, same size as `image`; 0 where there is no reading


@dataclass
class Scene:
    """The result of one scan: the frames that were looked at and every object in them."""

    seq: int
    kind: str  # "surround" or "hand"
    taken_at: float
    frames: dict[str, CameraFrame]
    objects: list[SceneObject]
    elapsed_ms: float = 0.0
    # False for segment-only scans (the grasp servo's): their labels are not classifications.
    classified: bool = True

    def object(self, object_id: int) -> SceneObject | None:
        return next((o for o in self.objects if o.id == object_id), None)

    def counts(self) -> dict[str, int]:
        counts = dict.fromkeys(GROUPS, 0)
        for obj in self.objects:
            counts[obj.group] += 1
        return counts


class VisionClient:
    """The CoreIO vision service, reached over an SSH channel (or a plain socket on the robot)."""

    def __init__(self) -> None:
        self.jpeg_quality = int(os.getenv("COREIO_VISION_JPEG_QUALITY", "85"))
        self._link = CoreIOLink(
            service_port=int(os.getenv("COREIO_VISION_PORT", "8784")),
            label="CoreIO vision",
            host=os.getenv("COREIO_HOST", "192.168.80.3"),
            port=int(os.getenv("COREIO_PORT", "20022")),
            username=os.getenv("COREIO_USER", "spot"),
            password=os.getenv("COREIO_PASSWORD", ""),
            timeout=float(os.getenv("COREIO_VISION_TIMEOUT", "12")),
            direct=_flag("COREIO_VISION_DIRECT", False),
            service_host=os.getenv("COREIO_VISION_HOST", "127.0.0.1"),
        )
        # paramiko opens channels on one transport; serialising keeps two threads from
        # racing on it. The service is serialised on its GPU anyway.
        self._lock = threading.Lock()

    def connect(self) -> None:
        with self._lock:
            self._link.connect()

    def close(self) -> None:
        with self._lock:
            self._link.close()

    def health(self) -> dict:
        with self._lock:
            return self._link.json("GET", "/health")

    def analyse(self, images: dict[str, np.ndarray], classify: bool = True) -> tuple[list[dict], float]:
        """Send named BGR frames, get back every object and the service's own timing.

        ``classify=False`` asks for segmentation only — outlines without groups, at half the
        cost — which is all the approach servo needs to keep hold of a target it has chosen.
        """
        cameras = list(images)
        header = json.dumps({"cameras": cameras, "classify": classify}).encode("utf-8")
        parts = [struct.pack(">I", len(header)), header]
        for camera in cameras:
            ok, buffer = cv2.imencode(".jpg", images[camera], [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            if not ok:
                raise CoreIOUnavailable(f"could not JPEG-encode the frame from {camera}")
            blob = buffer.tobytes()
            parts += [struct.pack(">I", len(blob)), blob]
        with self._lock:
            reply = self._link.json("POST", "/analyse", b"".join(parts), "application/octet-stream")
        return reply.get("objects", []), float(reply.get("elapsed_ms", 0.0))


class Vision:
    """Scans Spot's cameras through the CoreIO and tells its listeners what it found.

    One per console. Missions and the preview both scan through it: a mission's scan waits
    its turn, a preview scan (`wait=False`) is dropped when one is already running, because
    a preview frame is worth nothing by the time the mission ahead of it has finished.
    """

    def __init__(self, client: VisionClient | None = None) -> None:
        self.client = client or VisionClient()
        self.health: dict = {}
        self.error: str | None = None
        self.listeners: list = []
        # The spatial object registry (`tracker.ObjectTracker`), set up by `build_vision`.
        self.tracker = None
        self._gate = threading.Lock()
        self._seq = 0
        self._down_until = 0.0
        self.retry_after = float(os.getenv("COREIO_VISION_RETRY_SEC", "15"))

    # ── status ────────────────────────────────────────────────────────────────

    def probe(self) -> bool:
        """Connect and read /health. Never raises — the answer goes into `status()`."""
        try:
            self.client.connect()
            self.health = self.client.health()
            self.error = None
            logger.success(
                f"CoreIO vision ready: {self.health.get('pipeline')} on {self.health.get('backend')}, "
                f"groups {self.health.get('groups')}"
            )
            return True
        except CoreIOUnavailable as exc:
            # The console re-probes every 20 s; say it once, not every time, until it changes.
            warn = logger.debug if str(exc) == self.error else logger.warning
            self.error = str(exc)
            self.health = {}
            warn(f"CoreIO vision unavailable ({exc}) — start the spot-vision stack in Dockge")
            return False

    def status(self) -> dict:
        return {
            "ready": bool(self.health) and self.error is None,
            "backend": self.health.get("backend"),
            "pipeline": self.health.get("pipeline"),
            "error": self.error,
        }

    # ── scanning ──────────────────────────────────────────────────────────────

    def scan(self, agent: object, sources: tuple[str, ...] | list[str], depth: bool = False,
             wait: bool = True, classify: bool = True) -> Scene | None:
        """`scan_views`, returning the body cameras' scene — or the hand's, when only it was asked for."""
        views = self.scan_views(agent, sources, depth=depth, wait=wait, classify=classify)
        return views.get("cameras") or views.get("hand")

    def scan_views(self, agent: object, sources: tuple[str, ...] | list[str], depth: bool = False,
                   wait: bool = True, classify: bool = True) -> dict[str, Scene]:
        """Capture ``sources``, analyse them on the CoreIO in one request, publish and return
        the result as one scene per view: ``cameras`` (the body cameras) and ``hand``.

        The scenes of one request share a ``seq``: they are one look, so an object both the
        hand and a front camera saw counts as seen once, from two cameras.

        ``wait=False`` is the preview's entry: it returns nothing immediately if a scan is
        already in flight, and quietly during a back-off after a failure. A mission's scan
        (``wait=True``) always goes ahead and raises `CoreIOUnavailable` if it cannot.
        """
        if wait:
            self._gate.acquire()
        else:
            if time.time() < self._down_until or not self._gate.acquire(blocking=False):
                return {}
        try:
            frames = capture(agent, sources, depth=depth)
            if not frames:
                return {}
            upright = {name: _upright(frame) for name, frame in frames.items()}
            started = time.time()
            try:
                replies, elapsed = self.client.analyse({name: image for name, (image, _) in upright.items()}, classify)
            except CoreIOUnavailable as exc:
                self.error = str(exc)
                self._down_until = time.time() + self.retry_after
                raise
            if not self.health:
                self._refresh_health()
            self.error = None
            self._seq += 1
            arm = _arm_points(agent)
            objects = []
            for index, reply in enumerate(replies):
                frame = frames.get(reply.get("camera", ""))
                if frame is None:
                    continue
                obj = SceneObject.from_reply(index + 1, _turn_back(reply, upright[frame.source][1], frame.image.shape))
                if not _is_robot(frame, obj, arm):
                    objects.append(obj)
            scenes = {}
            for view, names in (("cameras", [n for n in frames if n != HAND]), ("hand", [n for n in frames if n == HAND])):
                if names:
                    scenes[view] = Scene(
                        seq=self._seq,
                        kind="hand" if view == "hand" else "surround",
                        taken_at=started,
                        frames={n: frames[n] for n in names},
                        objects=[o for o in objects if o.camera in names],
                        elapsed_ms=elapsed,
                        classified=classify,
                    )
        finally:
            self._gate.release()
        for scene in scenes.values():
            for listener in list(self.listeners):
                try:
                    listener(scene)
                except Exception as exc:
                    logger.debug(f"Vision listener failed: {exc}")
        return scenes

    def _refresh_health(self) -> None:
        try:
            self.health = self.client.health()
        except CoreIOUnavailable:
            pass


def _is_robot(frame: CameraFrame | None, obj: SceneObject, arm: list[tuple[float, float, float]]) -> bool:
    """Spot's own legs and arm, as seen by its body cameras.

    They segment perfectly well — they are objects — and a sorting demo must never offer
    one up for grasping. Legs: nothing else can be this close to a body camera, so the
    depth behind the outline is the test (frames scanned without depth keep everything).
    The arm, held out in front while the robot looks: an outline that covers where the
    gripper or the forearm projects into the picture is the arm, depth or not.
    """
    if frame is None or frame.source == HAND:
        return False
    distance = object_range(frame, obj)
    if distance is not None and distance < _SELF_RANGE_M:
        return True
    if len(obj.polygon) < 3 or not arm:
        return False
    outline = np.array(obj.polygon, dtype=np.float32)
    for point in arm:
        pixel = vision_to_pixel(frame, point)
        if pixel is not None and cv2.pointPolygonTest(outline, (float(pixel[0]), float(pixel[1])), False) >= 0:
            return True
    return False


def _arm_points(agent: object) -> list[tuple[float, float, float]]:
    """Points along the arm in the vision frame — the jaw tips, the hand, the forearm — while
    it is reached out in front of the body; empty when it is stowed or the state is unknown."""
    try:
        snapshot = agent.state_client.get_robot_state().kinematic_state.transforms_snapshot  # type: ignore[attr-defined]
        vision_T_hand = get_a_tform_b(snapshot, VISION_FRAME_NAME, HAND_FRAME_NAME)
        body_T_hand = get_a_tform_b(snapshot, BODY_FRAME_NAME, HAND_FRAME_NAME)
        vision_T_body = get_a_tform_b(snapshot, VISION_FRAME_NAME, BODY_FRAME_NAME)
    except Exception:
        return []
    if body_T_hand.x < _ARM_OUT_X:
        return []
    hand = np.array([vision_T_hand.x, vision_T_hand.y, vision_T_hand.z])
    shoulder = np.array(vision_T_body.transform_point(0.30, 0.0, 0.10))
    return [tuple(vision_T_hand.transform_point(0.12, 0.0, 0.0)), tuple(hand),
            tuple((hand + shoulder) / 2), tuple(0.25 * hand + 0.75 * shoulder)]


# The hand counts as reached out once it is this far ahead of the body's centre; stowed, it
# sits further back, out of the front cameras' view.
_ARM_OUT_X = 0.62
_SELF_RANGE_M = float(os.getenv("VISION_SELF_RANGE_M", "0.40"))


# ── capture ───────────────────────────────────────────────────────────────────


def capture(agent: object, sources, depth: bool = False, quality: int = 80) -> dict[str, CameraFrame]:
    """Grab colour frames — and optionally the depth behind each — in one image request.

    Colour is asked for explicitly: without a pixel format the body cameras answer in
    greyscale even on a robot whose fisheyes are RGB. Greyscale stays as the fallback for
    robots that only have that.
    """
    requests = [
        build_image_request(source, quality_percent=quality, pixel_format=image_pb2.Image.PIXEL_FORMAT_RGB_U8,
                            fallback_formats=[image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8])
        for source in sources
    ]
    if depth:
        requests += [
            build_image_request(DEPTH_OF[source], pixel_format=image_pb2.Image.PIXEL_FORMAT_DEPTH_U16)
            for source in sources if source in DEPTH_OF
        ]
    responses = agent.img_client.get_image(requests)  # type: ignore[union-attr]
    by_name = {r.source.name: r for r in responses}
    frames: dict[str, CameraFrame] = {}
    for source in sources:
        response = by_name.get(source)
        if response is None:
            continue
        image = cv2.imdecode(np.frombuffer(response.shot.image.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            continue
        frame = CameraFrame(source, image, response)
        depth_response = by_name.get(DEPTH_OF.get(source, ""))
        if depth_response is not None and _same_pose(response, depth_response):
            frame.depth = _decode_depth(depth_response, image.shape)
        frames[source] = frame
    return frames


# How far the camera may have moved between a colour frame and the depth behind it.
_POSE_TOL_M = 0.015
_POSE_TOL_RAD = math.radians(1.5)


def _same_pose(colour: object, depth: object) -> bool:
    """Whether a depth image was taken from where its colour image was.

    The body cameras' depth comes with the same acquisition time as the picture, but the
    hand's ToF depth can be seconds older than the hand colour image. With the arm still
    that is harmless; with the arm moving, the depth belongs to a different viewpoint and
    every pixel of it would be put in the wrong place — so it is dropped, and the caller
    falls back exactly as it would with no depth at all.
    """
    try:
        a = get_a_tform_b(colour.shot.transforms_snapshot, VISION_FRAME_NAME, colour.shot.frame_name_image_sensor)  # type: ignore[attr-defined]
        b = get_a_tform_b(depth.shot.transforms_snapshot, VISION_FRAME_NAME, depth.shot.frame_name_image_sensor)  # type: ignore[attr-defined]
    except Exception:
        return True
    moved = a.inverse() * b
    angle = 2.0 * math.acos(min(1.0, abs(moved.rotation.w)))
    same = math.hypot(moved.x, moved.y, moved.z) < _POSE_TOL_M and angle < _POSE_TOL_RAD
    if not same:
        logger.debug(f"Depth for {colour.source.name} dropped: the camera moved between colour and depth")  # type: ignore[attr-defined]
    return same


def _decode_depth(response: object, shape: tuple) -> np.ndarray | None:
    """A depth image in metres, the size of the colour frame it was aligned to."""
    shot = response.shot  # type: ignore[attr-defined]
    try:
        raw = np.frombuffer(shot.image.data, dtype=np.uint16).reshape(shot.image.rows, shot.image.cols)
    except ValueError:
        return None
    scale = response.source.depth_scale or 1000.0  # type: ignore[attr-defined]
    depth = raw.astype(np.float32) / scale
    if depth.shape[:2] != shape[:2]:
        depth = cv2.resize(depth, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return depth


# ── the right way up ──────────────────────────────────────────────────────────

# A roll this close to a quarter turn is taken as exactly one — lossless, and what the body
# cameras' fixed mounts are; the hand, rolled by the wrist, is turned by its actual angle.
_SNAP = math.radians(12.0)


def _upright(frame: CameraFrame) -> tuple[np.ndarray, np.ndarray | None]:
    """The frame turned so that down in the world is down in the picture, and the 2×3 map
    from the camera's pixels to the turned ones — None when it is already upright.

    The classifier was trained on upright photographs; a teddy bear lying on its side in
    the picture, because the camera is, is a worse teddy bear. Down comes from the image's
    own transforms snapshot, so it is right for the hand whatever the wrist is doing.
    """
    try:
        shot = frame.response.shot  # type: ignore[attr-defined]
        rotation = get_a_tform_b(shot.transforms_snapshot, VISION_FRAME_NAME,
                                 shot.frame_name_image_sensor).to_matrix()[:3, :3]
    except Exception:
        return frame.image, None
    down = rotation.T @ np.array([0.0, 0.0, -1.0])
    if math.hypot(down[0], down[1]) < 0.3:
        return frame.image, None  # looking straight down or up: there is no "up" to turn to
    angle = math.atan2(down[0], down[1])
    quarter = round(angle / (math.pi / 2)) * (math.pi / 2)
    snapped = abs(angle - quarter) < _SNAP
    if snapped:
        angle = quarter
    if abs(angle) < 1e-6:
        return frame.image, None
    turn = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    if snapped:
        turn = np.round(turn)
    height, width = frame.image.shape[:2]
    corners = np.array([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]], dtype=float) @ turn.T
    low, high = corners.min(axis=0), corners.max(axis=0)
    matrix = np.hstack([turn, -low[:, None]])
    size = (round(high[0] - low[0]) + 1, round(high[1] - low[1]) + 1)
    image = cv2.warpAffine(frame.image, matrix, size, flags=cv2.INTER_NEAREST if snapped else cv2.INTER_LINEAR)
    return image, matrix


def _turn_back(reply: dict, matrix: np.ndarray | None, shape: tuple) -> dict:
    """A reply about a turned frame, with its outline and grasp point in the camera's own pixels."""
    if matrix is None:
        return reply
    inverse = cv2.invertAffineTransform(matrix)
    height, width = shape[:2]

    def back(points: list) -> np.ndarray:
        mapped = np.asarray(points, dtype=float).reshape(-1, 2) @ inverse[:, :2].T + inverse[:, 2]
        return np.clip(np.round(mapped), 0, [width - 1, height - 1]).astype(int)

    reply = dict(reply)
    if reply.get("polygon"):
        corners = back(reply["polygon"])
        reply["polygon"] = corners.tolist()
    else:
        x0, y0, x1, y1 = reply.get("xyxy", [0, 0, 0, 0])
        corners = back([[x0, y0], [x1, y0], [x0, y1], [x1, y1]])
    reply["xyxy"] = [int(corners[:, 0].min()), int(corners[:, 1].min()), int(corners[:, 0].max()), int(corners[:, 1].max())]
    reply["cx"], reply["cy"] = (int(v) for v in back([[reply.get("cx", 0), reply.get("cy", 0)]])[0])
    return reply


# ── geometry ──────────────────────────────────────────────────────────────────


def object_range(frame: CameraFrame, obj: SceneObject) -> float | None:
    """Median depth over the inside of an object's outline, in metres, or None."""
    if frame.depth is None or len(obj.polygon) < 3:
        return None
    mask = np.zeros(frame.depth.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.array(obj.polygon, dtype=np.int32)], 1)
    # Eroded, so the edge pixels — where depth jumps to whatever is behind — do not vote.
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
    values = frame.depth[(mask > 0) & (frame.depth > 0.05)]
    if values.size < 12:
        return None
    return float(np.median(values))


def pixel_to_vision(frame: CameraFrame, u: float, v: float, range_m: float | None) -> tuple[float, float, float] | None:
    """The vision-frame point behind a pixel.

    With a range, it is that far along the pixel's ray. Without one — no depth reading —
    the ray is intersected with the ground plane, which is right for anything standing on
    the floor and the best guess available for anything else.
    """
    response = frame.response
    intrinsics = response.source.pinhole.intrinsics  # type: ignore[attr-defined]
    fx, fy = intrinsics.focal_length.x, intrinsics.focal_length.y
    px, py = intrinsics.principal_point.x, intrinsics.principal_point.y
    ray = np.array([(u - px) / fx, (v - py) / fy, 1.0])
    snapshot = response.shot.transforms_snapshot  # type: ignore[attr-defined]
    vision_tform_camera = get_a_tform_b(snapshot, VISION_FRAME_NAME, response.shot.frame_name_image_sensor)  # type: ignore[attr-defined]
    if range_m is not None:
        point = ray * range_m  # depth-in-visual-frame is the z distance, so scale the z=1 ray
        return tuple(vision_tform_camera.transform_point(*point))  # type: ignore[return-value]
    try:
        ground_z = get_a_tform_b(snapshot, VISION_FRAME_NAME, GROUND_PLANE_FRAME_NAME).z
    except Exception:
        return None
    origin = np.array([vision_tform_camera.x, vision_tform_camera.y, vision_tform_camera.z])
    direction = np.array(vision_tform_camera.transform_point(*ray)) - origin
    if direction[2] >= -1e-3:
        return None  # looking level or up — this ray never meets the floor
    t = (ground_z - origin[2]) / direction[2]
    return tuple(origin + t * direction)  # type: ignore[return-value]


def vision_to_pixel(frame: CameraFrame, point: tuple[float, float, float]) -> tuple[float, float] | None:
    """Where a vision-frame point lands in a camera's image, or None when it is behind it."""
    response = frame.response
    intrinsics = response.source.pinhole.intrinsics  # type: ignore[attr-defined]
    snapshot = response.shot.transforms_snapshot  # type: ignore[attr-defined]
    camera_tform_vision = get_a_tform_b(snapshot, response.shot.frame_name_image_sensor, VISION_FRAME_NAME)  # type: ignore[attr-defined]
    x, y, z = camera_tform_vision.transform_point(*point)
    if z <= 0.05:
        return None
    return (
        intrinsics.focal_length.x * x / z + intrinsics.principal_point.x,
        intrinsics.focal_length.y * y / z + intrinsics.principal_point.y,
    )


def project_box(frame: CameraFrame, point: tuple[float, float, float] | np.ndarray,
                size_m: float = 0.16) -> list[list[float]] | None:
    """A square around where a vision-frame point lands in a camera's image, sized as an
    object ``size_m`` across would be at that range — or None when it is out of view."""
    response = frame.response
    intrinsics = response.source.pinhole.intrinsics  # type: ignore[attr-defined]
    snapshot = response.shot.transforms_snapshot  # type: ignore[attr-defined]
    camera_tform_vision = get_a_tform_b(snapshot, response.shot.frame_name_image_sensor, VISION_FRAME_NAME)  # type: ignore[attr-defined]
    x, y, z = camera_tform_vision.transform_point(*point)
    if z <= 0.1:
        return None
    u = intrinsics.focal_length.x * x / z + intrinsics.principal_point.x
    v = intrinsics.focal_length.y * y / z + intrinsics.principal_point.y
    height, width = frame.image.shape[:2]
    if not (0 <= u < width and 0 <= v < height):
        return None
    r = min(140.0, max(10.0, intrinsics.focal_length.x * size_m / 2 / z))
    return [[u - r, v - r], [u + r, v - r], [u + r, v + r], [u - r, v + r]]


def locate(frame: CameraFrame, obj: SceneObject) -> tuple[float, float, float] | None:
    """An object's position in the vision frame, from its grasp point and its depth."""
    return pixel_to_vision(frame, obj.cx, obj.cy, object_range(frame, obj))


def bearing_and_distance(agent: object, point: tuple[float, float, float]) -> tuple[float, float]:
    """How far the body must turn (radians, left positive) and how far away a point is."""
    bx, by, byaw = agent.get_current_pose()  # type: ignore[attr-defined]
    dx, dy = point[0] - bx, point[1] - by
    bearing = math.atan2(dy, dx) - byaw
    return math.atan2(math.sin(bearing), math.cos(bearing)), math.hypot(dx, dy)


_vision: Vision | None = None


def build_vision() -> Vision:
    """The shared vision service for this process, probed once when first asked for."""
    global _vision
    if _vision is None:
        from .tracker import ObjectTracker

        _vision = Vision()
        # Every classified scan — preview or mission — registers its objects in space.
        _vision.tracker = ObjectTracker()
        _vision.listeners.append(_vision.tracker.update)
        _vision.probe()
    return _vision
