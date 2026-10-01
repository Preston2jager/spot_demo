"""3D reconstruction all round the robot, fused on this laptop from Spot's and the RealSense's depth.

Four depth sources, the front ones trusted most:

* the **hand** camera — colour plus ``hand_depth_in_hand_color_frame``;
* the two **front** body cameras — ``frontleft`` / ``frontright`` fisheye plus their depth
  in the visual frame;
* the **sides** — the ``left``, ``right`` and ``back`` body cameras, in their own slower loop
  so they never cost the front its rate: sampled more coarsely, trusted to a shorter range
  and fused with ``RECON_SIDE_WEIGHT``, so where they overlap the front's surface, the front
  decides where its points sit and what colour they are. They fill in whatever else they see;
* the CoreIO **RealSense** — colour plus aligned depth, from its bridge's ``/rgbd``.

Points inside Spot's own footprint (legs, body) are dropped from the body cameras.

Each frame is back-projected through its own intrinsics into Spot's vision frame. Spot's
cameras use the transforms snapshot taken *with* the image, so a moving robot still puts
every point where it was; hand depth taken from a different arm pose than its colour image
is dropped by `capture()`. The RealSense frames carry no robot timestamp, so they are placed
with the body pose at arrival through its mount, and only while the body is still — a
moving body would smear them.

The mount starts from ``RECON_RS_MOUNT`` and is meant to be **calibrated** (`calibrate()`,
the Calibrate button), with Spot still, in two steps:

1. **Floor and gravity** — the floor is found in the RealSense's own depth (RANSAC), and its
   normal must be the robot's "up": that fixes pitch, roll and height absolutely, from any
   starting guess. (An uncalibrated 15° pitch is what tipped the whole RealSense cloud.)
2. **Planar point-to-plane ICP** onto the cloud the front cameras have built refines only yaw and the
   horizontal offset — the floor already fixed tilt and height better than the front
   cameras' stereo can — accepted only if it converges and stays a small correction.

The result is saved to ``runtime/recon_rs_mount.json``, which takes precedence over the
environment.

Points go into a sparse voxel grid (``RECON_VOXEL_M``, 1 cm by default). Each voxel keeps a
weighted sum of where its points fell and what colour they were, weighted by 1/range²
because stereo and structured-light error grow with the square of range; the drawn point
is that weighted centroid, which is finer than the voxel. A voxel seen in fewer than
``RECON_MIN_HITS`` frames is held back as noise, and depth edges — the "flying pixels"
between a foreground edge and the background — are dropped before fusion.

**What leaves, leaves the map** (`VoxelMap.carve`). Every depth frame is also a statement
about empty space: a voxel that projects into it nearer than the surface it measured there
is seen *through*. A voxel seen through again and again, and not hit once, for
``RECON_CARVE_SEC`` (2 s) is deleted — so a person who stood in front of Spot and walked
away is gone a couple of seconds later, and the wall behind them fills back in. Occluded
voxels are never carved (the person in front of the wall measures nearer than the wall),
the margin grows with range as stereo noise does, and one stray reading cannot do it: any
hit in between resets the clock.

None of it runs on the robot. Spot's image service and the RealSense bridge are only asked
for frames they would serve anyway; every bit of arithmetic is numpy here. It runs only
while a browser has the 3D Recon view open (`want()`), and idles otherwise.
"""
import json
import math
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, GROUND_PLANE_FRAME_NAME, VISION_FRAME_NAME, get_a_tform_b
from loguru import logger

from .vision import FRONT, HAND, capture

SIDES = ("left_fisheye_image", "right_fisheye_image", "back_fisheye_image")
SOURCES = ("hand", "front", "sides", "realsense")

_VOXEL = float(os.getenv("RECON_VOXEL_M", "0.01"))
_NEAR = 0.15
_FAR = float(os.getenv("RECON_FAR_M", "3.0"))
# Body stereo is the noisiest of the four; past this it adds more speckle than surface.
_FAR_BODY = min(_FAR, float(os.getenv("RECON_FAR_BODY_M", "2.5")))
# The side and back cameras: shorter range, coarser sampling, less weight, a slower loop.
_FAR_SIDE = min(_FAR_BODY, float(os.getenv("RECON_FAR_SIDE_M", "2.0")))
_SIDE_WEIGHT = float(os.getenv("RECON_SIDE_WEIGHT", "0.25"))
_SIDE_PERIOD = float(os.getenv("RECON_SIDE_PERIOD", "1.0"))
_MIN_HITS = int(os.getenv("RECON_MIN_HITS", "2"))
_MAX_VOXELS = int(os.getenv("RECON_MAX_VOXELS", "1500000"))
# Seen through, and not hit, for this long → deleted. 0 turns carving off.
_CARVE_SEC = float(os.getenv("RECON_CARVE_SEC", "2.0"))
# How much farther than a voxel the measured surface must be for the voxel to count as seen
# through: a fixed part plus one growing with range², as stereo error does.
_CARVE_MARGIN = 0.05
_CARVE_PERIOD = 0.4
# At most this many frames per carving pass: the newest, which say most about now.
_CARVE_BACKLOG = 6
_CARVE_MARGIN_SQ = 0.03
_SEND_POINTS = int(os.getenv("RECON_SEND_POINTS", "600000"))
_RS_STEP = int(os.getenv("RECON_RS_STEP", "2"))
# A depth pixel whose 3×3 neighbourhood spans more than this fraction of its own range
# sits on an edge, where the sensor mixes foreground and background.
_EDGE = float(os.getenv("RECON_EDGE_REL", "0.03"))
# The RealSense is fused only while the body moves slower than these.
_STILL_MPS = 0.05
_STILL_RADPS = 0.08
_IDLE_SEC = 5.0
# ICP must see vertical surfaces facing both horizontal directions to be trusted on x/y.
_ICP_MIN_OBSERVED = 0.01
# Spot's own volume in its body frame (x forward, y left, z up): what a body camera sees of
# its legs. Generous in z, down to the floor beneath it.
_BODY_BOX = np.array([[-0.58, -0.30, -1.0], [0.58, 0.30, 0.35]])

_MOUNT_FILE = Path(__file__).resolve().parent.parent / "runtime" / "recon_rs_mount.json"

_OFFSET = 1 << 20
_MASK = (1 << 21) - 1
_KERNEL = np.ones((3, 3), np.uint8)
# Optical axes (x right, y down, z forward) in a frame looking along +x with z up.
_OPTICAL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def _mount() -> np.ndarray:
    """body_T_camera for the RealSense: the calibrated one if saved, else ``RECON_RS_MOUNT``."""
    try:
        return np.array(json.loads(_MOUNT_FILE.read_text())["body_T_camera"], dtype=float)
    except (OSError, ValueError, KeyError):
        return _env_mount()


def _env_mount() -> np.ndarray:
    """body_T_camera for the RealSense, from ``RECON_RS_MOUNT`` = "x,y,z,roll,pitch,yaw".

    Metres and degrees, in Spot's body frame (x forward, y left, z up), for the camera as
    the bridge serves it — already turned the right way up — with all angles zero meaning
    it looks straight ahead. Measure it on the robot; the default is only a starting point.
    """
    raw = os.getenv("RECON_RS_MOUNT", "0.30,0,0.10,0,15,0")
    try:
        x, y, z, roll, pitch, yaw = (float(v) for v in raw.split(","))
    except ValueError:
        logger.warning(f"RECON_RS_MOUNT='{raw}' is not x,y,z,roll,pitch,yaw — using the default")
        x, y, z, roll, pitch, yaw = 0.30, 0.0, 0.10, 0.0, 15.0, 0.0
    r, p, w = (math.radians(a) for a in (roll, pitch, yaw))
    rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    rz = np.array([[math.cos(w), -math.sin(w), 0], [math.sin(w), math.cos(w), 0], [0, 0, 1]])
    transform = np.eye(4)
    transform[:3, :3] = rz @ ry @ rx @ _OPTICAL
    transform[:3, 3] = (x, y, z)
    return transform


def _describe(body_T_camera: np.ndarray) -> str:
    """The mount as RECON_RS_MOUNT would write it: x,y,z in metres, roll,pitch,yaw in degrees."""
    r = body_T_camera[:3, :3] @ _OPTICAL.T
    roll = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -r[2, 0]))))
    yaw = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    x, y, z = body_T_camera[:3, 3]
    return f"{x:.3f},{y:.3f},{z:.3f},{roll:.1f},{pitch:.1f},{yaw:.1f}"


def floor_plane(camera_points: np.ndarray, expected_up: np.ndarray, tol: float = 0.015,
                iterations: int = 500) -> tuple[np.ndarray, float, float] | None:
    """The floor in a camera's own frame: (unit normal pointing up, camera height, inlier fraction).

    RANSAC over planes whose normal is within 40° of ``expected_up`` and that lie below the
    camera, then a least-squares refit on the inliers. None if no such plane holds 15%.
    """
    rng = np.random.default_rng(0)
    best_count, best = 0, None
    cos_limit = math.cos(math.radians(40))
    for _ in range(iterations):
        a, b, c = camera_points[rng.choice(len(camera_points), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-9:
            continue
        normal /= length
        if normal @ expected_up < 0:
            normal = -normal
        height = float(normal @ -a)  # camera above the plane, along the normal
        if normal @ expected_up < cos_limit or height <= 0.1:
            continue
        count = int((np.abs(camera_points @ normal + height) < tol).sum())
        if count > best_count:
            best_count, best = count, (normal, height)
    if best is None or best_count < 0.15 * len(camera_points):
        return None
    normal, height = best
    inliers = camera_points[np.abs(camera_points @ normal + height) < tol]
    centre = inliers.mean(axis=0)
    normal = np.linalg.svd(inliers - centre)[2][2]
    if normal @ expected_up < 0:
        normal = -normal
    return normal, float(normal @ -centre), len(inliers) / len(camera_points)


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The smallest rotation taking unit vector ``a`` onto unit vector ``b``."""
    axis = np.cross(a, b)
    sin, cos = np.linalg.norm(axis), float(a @ b)
    if sin < 1e-9:
        return np.eye(3)
    k = axis / sin
    skew = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    angle = math.atan2(sin, cos)
    return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * skew @ skew


def icp(source: np.ndarray, target: np.ndarray, iterations: int = 40) -> tuple[np.ndarray, float, float, float]:
    """Turn about z and shift in x, y taking ``source`` onto ``target``.

    Returns (4×4, inlier fraction, RMSE in metres, how well x/y were observed). Only those
    three degrees of freedom: the floor has already fixed tilt and height better than the
    target cloud can.

    Point-to-**plane**, against normals fitted to the target. Point-to-point was dominated
    by the floor, which is most of either cloud: each floor point's nearest neighbour lies
    straight below it, so the floor dragged every shift back towards none, and the fore-aft
    offset — which only vertical surfaces measure — came out short. Here a point counts only
    along its surface's normal, so the floor says nothing about x and y, and a bottle's or a
    wall's front face says exactly how far forward it is. The last value is the smallest
    eigenvalue of the inliers' horizontal-normal scatter: near 0 when nothing vertical faces
    some horizontal direction, and x/y along it are guesses.

    The correspondence radius shrinks from 40 cm to 2 cm so a mount that starts a hand's
    width out still converges. Nearest neighbours from OpenCV's FLANN.
    """
    target = np.ascontiguousarray(target, np.float32)
    index = cv2.flann_Index(target, {"algorithm": 1, "trees": 4})
    normals = _normals(target, index)
    transform = np.eye(4)
    fraction, rmse, observed = 0.0, float("inf"), 0.0
    for step in range(iterations):
        radius = max(0.02, 0.40 * (0.85 ** step))
        moved = source @ transform[:3, :3].T + transform[:3, 3]
        found, dist2 = index.knnSearch(np.ascontiguousarray(moved, np.float32), 1, params={})
        distance = np.sqrt(dist2[:, 0])
        inliers = distance < radius
        if inliers.sum() < 300:
            break
        a, b, n = moved[inliers], target[found[inliers, 0]], normals[found[inliers, 0]]
        residual = np.einsum("ij,ij->i", a - b, n)
        # d/dθ of a turned about z is (-a_y, a_x, 0), about the points' centre for conditioning.
        centre = a.mean(axis=0)
        rel = a - centre
        jacobian = np.stack((n[:, 1] * rel[:, 0] - n[:, 0] * rel[:, 1], n[:, 0], n[:, 1]), axis=1)
        try:
            angle, tx, ty = np.linalg.solve(jacobian.T @ jacobian + 1e-6 * np.eye(3), -jacobian.T @ residual)
        except np.linalg.LinAlgError:
            break
        rotation = np.array([[math.cos(angle), -math.sin(angle), 0.0],
                             [math.sin(angle), math.cos(angle), 0.0], [0.0, 0.0, 1.0]])
        step_transform = np.eye(4)
        step_transform[:3, :3] = rotation
        step_transform[:3, 3] = centre - rotation @ centre + (tx, ty, 0.0)
        transform = step_transform @ transform
        fraction = float(inliers.mean())
        rmse = float(np.sqrt(np.mean(residual ** 2)))
        observed = float(np.linalg.eigvalsh(n[:, :2].T @ n[:, :2] / len(n))[0])
    return transform, fraction, rmse, observed


def _normals(points: np.ndarray, index: object, k: int = 12) -> np.ndarray:
    """Unit surface normals: the least-spread direction of each point's ``k`` nearest neighbours."""
    found, _ = index.knnSearch(points, k, params={})  # type: ignore[attr-defined]
    neighbours = points[found]
    centred = neighbours - neighbours.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centred, centred)
    return np.linalg.eigh(covariance)[1][:, :, 0].astype(np.float32)


def backproject(depth: np.ndarray, colour: np.ndarray, intrinsics: tuple[float, float, float, float],
                world_tform_camera: np.ndarray, far: float, stride: int = 1
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Points (N×3, world frame), colours (N×3 RGB uint8) and weights (N) from one frame.

    ``depth`` is metres along the optical axis, the same size as ``colour``; 0 is no reading.
    """
    fx, fy, cx, cy = intrinsics
    if stride > 1:
        depth = depth[::stride, ::stride]
        colour = colour[::stride, ::stride]
    spread = cv2.dilate(depth, _KERNEL) - cv2.erode(depth, _KERNEL)
    valid = (depth > _NEAR) & (depth < far) & (spread < _EDGE * depth)
    vs, us = np.nonzero(valid)
    z = depth[vs, us]
    x = (us * stride - cx) / fx * z
    y = (vs * stride - cy) / fy * z
    camera = np.stack((x, y, z), axis=1)
    points = camera @ world_tform_camera[:3, :3].T + world_tform_camera[:3, 3]
    rgb = colour[vs, us][:, ::-1]
    return points.astype(np.float32), np.ascontiguousarray(rgb), (1.0 / (z * z)).astype(np.float32)


def outside_body(points: np.ndarray, vision_T_body: np.ndarray) -> np.ndarray:
    """Mask of the points (vision frame) that are not inside Spot's own footprint."""
    body = (points - vision_T_body[:3, 3]) @ vision_T_body[:3, :3]
    return ~np.all((body > _BODY_BOX[0]) & (body < _BODY_BOX[1]), axis=1)


class VoxelMap:
    """A sparse voxel grid of weighted centroids and colours, merged frame by frame.

    Keys are the three voxel indices packed into one int64, kept sorted so a frame merges
    with one `searchsorted` and one `insert`. Each row of ``_acc`` holds, weighted, the
    point's offset from its voxel's corner (small numbers, so float32 keeps sub-millimetre
    precision), the colour, the weight sum, how many frames hit the voxel — and for carving,
    when it was first and last seen through since its last hit (seconds on the map's own
    clock, -1 for never).
    """

    def __init__(self, voxel: float = _VOXEL) -> None:
        self.voxel = voxel
        self._lock = threading.Lock()
        self.clear()

    def clear(self) -> None:
        with self._lock:
            self._keys = np.empty(0, np.int64)
            self._acc = np.empty((0, _COLUMNS), np.float32)
            self.seq = 0
            self.carved = 0
            self._epoch = time.monotonic()

    def __len__(self) -> int:
        return len(self._keys)

    def fuse(self, points: np.ndarray, rgb: np.ndarray, weight: np.ndarray) -> None:
        if len(points) == 0:
            return
        index = np.floor(points / self.voxel).astype(np.int64)
        offset = points - index * self.voxel
        index += _OFFSET
        keys = (index[:, 0] << 42) | (index[:, 1] << 21) | index[:, 2]
        unique, inverse = np.unique(keys, return_inverse=True)
        n = len(unique)
        agg = np.empty((n, _COLUMNS), np.float32)
        columns = (offset[:, 0], offset[:, 1], offset[:, 2], rgb[:, 0], rgb[:, 1], rgb[:, 2])
        for col, values in enumerate(columns):
            agg[:, col] = np.bincount(inverse, weights=values * weight, minlength=n)
        agg[:, 6] = np.bincount(inverse, weights=weight, minlength=n)
        agg[:, 7] = 1.0
        agg[:, _SEEN_THROUGH:] = -1.0
        with self._lock:
            pos = np.searchsorted(self._keys, unique)
            hit = np.zeros(n, bool)
            inside = pos < len(self._keys)
            hit[inside] = self._keys[pos[inside]] == unique[inside]
            self._acc[pos[hit], :_SEEN_THROUGH] += agg[hit, :_SEEN_THROUGH]
            self._acc[pos[hit], _SEEN_THROUGH:] = -1.0  # hit: whatever saw through it was wrong
            fresh = ~hit
            if fresh.any():
                self._keys = np.insert(self._keys, pos[fresh], unique[fresh])
                self._acc = np.insert(self._acc, pos[fresh], agg[fresh], axis=0)
            if len(self._keys) > _MAX_VOXELS:
                # Over budget: keep the most-confirmed voxels, which is what a surface is.
                keep = np.sort(np.argpartition(self._acc[:, 7], -int(_MAX_VOXELS * 0.8))[-int(_MAX_VOXELS * 0.8):])
                self._keys, self._acc = self._keys[keep], self._acc[keep]
            self.seq += 1

    def carve(self, views: list[tuple[np.ndarray, tuple[float, float, float, float], np.ndarray]]) -> int:
        """Delete the voxels these depth frames have kept seeing through. Returns how many.

        ``views`` are (depth in metres, intrinsics fx fy cx cy, world_T_camera). Each voxel's
        centroid is projected into each frame; where the frame measured a surface farther
        than the voxel by more than the margin, and not on a depth edge, the voxel is seen
        through. Computed on a snapshot, then applied by key, so fusion is not held up.
        """
        if _CARVE_SEC <= 0 or not views:
            return 0
        with self._lock:
            keys, acc = self._keys, self._acc
        if len(keys) == 0:
            return 0
        index = np.stack(((keys >> 42) & _MASK, (keys >> 21) & _MASK, keys & _MASK), axis=1) - _OFFSET
        centroids = (index * self.voxel + acc[:, 0:3] / acc[:, 6:7]).astype(np.float32)
        through = np.zeros(len(keys), bool)
        for depth, (fx, fy, cx, cy), world_T_camera in views:
            rotation = world_T_camera[:3, :3].astype(np.float32)
            origin = world_T_camera[:3, 3].astype(np.float32)
            # Range along the optical axis first — one column — so only what is in front of
            # the camera and within range is projected at all.
            z = (centroids - origin) @ rotation[:, 2]
            ahead = np.nonzero((z > _NEAR) & (z < _FAR))[0]
            if len(ahead) == 0:
                continue
            relative = centroids[ahead] - origin
            z = z[ahead]
            u = np.rint(relative @ rotation[:, 0] / z * fx + cx).astype(np.int32)
            v = np.rint(relative @ rotation[:, 1] / z * fy + cy).astype(np.int32)
            height, width = depth.shape[:2]
            inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
            ahead, u, v, z = ahead[inside], u[inside], v[inside], z[inside]
            measured = _carving_depth(depth)[v, u]
            through[ahead[measured > z + _CARVE_MARGIN + _CARVE_MARGIN_SQ * z * z]] = True
        if not through.any():
            return 0
        now = time.monotonic() - self._epoch
        candidates = keys[through]
        with self._lock:
            pos = np.searchsorted(self._keys, candidates)
            pos = pos[pos < len(self._keys)]
            pos = pos[np.isin(self._keys[pos], candidates)]  # still there, not merged away
            first, last = self._acc[pos, _SEEN_THROUGH], self._acc[pos, _SEEN_THROUGH + 1]
            # A gap longer than the carve time since the last look-through starts the clock again.
            restart = (first < 0) | (now - last > _CARVE_SEC)
            self._acc[pos[restart], _SEEN_THROUGH] = now
            self._acc[pos, _SEEN_THROUGH + 1] = now
            gone = pos[~restart & (now - first >= _CARVE_SEC)]
            if len(gone):
                keep = np.ones(len(self._keys), bool)
                keep[gone] = False
                self._keys, self._acc = self._keys[keep], self._acc[keep]
                self.carved += len(gone)
                self.seq += 1
        return len(gone)

    def export(self, limit: int = _SEND_POINTS) -> tuple[np.ndarray, np.ndarray]:
        """Centroids (N×3 float32) and colours (N×3 uint8) of every confirmed voxel."""
        with self._lock:
            keys, acc = self._keys, self._acc.copy()
        good = acc[:, 7] >= _MIN_HITS
        keys, acc = keys[good], acc[good]
        if len(keys) > limit:
            best = np.argpartition(acc[:, 6], -limit)[-limit:]
            keys, acc = keys[best], acc[best]
        index = np.stack(((keys >> 42) & _MASK, (keys >> 21) & _MASK, keys & _MASK), axis=1) - _OFFSET
        weight = acc[:, 6:7]
        xyz = index * self.voxel + acc[:, 0:3] / weight
        rgb = np.clip(acc[:, 3:6] / weight, 0, 255)
        return xyz.astype(np.float32), rgb.astype(np.uint8)


class Reconstructor:
    """Pulls frames from the selected sources while someone is watching, and fuses them."""

    def __init__(self, agent: object, realsense: object) -> None:
        self.agent = agent
        # Called each time: the console may find the RealSense bridge after startup.
        self._realsense = realsense
        self.map = VoxelMap()
        # The front and hand cameras alone — what the RealSense is calibrated against.
        self.reference = VoxelMap()
        self.enabled = dict.fromkeys(SOURCES, True)
        self._notes = dict.fromkeys(SOURCES, "")
        self._rates = dict.fromkeys(SOURCES, 0.0)
        self._wanted = 0.0
        self._threads: list[threading.Thread] = []
        self._robot: dict | None = None
        self._rs_mount = _mount()
        self._packed: tuple[int, bytes] = (-1, b"")
        # Depth frames waiting to be carved with, per map — only the latest per camera pose
        # matters, so the capture loops never wait on carving.
        self._carving: dict[int, list] = {}
        self._carving_lock = threading.Lock()

    # ── control ───────────────────────────────────────────────────────────────

    def want(self) -> None:
        """A browser is looking: run (or keep running) for the next few seconds."""
        self._wanted = time.time()
        if not self._threads:
            loops = (("recon-front", self._spot_loop, ("hand", "front"), 0.0),
                     ("recon-sides", self._spot_loop, ("sides",), _SIDE_PERIOD),
                     ("recon-realsense", self._realsense_loop, (), 0.0),
                     ("recon-carve", self._carve_loop, (), 0.0))
            for name, loop, args, period in loops:
                thread = threading.Thread(target=loop, args=(args, period) if args else (), daemon=True, name=name)
                thread.start()
                self._threads.append(thread)
            logger.info(f"3D reconstruction running on this machine — {self.map.voxel * 100:.1f} cm voxels")

    @property
    def active(self) -> bool:
        return time.time() - self._wanted < _IDLE_SEC

    def clear(self) -> None:
        self.map.clear()
        self.reference.clear()
        self._packed = (-1, b"")

    def set_sources(self, changes: dict) -> dict:
        for name in SOURCES:
            if name in changes:
                self.enabled[name] = bool(changes[name])
        return dict(self.enabled)

    def status(self) -> dict:
        return {
            "state": "running" if self.active else "idle",
            "seq": self.map.seq,
            "voxels": len(self.map),
            "carved": self.map.carved,
            "voxel_m": self.map.voxel,
            "robot": self._robot,
            "rs_mount": _describe(self._rs_mount),
            "rs_calibrated": _MOUNT_FILE.is_file(),
            "sources": {name: {"enabled": self.enabled[name], "rate": round(self._rates[name], 1),
                               "note": self._notes[name]} for name in SOURCES},
        }

    def cloud(self) -> bytes:
        """uint32 count, then xyz float32 and rgb uint8 — re-packed only when the map changed."""
        seq = self.map.seq
        if self._packed[0] != seq:
            xyz, rgb = self.map.export()
            self._packed = (seq, len(xyz).to_bytes(4, "little") + xyz.tobytes() + rgb.tobytes())
        return self._packed[1]

    # ── the two capture loops ─────────────────────────────────────────────────

    def _spot_loop(self, groups: tuple[str, ...], period: float) -> None:
        """Spot's cameras in ``groups``, colour and depth in one image request, at most every ``period`` s."""
        while True:
            cameras = {camera: group for group in groups if self.enabled[group] for camera in _CAMERAS[group]}
            if not self.active or not cameras:
                for group in groups:
                    self._rates[group] = 0.0
                time.sleep(0.3)
                continue
            started = time.time()
            try:
                frames = capture(self.agent, list(cameras), depth=True, quality=85)
                carving, reference_carving = [], []
                for name, frame in frames.items():
                    if frame.depth is None:
                        continue
                    far, stride, gain, reference = _FUSION[cameras[name]]
                    snapshot = frame.response.shot.transforms_snapshot
                    pinhole = frame.response.source.pinhole.intrinsics
                    intrinsics = (pinhole.focal_length.x, pinhole.focal_length.y,
                                  pinhole.principal_point.x, pinhole.principal_point.y)
                    pose = _camera_pose(frame.response)
                    points, rgb, weight = backproject(frame.depth, frame.image, intrinsics, pose, far, stride=stride)
                    if name != HAND:  # the hand camera does not see the legs; it looks close in front of them
                        keep = outside_body(points, get_a_tform_b(snapshot, VISION_FRAME_NAME, BODY_FRAME_NAME).to_matrix())
                        points, rgb, weight = points[keep], rgb[keep], weight[keep]
                    weight = weight * gain
                    self.map.fuse(points, rgb, weight)
                    if reference:
                        self.reference.fuse(points, rgb, weight)
                        reference_carving.append((frame.depth, intrinsics, pose))
                    carving.append((frame.depth, intrinsics, pose))
                self._to_carve(self.map, carving)
                self._to_carve(self.reference, reference_carving)
                if frames:
                    self._robot = _body_pose(next(iter(frames.values())).response.shot.transforms_snapshot)
                rate = 1.0 / max(time.time() - started, period, 1e-3)
                for group in groups:
                    self._rates[group] = rate if self.enabled[group] else 0.0
                    self._notes[group] = ""
            except Exception as exc:
                for group in groups:
                    self._notes[group] = str(exc)
                logger.debug(f"Recon: Spot capture ({', '.join(groups)}) failed: {exc}")
                time.sleep(1.0)
                continue
            time.sleep(max(0.0, period - (time.time() - started)))

    def _to_carve(self, voxels: VoxelMap, views: list) -> None:
        if views and _CARVE_SEC > 0:
            with self._carving_lock:
                pending = self._carving.setdefault(id(voxels), [])
                pending.extend(views)
                del pending[:-_CARVE_BACKLOG]

    def _carve_loop(self) -> None:
        """Clear what has left the scene, a few times a second, off the capture threads."""
        while True:
            started = time.time()
            for voxels in (self.map, self.reference):
                with self._carving_lock:
                    views = self._carving.pop(id(voxels), [])
                try:
                    voxels.carve(views)
                except Exception as exc:
                    logger.debug(f"Recon: carving failed: {exc}")
            time.sleep(max(0.05, _CARVE_PERIOD - (time.time() - started)))

    def _realsense_loop(self) -> None:
        while True:
            client = self._realsense()
            if not self.active or not self.enabled["realsense"] or client is None:
                self._rates["realsense"] = 0.0
                self._notes["realsense"] = "" if client is not None else "no RealSense bridge"
                time.sleep(0.3)
                continue
            started = time.time()
            try:
                frame = client.rgbd(step=_RS_STEP)
                note = self._realsense_unusable(frame)
                if note:
                    self._notes["realsense"] = note
                    time.sleep(0.2)
                    continue
                state = self.agent.state_client.get_robot_state()  # type: ignore[attr-defined]
                if not _still(state):
                    self._notes["realsense"] = "paused while Spot moves"
                    time.sleep(0.1)
                    continue
                snapshot = state.kinematic_state.transforms_snapshot
                world = get_a_tform_b(snapshot, VISION_FRAME_NAME, BODY_FRAME_NAME).to_matrix() @ self._rs_mount
                intrinsics = (frame["fx"], frame["fy"], frame["cx"], frame["cy"])
                self.map.fuse(*backproject(frame["depth"], frame["color"], intrinsics, world, _FAR))
                self._to_carve(self.map, [(frame["depth"], intrinsics, world)])
                self._rates["realsense"] = 1.0 / max(time.time() - started, 1e-3)
                self._notes["realsense"] = ""
            except Exception as exc:
                self._notes["realsense"] = str(exc)
                logger.debug(f"Recon: RealSense capture failed: {exc}")
                time.sleep(2.0)

    def calibrate(self) -> dict:
        """Find the RealSense mount from the floor and gravity, refine it by ICP, save it.

        Spot must be still, with floor in the RealSense's view. ICP needs the 3D Recon view to
        have been open a few seconds so the front cameras have built a reference; without one
        the floor-and-gravity result is saved on its own.
        """
        client = self._realsense()
        if client is None:
            raise RuntimeError("no RealSense bridge")
        frame = client.rgbd(step=_RS_STEP)
        note = self._realsense_unusable(frame)
        if note:
            raise RuntimeError(f"RealSense: {note}")
        state = self.agent.state_client.get_robot_state()  # type: ignore[attr-defined]
        if not _still(state):
            raise RuntimeError("Spot is moving — calibrate with it standing still")
        snapshot = state.kinematic_state.transforms_snapshot
        vision_T_body = get_a_tform_b(snapshot, VISION_FRAME_NAME, BODY_FRAME_NAME).to_matrix()
        up_body = vision_T_body[:3, :3].T @ np.array([0.0, 0.0, 1.0])
        before = self._rs_mount
        # Height is anchored to the floor the front cameras measured when there is a
        # reference, so both clouds share one floor; Spot's own ground estimate is ~2 cm off it.
        target, _ = self.reference.export(limit=200000)
        if len(target) >= 5000:
            low = np.percentile(target[:, 2], 3)
            floor_z = float(np.median(target[target[:, 2] < low + 0.05, 2]))
            body_height = float(vision_T_body[2, 3] - floor_z)
        else:
            body_height = -get_a_tform_b(snapshot, BODY_FRAME_NAME, GROUND_PLANE_FRAME_NAME).z

        # 1. Floor and gravity: pitch, roll and height.
        depth = frame["depth"][::2, ::2]
        vs, us = np.nonzero(depth > 0.2)
        z = depth[vs, us]
        camera = np.stack(((us * 2 - frame["cx"]) / frame["fx"] * z, (vs * 2 - frame["cy"]) / frame["fy"] * z, z), 1)
        floor = floor_plane(camera, before[:3, :3].T @ up_body)
        if floor is None:
            raise RuntimeError("no floor found in the RealSense's view — point it at open floor and try again")
        normal, camera_height, floor_share = floor
        mount = before.copy()
        mount[:3, :3] = _rotation_between(mount[:3, :3] @ normal, up_body) @ mount[:3, :3]
        position = mount[:3, 3]
        mount[:3, 3] = position + (camera_height - body_height - up_body @ position) * up_body
        method = "floor"

        # 2. ICP against the front cameras: yaw and the horizontal offset.
        fraction = rmse = observed = None
        if len(target) >= 5000:
            intrinsics = (frame["fx"], frame["fy"], frame["cx"], frame["cy"])
            source, _, _ = backproject(frame["depth"], frame["color"], intrinsics, vision_T_body @ mount, _FAR_BODY)
            if len(source) > 30000:
                source = source[np.random.default_rng(0).choice(len(source), 30000, replace=False)]
            correction, fraction, rmse, observed = icp(source, target)
            refined = np.linalg.inv(vision_T_body) @ correction @ vision_T_body @ mount
            turned = math.degrees(math.acos(min(1.0, (np.trace(refined[:3, :3] @ mount[:3, :3].T) - 1) / 2)))
            shifted = float(np.linalg.norm(refined[:3, 3] - mount[:3, 3]))
            if observed < _ICP_MIN_OBSERVED:
                logger.warning(f"RealSense calibration: ICP not used — too little vertical surface in view to fix "
                               f"the horizontal offset ({observed:.3f}); put a box or a wall ~1 m in front and retry")
            elif fraction >= 0.3 and rmse <= 0.03 and turned < 8 and shifted < 0.15:
                mount, method = refined, "floor + ICP"
            else:
                logger.info(f"RealSense calibration: ICP not used (inliers {fraction:.0%}, RMSE {rmse * 1000:.0f} mm, "
                            f"{turned:.1f}°, {shifted * 100:.0f} cm) — keeping the floor result")

        moved_cm = float(np.linalg.norm(mount[:3, 3] - before[:3, 3]) * 100)
        turned_deg = math.degrees(math.acos(min(1.0, (np.trace(mount[:3, :3] @ before[:3, :3].T) - 1) / 2)))
        self._rs_mount = mount
        _MOUNT_FILE.parent.mkdir(parents=True, exist_ok=True)
        _MOUNT_FILE.write_text(json.dumps({
            "body_T_camera": mount.tolist(), "mount": _describe(mount), "method": method,
            "camera_height_m": camera_height, "floor_share": floor_share,
            "icp_inliers": fraction, "icp_rmse_m": rmse, "icp_observed": observed, "at": time.time(),
        }, indent=2))
        self.map.clear()
        self._packed = (-1, b"")
        logger.success(f"RealSense calibrated ({method}): RECON_RS_MOUNT={_describe(mount)} — "
                       f"turned {turned_deg:.1f}°, moved {moved_cm:.1f} cm; camera {camera_height:.2f} m above the floor")
        return {"mount": _describe(mount), "method": method, "turned_deg": round(turned_deg, 1),
                "moved_cm": round(moved_cm, 1), "camera_height_m": round(camera_height, 3)}

    @staticmethod
    def _realsense_unusable(frame: dict) -> str:
        if not frame.get("aligned"):
            return "depth is not aligned to colour — launch realsense2_camera with align_depth.enable:=true"
        if max(frame["color_age"], frame["depth_age"]) > 0.5:
            return "frames are stale"
        return ""


def _carving_depth(depth: np.ndarray) -> np.ndarray:
    """Depth for deciding what is seen through: the nearest reading in each 3×3, with no
    reading and depth edges as 0, so neither a hole nor a flying pixel carves anything."""
    filled = np.where(depth > 0, depth, np.inf).astype(np.float32)
    nearest = cv2.erode(filled, _KERNEL)
    spread = cv2.dilate(depth, _KERNEL) - nearest
    nearest[~np.isfinite(nearest) | (spread > _EDGE * nearest)] = 0.0
    return nearest


# The accumulator's columns: 0-2 offset, 3-5 colour, 6 weight, 7 hits; then, for carving,
# first and last time seen through since the last hit.
_SEEN_THROUGH = 8
_COLUMNS = 10

_CAMERAS = {"hand": (HAND,), "front": FRONT, "sides": SIDES}
# Per group: how far depth is trusted, pixel stride, weight gain, and whether it feeds the
# front/hand-only reference the RealSense is calibrated against.
_FUSION = {
    "hand": (_FAR, 2, 1.0, True),
    "front": (_FAR_BODY, 2, 1.0, True),
    "sides": (_FAR_SIDE, 3, _SIDE_WEIGHT, False),
}


def _camera_pose(response: object) -> np.ndarray:
    shot = response.shot  # type: ignore[attr-defined]
    return get_a_tform_b(shot.transforms_snapshot, VISION_FRAME_NAME, shot.frame_name_image_sensor).to_matrix()


def _body_pose(snapshot: object) -> dict:
    body = get_a_tform_b(snapshot, VISION_FRAME_NAME, BODY_FRAME_NAME)
    q = body.rotation
    yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    return {"x": body.x, "y": body.y, "z": body.z, "yaw": yaw}


def _still(state: object) -> bool:
    velocity = state.kinematic_state.velocity_of_body_in_vision  # type: ignore[attr-defined]
    linear = math.hypot(velocity.linear.x, velocity.linear.y)
    angular = abs(velocity.angular.z)
    return linear < _STILL_MPS and angular < _STILL_RADPS


__all__ = ["SOURCES", "Reconstructor", "VoxelMap", "backproject"]
