"""Turn a GraphNav graph into a top-down layout the browser can draw.

The waypoint map exists because "waypoint_67" means nothing to anyone. Picking where Spot
fetches from and where it drops off should be two clicks on a picture of the room, not two
strings typed into `.env` from a printout of `check_map.py`.

Positions come from the graph's own **anchoring** — the seed frame every waypoint is
already expressed in — when the map has one, which the maps recorded here do. Without it
the layout is walked out over the edges instead: each edge carries the transform between
the two waypoints it joins, so a breadth-first pass from any waypoint places the rest.
That second path drifts on long loops, which is exactly what an anchoring is for; it is a
fallback, not an equal.
"""
import hashlib
import os
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
from bosdyn.client.frame_helpers import ODOM_FRAME_NAME, get_a_tform_b
from bosdyn.client.math_helpers import SE3Pose
from loguru import logger

# Where a built cloud is kept between runs. Downloading 56 snapshots over Spot's wifi is a
# minute the operator should pay once, not once per console restart.
_CLOUD_CACHE = Path(__file__).resolve().parent.parent.parent / "runtime" / "clouds"
# Points to keep. The full set off a 56-waypoint map is around 600k, which the browser
# draws happily; the budget is here so a much larger map degrades by thinning rather than
# by running the laptop out of memory. Thinning is a stride, not a random sample: it keeps
# the sweep structure of each scan instead of turning it into noise.
_CLOUD_BUDGET = int(os.getenv("WEB_CLOUD_POINTS", "400000"))
# Points further than this from the camera that saw them are dropped. Spot builds these
# clouds from stereo depth, whose error grows with the square of range: past 8 m or so a
# return is a guess, and a map's worth of guesses is the starburst of speckle that hides
# the room. Measured in the sensor's own frame, before the point is placed in the map,
# because that is where "how far away was this" is a meaningful question.
_CLOUD_RANGE_M = float(os.getenv("WEB_CLOUD_RANGE_M", "8.0"))
# The robot's graph, and when it was fetched. The page asks for the map from three places
# at once — the layout, the cloud's progress poll, and again on every reopen — and each
# one downloading 56 waypoints afresh is a round trip and a log line per poll for a map
# that changes when somebody uploads a new one, which is not something that happens
# between two requests a second apart.
_GRAPH_TTL = float(os.getenv("WEB_GRAPH_TTL", "30"))
_ROBOT_GRAPH: tuple[object | None, float] = (None, 0.0)


def forget_graph() -> None:
    """Drop the cached robot graph — after a map has been uploaded, say."""
    global _ROBOT_GRAPH
    _ROBOT_GRAPH = (None, 0.0)


def load_graph(agent: object) -> object | None:
    """The graph Spot is holding — the localised one, else whatever GraphNav has — or None.

    **Only the robot's map, never this laptop's copy.** A freshly booted Spot holds no map,
    and drawing `SPOT_MAP_DIR` instead used to make it look as if it did: points got chosen
    on a map the robot did not have. Loading the local map onto Spot is an explicit step
    (`/api/map/load`), after which this finds it on the robot like any other.
    """
    global _ROBOT_GRAPH
    graph = getattr(agent, "graph", None)
    if graph is not None:
        return graph
    cached, at = _ROBOT_GRAPH
    if time.time() - at < _GRAPH_TTL:
        return cached
    try:
        graph = agent.get_current_graph()  # type: ignore[attr-defined]
    except Exception as exc:
        logger.debug(f"Could not download the robot's graph: {exc}")
        graph = None
    if graph is not None and not graph.waypoints:
        graph = None
    _ROBOT_GRAPH = (graph, time.time())
    return graph


def local_map() -> str | None:
    """The map directory that `/api/map/load` would upload, if there is one on disk."""
    map_dir = Path(os.getenv("SPOT_MAP_DIR", "./runtime/maps/amp_3"))
    return str(map_dir) if (map_dir / "graph").is_file() else None


def layout(graph: object) -> dict:
    """Project the graph onto the seed-frame XY plane, ready for an SVG."""
    positions, source = _positions(graph)
    heights = _heights(graph)
    waypoints = []
    for waypoint in graph.waypoints:  # type: ignore[union-attr]
        point = positions.get(waypoint.id)
        if point is None:
            # A waypoint no edge reaches cannot be placed. Listed without coordinates so
            # it is still selectable by name rather than quietly missing from the map.
            waypoints.append({
                "id": waypoint.id,
                "name": waypoint.annotations.name,
                "x": None,
                "y": None,
            })
            continue
        waypoints.append({
            "id": waypoint.id,
            "name": waypoint.annotations.name,
            "x": round(point[0], 3),
            "y": round(point[1], 3),
            # Height too, so the 3D view puts a marker where Spot actually stood rather
            # than flat on z=0 while the points around it climb a ramp.
            "z": round(heights.get(waypoint.id, 0.0), 3),
        })

    edges = [
        {"from": edge.id.from_waypoint, "to": edge.id.to_waypoint}
        for edge in graph.edges  # type: ignore[union-attr]
        if edge.id.from_waypoint in positions and edge.id.to_waypoint in positions
    ]

    fiducials = []
    for obj in getattr(graph, "anchoring", None).objects if _has_anchoring(graph) else []:
        pose = SE3Pose.from_proto(obj.seed_tform_object)
        fiducials.append({
            "id": obj.id,
            "x": round(pose.x, 3),
            "y": round(pose.y, 3),
            # Fiducials are on walls, well above the floor, so the height is the whole
            # point of drawing them in a 3D view rather than as dots on a plan.
            "z": round(pose.z, 3),
        })

    return {
        "source": source,
        "waypoints": waypoints,
        "edges": edges,
        "fiducials": fiducials,
        # Fiducials count towards the extent: they sit outside the walked route often
        # enough that fitting to the waypoints alone pushes them off the picture, and a
        # fiducial off the picture is the one landmark the operator was looking for.
        "bounds": _bounds(
            list(positions.values()) + [(f["x"], f["y"]) for f in fiducials]
        ),
    }


# ── internals ─────────────────────────────────────────────────────────────────


def _has_anchoring(graph: object) -> bool:
    """True when the graph carries a seed frame every waypoint is anchored to."""
    anchoring = getattr(graph, "anchoring", None)
    return bool(anchoring and anchoring.anchors)


def _heights(graph: object) -> dict[str, float]:
    """Seed-frame z per waypoint. Empty without an anchoring — the edge walk is 2D."""
    if not _has_anchoring(graph):
        return {}
    return {
        anchor.id: anchor.seed_tform_waypoint.position.z
        for anchor in graph.anchoring.anchors  # type: ignore[union-attr]
    }


def _positions(graph: object) -> tuple[dict[str, tuple[float, float]], str]:
    """Seed-frame XY per waypoint, and which method produced it."""
    if _has_anchoring(graph):
        anchors = {
            anchor.id: (anchor.seed_tform_waypoint.position.x, anchor.seed_tform_waypoint.position.y)
            for anchor in graph.anchoring.anchors  # type: ignore[union-attr]
        }
        return anchors, "anchoring"
    return _walk_edges(graph), "edges"


def _walk_edges(graph: object) -> dict[str, tuple[float, float]]:
    """Place waypoints by composing edge transforms outward from the first waypoint.

    Only used for maps with no anchoring. Each edge stores ``from_tform_to``, so one
    breadth-first pass reaches everything the edges connect; disconnected components are
    left unplaced rather than piled on the origin.
    """
    if not graph.waypoints:  # type: ignore[union-attr]
        return {}
    neighbours: dict[str, list[tuple[str, SE3Pose]]] = {}
    for edge in graph.edges:  # type: ignore[union-attr]
        tform = SE3Pose.from_proto(edge.from_tform_to)
        neighbours.setdefault(edge.id.from_waypoint, []).append((edge.id.to_waypoint, tform))
        neighbours.setdefault(edge.id.to_waypoint, []).append((edge.id.from_waypoint, tform.inverse()))

    root = graph.waypoints[0].id  # type: ignore[union-attr]
    poses: dict[str, SE3Pose] = {root: SE3Pose(0.0, 0.0, 0.0, SE3Pose.from_identity().rot)}
    pending = deque([root])
    while pending:
        current = pending.popleft()
        for neighbour, tform in neighbours.get(current, []):
            if neighbour in poses:
                continue
            poses[neighbour] = poses[current] * tform
            pending.append(neighbour)
    return {wp_id: (pose.x, pose.y) for wp_id, pose in poses.items()}


def _bounds(points: object) -> dict:
    """The extent of the layout, so the page can fit it without guessing a scale."""
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    if not xs:
        return {"min_x": -1.0, "max_x": 1.0, "min_y": -1.0, "max_y": 1.0}
    return {"min_x": min(xs), "max_x": max(xs), "min_y": min(ys), "max_y": max(ys)}


# ── the point cloud ───────────────────────────────────────────────────────────
#
# Every waypoint snapshot carries the points Spot's depth cameras saw while the map was
# being recorded — around 10,000 per waypoint, in the frame of the sensor that saw them.
# Put through the two transforms below they all land in the same seed frame the waypoints
# are anchored in, which is what makes a single 3D scene out of 56 separate scans.
#
# This robot has no lidar payload, so this *is* the three-dimensional record of the space:
# there is no other point source on it to prefer.


def _snapshot(agent: object, snapshot_id: str) -> object | None:
    """One waypoint snapshot, from disk when this laptop already has that exact one.

    Snapshot ids are content-addressed, so a local file with a matching name is not merely
    a plausible copy of the robot's snapshot — it is the same bytes. Reading it saves a
    couple of hundred KB over Spot's wifi, and a whole map's worth of that is the
    difference between a few seconds and a couple of minutes.
    """
    from bosdyn.api.graph_nav import map_pb2

    map_dir = Path(os.getenv("SPOT_MAP_DIR", "./runtime/maps/amp_3"))
    for candidate in (map_dir / "waypoint_snapshots" / snapshot_id, map_dir / f"wp_{snapshot_id}"):
        if candidate.is_file():
            snapshot = map_pb2.WaypointSnapshot()
            snapshot.ParseFromString(candidate.read_bytes())
            return snapshot

    client = getattr(agent, "graph_nav_client", None)
    if client is None:
        return None
    return client.download_waypoint_snapshot(snapshot_id)


def _seed_tform_cloud(waypoint: object, anchors: dict, cloud: object) -> object | None:
    """Where a snapshot's points sit in the seed frame.

    Three frames, two hops: the points are in the sensor's frame, the snapshot knows that
    sensor against the odometry frame it was recorded in, the waypoint knows itself
    against that same odometry frame, and the anchoring knows the waypoint against the
    seed. Skipping the middle hop — treating the points as if they were already in the
    waypoint's frame — leaves each scan rotated by however Spot happened to be standing.
    """
    anchor = anchors.get(waypoint.id)
    if anchor is None:
        return None
    odom_tform_cloud = get_a_tform_b(
        cloud.source.transforms_snapshot, ODOM_FRAME_NAME, cloud.source.frame_name_sensor
    )
    if odom_tform_cloud is None:
        return None
    waypoint_tform_cloud = SE3Pose.from_proto(waypoint.waypoint_tform_ko) * odom_tform_cloud
    return SE3Pose.from_proto(anchor) * waypoint_tform_cloud


class CloudBuilder:
    """Builds the map's point cloud once, in the background, and remembers it.

    Background because it is a download: 56 snapshots off the robot is long enough that
    doing it inside a request would hang the page, and the operator would rather watch a
    count climb than a spinner. Once, because the result is keyed by the snapshot ids that
    produced it — the same map rebuilt is the same file, and a different map is a
    different key rather than a stale cache anybody has to remember to clear.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._key: str | None = None
        self._points: bytes | None = None
        self._state = "idle"
        self._done = 0
        self._total = 0
        self._error: str | None = None

    # ── what the console asks ─────────────────────────────────────────────────

    def status(self) -> dict:
        with self._lock:
            return self._status()

    def _status(self) -> dict:
        """The status dict, for a caller that already holds the lock.

        Separate from `status()` because `threading.Lock` is not reentrant: `request()`
        decides what to do while holding the lock and then has to describe the result,
        and calling the public method there deadlocks the request that asked.
        """
        return {
            "state": self._state,
            "done": self._done,
            "total": self._total,
            "points": (len(self._points) // 12) if self._points else 0,
            "error": self._error,
        }

    def data(self) -> bytes | None:
        """The cloud as packed float32 xyz triplets, or None while it is being built."""
        with self._lock:
            return self._points

    def request(self, agent: object, graph: object) -> dict:
        """Start a build if this map has not been built yet. Cheap to call repeatedly."""
        key = _graph_key(graph)
        with self._lock:
            if self._key == key and self._state in {"ready", "building"}:
                return self._status()
            cached = _CLOUD_CACHE / f"{key}.f32"
            if cached.is_file():
                self._key, self._points, self._state = key, cached.read_bytes(), "ready"
                self._error = None
                return self._status()
            self._key, self._points, self._state = key, None, "building"
            self._done, self._total, self._error = 0, len(graph.waypoints), None
            started = self._status()
        threading.Thread(
            target=self._build, args=(agent, graph, key), daemon=True, name="map-cloud"
        ).start()
        return started

    # ── the work ──────────────────────────────────────────────────────────────

    def _build(self, agent: object, graph: object, key: str) -> None:
        anchors = {
            anchor.id: anchor.seed_tform_waypoint
            for anchor in getattr(graph, "anchoring", None).anchors
        } if _has_anchoring(graph) else {}
        chunks: list[np.ndarray] = []
        skipped = 0

        for waypoint in graph.waypoints:
            try:
                snapshot = _snapshot(agent, waypoint.snapshot_id)
                if snapshot is not None and snapshot.point_cloud.data:
                    transform = _seed_tform_cloud(waypoint, anchors, snapshot.point_cloud)
                    if transform is not None:
                        points = np.frombuffer(snapshot.point_cloud.data, dtype=np.float32)
                        points = points.reshape(-1, 3)
                        if _CLOUD_RANGE_M > 0:
                            keep = np.einsum("ij,ij->i", points, points) <= _CLOUD_RANGE_M ** 2
                            points = points[keep]
                        if len(points):
                            chunks.append(transform.transform_cloud(points).astype(np.float32))
                    else:
                        skipped += 1
                else:
                    skipped += 1
            except Exception as exc:
                # One bad snapshot is a hole in the scene, not a failed map.
                skipped += 1
                logger.debug(f"Snapshot {waypoint.snapshot_id[:8]} skipped: {exc}")
            with self._lock:
                self._done += 1

        if not chunks:
            with self._lock:
                self._state = "error"
                self._error = "no snapshot in this map carries a point cloud"
            logger.warning("Map cloud: nothing to draw — no snapshot carried points")
            return

        cloud = np.concatenate(chunks)
        if len(cloud) > _CLOUD_BUDGET:
            cloud = cloud[:: max(1, len(cloud) // _CLOUD_BUDGET)]
        payload = np.ascontiguousarray(cloud, dtype=np.float32).tobytes()

        try:
            _CLOUD_CACHE.mkdir(parents=True, exist_ok=True)
            (_CLOUD_CACHE / f"{key}.f32").write_bytes(payload)
        except OSError as exc:
            logger.debug(f"Map cloud not cached to disk: {exc}")

        with self._lock:
            self._points, self._state = payload, "ready"
        logger.info(
            f"Map cloud ready — {len(cloud):,} points from "
            f"{len(graph.waypoints) - skipped}/{len(graph.waypoints)} waypoints"
        )


def _graph_key(graph: object) -> str:
    """A name for this exact map: its snapshot ids, hashed.

    Content-addressed all the way down, so two consoles pointed at the same robot build
    the same key, and a map that has been re-recorded never collides with the old one.
    """
    digest = hashlib.sha1()
    for waypoint in sorted(graph.waypoints, key=lambda w: w.id):
        digest.update(waypoint.snapshot_id.encode())
    digest.update(f"|range={_CLOUD_RANGE_M}|budget={_CLOUD_BUDGET}".encode())
    return digest.hexdigest()[:16]
