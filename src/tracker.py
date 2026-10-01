"""Objects registered in space, across scans — not decided from one frame.

The detector sees an object in one scan and misses it in the next; a single frame is a poor
basis for choosing what to grasp. So every classified scan — the AI preview's and the
missions' — is fed here, each detection is placed in Spot's vision frame (`locate`: depth
behind the mask, else where the pixel's ray meets the floor), and detections within
``TRACK_RADIUS_M`` of each other are the same object. A track seen in at least
``TRACK_MIN_SCANS`` separate scans — or once, classified at ``TRACK_SURE_CONF`` or better —
is **registered**; those are what the operator, or the automatic mode, chooses from — at
most ten, best first: classified before unclaimed, then by confidence, then how often seen.

**What it is** is voted on by classification evidence only. The CoreIO reports a mask no
prompt claimed as ``other``/uncertain under the segmenter's own label; that is "not
recognised this time", not a vote for the Other bin. So a teddy bear recognised once at
0.6 and only segmented in the next scans stays a teddy bear: those scans add to where it
is and how often it was seen, plus whatever below-threshold class scores came with them.

One object seen by two cameras in the same scan (the front pair overlap) joins one track
from both, which is why the page can show several outlines for one choice. Positions are
a confidence-weighted running mean, so a track gets steadier the more it is seen. Tracks
not seen for ``TRACK_TTL_SEC`` are forgotten, and a track that has been picked up is
dropped (`forget`), since it is no longer where it was.
"""
import math
import os
import threading
import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, get_a_tform_b

from .vision import GROUPS, Scene, SceneObject, locate, object_range

RADIUS = float(os.getenv("TRACK_RADIUS_M", "0.25"))
MIN_SCANS = int(os.getenv("TRACK_MIN_SCANS", "2"))
# One sighting classified this confidently is enough on its own.
SURE_CONF = float(os.getenv("TRACK_SURE_CONF", "0.5"))
# A survey also offers what was seen this long before it started (the AI preview's scans):
# a detector that recognised something a moment ago and misses it now has not made it vanish.
RECENT = float(os.getenv("TRACK_RECENT_SEC", "30"))
TTL = float(os.getenv("TRACK_TTL_SEC", "120"))
MAX_RANGE = 4.0
LIMIT = 10
_KEEP_SIGHTINGS = 24
# The vocab's material_defaults, for an object only ever recognised below the threshold.
_MATERIAL = {"teddy_bear": "soft"}


@dataclass
class Track:
    """One object in space, and every time it has been seen."""

    id: int
    position: np.ndarray
    weight: float
    first_seen: float
    last_seen: float
    scans: set[int] = field(default_factory=set)
    labels: Counter = field(default_factory=Counter)  # classified labels only
    seg_labels: Counter = field(default_factory=Counter)  # the segmenter's, for the unrecognised
    groups: Counter = field(default_factory=Counter)  # classification evidence only
    materials: Counter = field(default_factory=Counter)
    best_conf: float = 0.0  # any sighting, the unclaimed included
    best_class: float = 0.0  # the strongest classification evidence
    certain_votes: int = 0
    # scene seq -> the detections of this object in that scan (one per camera that saw it)
    sightings: dict[int, list[SceneObject]] = field(default_factory=dict)
    latest: tuple[Scene, SceneObject] | None = None
    # camera -> (outline polygon in its pixels, vision_T_camera when it was found)
    outlines: dict[str, tuple[list, np.ndarray]] = field(default_factory=dict)

    @property
    def hits(self) -> int:
        return len(self.scans)

    @property
    def registered(self) -> bool:
        return self.hits >= MIN_SCANS or self.best_class >= SURE_CONF

    @property
    def classified(self) -> bool:
        return bool(self.groups)

    @property
    def label(self) -> str:
        for votes in (self.labels, self.seg_labels):
            if votes:
                return votes.most_common(1)[0][0]
        return "object"

    @property
    def group(self) -> str:
        return self.groups.most_common(1)[0][0] if self.groups else "other"

    @property
    def material(self) -> str:
        return self.materials.most_common(1)[0][0] if self.materials else _MATERIAL.get(self.group, "medium")

    @property
    def uncertain(self) -> bool:
        """Never classified with certainty, or the votes for its bin are close."""
        if not self.certain_votes:
            return True
        ranked = self.groups.most_common(2)
        return len(ranked) > 1 and ranked[1][1] > 0.6 * ranked[0][1]

    @property
    def conf(self) -> float:
        total = sum(self.groups.values())
        return self.groups[self.group] / total * self.best_class if total else self.best_conf

    def summary(self) -> str:
        p = self.position
        return (f"#{self.id} '{self.label}'{'?' if self.uncertain else ''} → {GROUPS[self.group]['label']} bin, "
                f"seen in {self.hits} scans, at ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})")


class ObjectTracker:
    """Registers detections from every classified scan into tracks in the vision frame."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tracks: dict[int, Track] = {}
        self._next = 1

    def update(self, scene: Scene) -> None:
        if not scene.classified or not scene.objects:
            return
        placed = []
        for obj in scene.objects:
            frame = scene.frames.get(obj.camera)
            if frame is None:
                continue
            point = locate(frame, obj)
            if point is None:
                continue
            point = np.asarray(point, dtype=float)
            pose = camera_pose(frame)
            if pose is not None and np.linalg.norm(point - pose[:3, 3]) > MAX_RANGE:
                continue
            # A depth reading pins an object down far better than the floor guess does.
            weight = max(obj.conf, 0.05) * (1.0 if object_range(frame, obj) is not None else 0.4)
            placed.append((obj, point, weight, pose))
        now = time.time()
        with self._lock:
            self._prune(now)
            taken: dict[int, set[str]] = {}
            # Most confident first, so a strong detection claims a track before a weak one.
            for obj, point, weight, pose in sorted(placed, key=lambda item: -item[2]):
                track = self._nearest_unused(point, obj.camera, taken)
                if track is None:
                    track = Track(self._next, point, 0.0, now, now)
                    self._tracks[track.id] = track
                    self._next += 1
                taken.setdefault(track.id, set()).add(obj.camera)
                track.position = (track.position * track.weight + point * weight) / (track.weight + weight)
                track.weight += weight
                track.last_seen = now
                track.scans.add(scene.seq)
                _vote(track, obj)
                track.sightings.setdefault(scene.seq, []).append(obj)
                if len(track.sightings) > _KEEP_SIGHTINGS:
                    del track.sightings[min(track.sightings)]
                track.latest = (scene, obj)
                if pose is not None and len(obj.polygon) >= 3:
                    track.outlines[obj.camera] = (obj.polygon, pose)

    def _nearest_unused(self, point: np.ndarray, camera: str, taken: dict[int, set[str]]) -> Track | None:
        """The closest track within RADIUS that this scan has not already matched from ``camera``."""
        best, best_distance = None, RADIUS
        for track in self._tracks.values():
            if camera in taken.get(track.id, ()):
                continue
            distance = float(np.linalg.norm(track.position - point))
            if distance < best_distance:
                best, best_distance = track, distance
        return best

    def _prune(self, now: float) -> None:
        for track_id in [t.id for t in self._tracks.values() if now - t.last_seen > TTL]:
            del self._tracks[track_id]

    # ── what the missions and the page ask ────────────────────────────────────

    def candidates(self, since: float = 0.0, limit: int = LIMIT) -> list[Track]:
        """Registered tracks seen since ``RECENT`` s before ``since``, best first.

        Classified before unclaimed, then confidence, then how often seen — ranking by hits
        first let a dozen patches of floor, segmented every scan, bury a bear recognised once.
        """
        since = since - RECENT if since else 0.0
        with self._lock:
            pool = [t for t in self._tracks.values() if t.registered and t.last_seen >= since and t.latest]
        pool.sort(key=lambda t: (t.classified, not t.uncertain, round(t.conf, 1), t.hits), reverse=True)
        return pool[:limit]

    def get(self, track_id: int) -> Track | None:
        with self._lock:
            return self._tracks.get(track_id)

    def forget(self, track_id: int) -> None:
        with self._lock:
            self._tracks.pop(track_id, None)

    def clear(self) -> None:
        with self._lock:
            self._tracks.clear()

    def __len__(self) -> int:
        return len(self._tracks)


def _vote(track: Track, obj: SceneObject) -> None:
    """Add one sighting's say on what the object is."""
    track.best_conf = max(track.best_conf, obj.conf)
    unclaimed = obj.uncertain and obj.group == "other" and obj.label == obj.seg_label
    if unclaimed:
        # No prompt claimed it — but the class scores under the threshold still count.
        if obj.seg_label:
            track.seg_labels[obj.seg_label] += obj.conf
        for group, score in obj.scores.items():
            if group in GROUPS:
                track.groups[group] += score
                track.best_class = max(track.best_class, score)
        return
    track.labels[obj.label] += obj.conf
    track.groups[obj.group] += obj.conf
    track.materials[obj.material] += obj.conf
    track.best_class = max(track.best_class, obj.conf)
    track.certain_votes += int(not obj.uncertain)


def camera_pose(frame: object) -> np.ndarray | None:
    """vision_T_camera for a frame, as a 4×4, or None."""
    try:
        shot = frame.response.shot  # type: ignore[attr-defined]
        return get_a_tform_b(shot.transforms_snapshot, VISION_FRAME_NAME, shot.frame_name_image_sensor).to_matrix()
    except Exception:
        return None


def same_view(a: np.ndarray, b: np.ndarray, metres: float = 0.03, degrees: float = 2.0) -> bool:
    """Whether two camera poses are close enough that an outline from one still fits the other."""
    turned = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(a[:3, :3] @ b[:3, :3].T) - 1) / 2))))
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3])) < metres and turned < degrees


def distance_to(agent: object, track: Track) -> float:
    bx, by, _ = agent.get_current_pose()  # type: ignore[attr-defined]
    return math.hypot(track.position[0] - bx, track.position[1] - by)


__all__ = ["LIMIT", "ObjectTracker", "Track", "camera_pose", "distance_to", "same_view"]
