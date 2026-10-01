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
from both, which is why the page can show several outlines for one choice. A detection
joins a track more than ``TRACK_MERGE_M`` away only if the track lands on its outline (box
grown by half) — in a pile, the neighbour's track is near, but somewhere else in the picture.

**Close objects — a pile.** Distance alone cannot tell two touching objects from one object
the segmenter split (a bear's head and body), so the cameras vote, per pair of nearby
tracks: *apart* when one camera in one scan outlined both separately (the CoreIO already
drops a mask lying inside a bigger one, so two outlines are two things), *together* when
one outline covers the other's position at the other's depth (not merely hiding it, which
would read nearer). Two tracks within ``TRACK_MERGE_M`` merge into the older — whose id the
page keeps working with — only when together outvotes apart, and never when both are
certainly classified into different bins. A pile, outlined piece by piece scan after scan,
stays several objects; a bear split once and seen whole every other time is one. Positions are
a confidence-weighted running mean, so a track gets steadier the more it is seen. Tracks
not seen for ``TRACK_TTL_SEC`` are forgotten, and a track that has been picked up is
dropped (`forget`), since it is no longer where it was.

**What has gone, goes.** A scan that should have seen a track and did not is a miss: the
track lands well inside a camera's picture, within ``TRACK_GONE_RANGE_M``, and the depth
there is not nearer than the track (nothing stands in front of it). ``TRACK_GONE_SCANS``
misses spanning ``TRACK_GONE_SEC``, with no sighting in between, and the track is deleted
— an object picked up and put somewhere else loses its old tag a few scans later, while
one the detector merely misses now and then is kept, since any sighting resets the count.
"""
import math
import os
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations

import cv2
import numpy as np
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, get_a_tform_b
from loguru import logger

from .vision import GROUPS, CameraFrame, Scene, SceneObject, locate, object_range, vision_to_pixel

RADIUS = float(os.getenv("TRACK_RADIUS_M", "0.25"))
# Two tracks closer than this are one object, if the cameras' votes agree (see above).
MERGE = float(os.getenv("TRACK_MERGE_M", "0.12"))
# How near in depth another track must be to an outline covering it to count as part of it.
_SAME_DEPTH_M = 0.06
MIN_SCANS = int(os.getenv("TRACK_MIN_SCANS", "2"))
# One sighting classified this confidently is enough on its own.
SURE_CONF = float(os.getenv("TRACK_SURE_CONF", "0.5"))
# A survey also offers what was seen this long before it started (the AI preview's scans):
# a detector that recognised something a moment ago and misses it now has not made it vanish.
RECENT = float(os.getenv("TRACK_RECENT_SEC", "30"))
TTL = float(os.getenv("TRACK_TTL_SEC", "120"))
# Looked for, in clear view, and not found this many scans over at least this long → gone.
GONE_SCANS = int(os.getenv("TRACK_GONE_SCANS", "3"))
GONE_SEC = float(os.getenv("TRACK_GONE_SEC", "2.0"))
# Only a camera this close (along its axis) counts as having looked: farther, small
# objects are missed by the detector rather than gone.
GONE_RANGE = float(os.getenv("TRACK_GONE_RANGE_M", "2.5"))
# A track this near the picture's edge (share of its size) is not clearly in view.
_GONE_BORDER = 0.06
MAX_RANGE = 4.0
LIMIT = 10
_KEEP_SIGHTINGS = 24
# A hand camera's close look, aimed at one track, outweighs a body camera's glimpse: its
# classification is voted this many times over (the ordinary vote included).
LOOK_WEIGHT = float(os.getenv("TRACK_LOOK_WEIGHT", "3.0"))
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
    # Scans that should have seen it and did not, since it was last seen: how many, from when.
    misses: int = 0
    first_miss: float = 0.0
    last_miss_seq: int = -1
    # The hand's close looks (`ObjectTracker.confirm`): +conf per look that recognised it as
    # what it is taken for, 0 per look that found it but did not, -1 per look at empty floor.
    looks: int = 0
    look_score: float = 0.0

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
        # A merged-away track's id -> the track it went into, so a chip still resolves.
        self._merged: dict[int, int] = {}
        # Per pair of track ids (low, high): scans that saw them as two, and as one.
        self._apart: Counter = Counter()
        self._together: Counter = Counter()

    def update(self, scene: Scene) -> None:
        if not scene.classified:
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
            joined: list[tuple[SceneObject, Track]] = []
            # Most confident first, so a strong detection claims a track before a weak one.
            for obj, point, weight, pose in sorted(placed, key=lambda item: -item[2]):
                track = self._nearest_unused(point, obj, scene.frames[obj.camera], taken)
                if track is None:
                    track = Track(self._next, point, 0.0, now, now)
                    self._tracks[track.id] = track
                    self._next += 1
                taken.setdefault(track.id, set()).add(obj.camera)
                joined.append((obj, track))
                track.position = (track.position * track.weight + point * weight) / (track.weight + weight)
                track.weight += weight
                track.last_seen = now
                track.misses = 0
                track.scans.add(scene.seq)
                _vote(track, obj)
                track.sightings.setdefault(scene.seq, []).append(obj)
                if len(track.sightings) > _KEEP_SIGHTINGS:
                    del track.sightings[min(track.sightings)]
                track.latest = (scene, obj)
                if pose is not None and len(obj.polygon) >= 3:
                    track.outlines[obj.camera] = (obj.polygon, pose)
            self._count_misses(scene, now)
            self._count_votes(scene, joined, taken)
            self._merge_close()

    def _nearest_unused(self, point: np.ndarray, obj: SceneObject, frame: CameraFrame,
                        taken: dict[int, set[str]]) -> Track | None:
        """The closest track within RADIUS that this scan has not already matched from this
        camera — and, beyond MERGE, only one that lands on this detection's outline."""
        best, best_distance = None, RADIUS
        for track in self._tracks.values():
            if obj.camera in taken.get(track.id, ()):
                continue
            distance = float(np.linalg.norm(track.position - point))
            if distance < best_distance and (distance < MERGE or _lands_on(frame, obj, track.position)):
                best, best_distance = track, distance
        return best

    def _count_misses(self, scene: Scene, now: float) -> None:
        """A miss for every track this look should have seen and did not; enough → deleted.

        The hand and body scenes of one look share a seq: a track either saw counts as seen,
        and one look is at most one miss.
        """
        for track in list(self._tracks.values()):
            if scene.seq in track.scans or track.last_miss_seq == scene.seq:
                continue
            if not any(_in_clear_view(frame, track.position) for frame in scene.frames.values()):
                continue
            if track.misses == 0:
                track.first_miss = now
            track.misses += 1
            track.last_miss_seq = scene.seq
            if track.misses >= GONE_SCANS and now - track.first_miss >= GONE_SEC:
                logger.info(f"Tracker: {track.summary()} — not there any more, forgotten")
                del self._tracks[track.id]

    def _count_votes(self, scene: Scene, joined: list[tuple[SceneObject, Track]],
                     taken: dict[int, set[str]]) -> None:
        """What this scan says about each pair of nearby tracks: seen as two, or as one."""
        by_camera: dict[str, list[Track]] = {}
        for obj, track in joined:
            by_camera.setdefault(obj.camera, []).append(track)
        for tracks in by_camera.values():
            for a, b in combinations(tracks, 2):
                if np.linalg.norm(a.position - b.position) < RADIUS:
                    self._apart[_pair(a.id, b.id)] += 1
        for obj, track in joined:
            frame = scene.frames[obj.camera]
            for other in self._tracks.values():
                if (other is track or obj.camera in taken.get(other.id, ())
                        or np.linalg.norm(other.position - track.position) >= RADIUS):
                    continue
                if _covers(frame, obj, other.position):
                    self._together[_pair(track.id, other.id)] += 1

    def _merge_close(self) -> None:
        """Fold every track within MERGE of an older one into it, where the votes say one object."""
        tracks = sorted(self._tracks.values(), key=lambda t: t.id)
        for index, older in enumerate(tracks):
            if older.id not in self._tracks:
                continue
            for newer in tracks[index + 1:]:
                if newer.id not in self._tracks or np.linalg.norm(newer.position - older.position) >= MERGE:
                    continue
                pair = _pair(older.id, newer.id)
                if self._together[pair] <= self._apart[pair] or _different(older, newer):
                    continue
                _absorb(older, newer)
                del self._tracks[newer.id]
                self._merged[newer.id] = older.id
                self._rekey(newer.id, older.id)

    def _rekey(self, gone: int, into: int) -> None:
        """Hand the merged-away track's votes on to the track it went into."""
        for votes in (self._apart, self._together):
            for pair in [p for p in votes if gone in p]:
                other = pair[0] if pair[1] == gone else pair[1]
                count = votes.pop(pair)
                if other != into:
                    votes[_pair(into, other)] += count

    def _prune(self, now: float) -> None:
        for track_id in [t.id for t in self._tracks.values() if now - t.last_seen > TTL]:
            del self._tracks[track_id]
        self._merged = {old: new for old, new in self._merged.items() if new in self._tracks}
        for votes in (self._apart, self._together):
            for pair in [p for p in votes if p[0] not in self._tracks or p[1] not in self._tracks]:
                del votes[pair]

    # ── what the missions and the page ask ────────────────────────────────────

    def candidates(self, since: float = 0.0, limit: int = LIMIT) -> list[Track]:
        """Registered tracks seen since ``RECENT`` s before ``since``, best first.

        Classified before unclaimed, then confidence, then how often seen — ranking by hits
        first let a dozen patches of floor, segmented every scan, bury a bear recognised once.
        """
        since = since - RECENT if since else 0.0
        with self._lock:
            pool = [t for t in self._tracks.values() if t.registered and t.last_seen >= since and t.latest]
        # A track the hand looked straight at and found nothing goes last; one it confirmed
        # goes ahead of the equally certain ones it never looked at.
        pool.sort(key=lambda t: (t.look_score >= 0, t.classified, not t.uncertain, t.look_score > 0,
                                 round(t.conf, 1), t.hits), reverse=True)
        return pool[:limit]

    def worth_a_look(self, since: float, groups) -> list[Track]:
        """Tracks seen since ``since`` that a prompt claimed at least once — a match, sure or
        not, registered or not — for one of ``groups``: what a close look could settle. Best first."""
        with self._lock:
            pool = [t for t in self._tracks.values()
                    if t.last_seen >= since and t.labels and t.group in groups and t.latest]
        pool.sort(key=lambda t: (not t.uncertain, round(t.conf, 1), t.hits), reverse=True)
        return pool

    def confirm(self, track_id: int, scene: Scene) -> str:
        """Score a track from a classified hand scan aimed straight at it.

        `update` has already filed the scan like any other; this adds the look's verdict:
        ``seen`` (recognised as its bin — voted `LOOK_WEIGHT` times over), ``found`` (outlined
        but not recognised as that), ``empty`` (in clear view, nothing there) or ``unseen``
        (the hand could not tell: out of the picture, or no depth there).
        """
        with self._lock:
            track = self._tracks.get(self._resolve(track_id))
            if track is None:
                return "empty"  # the miss that look added was its last: gone
            track.looks += 1
            taken_for = track.group  # before the extra votes below (the ordinary one is already in)
            mine = [o for o in track.sightings.get(scene.seq, []) if o.camera in scene.frames]
            if mine:
                best = max(mine, key=lambda o: o.conf)
                claimed = not (best.uncertain and best.group == "other" and best.label == best.seg_label)
                if claimed:
                    _vote(track, best, weight=LOOK_WEIGHT - 1.0)
                if claimed and best.group == taken_for:
                    track.look_score += best.conf
                    return "seen"
                return "found"
            frame = next(iter(scene.frames.values()), None)
            if frame is not None and _in_clear_view(frame, track.position):
                track.look_score -= 1.0
                return "empty"
            return "unseen"

    def get(self, track_id: int) -> Track | None:
        with self._lock:
            return self._tracks.get(self._resolve(track_id))

    def forget(self, track_id: int) -> None:
        with self._lock:
            self._tracks.pop(self._resolve(track_id), None)

    def clear(self) -> None:
        with self._lock:
            self._tracks.clear()
            self._merged.clear()
            self._apart.clear()
            self._together.clear()

    def _resolve(self, track_id: int) -> int:
        while track_id in self._merged:
            track_id = self._merged[track_id]
        return track_id

    def __len__(self) -> int:
        return len(self._tracks)


def _pair(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _different(a: Track, b: Track) -> bool:
    """Both certainly classified, into different bins."""
    return not a.uncertain and not b.uncertain and a.group != b.group


def _lands_on(frame: CameraFrame, obj: SceneObject, position: np.ndarray) -> bool:
    """Whether a point projects onto this detection's box, grown by half its size each way."""
    pixel = vision_to_pixel(frame, tuple(position))
    if pixel is None or not obj.xyxy:
        return False
    x0, y0, x1, y1 = obj.xyxy
    grow_x, grow_y = (x1 - x0) / 2, (y1 - y0) / 2
    return x0 - grow_x <= pixel[0] <= x1 + grow_x and y0 - grow_y <= pixel[1] <= y1 + grow_y


def _in_clear_view(frame: CameraFrame, position: np.ndarray) -> bool:
    """Whether this frame should show the point: well inside the picture, in range, and with
    depth there that is not nearer than the point (else something stands in front of it)."""
    if frame.depth is None:
        return False
    pose = camera_pose(frame)
    if pose is None:
        return False
    along = float((position - pose[:3, 3]) @ pose[:3, 2])
    if not 0.2 < along < GONE_RANGE:
        return False
    pixel = vision_to_pixel(frame, tuple(position))
    if pixel is None:
        return False
    height, width = frame.depth.shape[:2]
    u, v = round(pixel[0]), round(pixel[1])
    if not (_GONE_BORDER * width <= u < (1 - _GONE_BORDER) * width
            and _GONE_BORDER * height <= v < (1 - _GONE_BORDER) * height):
        return False
    measured = frame.depth[v - 3:v + 4, u - 3:u + 4]
    measured = measured[measured > 0]
    if measured.size < 5:
        return False  # no depth there: cannot tell gone from unseen
    return float(np.median(measured)) > along - (0.15 + 0.03 * along * along)


def _covers(frame: CameraFrame, obj: SceneObject, position: np.ndarray) -> bool:
    """Whether this outline covers a point that it is not merely hiding: the point projects
    inside it and the depth there is the point's own. No depth there, no vote."""
    if frame.depth is None or len(obj.polygon) < 3:
        return False
    pixel = vision_to_pixel(frame, tuple(position))
    if pixel is None:
        return False
    u, v = round(pixel[0]), round(pixel[1])
    height, width = frame.depth.shape[:2]
    if not (0 <= u < width and 0 <= v < height):
        return False
    if cv2.pointPolygonTest(np.array(obj.polygon, dtype=np.float32), (float(u), float(v)), False) < 0:
        return False
    pose = camera_pose(frame)
    measured = frame.depth[max(0, v - 2):v + 3, max(0, u - 2):u + 3]
    measured = measured[measured > 0]
    if pose is None or measured.size == 0:
        return False
    expected = float((position - pose[:3, 3]) @ pose[:3, 2])  # along the optical axis, as depth is
    return abs(float(np.median(measured)) - expected) < _SAME_DEPTH_M


def _absorb(into: Track, other: Track) -> None:
    """Merge ``other``'s evidence into ``into``: position by weight, every vote, every sighting."""
    total = into.weight + other.weight
    if total > 0:
        into.position = (into.position * into.weight + other.position * other.weight) / total
    into.weight = total
    into.first_seen = min(into.first_seen, other.first_seen)
    into.last_seen = max(into.last_seen, other.last_seen)
    into.scans |= other.scans
    for mine, theirs in ((into.labels, other.labels), (into.seg_labels, other.seg_labels),
                         (into.groups, other.groups), (into.materials, other.materials)):
        mine.update(theirs)
    into.best_conf = max(into.best_conf, other.best_conf)
    into.best_class = max(into.best_class, other.best_class)
    into.certain_votes += other.certain_votes
    into.looks += other.looks
    into.look_score += other.look_score
    for seq, objects in other.sightings.items():
        into.sightings.setdefault(seq, []).extend(objects)
    while len(into.sightings) > _KEEP_SIGHTINGS:
        del into.sightings[min(into.sightings)]
    if other.latest is not None and (into.latest is None or other.latest[0].seq > into.latest[0].seq):
        into.latest = other.latest
    for camera, outline in other.outlines.items():
        into.outlines.setdefault(camera, outline)


def _vote(track: Track, obj: SceneObject, weight: float = 1.0) -> None:
    """Add one sighting's say on what the object is (``weight`` times over)."""
    track.best_conf = max(track.best_conf, obj.conf)
    unclaimed = obj.uncertain and obj.group == "other" and obj.label == obj.seg_label
    if unclaimed:
        # No prompt claimed it — but the class scores under the threshold still count.
        if obj.seg_label:
            track.seg_labels[obj.seg_label] += obj.conf * weight
        for group, score in obj.scores.items():
            if group in GROUPS:
                track.groups[group] += score * weight
                track.best_class = max(track.best_class, score)
        return
    track.labels[obj.label] += obj.conf * weight
    track.groups[obj.group] += obj.conf * weight
    track.materials[obj.material] += obj.conf * weight
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


__all__ = ["LIMIT", "LOOK_WEIGHT", "ObjectTracker", "Track", "camera_pose", "distance_to", "same_view"]
