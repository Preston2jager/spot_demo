"""Drawing a `Scene`: objects kept in colour, everything else turned grey, cameras stitched.

The segmentation masks are what make this view worth having. A box around a bottle says
roughly where it is; a bottle in full colour on a grey room says *which pixels* the model
thinks are the bottle, and an operator about to confirm a grasp can see at a glance whether
the outline is the bottle or the bottle and half the table.

Everything is drawn in each camera's own pixel frame first — outlines, fills — then rotated
into its cell of the panorama, and only the text is drawn afterwards, in panorama space, so
the labels on the sideways-mounted front cameras read upright. `Panorama.layout` keeps the
transform for every cell so the page can map a click back to an object and the server can
draw the page's selection.

The body cameras are laid out as a 2×2 grid rather than a strip: the front pair together in
one quadrant (they are cross-eyed, so side by side they make one forward view), then back,
left and right — four pictures the size of a quarter of the panel instead of five slivers.
"""
from dataclasses import dataclass

import cv2
import numpy as np

from .vision import GROUPS, UNCERTAIN_BGR, Scene, SceneObject, project_box

# How the body cameras are mounted: the front pair lie on their sides, the right one is
# upside down. Rotating by these puts every picture the right way up.
_ROTATE = {
    "frontleft_fisheye_image": cv2.ROTATE_90_CLOCKWISE,
    "frontright_fisheye_image": cv2.ROTATE_90_CLOCKWISE,
    "right_fisheye_image": cv2.ROTATE_180,
}
# The grid, row by row. The front pair are listed right-then-left because they are
# cross-eyed: the right camera sees the left half of what is ahead.
_QUADRANTS = (
    ("FRONT", ("frontright_fisheye_image", "frontleft_fisheye_image")),
    ("BACK", ("back_fisheye_image",)),
    ("LEFT", ("left_fisheye_image",)),
    ("RIGHT", ("right_fisheye_image",)),
)
_QUAD_W, _QUAD_H = 720, 500
_GAP = 4


def highlight(image: np.ndarray, objects: list[SceneObject], selected: int | None = None,
              labels: bool = False) -> np.ndarray:
    """Grey the frame, restore colour inside every object's mask, outline each by its group.

    Drawn on every live frame while AI is on, so the fills go into one tint layer and one
    blend rather than a full-frame copy per object. ``labels`` adds each object's chip; the
    panorama leaves it off and places the chips itself, in its own coordinates.
    """
    grey = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    canvas = (grey * 0.45).astype(np.uint8)
    shapes = [(obj, np.array(obj.polygon, dtype=np.int32)) for obj in objects if len(obj.polygon) >= 3]
    if not shapes:
        return canvas
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    tint = image.copy()
    for obj, points in shapes:
        cv2.fillPoly(mask, [points], 255)
        # A thin tint of the group colour over the object, so the classification reads even
        # where the outline is hidden behind a label.
        cv2.fillPoly(tint, [points], UNCERTAIN_BGR if obj.uncertain else GROUPS[obj.group]["bgr"])
    blended = cv2.addWeighted(image, 0.8, tint, 0.2, 0)
    np.copyto(canvas, blended, where=mask[..., None] > 0)
    for obj, points in shapes:
        colour = UNCERTAIN_BGR if obj.uncertain else GROUPS[obj.group]["bgr"]
        thickness = 5 if obj.id == selected else 2
        if obj.id == selected:
            cv2.polylines(canvas, [points], True, (255, 255, 255), thickness + 3, cv2.LINE_AA)
        cv2.polylines(canvas, [points], True, colour, thickness, cv2.LINE_AA)
        cv2.drawMarker(canvas, (obj.cx, obj.cy), (255, 255, 255), cv2.MARKER_CROSS, 14, 2)
    if labels:
        for obj, points in shapes:
            _label(canvas, obj, int(points[:, 0].min()), int(points[:, 1].min()) - 6, obj.id == selected)
    return canvas


def _label(canvas: np.ndarray, obj: SceneObject, x: int, y: int, selected: bool) -> None:
    """A filled chip above the object: id, what the AI calls it, confidence, ? when unsure.

    The name is the model's own class; the bin it sorts into is the chip's colour.
    """
    colour = UNCERTAIN_BGR if obj.uncertain else GROUPS[obj.group]["bgr"]
    text = f"{obj.id} {obj.label}{' ?' if obj.uncertain else ''} {obj.conf:.2f}"
    scale = 0.5
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x = int(max(0, min(x, canvas.shape[1] - tw - 8)))
    y = int(max(th + 6, y))
    cv2.rectangle(canvas, (x, y - th - 6), (x + tw + 8, y + 2), colour, -1)
    if selected:
        cv2.rectangle(canvas, (x - 2, y - th - 8), (x + tw + 10, y + 4), (255, 255, 255), 2)
    cv2.putText(canvas, text, (x + 4, y - 3), cv2.FONT_HERSHEY_SIMPLEX, scale, (10, 10, 10), 1, cv2.LINE_AA)


@dataclass
class Cell:
    camera: str
    x: int
    y: int
    w: int
    h: int
    src_w: int
    src_h: int
    rotate: int | None

    def to_cell(self, px: float, py: float) -> tuple[float, float]:
        """A pixel of the camera's own image, in panorama coordinates."""
        rw, rh = (self.src_h, self.src_w) if self.rotate == cv2.ROTATE_90_CLOCKWISE else (self.src_w, self.src_h)
        if self.rotate == cv2.ROTATE_90_CLOCKWISE:
            px, py = self.src_h - 1 - py, px
        elif self.rotate == cv2.ROTATE_180:
            px, py = self.src_w - 1 - px, self.src_h - 1 - py
        return self.x + px * self.w / rw, self.y + py * self.h / rh


class Panorama:
    """The surround cameras in a 2×2 grid, or one camera at its own size — plain or annotated.

    ``annotate=False`` is the Body Cameras view with AI off: the same layout, so turning AI
    on changes what is drawn and never where anything is.
    """

    def __init__(self, scene: Scene, selected: int | None = None, annotate: bool = True) -> None:
        self.scene = scene
        self.selected = selected
        self.annotate = annotate
        self.cells: list[Cell] = []
        self.image = self._render()

    def _tile(self, camera: str) -> tuple[np.ndarray, int | None]:
        """One camera's picture, highlighted if annotating, turned the right way up."""
        frame = self.scene.frames[camera]
        objects = [o for o in self.scene.objects if o.camera == camera]
        drawn = highlight(frame.image, objects, self.selected) if self.annotate else frame.image
        rotate = _ROTATE.get(camera)
        return (cv2.rotate(drawn, rotate) if rotate is not None else drawn), rotate

    def _render(self) -> np.ndarray:
        canvas = self._grid() if self.scene.kind == "surround" else self._single()
        for obj in self.scene.objects if self.annotate else ():
            cell = self.cell(obj.camera)
            if cell is None or not obj.polygon:
                continue
            points = [cell.to_cell(px, py) for px, py in obj.polygon]
            top = min(p[1] for p in points)
            left = min(p[0] for p in points)
            _label(canvas, obj, int(left), int(top) - 6, obj.id == self.selected)
        return canvas

    def _single(self) -> np.ndarray:
        """A non-surround scan — the hand camera — at its native size, one cell."""
        if not self.scene.frames:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        camera = next(iter(self.scene.frames))
        tile, rotate = self._tile(camera)
        src_h, src_w = self.scene.frames[camera].image.shape[:2]
        self.cells.append(Cell(camera, 0, 0, tile.shape[1], tile.shape[0], src_w, src_h, rotate))
        return tile if tile is not self.scene.frames[camera].image else tile.copy()

    def _grid(self) -> np.ndarray:
        canvas = np.full((2 * _QUAD_H + _GAP, 2 * _QUAD_W + _GAP, 3), 12, dtype=np.uint8)
        for index, (title, cameras) in enumerate(_QUADRANTS):
            qx = (index % 2) * (_QUAD_W + _GAP)
            qy = (index // 2) * (_QUAD_H + _GAP)
            tiles = [(camera, *self._tile(camera)) for camera in cameras if camera in self.scene.frames]
            if tiles:
                width = sum(t.shape[1] for _, t, _ in tiles) + _GAP * (len(tiles) - 1)
                height = max(t.shape[0] for _, t, _ in tiles)
                scale = min(_QUAD_W / width, _QUAD_H / height)
                x = qx + round((_QUAD_W - width * scale) / 2)
                y = qy + round((_QUAD_H - height * scale) / 2)
                for camera, tile, rotate in tiles:
                    w, h = round(tile.shape[1] * scale), round(tile.shape[0] * scale)
                    canvas[y: y + h, x: x + w] = cv2.resize(tile, (w, h), interpolation=cv2.INTER_AREA)
                    src_h, src_w = self.scene.frames[camera].image.shape[:2]
                    self.cells.append(Cell(camera, x, y, w, h, src_w, src_h, rotate))
                    x += w + round(_GAP * scale)
            cv2.putText(canvas, title, (qx + 10, qy + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(canvas, title, (qx + 10, qy + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 2, cv2.LINE_AA)
        return canvas

    def cell(self, camera: str) -> Cell | None:
        return next((c for c in self.cells if c.camera == camera), None)

    def describe(self) -> dict:
        """The layout and every object's outline in panorama pixels — what the page hit-tests."""
        objects = []
        for obj in self.scene.objects:
            cell = self.cell(obj.camera)
            if cell is None:
                continue
            outline = [[round(v, 1) for v in cell.to_cell(px, py)] for px, py in obj.polygon]
            objects.append({
                "id": obj.id,
                "camera": obj.camera,
                "group": obj.group,
                "group_label": GROUPS[obj.group]["label"],
                "label": obj.label,
                "seg_label": obj.seg_label,
                "conf": round(obj.conf, 3),
                "uncertain": obj.uncertain,
                "material": obj.material,
                "scores": obj.scores,
                "source": obj.source,
                "outline": outline,
            })
        return {
            "seq": self.scene.seq,
            "kind": self.scene.kind,
            "taken_at": self.scene.taken_at,
            "elapsed_ms": self.scene.elapsed_ms,
            "width": int(self.image.shape[1]),
            "height": int(self.image.shape[0]),
            "objects": objects,
            "counts": self.scene.counts(),
            "selected": self.selected,
        }


def track_objects(tracks: list, scene: Scene) -> list[SceneObject]:
    """Registered objects as the objects of a held scan — numbered by track, not by detection.

    For each camera of ``scene``, an object's outline is, in order of preference: its
    detection in this scan; the last outline the AI found for it in that camera, if the
    camera has not moved since ("remembered"); or a box where its registered position
    projects ("projected"). One object can have an outline in several cameras.
    """
    from .tracker import camera_pose, same_view

    objects = []
    for track in tracks:
        for camera, frame in scene.frames.items():
            sighting = next((o for o in track.sightings.get(scene.seq, []) if o.camera == camera), None)
            if sighting is not None:
                polygon, source = sighting.polygon, "detected"
            else:
                stored = track.outlines.get(camera)
                pose = camera_pose(frame) if stored is not None else None
                if stored is not None and pose is not None and same_view(stored[1], pose):
                    polygon, source = stored[0], "remembered"
                else:
                    box = project_box(frame, track.position)
                    if box is None:
                        continue
                    polygon, source = [[round(u), round(v)] for u, v in box], "projected"
            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            objects.append(SceneObject(
                id=track.id, camera=camera, group=track.group, label=track.label, seg_label="",
                conf=track.conf, uncertain=track.uncertain, material=track.material,
                xyxy=[min(xs), min(ys), max(xs), max(ys)], polygon=[list(p) for p in polygon],
                cx=round(sum(xs) / len(xs)), cy=round(sum(ys) / len(ys)), area=0.0,
                scores={}, source=source,
            ))
    return objects


def describe_track(track: object) -> dict:
    """What the page lists for a registered object; its outlines are the held scan's objects."""
    return {
        "id": track.id,  # type: ignore[attr-defined]
        "label": track.label,  # type: ignore[attr-defined]
        "group": track.group,  # type: ignore[attr-defined]
        "group_label": GROUPS[track.group]["label"],  # type: ignore[attr-defined]
        "conf": round(track.conf, 3),  # type: ignore[attr-defined]
        "hits": track.hits,  # type: ignore[attr-defined]
        "uncertain": track.uncertain,  # type: ignore[attr-defined]
        "material": track.material,  # type: ignore[attr-defined]
    }
