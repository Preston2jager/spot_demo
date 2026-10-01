"""Segment-then-classify vision service, run in the `spot-vision` image on Spot's CoreIO.

Replaces `yolo_server.py`, which served stock COCO YOLOv8: 80 fixed classes, boxes only,
and no way to say "I see *something* here but I do not know what". The sorting demo needs
all three, so this is two stages on the same frame:

1. **segment** — YOLOE prompt-free (`segment.pt`). Separates every object into its own
   mask whatever it is. This is what decides *that* there is an object, and its outline.
2. **classify** — YOLOE with the demo's text prompts baked in (`classify.pt`, written by
   `coreio/vision/bake_vocab.py`). Each prompt belongs to a group — `teddy_bear`, `bottle`,
   `other`, or `ignore` — and carries a grip material.

Each segmented object is matched against the classify boxes it overlaps and scored per
group. The best group wins if it is confident enough; if no group is, the object is still
reported, as `other` with ``uncertain: true`` — the brief's "mark it when the class is not
settled". Two groups scoring close together are also uncertain, because that is exactly the
object an operator should look at before the robot picks it up.

The material (`soft` / `medium` / `hard`) comes from the winning prompt, falling back to
the group's default. It is what the laptop turns into gripper torque.

Wire format is the length-prefixed JSON header + length-prefixed JPEGs of the old service.
Coordinates — boxes, polygons, grasp points — are in the *original* image's pixels.

Python 3.8 — no match statements, no `X | None` annotations.
"""
import json
import os
import struct
import threading
import time
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

# TensorRT 8's Python bindings use `np.bool`, removed in numpy 1.24. Restored before
# ultralytics imports tensorrt; harmless for the .pt path. See the old yolo_server.py.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    for _alias, _builtin in (("bool", bool), ("int", int), ("float", float),
                             ("object", object), ("str", str)):
        if not hasattr(np, _alias):
            setattr(np, _alias, _builtin)

import torch
import torch.nn.functional as F
from ultralytics import YOLOE
from ultralytics.engine.model import Model
from ultralytics.utils import ops

WEIGHTS_DIR = os.environ.get("VISION_WEIGHTS_DIR", "/weights")
SEGMENT_WEIGHTS = os.environ.get("VISION_SEGMENT_WEIGHTS", WEIGHTS_DIR + "/segment.pt")
CLASSIFY_WEIGHTS = os.environ.get("VISION_CLASSIFY_WEIGHTS", WEIGHTS_DIR + "/classify.pt")
VOCAB_FILE = os.environ.get("VISION_VOCAB", WEIGHTS_DIR + "/vocab.json")
DEVICE = os.environ.get("VISION_DEVICE", "0")
HALF = os.environ.get("VISION_HALF", "1") == "1"
IMGSZ = int(os.environ.get("VISION_IMGSZ", "640"))
HOST = os.environ.get("VISION_HOST", "127.0.0.1")
PORT = int(os.environ.get("VISION_PORT", "8784"))


def _env_float(name, default):
    return float(os.environ.get(name, str(default)))


# Stage 1. Wide open: an object the segmenter misses cannot be classified at all, while a
# spurious mask is cheap — it simply fails to match anything and is filtered below.
SEG_CONF = _env_float("VISION_SEG_CONF", 0.12)
# Stage 2 runs even wider; its scores are evidence, weighed per object, not a verdict.
CLS_CONF = _env_float("VISION_CLS_CONF", 0.03)
# A group needs this much evidence to be the answer at all...
ACCEPT = _env_float("VISION_ACCEPT", 0.20)
# ...and this much, clear of the runner-up by MARGIN, to be reported as certain.
CERTAIN = _env_float("VISION_CERTAIN", 0.40)
MARGIN = _env_float("VISION_MARGIN", 0.10)
# How well a classify box must line up with a segmented object to count as describing it.
MATCH_IOU = _env_float("VISION_MATCH_IOU", 0.45)
# Masks covering more than this share of the frame are floor, wall or table top; smaller
# than MIN_AREA is noise, or an object too far away to pick up anyway.
MAX_AREA = _env_float("VISION_MAX_AREA", 0.30)
MIN_AREA = _env_float("VISION_MIN_AREA", 0.0006)
# A mask this much inside a larger one is a part of it — a bottle's label, a bear's eye.
PART_OF = _env_float("VISION_PART_OF", 0.85)
# An object no prompt claims is still reported — but only if the segmenter itself was sure
# of it. Below this it is clutter: shadows, reflections, a scuff on the floor.
UNCLAIMED_MIN_SEG = _env_float("VISION_UNCLAIMED_MIN_SEG", 0.20)
MASK_SCALE = _env_float("VISION_MASK_SCALE", 0.5)
# Polygon simplification in pixels. The page draws these; a few px is invisible.
POLY_EPS = _env_float("VISION_POLY_EPS", 1.5)
# Prompt-free labels that name surfaces and people rather than things to pick up. The
# classify stage's `ignore` group catches most of these; this catches what it does not
# describe at all, so a patch of carpet is never offered as an "other".
STUFF = {s.strip() for s in os.environ.get(
    "VISION_STUFF",
    "floor,flooring,wall,ceiling,carpet,rug,mat,ground,tile,road,pavement,sky,grass,"
    "table,desk,countertop,shelf,person,man,woman,boy,girl,child,people,hand,arm,leg,"
    "foot,shadow,window,door,curtain,building,room,wood floor,hardwood,home interior,"
    "living room,playroom,childs room,kitchen,office,interior,siding,blind",
).split(",") if s.strip()}

_seg_model = None
_cls_model = None
_vocab = {}
_cls_names = {}
# One GPU, so inference is serialised; the server stays threaded only so a client that
# dies mid-request cannot wedge the next one.
_infer_lock = threading.Lock()


def _load_vocab():
    global _vocab
    try:
        with open(VOCAB_FILE) as handle:
            _vocab = json.load(handle)
    except (OSError, ValueError) as exc:
        print(f"[vision] WARNING: no usable vocab at {VOCAB_FILE} ({exc}) — every object will be 'other'", flush=True)
        _vocab = {}


def _group_of(label):
    return _vocab.get("label_to_group", {}).get(label, _vocab.get("fallback_group", "other"))


def _material_of(label, group):
    """The grip material for a classify label, else a keyword guess, else the group default."""
    by_label = _vocab.get("label_to_material", {})
    if label in by_label:
        return by_label[label]
    lowered = (label or "").lower()
    for known, material in by_label.items():
        if known in lowered or (lowered and lowered in known):
            return material
    return _vocab.get("material_defaults", {}).get(group, "medium")


def _predict(model, images, **kwargs):
    """Inference through the model's one predictor, set up on the first call and kept.

    Not ``model(...)``: in ultralytics 8.3.156 ``YOLOE.predict`` builds a fresh predictor
    and calls ``setup_model`` on *every* call — fusing the layers again and running a warmup
    pass — so each request paid for two model loads before any real inference. That was
    most of the ~0.8 s a single hand frame took. ``Model.predict`` is the base-class path,
    which reuses the predictor and only updates its arguments; with no visual prompts it is
    exactly what ``YOLOE.predict`` ends up calling anyway. It also letterboxes ``rect`` —
    640x480 rather than padded to 640x640 — a quarter fewer pixels through the network.
    """
    return Model.predict(model, source=images, device=DEVICE, half=HALF, verbose=False, **kwargs)


def _warm(model):
    result = _predict(model, [np.zeros((480, 640, 3), dtype=np.uint8)], imgsz=IMGSZ)
    return dict(result[0].names)


def load():
    """Load both stages once and warm CUDA before answering anything."""
    global _seg_model, _cls_model, _cls_names
    started = time.time()
    print(f"[vision] loading {SEGMENT_WEIGHTS} + {CLASSIFY_WEIGHTS} on device {DEVICE} "
          f"(half={HALF}, imgsz={IMGSZ})", flush=True)
    _load_vocab()
    _seg_model = YOLOE(SEGMENT_WEIGHTS)
    seg_names = _warm(_seg_model)
    _cls_model = YOLOE(CLASSIFY_WEIGHTS)
    _cls_names = _warm(_cls_model)
    counts = {}
    for label in _cls_names.values():
        group = _group_of(label)
        counts[group] = counts.get(group, 0) + 1
    print(f"[vision] segment vocabulary {len(seg_names)} labels; classify vocabulary "
          f"{len(_cls_names)} prompts {counts}", flush=True)
    print(f"[vision] ready in {time.time() - started:.1f}s", flush=True)


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _objects_from_masks(seg, shape):
    """Turn one segmentation result into candidate objects with polygons in image pixels."""
    height, width = shape[:2]
    if seg.boxes is None or len(seg.boxes) == 0 or seg.masks is None:
        return []
    # Scaled down on the GPU before it crosses to the CPU: contours, the grasp point and
    # the part test are all fine at half size, and on a CoreIO whose six cores are shared
    # with the ROS 2 stack, copying and scanning full-size masks was a third of a request.
    with torch.no_grad():
        data = seg.masks.data
        if MASK_SCALE < 1.0:
            data = F.interpolate(data[None].float(), scale_factor=MASK_SCALE, mode="bilinear",
                                 align_corners=False)[0]
        masks = data.cpu().numpy() > 0.5                # (N, h, w) in letterbox space
    mask_shape = masks.shape[1:]
    boxes = seg.boxes.xyxy.cpu().numpy()
    confs = seg.boxes.conf.cpu().numpy()
    classes = seg.boxes.cls.cpu().numpy().astype(int)
    names = seg.names
    frame_area = float(width * height)

    candidates = []
    for i in range(len(boxes)):
        label = names.get(int(classes[i]), "object")
        if label in STUFF:
            continue
        mask = masks[i].astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        # The point furthest from the mask's edge: inside the object even when it is
        # concave, which a centroid is not — a grasp aimed at a mug's centroid is air.
        # Padded with zeros: OpenCV counts outside the image as foreground, so a mask cut
        # off by the frame edge would otherwise put its "inside" point on that edge.
        dist = cv2.distanceTransform(cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0),
                                     cv2.DIST_L2, 3)[1:-1, 1:-1]
        _, _, _, inner = cv2.minMaxLoc(dist)
        mapped = ops.scale_coords(
            mask_shape,
            np.concatenate([contour.reshape(-1, 2), np.array([inner])]).astype(np.float32),
            (height, width),
        )
        polygon = mapped[:-1]
        area = cv2.contourArea(polygon.astype(np.float32)) / frame_area
        if area < MIN_AREA or area > MAX_AREA:
            continue
        simplified = cv2.approxPolyDP(polygon.astype(np.float32), POLY_EPS, True).reshape(-1, 2)
        if len(simplified) < 3:
            continue
        x1, y1, x2, y2 = (float(v) for v in boxes[i])
        candidates.append({
            "mask": masks[i],
            "mask_px": int(masks[i].sum()),
            "seg_label": label,
            "seg_conf": float(confs[i]),
            "xyxy": [x1, y1, x2, y2],
            "polygon": [[round(x), round(y)] for x, y in simplified],
            "grasp": [round(mapped[-1][0]), round(mapped[-1][1])],
            "area": area,
        })
    return _drop_parts(candidates)


def _drop_parts(candidates):
    """Remove masks that sit almost entirely inside a larger one — parts, not objects."""
    candidates.sort(key=lambda c: c["mask_px"], reverse=True)
    kept = []
    for small in candidates:
        is_part = False
        for big in kept:
            if _iou(small["xyxy"], big["xyxy"]) <= 0.0:
                continue
            if small["mask_px"] > 0.6 * big["mask_px"]:
                continue
            inside = np.logical_and(small["mask"], big["mask"]).sum() / float(max(1, small["mask_px"]))
            if inside >= PART_OF:
                is_part = True
                break
        if not is_part:
            kept.append(small)
    return kept


def _classify(candidates, cls):
    """Score each candidate per group from the classify boxes that line up with it."""
    evidence = []
    if cls is not None and cls.boxes is not None and len(cls.boxes):
        for box, conf, index in zip(cls.boxes.xyxy.cpu().numpy(), cls.boxes.conf.cpu().numpy(),
                                    cls.boxes.cls.cpu().numpy().astype(int)):
            label = _cls_names.get(int(index), "unknown")
            evidence.append(([float(v) for v in box], float(conf), label))

    ignored = set(_vocab.get("ignore_groups", []))
    fallback = _vocab.get("fallback_group", "other")
    objects = []
    for cand in candidates:
        scores = {}
        labels = {}
        for box, conf, label in evidence:
            overlap = _iou(cand["xyxy"], box)
            if overlap < MATCH_IOU:
                continue
            score = conf * min(1.0, overlap / 0.75)
            group = _group_of(label)
            if score > scores.get(group, 0.0):
                scores[group] = score
                labels[group] = label
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_group, best = ranked[0] if ranked else (None, 0.0)
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0

        if best_group is not None and best >= ACCEPT:
            if best_group in ignored:
                continue
            group = best_group
            label = labels[best_group]
            uncertain = best < CERTAIN or (best - runner_up) < MARGIN
            conf = best
        else:
            # Segment-only callers match by position and want every mask they can get.
            if cls is not None and cand["seg_conf"] < UNCLAIMED_MIN_SEG:
                continue
            # Found by the segmenter, claimed by no prompt with any confidence. Still an
            # object — reported as the fallback bin and flagged, never silently dropped.
            group = fallback
            label = cand["seg_label"]
            uncertain = True
            conf = cand["seg_conf"] * 0.5
        objects.append({
            "group": group,
            "label": label,
            "seg_label": cand["seg_label"],
            "conf": round(conf, 4),
            "seg_conf": round(cand["seg_conf"], 4),
            "uncertain": bool(uncertain),
            "scores": {g: round(s, 3) for g, s in ranked[:3]},
            "material": _material_of(label, group),
            "xyxy": [int(v) for v in cand["xyxy"]],
            "polygon": cand["polygon"],
            "cx": cand["grasp"][0],
            "cy": cand["grasp"][1],
            "area": round(cand["area"], 5),
        })
    return objects


def analyse(images, cameras, classify=True, imgsz=None):
    """Run the stages over a batch and return every object, sorted by confidence.

    ``classify=False`` is segmentation alone, for a caller that already knows what it is
    looking at and only needs to know where it went — the approach servo, which re-finds
    its chosen target twice a second and would otherwise pay for the vocabulary each time.
    """
    size = int(imgsz or IMGSZ)
    with _infer_lock:
        seg_results = _predict(_seg_model, images, imgsz=size, conf=SEG_CONF, iou=0.5, agnostic_nms=True)
        cls_results = [None] * len(images)
        if classify:
            cls_results = _predict(_cls_model, images, imgsz=size, conf=CLS_CONF, iou=0.6, max_det=100)
    objects = []
    for index, (seg, cls) in enumerate(zip(seg_results, cls_results)):
        for obj in _classify(_objects_from_masks(seg, images[index].shape), cls):
            obj["camera"] = cameras[index]
            obj["width"] = int(images[index].shape[1])
            obj["height"] = int(images[index].shape[0])
            objects.append(obj)
    # Certain before uncertain, then by confidence: the order an automatic pick walks.
    objects.sort(key=lambda o: (o["uncertain"], -o["conf"]))
    return objects


def _unpack(body):
    """Split a request body into (header dict, [BGR image, ...])."""
    if len(body) < 4:
        raise ValueError("body too short")
    (header_len,) = struct.unpack_from(">I", body, 0)
    offset = 4
    header = json.loads(body[offset:offset + header_len].decode("utf-8"))
    offset += header_len
    images = []
    while offset < len(body):
        (blob_len,) = struct.unpack_from(">I", body, offset)
        offset += 4
        blob = body[offset:offset + blob_len]
        offset += blob_len
        image = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("undecodable JPEG in request")
        images.append(image)
    cameras = header.get("cameras", [])
    if len(cameras) != len(images):
        raise ValueError(f"header lists {len(cameras)} cameras but body carries {len(images)} images")
    return header, images


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        return

    def _send_json(self, status, payload):
        blob = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/health":
            self._send_json(200, {
                "status": "ok",
                "pipeline": "segment+classify",
                "segment": os.path.basename(SEGMENT_WEIGHTS),
                "classify": os.path.basename(CLASSIFY_WEIGHTS),
                "backend": "tensorrt" if CLASSIFY_WEIGHTS.endswith(".engine") else "pytorch",
                "device": DEVICE,
                "half": HALF,
                "imgsz": IMGSZ,
                "groups": sorted(set(_vocab.get("label_to_group", {}).values())),
                "ignore_groups": _vocab.get("ignore_groups", []),
                "vocab": _vocab.get("vocab"),
                "prompts": len(_cls_names),
            })
        elif path == "/vocab":
            self._send_json(200, _vocab)
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path.split("?")[0] != "/analyse":
            self._send_json(404, {"error": "not found"})
            return
        try:
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            header, images = _unpack(body)
        except Exception as exc:
            self._send_json(400, {"error": f"bad request: {exc}"})
            return
        started = time.time()
        try:
            objects = analyse(images, header.get("cameras", []),
                              classify=header.get("classify", True), imgsz=header.get("imgsz"))
        except Exception as exc:
            # A server fault, not an empty scene — the client must be able to tell.
            print(f"[vision] analysis failed: {exc}", flush=True)
            self._send_json(500, {"error": f"analysis failed: {exc}"})
            return
        elapsed_ms = (time.time() - started) * 1000
        counts = {}
        for obj in objects:
            counts[obj["group"]] = counts.get(obj["group"], 0) + 1
        print(f"[vision] {len(images)} image(s) -> {len(objects)} object(s) {counts} in {elapsed_ms:.0f} ms",
              flush=True)
        self._send_json(200, {"objects": objects, "counts": counts, "elapsed_ms": round(elapsed_ms, 1)})


def main():
    load()
    server = ThreadingHTTPServer((HOST, PORT), _Handler)
    server.daemon_threads = True
    print(f"[vision] listening on {HOST}:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
