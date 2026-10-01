"""The video panel's frames: Spot's cameras, the RealSense, and what the vision service saw.

Every view is an MJPEG stream the page attaches an `<img>` to, and every view is captured
only while some browser is pulling it (`wanted`) — the body cameras and the RealSense cost
nothing when nobody has them open.

Views:

* ``hand`` — the gripper camera.
* ``cameras`` — the body cameras in a 2×2 grid (`Panorama`), live.
* ``rs_blend`` — the CoreIO RealSense, colour and depth alpha-blended; the page picks the
  alpha, so 0 is the colour picture and 1 the depth map.

**AI is a layer, not a view.** With it on, ``hand`` and ``cameras`` keep streaming live and
the latest scan of each is drawn over the live picture: every segmented object in colour on
a grey scene, outlined and labelled with its bin, confidence and material. An overlay older
than its TTL is dropped, so a stalled service shows as outlines vanishing rather than as
stale ones on a picture that has moved on.

Scans come from two places: a **preview** loop that rescans back to back while AI is on and
a view is open, and a mission's own scans, which are drawn whether or not AI is on. The
preview pauses while a mission scans, so it never takes the CoreIO from the grasp servo.
While a mission waits for the operator to choose, the view being chosen from — ``hand`` or
``cameras`` — is *held* on that scan: the one picture that must not change under a click.
"""
import dataclasses
import os
import threading
import time

import cv2
import numpy as np
from loguru import logger

from ..vision import HAND, SURROUND, Scene, capture
from ..vision_view import Panorama, describe_track, track_objects

_CAPTURE_FPS = float(os.getenv("WEB_CAPTURE_FPS", "10"))
# The pause between preview scans. The CoreIO answers one at a time and a preview is
# dropped while another scan is in flight, so this only keeps the loop from spinning.
_SCAN_INTERVAL = float(os.getenv("WEB_SCAN_INTERVAL", "0.3"))
# How long a scan stays drawn over the live picture. The surround scan is slower and the
# body cameras move less than the hand, so it is kept longer.
_TTL = {"hand": 2.5, "cameras": float(os.getenv("WEB_OVERLAY_TTL", "4.0"))}
_VIEW_IDLE = 5.0
_RS_RETRY = 5.0
# The robot keeps answering with the last picture when a camera dies (a "hand_color_camera:
# Unable to initialize device" fault does exactly that), so a frozen feed looks alive.
_FROZEN_AFTER = 3.0

VIEWS = {"hand", "cameras", "rs_blend"}


def view_of(scene: Scene) -> str:
    """Which view a scan belongs to."""
    return "hand" if scene.kind == "hand" else "cameras"


class LiveView:
    """Captures what the page asks for and serves it as MJPEG."""

    def __init__(self, agent: object, vision: object) -> None:
        self.agent = agent
        self.vision = vision
        self._lock = threading.Lock()
        self._raw: dict[str, bytes] = {}
        self._requested: dict[str, float] = {}
        self._running = False
        self._placeholders: dict[str, bytes] = {}
        self.realsense: object | None = None
        self._rs_failed_at = 0.0
        self._rs_pair: tuple[np.ndarray, np.ndarray] | None = None
        self._rs_seq = 0
        # The AI switch, and whether a mission is scanning — either draws scans over the views.
        self.detecting = False
        self.mission_scans = False
        # Set by the console: the cameras to scan for the 3D Recon view's labels, or None
        # while nobody is looking at it.
        self.label_scan = lambda: None
        # The latest scan of each view, laid out the way that view draws it.
        self._panoramas: dict[str, Panorama] = {}
        self._held: str | None = None
        # While a choice is held: the registered objects to offer (set by the console).
        self.candidates = None
        self.selected: int | None = None
        self._hand_stamp: tuple[int, int] | None = None
        self._hand_changed = time.time()
        self._hand_frozen = False
        vision.listeners.append(self._on_scene)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        threading.Thread(target=self._capture_loop, daemon=True, name="live-capture").start()
        threading.Thread(target=self._preview_loop, daemon=True, name="live-preview").start()

    # ── what the browser asks for ─────────────────────────────────────────────

    def frames(self, view: str, alpha: float = 0.5):
        """Yield an endless MJPEG part stream for one view."""
        last_seq = -1
        while True:
            self._requested[view] = time.time()
            if view == "rs_blend":
                frame, last_seq = self._blend(alpha, last_seq)
            else:
                with self._lock:
                    frame = self._raw.get(view)
                frame = frame if frame is not None else self._placeholder(view)
            if frame is not None:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(1.0 / _CAPTURE_FPS)

    def wanted(self, view: str) -> bool:
        return time.time() - self._requested.get(view, 0.0) < _VIEW_IDLE

    # ── scans ─────────────────────────────────────────────────────────────────

    @property
    def annotating(self) -> bool:
        return self.detecting or self.mission_scans or self._held is not None

    @property
    def held(self) -> str | None:
        return self._held

    def _on_scene(self, scene: Scene) -> None:
        """A scan finished somewhere — preview or mission. Keep it for the overlay."""
        view = view_of(scene)
        if self._held != view:
            self._adopt(view, scene)

    def _adopt(self, view: str, scene: Scene) -> None:
        """Make ``scene`` the scan ``view`` draws and the page hit-tests.

        A held scan with candidates is drawn as the registered objects, numbered by track,
        with remembered or projected outlines where this scan missed one.
        """
        candidates = self.candidates
        if self._held == view and candidates is not None:
            scene = dataclasses.replace(scene, objects=track_objects(candidates(), scene))
        panorama = Panorama(scene, self.selected)
        with self._lock:
            self._panoramas[view] = panorama
        if self._held == view:
            encoded = _encode(panorama.image, quality=80)
            if encoded is not None:
                with self._lock:
                    self._raw[view] = encoded

    def _overlay(self, view: str) -> Scene | None:
        """The scan to draw over ``view`` right now, or None."""
        with self._lock:
            panorama = self._panoramas.get(view)
        if panorama is None or not self.annotating:
            return None
        scene = panorama.scene
        return scene if self._held == view or time.time() - scene.taken_at < _TTL[view] else None

    def hold(self, scene: Scene) -> None:
        """Freeze one view on one scan — the one the operator is choosing from."""
        self._held = view_of(scene)
        self.selected = None
        self._adopt(self._held, scene)

    def release(self) -> None:
        self._held = None

    def select(self, object_id: int | None) -> None:
        """Mark an object as the chosen one and redraw the held scan."""
        self.selected = object_id
        if self._held is not None:
            with self._lock:
                panorama = self._panoramas.get(self._held)
            if panorama is not None:
                self._adopt(self._held, panorama.scene)

    def scene_description(self, view: str) -> dict | None:
        with self._lock:
            panorama = self._panoramas.get(view)
        if panorama is None:
            return None
        payload = panorama.describe()
        payload["age"] = round(time.time() - panorama.scene.taken_at, 1)
        payload["held"] = self._held == view
        # Whether the outlines are on screen: the page draws its hover layer only then.
        payload["shown"] = self._overlay(view) is not None
        tracker = getattr(self.vision, "tracker", None)
        payload["registered"] = len(tracker.candidates()) if tracker is not None else 0
        candidates = self.candidates
        if self._held == view and candidates is not None:
            payload["tracks"] = [describe_track(track) for track in candidates()]
        return payload

    # ── loops ─────────────────────────────────────────────────────────────────

    def _capture_loop(self) -> None:
        interval = 1.0 / _CAPTURE_FPS
        while self._running:
            started = time.time()
            try:
                if self.wanted("hand") and self._held != "hand":
                    self._capture_hand()
                if self.wanted("cameras") and self._held != "cameras":
                    self._capture_cameras()
                if self.wanted("rs_blend"):
                    self._capture_realsense()
            except Exception as exc:
                logger.debug(f"Live view capture failed: {exc}")
                time.sleep(0.5)
            time.sleep(max(0.0, interval - (time.time() - started)))

    def _preview_loop(self) -> None:
        """While AI is on, rescan whichever camera views are open, one scan after another.

        With the 3D Recon view open instead, scan what a grasp survey would — with depth,
        since those objects are placed in 3D and labelled there.
        """
        while self._running:
            started = time.time()
            if self.detecting and not self.mission_scans:
                try:
                    labelling = self.label_scan()
                    if labelling:
                        self.vision.scan_views(self.agent, labelling, depth=True, wait=False)
                    if not labelling and self.wanted("cameras") and self._held != "cameras":
                        self.vision.scan(self.agent, SURROUND, wait=False)
                    if not labelling and self.wanted("hand") and self._held != "hand":
                        self.vision.scan(self.agent, (HAND,), wait=False)
                except Exception as exc:
                    logger.debug(f"Preview scan failed: {exc}")
                    time.sleep(1.0)
            time.sleep(max(0.05, _SCAN_INTERVAL - (time.time() - started)))

    # ── capture ───────────────────────────────────────────────────────────────

    def _render(self, view: str, kind: str, frames: dict) -> np.ndarray:
        """Live frames laid out like the view's scans, with the current overlay drawn on."""
        overlay = self._overlay(view)
        live = Scene(seq=0, kind=kind, taken_at=time.time(), frames=frames,
                     objects=overlay.objects if overlay is not None else [])
        return Panorama(live, self.selected, annotate=overlay is not None).image

    def _capture_hand(self) -> None:
        frame = capture(self.agent, (HAND,), quality=70).get(HAND)
        if frame is None:
            return
        image = self._render("hand", "hand", {HAND: frame})
        if self._hand_is_frozen(frame):
            image = image.copy()
            cv2.rectangle(image, (0, 0), (image.shape[1], 34), (0, 0, 200), -1)
            cv2.putText(image, f"HAND CAMERA FROZEN - no new frame for {time.time() - self._hand_changed:.0f}s",
                        (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        encoded = _encode(image)
        with self._lock:
            if encoded is not None and self._held != "hand":
                self._raw["hand"] = encoded

    def _hand_is_frozen(self, frame: object) -> bool:
        """Whether the robot has been handing back the same hand picture for a while."""
        shot = frame.response.shot  # type: ignore[attr-defined]
        stamp = (shot.acquisition_time.seconds, shot.acquisition_time.nanos)
        if stamp != self._hand_stamp:
            self._hand_stamp = stamp
            self._hand_changed = time.time()
        frozen = time.time() - self._hand_changed > _FROZEN_AFTER
        if frozen != self._hand_frozen:
            self._hand_frozen = frozen
            if frozen:
                logger.warning("Hand camera is frozen: the robot keeps returning the same frame — "
                               "check the robot's system faults (a power cycle clears a hand_color_camera fault)")
            else:
                logger.info("Hand camera is delivering new frames again")
        return frozen

    def _capture_cameras(self) -> None:
        """The body cameras live, laid out by the same `Panorama` as the scan, so the
        outlines the page hit-tests line up with the live picture."""
        frames = capture(self.agent, SURROUND, quality=60)
        if not frames:
            return
        encoded = _encode(self._render("cameras", "surround", frames))
        with self._lock:
            if encoded is not None and self._held != "cameras":
                self._raw["cameras"] = encoded

    def _capture_realsense(self) -> None:
        """Pull colour and depth together; the blend is made per viewer in `_blend`."""
        client = self.realsense
        if client is None or time.time() - self._rs_failed_at < _RS_RETRY:
            return
        try:
            colour = cv2.imdecode(np.frombuffer(client.color_jpeg(), np.uint8), cv2.IMREAD_COLOR)
            depth = cv2.imdecode(np.frombuffer(client.depth_jpeg(), np.uint8), cv2.IMREAD_COLOR)
        except Exception as exc:
            self._rs_failed_at = time.time()
            logger.debug(f"RealSense capture failed: {exc}")
            return
        if colour is not None and depth is not None:
            self._rs_pair = (colour, depth)
            self._rs_seq += 1

    def _blend(self, alpha: float, last_seq: int) -> tuple[bytes | None, int]:
        """Colour under depth at ``alpha`` (0 = all colour, 1 = all depth), per viewer.

        Blended per request rather than once, because two browsers can want two different
        alphas; re-encoded only when a new pair has arrived.
        """
        pair, seq = self._rs_pair, self._rs_seq
        if pair is None:
            return self._placeholder("rs_blend"), last_seq
        if seq == last_seq:
            return None, last_seq
        colour, depth = pair
        if depth.shape[:2] != colour.shape[:2]:
            depth = cv2.resize(depth, (colour.shape[1], colour.shape[0]), interpolation=cv2.INTER_NEAREST)
        alpha = min(max(alpha, 0.0), 1.0)
        if alpha <= 0.0:
            return _encode(colour), seq
        mixed = cv2.addWeighted(colour, 1.0 - alpha, depth, alpha, 0)
        # Pixels with no depth reading are black in the depth map; blending black in only
        # darkens the colour there, so those keep the colour picture untouched.
        if alpha < 1.0:
            void = depth.max(axis=2) < 8
            mixed[void] = colour[void]
        return _encode(mixed), seq

    def _placeholder(self, view: str) -> bytes | None:
        caption = {
            "cameras": "waiting for the body cameras...",
            "rs_blend": "waiting for RealSense colour and depth...",
        }.get(view, "waiting for the first frame...")
        cached = self._placeholders.get(view)
        if cached is None:
            frame = np.full((480, 640, 3), 18, dtype=np.uint8)
            cv2.putText(frame, caption, (30, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1, cv2.LINE_AA)
            cached = _encode(frame) or b""
            self._placeholders[view] = cached
        return cached or None


def _encode(frame: np.ndarray, quality: int = 72) -> bytes | None:
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buffer.tobytes() if ok else None
