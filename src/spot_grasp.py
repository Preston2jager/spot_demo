"""Walk up to an object, line up on it, then grasp — instead of grasping from where we stand.

The old fetch handed the manipulation API a pixel from wherever the robot happened to be
and let it walk the rest of the way. The walk changes the camera's view of the object, the
API does not look again, and the grasp lands on where the object *used to be* from an angle
that no longer exists. So the approach is ours now, and the API only does the last few
centimetres:

1. **locate** — the chosen object's pixel and the depth behind it give a point in the vision
   frame, which does not move when the robot does.
2. **approach** — walk to a standoff short of that point, facing it.
3. **servo** — every ``SORT_SERVO_PERIOD`` seconds (1.0 by default): aim the hand camera at
   the point with a gaze, find the object again in the hand image, re-measure the point,
   and step the body so the object sits dead ahead at grasping range. Two consecutive
   checks inside tolerance and the robot is lined up.
4. **grasp** — `grasp_in_image` on the latest hand frame with auto-walk *off*, so nothing
   moves the body between the last look and the grasp.
5. **back off and retry** — a grasp that misses (or a servo that loses the object) backs the
   body ``SORT_BACKOFF_M`` straight away from the object, still facing it, and goes round
   the servo and grasp again: from further back the hand sees the whole object and a fresh
   depth reading, instead of trying the same pose twice. ``SORT_GRASP_RETRIES`` times; only
   the last attempt may let the API auto-walk.
6. **grip** — re-close at the torque the object's material calls for, then carry.

Every step goes through `SpotAgent._command`, so a Stop freezes the robot mid-servo exactly
as it would anywhere else.
"""
import math
import os
import time

import cv2
import numpy as np
from loguru import logger

from .spot_agent import MissionStopped, SpotAgent
from .vision import HAND, CameraFrame, CoreIOUnavailable, Scene, SceneObject, Vision, locate, vision_to_pixel


def _env(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


# Where the body should end up: the object this far from the body's centre, dead ahead.
GRASP_RANGE = _env("SORT_GRASP_RANGE_M", 0.85)
RANGE_TOL = _env("SORT_RANGE_TOL_M", 0.10)
YAW_TOL = math.radians(_env("SORT_YAW_TOL_DEG", 7.0))
# The first walk stops this far away, so the hand camera can see the object before the
# servo takes over — close enough to see it well, far enough not to have walked onto it.
STANDOFF = _env("SORT_STANDOFF_M", 1.25)
SERVO_PERIOD = _env("SORT_SERVO_PERIOD", 1.0)
SERVO_TIMEOUT = _env("SORT_SERVO_TIMEOUT", 25.0)
# One servo step never moves the body further than this: the estimate is refreshed once a
# second, so a big correction is several small ones, each checked against a new look.
MAX_STEP = _env("SORT_MAX_STEP_M", 0.30)
MAX_MISSES = int(_env("SORT_MAX_MISSES", 5))
# A missed grasp backs off this far — a step or two — and lines up again, this many times.
BACKOFF = _env("SORT_BACKOFF_M", 0.35)
GRASP_RETRIES = int(_env("SORT_GRASP_RETRIES", 2))


class GraspPlanner:
    """The approach-and-grasp routine for one object, on one agent, through one vision service."""

    def __init__(self, agent: SpotAgent, vision: Vision, phase=None) -> None:
        self.agent = agent
        self.vision = vision
        # Called with a short description of each stage — the console shows it on screen.
        self.phase = phase or (lambda text: None)

    def pick(self, scene: Scene, obj: SceneObject, target: tuple[float, float, float] | None = None) -> bool:
        """Approach, line up on and grasp one object from a scan. True when it is in the gripper.

        ``target`` — the object's registered position, fused over several scans — is used
        instead of placing it from this one detection when given.
        """
        if target is None:
            frame = scene.frames.get(obj.camera)
            target = locate(frame, obj) if frame is not None else None
        else:
            target = tuple(float(v) for v in target)
        if target is None:
            logger.error(f"Cannot place {obj.summary()} in 3D — no depth and no floor under it")
            return False
        logger.info(f"Target {obj.summary()} at ({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}) in vision")

        self.phase(f"approaching {obj.label}")
        self.agent.arm_ready()
        self._walk_to_standoff(target)

        attempts = 1 + GRASP_RETRIES
        for attempt in range(1, attempts + 1):
            if attempt > 1:
                self.phase(f"backing off from {obj.label} (try {attempt}/{attempts})")
                self._back_off(target, attempt - 1)

            self.phase(f"lining up on {obj.label}")
            lined_up = self._servo(target, obj)
            if lined_up is None:
                logger.warning(f"Lost the target while lining up (try {attempt}/{attempts}).")
                self.agent.release_to_ready()
                continue
            hand_frame, hand_obj, target = lined_up

            self.phase(f"grasping {obj.label}")
            if self.agent.grasp_in_image(hand_frame.response, hand_obj.cx, hand_obj.cy, obj.label):
                break
            if attempt == attempts:
                logger.info("No grasp on the last try — allowing auto-walk once.")
                if self.agent.grasp_in_image(hand_frame.response, hand_obj.cx, hand_obj.cy, obj.label,
                                             auto_walk=True):
                    break
            logger.warning(f"Grasp missed (try {attempt}/{attempts}).")
        else:
            return False
        self.agent.set_grip(obj.material)
        self.agent.carry()
        return True

    # ── stages ────────────────────────────────────────────────────────────────

    def _walk_to_standoff(self, target: tuple[float, float, float]) -> None:
        bx, by, _ = self.agent.get_current_pose()
        dx, dy = target[0] - bx, target[1] - by
        distance = math.hypot(dx, dy)
        heading = math.atan2(dy, dx)
        if distance <= STANDOFF:
            # Already close: only turn to face it, never back away first.
            self.agent.move_to_goal(bx, by, heading, timeout_sec=8.0)
            return
        goal_x = target[0] - STANDOFF * math.cos(heading)
        goal_y = target[1] - STANDOFF * math.sin(heading)
        if not self.agent.move_to_goal(goal_x, goal_y, heading, timeout_sec=15.0, tolerance_m=0.25):
            logger.warning("Did not quite reach the standoff — the servo will make up the difference.")

    def _back_off(self, target: tuple[float, float, float], steps: int) -> None:
        """Step straight back from ``target``, still facing it, before lining up again.

        Measured from the target, not from where the body stands: GRASP_RANGE plus one
        BACKOFF per retry, so the second retry starts a step further back than the first.
        """
        self.agent.arm_ready()
        bx, by, _ = self.agent.get_current_pose()
        heading = math.atan2(target[1] - by, target[0] - bx)
        distance = GRASP_RANGE + BACKOFF * steps
        goal_x = target[0] - distance * math.cos(heading)
        goal_y = target[1] - distance * math.sin(heading)
        logger.info(f"Backing off to {distance:.2f} m from the target.")
        if not self.agent.move_to_goal(goal_x, goal_y, heading, timeout_sec=8.0, tolerance_m=0.15):
            logger.warning("Did not quite reach the back-off pose — lining up from here.")

    def _servo(self, target: tuple[float, float, float],
               chosen: SceneObject) -> tuple[CameraFrame, SceneObject, tuple[float, float, float]] | None:
        """Close in on ``target`` until it sits straight ahead at grasping range.

        Returns the hand frame and the object as last seen in it — the pair the grasp is
        aimed with — and the refined target, or None if the object could not be kept in view.
        """
        started = time.time()
        misses = 0
        settled = 0
        last_seen: tuple[CameraFrame, SceneObject, tuple[float, float, float]] | None = None
        while time.time() - started < SERVO_TIMEOUT:
            self.agent.check_halt()
            tick = time.time()
            self.agent.gaze_at(*target)

            try:
                scene = self.vision.scan(self.agent, (HAND,), depth=True, classify=False)
            except CoreIOUnavailable as exc:
                logger.warning(f"Servo scan failed ({exc})")
                scene = None
            match = self._find(scene, target) if scene is not None else None
            if match is None:
                misses += 1
                if misses >= MAX_MISSES:
                    return None
                self._sleep_until(tick + SERVO_PERIOD)
                continue
            misses = 0
            frame, obj = match
            measured = locate(frame, obj)
            if measured is not None:
                # Blended rather than replaced: one noisy depth reading should nudge the
                # estimate, not throw the robot a metre sideways.
                target = tuple(0.4 * t + 0.6 * m for t, m in zip(target, measured))  # type: ignore[assignment]
            last_seen = (frame, obj, target)

            bx, by, byaw = self.agent.get_current_pose()
            dx, dy = target[0] - bx, target[1] - by
            distance = math.hypot(dx, dy)
            heading = math.atan2(dy, dx)
            bearing = math.atan2(math.sin(heading - byaw), math.cos(heading - byaw))
            range_error = distance - GRASP_RANGE
            logger.info(f"Servo: range {distance:.2f} m ({range_error:+.2f}), bearing {math.degrees(bearing):+.1f}°")

            if abs(range_error) < RANGE_TOL and abs(bearing) < YAW_TOL:
                settled += 1
                if settled >= 2:
                    logger.success("Lined up on the target.")
                    return frame, obj, target
            else:
                settled = 0
                step = max(-MAX_STEP, min(MAX_STEP, range_error))
                goal_x = bx + step * math.cos(heading)
                goal_y = by + step * math.sin(heading)
                self.agent.step_towards(goal_x, goal_y, heading, gaze=target, duration_sec=SERVO_PERIOD * 4)
            self._sleep_until(tick + SERVO_PERIOD)

        # Out of time. Close enough to try is better than walking away from a reachable
        # object; the grasp itself will say if it is not.
        if last_seen is not None:
            logger.warning("Servo timed out — grasping from the last good view.")
        return last_seen

    def _find(self, scene: Scene, target: tuple[float, float, float]) -> tuple[CameraFrame, SceneObject] | None:
        """The object in a hand scan that is the one being approached.

        Re-identified by position, not by name: the segmenter's label for the same bottle
        changes from frame to frame, but where the target projects into the hand image only
        changes as fast as the robot moves. An outline the target lands *inside* wins over
        one whose grasp point is merely nearer: in a pile, the neighbour's grasp point can be
        closer to where the target projects than the target's own.
        """
        frame = scene.frames.get(HAND)
        if frame is None or not scene.objects:
            return None
        expected = vision_to_pixel(frame, target)
        height, width = frame.image.shape[:2]
        if expected is None:
            expected = (width / 2, height / 2)
        diagonal = math.hypot(width, height)

        def cost(obj: SceneObject) -> float:
            return math.hypot(obj.cx - expected[0], obj.cy - expected[1]) / diagonal

        def inside(obj: SceneObject) -> bool:
            return len(obj.polygon) >= 3 and cv2.pointPolygonTest(
                np.array(obj.polygon, dtype=np.float32), (float(expected[0]), float(expected[1])), False) >= 0

        best = min(scene.objects, key=lambda obj: (not inside(obj), cost(obj)))
        if cost(best) > 0.35:
            return None
        return frame, best

    def _sleep_until(self, deadline: float) -> None:
        remaining = deadline - time.time()
        if remaining > 0:
            time.sleep(remaining)
        self.agent.check_halt()


__all__ = ["GraspPlanner", "MissionStopped"]
