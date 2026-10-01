"""Mission handlers. Each takes a `SpotAgent` and does one job; the console decides when.

Two rules hold for all of them.

**Stopping freezes, it never sits.** The operator's Stop latches the agent (`halt()`): the
robot holds where it is and the mission's next command raises `MissionStopped`, which
unwinds it. The safety fallback below is the same idea for errors — hold position, report,
return — rather than stowing, sitting or powering off, any of which is a movement nobody
asked for at the moment something has already gone wrong.

**Missions do not pick their own inputs.** Waypoints come from the map and the sorting
missions take a `choose` callback, so the same loop runs with an operator confirming each
object in the browser or with the most confident object taken automatically.
"""
import functools
import math
import os
import time
from dataclasses import dataclass

from loguru import logger

from .spot_agent import MissionStopped, SpotAgent
from .spot_grasp import GraspPlanner
from .tracker import Track, distance_to
from .vision import GROUPS, CoreIOUnavailable, Scene, SceneObject, Vision, bearing_and_distance, survey_sources


def safety_fallback(func):
    """Hold position and report on any error; let a Stop through untouched.

    Returns False on an error so the console reports "did not complete" rather than success.
    """
    @functools.wraps(func)
    def wrapper(agent: SpotAgent, *args, **kwargs):
        try:
            return func(agent, *args, **kwargs)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.exception(f"Safety fallback: {type(e).__name__}: {e} — holding position")
            agent.hold_position()
            return False
    return wrapper


# ── basics ────────────────────────────────────────────────────────────────────


@safety_fallback
def run_initialisation(agent: SpotAgent) -> None:
    """The full reset, after the stand and announcement: report the battery, sit, power off."""
    battery = agent.battery()
    if battery:
        runtime = battery.get("runtime_sec", 0)
        logger.info(f"Battery {battery['percentage']}% ({battery['status']}), about {runtime // 60} min left")
    else:
        logger.warning("No battery state reported.")
    time.sleep(2.0)
    agent.arm_in()
    agent.sit()
    agent.power_off()


@safety_fallback
def run_sit_down(agent: SpotAgent) -> None:
    agent.arm_in()
    time.sleep(1.5)
    agent.sit()
    agent.power_off()


@safety_fallback
def run_go_to(agent: SpotAgent, waypoint_name: str) -> bool:
    """Walk to one named waypoint. Returns whether Spot arrived."""
    wp_id = agent.get_waypoint_id_by_name(agent.graph, waypoint_name)
    if not wp_id:
        logger.error(f"No waypoint named '{waypoint_name}' in the robot's map")
        return False
    return bool(agent.navigate_to_waypoint(wp_id))


@safety_fallback
def run_patrol_loop(agent: SpotAgent, route: list[str], pause_sec: float = 1.0) -> bool:
    """Walk a route of waypoints in order, lap after lap, until the operator stops it.

    Names are resolved up front, so a typo stops the mission before Spot moves. A leg that
    cannot be walked is logged and skipped: the next leg routes from wherever Spot got to.
    """
    if len(route) < 2:
        logger.error("A patrol needs at least two waypoints — aborting mission.")
        return False
    legs = []
    for name in route:
        wp_id = agent.get_waypoint_id_by_name(agent.graph, name)
        if not wp_id:
            logger.error(f"Waypoint not found: {name} — aborting patrol.")
            return False
        legs.append((name, wp_id))
    logger.info(f"Patrol: {' → '.join(name for name, _ in legs)}. Stop to end.")
    lap = 0
    while True:
        lap += 1
        for name, wp_id in legs:
            logger.info(f"[Lap {lap}] heading for {name}")
            if not agent.navigate_to_waypoint(wp_id):
                logger.warning(f"[Lap {lap}] could not reach {name} — carrying on to the next point")
            time.sleep(pause_sec)


# ── the sorting demo ──────────────────────────────────────────────────────────


@dataclass
class SortPoints:
    """The four waypoints the sorting demo runs between."""

    pick: str
    bins: dict[str, str]  # group -> waypoint name

    def missing(self) -> list[str]:
        needed = ["pick"] + [f"bin_{group}" for group in GROUPS]
        have = {"pick": self.pick, **{f"bin_{g}": self.bins.get(g, "") for g in GROUPS}}
        return [name for name in needed if not have[name]]


# How many scans a survey takes before choosing. An object has to turn up in TRACK_MIN_SCANS
# of them (or of the preview's scans just before) to be registered and offered.
SURVEY_SCANS = int(os.getenv("SURVEY_SCANS", "3"))


# With surround vision off, what counts as "in front": within this many degrees either side
# of straight ahead — roughly what the front pair and the hand see between them.
FRONT_HALF_ANGLE = math.radians(float(os.getenv("VISION_FRONT_HALF_DEG", "60")))


@dataclass
class Survey:
    """A few scans from where Spot stands, and the registered objects they confirmed."""

    scene: Scene  # the last scan of the body cameras — what the operator is shown first
    tracks: list[Track]  # registered objects seen during the survey, best first (at most 10)
    started: float
    hand: Scene | None = None  # the hand camera's part of that scan


@dataclass
class Choice:
    """What to pick up and which bin it goes in. ``group`` may overrule the classifier.

    ``track`` is the registered object; ``obj`` and ``scene`` are its most recent sighting,
    which the grasp starts from, aimed at the track's fused position.
    """

    track: Track
    group: str

    @property
    def obj(self) -> SceneObject:
        return self.track.latest[1]  # type: ignore[index]

    @property
    def scene(self) -> Scene:
        return self.track.latest[0]  # type: ignore[index]


class SortHooks:
    """How a grasping mission talks to whoever is running it. The defaults do nothing."""

    def phase(self, text: str) -> None:
        """A short description of what the robot is doing now."""

    def choose(self, survey: Survey) -> Choice | None:
        """Pick a registered object from a survey, or None to survey again."""
        raise NotImplementedError

    def surround(self) -> bool:
        """Survey with every body camera (True) or only the front pair (False). Both add the hand."""
        return True


class AutoChoice(SortHooks):
    """The automatic mode: the best registered object, in the bin it was classified into."""

    def choose(self, survey: Survey) -> Choice | None:
        if not survey.tracks:
            return None
        best = survey.tracks[0]
        if best.uncertain:
            logger.warning(f"Best registered object is uncertain — taking it anyway: {best.summary()}")
        return Choice(best, best.group)


def survey(agent: SpotAgent, vision: Vision, keep=None, scans: int = SURVEY_SCANS,
           surround: bool = True) -> Survey | None:
    """Scan ``scans`` times and return what got registered.

    The arm is put out in front with the gripper open first, so the hand camera looks along
    with the body cameras — every body camera with ``surround``, else the front pair, whose
    registered objects are then only those ahead of the robot (`FRONT_HALF_ANGLE`), not ones
    an earlier all-round look left behind. ``keep`` filters further (grasp-here keeps only
    what is within reach). None if no scan came back at all.
    """
    agent.arm_ready()
    started = time.time()
    views: dict[str, Scene] = {}
    for _ in range(max(1, scans)):
        agent.check_halt()
        views.update(vision.scan_views(agent, survey_sources(surround), depth=True))
    scene = views.get("cameras") or views.get("hand")
    if scene is None:
        return None
    tracks = vision.tracker.candidates(since=started)  # type: ignore[union-attr]
    if not surround:
        tracks = [t for t in tracks if abs(bearing_and_distance(agent, t.position)[0]) <= FRONT_HALF_ANGLE]
    if keep is not None:
        tracks = [t for t in tracks if keep(t)]
    return Survey(scene, tracks, started, hand=views.get("hand"))


def _picked(vision: Vision, choice: Choice) -> None:
    """The object has left where it was registered: stop offering it."""
    vision.tracker.forget(choice.track.id)  # type: ignore[union-attr]


@safety_fallback
def run_sort_loop(agent: SpotAgent, vision: Vision, points: SortPoints, hooks: SortHooks,
                  rescan_sec: float = 3.0) -> bool:
    """Pick objects at the pick point and sort them into three bins, until stopped.

    Each cycle: at the pick point, survey with the body cameras and the hand (several
    scans, objects registered in space); ask `hooks.choose` which registered object to take (the
    operator's click, or the best one); approach, line up and grasp it (`GraspPlanner`,
    aimed at the fused position); walk to its group's bin and drop it in; walk back.
    """
    missing = points.missing()
    if missing:
        logger.error(f"Sorting needs every point chosen on the map — missing {', '.join(missing)}")
        return False
    pick_id = agent.get_waypoint_id_by_name(agent.graph, points.pick)
    bin_ids = {group: agent.get_waypoint_id_by_name(agent.graph, name) for group, name in points.bins.items()}
    if not pick_id or not all(bin_ids.values()):
        logger.error("A sorting waypoint is not in the robot's map — aborting before moving.")
        return False

    planner = GraspPlanner(agent, vision, phase=hooks.phase)
    at_pick = False
    cycle = 0
    while True:
        agent.check_halt()
        if not at_pick:
            hooks.phase("walking to the pick point")
            if not agent.navigate_to_waypoint(pick_id):
                logger.warning("Could not reach the pick point — retrying in 3 s")
                time.sleep(3.0)
                continue
            at_pick = True

        hooks.phase("surveying")
        try:
            found = survey(agent, vision, surround=hooks.surround())
        except CoreIOUnavailable as exc:
            logger.error(f"Vision service unavailable ({exc}) — retrying in {rescan_sec:.0f} s")
            time.sleep(rescan_sec)
            continue
        if found is None or not found.tracks:
            hooks.phase("nothing registered to sort — watching")
            logger.info(f"No object seen consistently. Surveying again in {rescan_sec:.0f} s.")
            time.sleep(rescan_sec)
            continue
        logger.info(f"Survey: {len(found.tracks)} registered object(s) — "
                    + ", ".join(f"#{t.id} {t.label}" for t in found.tracks))

        choice = hooks.choose(found)
        if choice is None:
            continue
        cycle += 1
        obj, group = choice.obj, choice.group
        logger.info(f"[Cycle {cycle}] taking {choice.track.summary()} → {GROUPS[group]['label']} bin")

        at_pick = False  # whatever happens next, the robot will have walked off the point
        if not planner.pick(choice.scene, obj, target=choice.track.position):
            logger.warning(f"[Cycle {cycle}] grasp failed — back to the pick point to look again")
            continue

        _picked(vision, choice)
        hooks.phase(f"carrying to the {GROUPS[group]['label']} bin")
        if not agent.navigate_to_waypoint(bin_ids[group]):
            logger.error(f"[Cycle {cycle}] could not reach the {group} bin — holding the object, stopping")
            agent.hold_position()
            return False
        hooks.phase(f"dropping into the {GROUPS[group]['label']} bin")
        agent.drop_in_bin()
        logger.success(f"[Cycle {cycle}] {obj.label} sorted into {GROUPS[group]['label']}")


# ── grasp here, no map ────────────────────────────────────────────────────────

# How far from the body an object may be and still count as "in front of Spot". The grasp
# planner walks the rest itself, on odometry, so this bounds the walk, not the arm.
GRAB_REACH = float(os.getenv("GRAB_REACH_M", "2.0"))


@safety_fallback
def run_grasp_here(agent: SpotAgent, vision: Vision, hooks: SortHooks, attempts: int = 3,
                   rescan_sec: float = 2.0) -> bool:
    """Pick up one object near where Spot stands, set it back down in front, stow. No map.

    Survey with the body cameras and the hand, keep the registered objects within `GRAB_REACH`, and ask
    `hooks.choose` for one (the operator's click, or the best). The grasp is the sorting
    demo's — approach, line up, grasp, grip by material — which runs on odometry and never
    needs GraphNav. Then `set_down()` and done. Gives up after ``attempts`` surveys with
    nothing in reach; an operator's rescan does not count against that.
    """
    planner = GraspPlanner(agent, vision, phase=hooks.phase)
    empty = 0
    while True:
        agent.check_halt()
        hooks.phase("surveying around Spot")
        try:
            found = survey(agent, vision, keep=lambda t: distance_to(agent, t) <= GRAB_REACH,
                           surround=hooks.surround())
        except CoreIOUnavailable as exc:
            logger.error(f"Vision service unavailable ({exc})")
            return False
        if found is None or not found.tracks:
            empty += 1
            if empty >= attempts:
                logger.warning(f"Nothing registered within {GRAB_REACH:.1f} m after {attempts} surveys — giving up.")
                return False
            logger.info(f"Nothing registered within {GRAB_REACH:.1f} m. Surveying again in {rescan_sec:.0f} s.")
            time.sleep(rescan_sec)
            continue
        logger.info(f"Survey: {len(found.tracks)} registered object(s) within {GRAB_REACH:.1f} m — "
                    + ", ".join(f"#{t.id} {t.label}" for t in found.tracks))

        choice = hooks.choose(found)
        if choice is None:
            continue
        logger.info(f"Taking {choice.track.summary()}")
        if not planner.pick(choice.scene, choice.obj, target=choice.track.position):
            logger.warning(f"Could not grasp {choice.track.label}.")
            return False
        _picked(vision, choice)

        hooks.phase(f"setting {choice.track.label} down")
        agent.set_down()
        logger.success(f"Picked up {choice.track.label} and set it down.")
        return True


__all__ = [
    "AutoChoice",
    "Choice",
    "MissionStopped",
    "SortHooks",
    "SortPoints",
    "Survey",
    "run_go_to",
    "run_grasp_here",
    "run_initialisation",
    "run_patrol_loop",
    "run_sit_down",
    "run_sort_loop",
    "survey",
]
