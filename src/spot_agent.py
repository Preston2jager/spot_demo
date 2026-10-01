import math
import os
import threading
import time
from pathlib import Path
from typing import Self

import bosdyn.client
from bosdyn.api import geometry_pb2, manipulation_api_pb2, robot_state_pb2
from bosdyn.api.graph_nav import graph_nav_pb2, nav_pb2
from bosdyn.client.frame_helpers import (
    BODY_FRAME_NAME,
    GRAV_ALIGNED_BODY_FRAME_NAME,
    GROUND_PLANE_FRAME_NAME,
    HAND_FRAME_NAME,
    ODOM_FRAME_NAME,
    VISION_FRAME_NAME,
    get_a_tform_b,
    math_helpers,
)
from bosdyn.client.graph_nav import GraphNavClient
from bosdyn.client.image import ImageClient
from bosdyn.client.lease import Lease, LeaseClient, LeaseKeepAlive
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import (
    CommandFailedErrorWithFeedback,
    RobotCommandBuilder,
    RobotCommandClient,
    blocking_sit,
    blocking_stand,
)
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.sdk import generate_client_name
from loguru import logger

from .config import load_robot_credentials
from .janitor import Janitor
from .speaker import CoreIOSpeaker, SpeakerDeviceMissing, resolve_audio
from .spot_tracker import SpotTracker

# The resource the robot reports for the body itself. A client holding "all-leases" holds
# a super-lease of this one, which is why ownership is compared rather than name-matched.
_BODY_RESOURCE = "body"

# Body height above the ground plane that separates sitting from standing. Spot stands with
# its body frame about half a metre up and sits with it well under a quarter.
_SIT_HEIGHT_M = 0.33

# The ready pose: the hand this far ahead of / above the body centre, in the gravity-aligned
# body frame, level unless ARM_READY_PITCH_DEG tips it down (positive) to look at the floor.
_READY_HAND = (0.80, 0.0, 0.20)
_READY_PITCH_DEG = float(os.getenv("ARM_READY_PITCH_DEG", "0"))

# Gripper torque by grip material, from the vision service's soft/medium/hard. Applied as a
# torque-limited re-close once the grasp has succeeded — the manipulation API's own grasp
# has no force setting — so a teddy bear is not carried crushed and a glass bottle is not
# carried loose. Tune on the real objects; the defaults are deliberately gentle.
_GRIP_TORQUE = {
    "soft": float(os.getenv("GRIP_TORQUE_SOFT", "1.5")),
    "medium": float(os.getenv("GRIP_TORQUE_MEDIUM", "4.0")),
    "hard": float(os.getenv("GRIP_TORQUE_HARD", "8.0")),
}


class MissionStopped(KeyboardInterrupt):
    """The operator pressed Stop.

    A KeyboardInterrupt on purpose: every mission already unwinds on one, and nothing that
    catches `Exception` — the safety fallback, a tracked method — can mistake it for an
    error and carry on.
    """


class SpotAgent:
    """Context-manager controller: authentication, lease, navigation, arm, and vision for Spot."""

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    @SpotTracker("Spot agent initialisation", exit_on_fail=True)
    def __init__(
        self,
        *,
        client_name: str = "SpotAgent",
        keep_alive_period_sec: float = 2.0,
        force_lease: bool = True,
        navigation: bool = False,
        speaker: bool = True,
        lease: bool = True,
    ) -> None:
        """Store connection parameters; the actual connection happens in __enter__.

        ``lease=False`` connects without taking the body lease: cameras, robot state and the
        map can all be read without it, and it is taken on the first command instead, by
        ``ensure_lease()``. That is how the web console runs — opening the page must not
        pull the robot away from the ROS 2 driver or the tablet.
        """
        creds = load_robot_credentials()
        self.hostname: str = creds["hostname"]
        self.username: str = creds["username"]
        self.password: str = creds["password"]
        self.navigation_state = navigation
        self.speaker_state = speaker
        self.lease_state = lease
        self.client_name = client_name
        # The name the robot reports as our lease owner: the SDK builds it from the prefix,
        # the host and this process, so the same call here gives the same string.
        self.owner_name = generate_client_name(client_name)
        self.keep_alive_period_sec = keep_alive_period_sec
        self.force_lease = force_lease
        self.sdk: bosdyn.client.Sdk | None = None
        self.robot: bosdyn.client.Robot | None = None
        self.lease_client: LeaseClient | None = None
        self.cmd_client: RobotCommandClient | None = None
        self.img_client: ImageClient | None = None
        self.state_client: RobotStateClient | None = None
        self.graph_nav_client: GraphNavClient | None = None
        self.manip_client: ManipulationApiClient | None = None
        self._lease_keepalive: LeaseKeepAlive | None = None
        self._speaker: CoreIOSpeaker | None = None
        self.graph: object | None = None
        # The stop latch. Set by `halt()` from whichever thread the operator's Stop came in
        # on; while it is set every command a mission tries to send raises MissionStopped
        # instead of reaching the robot. Cleared only by `resume()`, which the console calls
        # when the operator dispatches the next thing.
        self._halted = threading.Event()

    @SpotTracker("Spot agent services startup", exit_on_fail=True)
    def __enter__(self) -> Self:
        """Authenticate, optionally take the lease, and bring up navigation and the speaker.

        Connecting does not move the robot — callers stand it up with ``power_on_and_stand()``.
        With ``lease=False`` the lease and the fault clearing wait for the first command.
        """
        self._auto_login(self.username, self.password)
        if self.lease_state:
            self._get_lease(force=self.force_lease)
            self.status_check()
        with SpotTracker("Speaker startup", exit_on_fail=False):
            if self.speaker_state:
                self._start_speaker()
        if self.navigation_state:
            self.ensure_navigation()
        return self

    def __exit__(self, *args: object) -> None:
        """Release lease and shut down on context exit."""
        self._shutdown()

    # ── Stopping ──────────────────────────────────────────────────────────────

    def halt(self) -> None:
        """Freeze the robot where it is, now, and refuse further mission commands.

        The highest-priority thing the console can do, so it is built to depend on nothing
        a mission might be holding: no lease check, no tracker, no lock — the latch first,
        so a mission thread racing this cannot slip one more command in, then a *stop*
        command straight to the command client.

        A stop, not a stand and not a sit. `synchro_stand_command` resets the body to its
        nominal pose, which is a movement; sitting is a bigger one. `stop_command` holds the
        robot as it is — mid-stride it settles its feet, the arm holds its pose, the gripper
        keeps what it has. It also supersedes whatever GraphNav or the manipulation API was
        driving, because both command the robot through this same service.
        """
        self._halted.set()
        if self.cmd_client is None:
            return
        try:
            self.cmd_client.robot_command(RobotCommandBuilder.stop_command())
        except Exception as e:
            logger.warning(f"Stop command could not be sent ({e})")

    def resume(self) -> None:
        """Lift the stop latch — the operator has asked for something new."""
        self._halted.clear()

    @property
    def halted(self) -> bool:
        return self._halted.is_set()

    def check_halt(self) -> None:
        """Raise MissionStopped if the operator has pressed Stop. Called in every poll loop."""
        if self._halted.is_set():
            raise MissionStopped()

    def hold_position(self) -> None:
        """Stop in place without latching — what the safety fallback does after an error."""
        if self.cmd_client is None:
            return
        try:
            self.cmd_client.robot_command(RobotCommandBuilder.stop_command())
        except Exception as e:
            logger.warning(f"Hold position failed ({e})")

    def _command(self, command: object, end_time_secs: float | None = None) -> int:
        """Send a robot command, unless the operator has stopped the robot.

        Every command this class sends goes through here. The check is what makes Stop
        stick: a mission interrupted mid-RPC wakes up, and its very next command — a stow,
        a stand, the next step of a walk — raises instead of undoing the stop.
        """
        self.check_halt()
        return self.cmd_client.robot_command(command, end_time_secs=end_time_secs)  # type: ignore[union-attr]

    # ── Lease ─────────────────────────────────────────────────────────────────

    def status_check(self, verbose: bool = False) -> list[str]:
        """The soft reset: everything a command needs, checked and tidied — nothing moved.

        Re-establishes time sync if it has dropped, takes the lease back if another client
        has it, clears behaviour faults, and reports what would stop the robot obeying: an
        E-Stop, a system fault, low battery. No stand, no sit, no sound. Every command the
        console dispatches runs this first; ``verbose`` is the operator's own Status Check,
        which logs every line instead of only the problems.

        Returns the problems found — empty means ready. Raises RuntimeError only when Spot
        does not answer at all, which no command can get past.
        """
        say = logger.info if verbose else logger.debug
        problems: list[str] = []
        if self.robot is None or self.state_client is None:
            raise RuntimeError("Not connected to Spot.")
        try:
            if not self.robot.time_sync.has_established_time_sync:  # type: ignore[union-attr]
                self.robot.time_sync.wait_for_sync(timeout_sec=5.0)  # type: ignore[union-attr]
            say("Time sync: established")
        except Exception as e:
            problems.append(f"time sync failed ({e})")
        try:
            state = self.state_client.get_robot_state()
        except Exception as e:
            raise RuntimeError(f"Spot is not answering ({e})") from e
        say("Connection: robot state reachable")

        self.ensure_lease()
        lease = self.lease_status()
        if lease.get("held"):
            say("Lease: held by the console")
        else:
            problems.append(f"lease not held (owner: {lease.get('owner', 'unknown')})")

        if not self._clear_behavior_faults():
            problems.append("behaviour faults remain after clearing")

        estopped = [s.name for s in state.estop_states if s.state == robot_state_pb2.EStopState.STATE_ESTOPPED]
        if estopped:
            problems.append(f"E-Stop engaged: {', '.join(estopped)}")
        else:
            say("E-Stop: clear")
        for fault in state.system_fault_state.faults:
            problems.append(f"system fault: {fault.name} ({fault.error_message or 'no detail'})")

        motors = robot_state_pb2.PowerState.MotorPowerState.Name(state.power_state.motor_power_state)
        say(f"Motors: {motors.replace('STATE_', '').lower()}")
        battery = self.battery()
        if battery:
            say(f"Battery: {battery['percentage']}% ({battery['status'].lower()})")
            if battery["percentage"] < 15:
                problems.append(f"battery low ({battery['percentage']}%)")
        for problem in problems:
            logger.warning(f"Status check: {problem}")
        return problems

    def _lease_state(self) -> tuple[bool, str]:
        """Whether the body lease in our wallet is still the robot's current one, and who owns it."""
        active = next(
            (r for r in self.lease_client.list_leases() if r.resource == _BODY_RESOURCE),  # type: ignore[union-attr]
            None,
        )
        if active is None:
            return False, "nobody"
        owner = active.lease_owner.client_name or "an unnamed client"
        # Ownership is decided by name first. Comparing lease sequences alone said "not
        # ours" about our own lease once commands had advanced the robot's copy, and the
        # retake that followed handed the lease back — which sits a standing robot down.
        if owner == self.owner_name:
            return True, owner
        try:
            mine = self.robot.lease_wallet.get_lease(_BODY_RESOURCE)  # type: ignore[union-attr]
        except Exception:
            return False, owner
        held = mine.compare(Lease(active.lease)) in (
            Lease.CompareResult.SAME,
            Lease.CompareResult.SUPER_LEASE,
            Lease.CompareResult.SUB_LEASE,
        )
        return held, owner

    @SpotTracker("Lease check", exit_on_fail=False)
    def ensure_lease(self) -> bool:
        """Take the body lease back when another client has taken it since the last mission.

        The ROS 2 driver takes the lease whenever it navigates, and Spot recognises exactly
        one holder. From then on our ``LeaseKeepAlive`` renews a superseded lease and every
        command comes back as a ``LeaseUseError``. Taking it back is right *here* because
        this runs when the operator dispatches something — the lease never moves behind
        anyone's back.
        """
        if self.lease_client is None:
            return False
        if self._lease_keepalive is None:
            logger.info("Taking Spot's lease for the first command")
            self._get_lease(force=self.force_lease)
            return True
        try:
            held, owner = self._lease_state()
        except Exception as e:
            logger.debug(f"Lease state unreadable ({e}) — taking the lease to be sure")
            held, owner = False, "an unknown client"
        if held:
            return True
        logger.warning(f"Lease was taken by {owner} — taking it back")
        self._get_lease(force=True)
        return True

    def lease_status(self) -> dict:
        """Who holds the body lease, and whether it is us — polled, so untracked and quiet."""
        if self.lease_client is None:
            return {}
        try:
            held, owner = self._lease_state()
        except Exception as e:
            logger.debug(f"Lease state poll failed: {e}")
            return {}
        return {"held": held and self._lease_keepalive is not None, "owner": owner}

    @SpotTracker("Take Spot lease", exit_on_fail=True)
    def _get_lease(self, force: bool = False) -> None:
        """Acquire the robot lease, optionally force-taking it from another holder.

        Taken *before* the old keepalive is stopped, and the old one never returns its
        lease: a lease returned even for a moment leaves a standing robot with no owner,
        and Spot answers that by sitting down.
        """
        if force:
            try:
                self.lease_client.take()  # type: ignore[union-attr]
            except Exception as e:
                logger.warning(f"Lease take failed: {e}")
        if self._lease_keepalive:
            self._lease_keepalive._return_at_exit = False
            try:
                self._lease_keepalive.shutdown()
            except Exception as e:
                logger.debug(f"Stale lease keepalive did not shut down cleanly: {e}")
        self._lease_keepalive = LeaseKeepAlive(
            self.lease_client, must_acquire=True, return_at_exit=True
        )

    # ── Navigation startup ────────────────────────────────────────────────────

    @SpotTracker("Navigation startup", exit_on_fail=False)
    def ensure_navigation(self) -> bool:
        """Upload the map and localise to a fiducial, at most once per agent.

        Lazy: localisation needs a fiducial in view, which is only true once the operator has
        parked Spot for a navigation mission. Returns False (and clears the graph) on failure
        so the next attempt retries instead of navigating unlocalised.
        """
        if self.graph:
            logger.debug("Graph already loaded and localised — reusing it.")
            return True
        self.graph = self.get_current_graph()
        if not self.graph:
            map_dir = os.getenv("SPOT_MAP_DIR", "./runtime/maps/amp_3")
            self.graph = self.upload_graph_and_snapshots(map_dir)
        if not self.graph:
            logger.warning("No graph available. Check connection and graph path.")
            return False
        if not self.initialize_graphnav_to_fiducial():
            self.graph = None
            logger.warning("Failed to initialise GraphNav. Check fiducial visibility.")
            return False
        return True

    # ── Private: admin ────────────────────────────────────────────────────────

    @SpotTracker("Login and Time Sync")
    def _auto_login(self, username: str, password: str) -> None:
        """Create SDK, authenticate with the robot, sync time, and initialise all service clients."""
        self.sdk = bosdyn.client.create_standard_sdk(self.client_name)
        self.robot = self.sdk.create_robot(self.hostname)
        self.robot.authenticate(username, password)
        self.robot.time_sync.start()  # type: ignore[union-attr]
        for i in range(3):
            try:
                self.robot.time_sync.wait_for_sync(timeout_sec=5.0)  # type: ignore[union-attr]
                break
            except Exception:
                logger.warning(f"Time sync failed, attempt ({i + 1}/3)...")
                time.sleep(1)
        else:
            raise RuntimeError("Cannot establish time sync with the robot.")
        self.lease_client = self.robot.ensure_client(LeaseClient.default_service_name)
        self.cmd_client = self.robot.ensure_client(RobotCommandClient.default_service_name)
        self.img_client = self.robot.ensure_client(ImageClient.default_service_name)
        self.state_client = self.robot.ensure_client(RobotStateClient.default_service_name)
        self.graph_nav_client = self.robot.ensure_client(GraphNavClient.default_service_name)
        self.manip_client = self.robot.ensure_client(ManipulationApiClient.default_service_name)

    @SpotTracker("Shutdown", exit_on_fail=False)
    def _shutdown(self) -> None:
        """Release the lease keepalive and close the speaker session."""
        if self._lease_keepalive:
            self._lease_keepalive.shutdown()
        self._close_speaker()

    @SpotTracker("Clear Behavior Faults", exit_on_fail=False)
    def _clear_behavior_faults(self) -> bool:
        """Query and clear any active behavior faults before issuing commands."""
        if self.state_client is None or self.cmd_client is None:
            logger.warning("Clients not initialised — cannot clear faults.")
            return False
        try:
            faults = self.state_client.get_robot_state().behavior_fault_state.faults
            if not faults:
                logger.info("No behavior faults. Spot is ready.")
                return True
            logger.info(f"{len(faults)} fault(s) detected, clearing...")
            for fault in faults:
                logger.info(f"Fault ID: {fault.behavior_fault_id} | Cause: {fault.cause}")
                self.cmd_client.clear_behavior_fault(behavior_fault_id=fault.behavior_fault_id)
                time.sleep(0.5)
            time.sleep(1.0)
            remaining = self.state_client.get_robot_state().behavior_fault_state.faults
            if not remaining:
                logger.info("All faults cleared.")
                return True
            logger.warning(f"Still {len(remaining)} fault(s) remaining — check with tablet.")
            return False
        except Exception as e:
            logger.error(f"Error clearing faults: {e}")
            return False

    # ── Posture ───────────────────────────────────────────────────────────────

    def posture(self) -> str:
        """What the robot is actually doing: ``off``, ``sitting``, ``standing`` or ``unknown``.

        Asked of the robot rather than remembered, because the tablet, a behaviour fault or
        the ROS 2 driver can all change it behind the console's back. Motors off is sitting
        by definition; with motors on, the body's height above the ground plane tells the
        two postures apart.
        """
        try:
            state = self.state_client.get_robot_state()  # type: ignore[union-attr]
        except Exception as e:
            logger.debug(f"Posture poll failed: {e}")
            return "unknown"
        if state.power_state.motor_power_state != robot_state_pb2.PowerState.MOTOR_POWER_STATE_ON:
            return "off"
        try:
            snapshot = state.kinematic_state.transforms_snapshot
            height = get_a_tform_b(snapshot, GROUND_PLANE_FRAME_NAME, BODY_FRAME_NAME).z
        except Exception:
            return "unknown"
        return "standing" if height > _SIT_HEIGHT_M else "sitting"

    @SpotTracker("Power on and Stand", exit_on_fail=True)
    def power_on_and_stand(self) -> None:
        """Power on the robot and command it to stand."""
        self.check_halt()
        if not self.robot.is_powered_on():  # type: ignore[union-attr]
            self.robot.power_on(timeout_sec=20)  # type: ignore[union-attr]
        self.check_halt()
        try:
            blocking_stand(self.cmd_client, timeout_sec=10)
        except CommandFailedErrorWithFeedback as e:
            # STATUS_COMMAND_OVERRIDDEN: something else — the ROS 2 driver, typically —
            # commanded the robot while this stand was in flight. Retaken and retried once;
            # two programs taking turns is a situation for a person, not a loop.
            logger.warning(f"Stand was overridden by another client ({e}) — retaking and retrying")
            self.ensure_lease()
            self.check_halt()
            blocking_stand(self.cmd_client, timeout_sec=10)

    @SpotTracker("Sit down", exit_on_fail=False)
    def sit(self) -> None:
        """Command the robot to sit down."""
        self.check_halt()
        blocking_sit(self.cmd_client, timeout_sec=10.0)
        logger.info("Spot sat down.")

    @SpotTracker("Power off", exit_on_fail=False)
    def power_off(self) -> None:
        """Power off the robot."""
        self.check_halt()
        if self.robot.is_powered_on():  # type: ignore[union-attr]
            self.robot.power_off(cut_immediately=False, timeout_sec=20.0)  # type: ignore[union-attr]
            logger.info("Spot powered off.")

    def battery(self) -> dict:
        """Charge percentage and charging status, or ``{}`` — polled, so untracked and quiet."""
        try:
            state = self.state_client.get_robot_state()  # type: ignore[union-attr]
            if not state.battery_states:
                return {}
            battery = state.battery_states[0]
            return {
                "percentage": round(battery.charge_percentage.value, 1),
                "status": battery.Status.Name(battery.status).replace("STATUS_", ""),
                "runtime_sec": battery.estimated_runtime.seconds,
            }
        except Exception as e:
            logger.debug(f"Battery state poll failed: {e}")
            return {}

    # ── Arm ───────────────────────────────────────────────────────────────────

    @SpotTracker("Arm In", exit_on_fail=False)
    def arm_in(self) -> None:
        """Close the gripper and stow the arm."""
        try:
            self._command(RobotCommandBuilder.claw_gripper_close_command())
            self._command(RobotCommandBuilder.arm_stow_command())
            logger.info("Arm stowing...")
        except MissionStopped:
            raise
        except Exception as e:
            logger.error(f"Arm stow failed: {e}")

    @SpotTracker("Arm ready", exit_on_fail=False)
    def arm_ready(self) -> None:
        """Hand straight ahead, gripper open — the pose every action starts and looks from.

        The teleop pose too. The hand camera then sees what is in front of the robot and
        joins the body cameras in recognition, and the open gripper is ready to grasp. Left
        alone when the hand is already there, so calling it before every survey costs nothing.
        """
        try:
            if self._hand_ready():
                return
            x, y, z = _READY_HAND
            q = math_helpers.Quat.from_pitch(math.radians(_READY_PITCH_DEG))
            command = RobotCommandBuilder.arm_pose_command(x, y, z, q.w, q.x, q.y, q.z,
                                                           GRAV_ALIGNED_BODY_FRAME_NAME, 1.5)
            self._command(RobotCommandBuilder.claw_gripper_open_command(build_on_command=command))
            deadline = time.time() + 3.0
            while time.time() < deadline:
                self.check_halt()
                time.sleep(0.2)
                if self._hand_ready():
                    break
            logger.info("Arm ready: hand ahead, gripper open.")
        except MissionStopped:
            raise
        except Exception as e:
            logger.error(f"Arm ready failed: {e}")

    def _hand_ready(self) -> bool:
        """Whether the hand is already in the ready pose with the gripper open."""
        state = self.state_client.get_robot_state()  # type: ignore[union-attr]
        if state.manipulator_state.gripper_open_percentage < 50.0:
            return False
        hand = get_a_tform_b(state.kinematic_state.transforms_snapshot, GRAV_ALIGNED_BODY_FRAME_NAME, HAND_FRAME_NAME)
        target = math_helpers.Quat.from_pitch(math.radians(_READY_PITCH_DEG))
        q = hand.rotation
        turned = 2.0 * math.acos(min(1.0, abs(q.w * target.w + q.x * target.x + q.y * target.y + q.z * target.z)))
        return math.dist((hand.x, hand.y, hand.z), _READY_HAND) < 0.04 and turned < math.radians(8.0)

    def gaze_at(self, x: float, y: float, z: float, open_gripper: bool = True) -> None:
        """Point the hand camera at a point in the vision frame — untracked, called every servo step."""
        gaze = RobotCommandBuilder.arm_gaze_command(x, y, z, VISION_FRAME_NAME)
        if open_gripper:
            gaze = RobotCommandBuilder.claw_gripper_open_command(build_on_command=gaze)
        self._command(gaze)

    def wait_hand_still(self, timeout_sec: float = 2.5, metres: float = 0.005, degrees: float = 1.0) -> bool:
        """Wait until the hand has stopped moving (after a gaze), so its camera is not blurred."""
        last = None
        deadline = time.time() + timeout_sec
        time.sleep(0.3)  # let the command start the motion before judging stillness
        while time.time() < deadline:
            self.check_halt()
            state = self.state_client.get_robot_state()  # type: ignore[union-attr]
            hand = get_a_tform_b(state.kinematic_state.transforms_snapshot, VISION_FRAME_NAME, HAND_FRAME_NAME)
            if last is not None:
                q, p = hand.rotation, last.rotation
                turned = 2.0 * math.acos(min(1.0, abs(q.w * p.w + q.x * p.x + q.y * p.y + q.z * p.z)))
                if math.dist((hand.x, hand.y, hand.z), (last.x, last.y, last.z)) < metres \
                        and turned < math.radians(degrees):
                    return True
            last = hand
            time.sleep(0.1)
        return False

    def step_towards(self, x: float, y: float, heading: float, gaze: tuple | None = None,
                     duration_sec: float = 2.0) -> None:
        """Start walking to an SE2 pose in the vision frame without waiting to arrive.

        The approach servo re-plans every servo period (about a second), so each step is a short-lived goal:
        if the next correction never comes the walk simply ends where it was going. The gaze
        rides in the same synchronized command so the hand keeps looking at the target while
        the body turns under it.
        """
        command = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=x, goal_y=y, goal_heading=heading, frame_name=VISION_FRAME_NAME
        )
        if gaze is not None:
            command = RobotCommandBuilder.build_synchro_command(
                command, RobotCommandBuilder.arm_gaze_command(*gaze, VISION_FRAME_NAME)
            )
        self._command(command, end_time_secs=time.time() + duration_sec)

    @SpotTracker("Grip", exit_on_fail=False)
    def set_grip(self, material: str) -> None:
        """Re-close the gripper at the torque the object's material calls for."""
        torque = _GRIP_TORQUE.get(material, _GRIP_TORQUE["medium"])
        self._command(RobotCommandBuilder.claw_gripper_close_command(max_torque=torque))
        logger.info(f"Holding a {material} object at {torque:.1f} N·m")

    @SpotTracker("Carry", exit_on_fail=False)
    def carry(self) -> None:
        """Lift what is in the gripper to the carry position."""
        self._command(RobotCommandBuilder.arm_carry_command())
        time.sleep(1.0)

    @SpotTracker("Drop in bin", exit_on_fail=False)
    def drop_in_bin(self) -> None:
        """Reach over the bin in front, open the gripper, stow.

        The pose is ahead and above the body, pitched down, so the object falls into a bin
        standing in front of the robot rather than being set down beside it. Distances are
        environment settings because they depend on the bins actually used.
        """
        reach = float(os.getenv("SORT_BIN_DROP_X", "0.85"))
        height = float(os.getenv("SORT_BIN_DROP_Z", "0.15"))
        q = math_helpers.Quat.from_pitch(math.radians(50.0))
        self._command(RobotCommandBuilder.arm_pose_command(
            reach, 0.0, height, q.w, q.x, q.y, q.z, GRAV_ALIGNED_BODY_FRAME_NAME, 2.0,
        ))
        time.sleep(2.2)
        self._command(RobotCommandBuilder.claw_gripper_open_command())
        time.sleep(1.2)
        self._command(RobotCommandBuilder.arm_stow_command())
        time.sleep(1.2)
        logger.info("Released into the bin and stowed.")

    @SpotTracker("Set down", exit_on_fail=False)
    def set_down(self) -> None:
        """Lower what is in the gripper to just above the floor in front, let go, stow.

        The pose is in the gravity-aligned body frame, whose origin is the body centre — about
        half a metre up on a standing Spot — so ``SET_DOWN_Z`` is negative. The hand points
        almost straight down, so the object is released low rather than dropped from carry
        height. Tune both on the real floor: too low presses the object into the ground.
        """
        reach = float(os.getenv("SET_DOWN_X", "0.80"))
        height = float(os.getenv("SET_DOWN_Z", "-0.25"))
        q = math_helpers.Quat.from_pitch(math.radians(80.0))
        self._command(RobotCommandBuilder.arm_pose_command(
            reach, 0.0, height, q.w, q.x, q.y, q.z, GRAV_ALIGNED_BODY_FRAME_NAME, 2.5,
        ))
        time.sleep(2.7)
        self._command(RobotCommandBuilder.claw_gripper_open_command())
        time.sleep(1.0)
        self._command(RobotCommandBuilder.arm_stow_command())
        time.sleep(1.2)
        logger.info("Set the object down in front and stowed.")

    @SpotTracker("Grasp", exit_on_fail=False)
    def grasp_in_image(
        self,
        image_response: object,
        px: int,
        py: int,
        label: str,
        auto_walk: bool = False,
        timeout_sec: float = 25.0,
    ) -> bool:
        """Ask the manipulation API to grasp what is at a pixel, and wait for the verdict.

        ``auto_walk=False`` is the normal case now: the approach servo has already put the
        body where the grasp should happen, and letting the API walk again is exactly the
        re-positioning that used to change the angle and miss. It still gazes — the hand
        settling onto the target is a few centimetres, not a walk.
        """
        logger.info(f"Grasping '{label}' at ({px}, {py}) in {image_response.source.name}")  # type: ignore[union-attr]
        pick = manipulation_api_pb2.PickObjectInImage(
            pixel_xy=geometry_pb2.Vec2(x=px, y=py),
            transforms_snapshot_for_camera=image_response.shot.transforms_snapshot,  # type: ignore[union-attr]
            frame_name_image_sensor=image_response.shot.frame_name_image_sensor,  # type: ignore[union-attr]
            camera_model=image_response.source.pinhole,  # type: ignore[union-attr]
            walk_gaze_mode=(
                manipulation_api_pb2.PICK_AUTO_WALK_AND_GAZE if auto_walk else manipulation_api_pb2.PICK_AUTO_GAZE
            ),
        )
        pick.grasp_params.grasp_params_frame_name = VISION_FRAME_NAME
        self.check_halt()
        try:
            response = self.manip_client.manipulation_api_command(  # type: ignore[union-attr]
                manipulation_api_request=manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=pick)
            )
            started = time.time()
            last = ""
            while time.time() - started < timeout_sec:
                self.check_halt()
                feedback = self.manip_client.manipulation_api_feedback_command(  # type: ignore[union-attr]
                    manipulation_api_feedback_request=manipulation_api_pb2.ManipulationApiFeedbackRequest(
                        manipulation_cmd_id=response.manipulation_cmd_id
                    )
                )
                name = manipulation_api_pb2.ManipulationFeedbackState.Name(feedback.current_state)
                if name != last:
                    logger.info(f"Grasp state: {name}")
                    last = name
                if feedback.current_state in (
                    manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED,
                    manipulation_api_pb2.MANIP_STATE_DONE,
                ):
                    time.sleep(0.4)
                    opening = self.state_client.get_robot_state().manipulator_state.gripper_open_percentage  # type: ignore[union-attr]
                    if opening < 3.0:
                        logger.warning("Gripper closed on nothing.")
                        self.release_to_ready()
                        return False
                    return True
                if "FAILED" in name or "NO_SOLUTION" in name:
                    logger.warning(f"Grasp failed: {name}")
                    self.release_to_ready()
                    return False
                time.sleep(0.25)
            logger.warning(f"Grasp timed out ({timeout_sec:.0f}s).")
            self.release_to_ready()
            return False
        except MissionStopped:
            raise
        except Exception as e:
            logger.exception(f"Grasp error: {e}")
            self.release_to_ready()
            return False

    def release_to_ready(self) -> None:
        """Open the gripper and bring the hand back to the ready pose after a failed grasp."""
        try:
            self._command(RobotCommandBuilder.claw_gripper_open_command())
            time.sleep(0.5)
            self.arm_ready()
        except MissionStopped:
            raise
        except Exception as e:
            logger.error(f"Arm recovery failed: {e}")

    # ── Speaker ───────────────────────────────────────────────────────────────

    @SpotTracker("Speaker announcement", exit_on_fail=False)
    def play(self, audio: str | Path | None, attempts: int = 2) -> bool:
        """Play an audio file on the CoreIO's USB speaker. Best effort — never aborts a mission."""
        if not audio:
            logger.debug("Speaker: no audio configured, skipping announcement")
            return False
        try:
            resolve_audio(audio)
        except FileNotFoundError as e:
            logger.warning(f"Speaker: {e} — continuing without audio")
            return False
        for attempt in range(1, attempts + 1):
            try:
                self._open_speaker().play(audio)
                return True
            except SpeakerDeviceMissing as e:
                logger.warning(f"Speaker: {e}")
                return False
            except Exception as e:
                self._close_speaker()
                if attempt < attempts:
                    logger.warning(f"Speaker: attempt {attempt}/{attempts} failed ({e}) — retrying")
                    time.sleep(2.0)
                else:
                    logger.warning(f"Speaker: announcement failed ({e}) — continuing without audio")
        return False

    def _start_speaker(self) -> None:
        """Connect to the CoreIO and confirm the USB sound card is there, without playing."""
        try:
            self._open_speaker().check_device()
        except SpeakerDeviceMissing as e:
            logger.warning(f"Speaker: {e}")
        except Exception as e:
            logger.warning(f"Speaker: connection failed ({e}) — will retry when audio is played")

    def _open_speaker(self) -> CoreIOSpeaker:
        if self._speaker is None:
            speaker = CoreIOSpeaker()
            speaker.connect()
            self._speaker = speaker
        return self._speaker

    def _close_speaker(self) -> None:
        if self._speaker is not None:
            self._speaker.close()
            self._speaker = None

    # ── Navigation ────────────────────────────────────────────────────────────

    @SpotTracker("Navigation initialisation", exit_on_fail=False)
    def initialize_graphnav_to_fiducial(self, fiducial_id: int | None = None) -> bool:
        """Localise the robot to the nearest (or specified) fiducial on the loaded graph."""
        logger.info("Determining initial GraphNav position...")
        try:
            initial_guess = nav_pb2.Localization()
            if fiducial_id is not None:
                self.graph_nav_client.set_localization(  # type: ignore[union-attr]
                    initial_guess_localization=initial_guess,
                    fiducial_init=graph_nav_pb2.SetLocalizationRequest.FIDUCIAL_INIT_SPECIFIC,
                    use_fiducial_id=fiducial_id,
                )
            else:
                self.graph_nav_client.set_localization(  # type: ignore[union-attr]
                    initial_guess_localization=initial_guess,
                    fiducial_init=graph_nav_pb2.SetLocalizationRequest.FIDUCIAL_INIT_NEAREST,
                )
            state = self.graph_nav_client.get_localization_state()  # type: ignore[union-attr]
            if not state.localization.waypoint_id:
                logger.warning("No fiducial detected. Ensure Spot can see a fiducial marker.")
                return False
            logger.info(f"Localised near waypoint {state.localization.waypoint_id[:6]}")
            return True
        except Exception as e:
            logger.error(f"Localisation error: {e}")
            return False

    @SpotTracker("Navigate to waypoint", exit_on_fail=False)
    def navigate_to_waypoint(self, destination_wp_id: str, timeout_sec: float = 60.0) -> bool:
        """Command navigation to a waypoint ID and poll until reached, lost, or timed out."""
        logger.info(f"Navigating to: {destination_wp_id[:6]}...")
        self.check_halt()
        try:
            nav_cmd_id = self.graph_nav_client.navigate_to(  # type: ignore[union-attr]
                destination_waypoint_id=destination_wp_id,
                cmd_duration=timeout_sec,
            )
            start_time = time.time()
            while time.time() - start_time < timeout_sec:
                self.check_halt()
                status = self.graph_nav_client.navigation_feedback(nav_cmd_id).status  # type: ignore[union-attr]
                if status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_REACHED_GOAL:
                    logger.info("Reached destination.")
                    return True
                if status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_LOST:
                    logger.warning("Spot lost localisation.")
                    return False
                if status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_STUCK:
                    logger.warning("Spot is stuck.")
                time.sleep(0.5)
            logger.warning(f"Navigation timeout ({timeout_sec}s).")
            return False
        except MissionStopped:
            raise
        except Exception as e:
            logger.error(f"Navigation error: {e}")
            return False

    @SpotTracker("Loading graph", exit_on_fail=False)
    def get_current_graph(self) -> object | None:
        """Download the current graph from robot memory; return None if the graph is empty."""
        logger.info("Reading graph from robot memory...")
        try:
            graph = self.graph_nav_client.download_graph()  # type: ignore[union-attr]
            if not graph.waypoints:
                logger.warning("Graph has no waypoints.")
                return None
            logger.info(f"Graph loaded: {len(graph.waypoints)} waypoints, {len(graph.edges)} edges.")
            return graph
        except Exception as e:
            logger.error(f"Error loading graph: {e}")
            return None

    def get_waypoint_id_by_name(self, graph: object, target_name: str) -> str | None:
        """Look up a full waypoint ID from its human-readable annotation name."""
        available = []
        for wp in graph.waypoints:  # type: ignore[union-attr]
            name = wp.annotations.name
            available.append(name)
            if name.lower() == target_name.lower():
                return wp.id
        logger.warning(f"Waypoint '{target_name}' not found. Available: {', '.join(available)}")
        return None

    @SpotTracker("Uploading graph and snapshots", exit_on_fail=False)
    def upload_graph_and_snapshots(self, save_dir: str) -> object | None:
        """Load graph and all snapshot files from disk via Janitor and upload them to the robot."""
        logger.info(f"Uploading '{save_dir}' to Spot...")
        self.graph_nav_client.clear_graph()  # type: ignore[union-attr]
        try:
            graph = Janitor.load_graph(save_dir)
        except FileNotFoundError as e:
            logger.error(str(e))
            return None
        self.graph_nav_client.upload_graph(graph=graph, generate_new_anchoring=True)  # type: ignore[union-attr]
        logger.info("Uploading waypoint snapshots...")
        for snap in Janitor.load_waypoint_snapshots(save_dir, graph):
            self.graph_nav_client.upload_waypoint_snapshot(snap)  # type: ignore[union-attr]
        logger.info("Uploading edge snapshots...")
        for snap in Janitor.load_edge_snapshots(save_dir, graph):
            self.graph_nav_client.upload_edge_snapshot(snap)  # type: ignore[union-attr]
        logger.info("Graph upload complete.")
        return graph

    def get_current_pose(self, frame: str = "vision") -> tuple[float, float, float]:
        """Current x, y and yaw (radians) of the body in the vision or odom frame."""
        target_frame = VISION_FRAME_NAME if frame == "vision" else ODOM_FRAME_NAME
        transforms = self.state_client.get_robot_state().kinematic_state.transforms_snapshot  # type: ignore[union-attr]
        tform = get_a_tform_b(transforms, target_frame, BODY_FRAME_NAME)
        q = tform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return tform.position.x, tform.position.y, yaw

    def drive(self, v_x: float = 0.0, v_y: float = 0.0, v_rot: float = 0.0, duration_sec: float = 0.6) -> None:
        """One self-expiring velocity command — teleop's dead-man switch. Untracked: called at 8 Hz."""
        cmd = RobotCommandBuilder.synchro_velocity_command(v_x=v_x, v_y=v_y, v_rot=v_rot)
        self._command(cmd, end_time_secs=time.time() + duration_sec)

    @SpotTracker("Move to coordinate goal", exit_on_fail=False)
    def move_to_goal(self, x: float, y: float, heading: float, timeout_sec: float = 15.0,
                     tolerance_m: float = 0.15) -> bool:
        """Walk to an SE2 pose in the vision frame (heading in radians) and wait to arrive."""
        self._command(
            RobotCommandBuilder.synchro_se2_trajectory_point_command(
                goal_x=x, goal_y=y, goal_heading=heading, frame_name=VISION_FRAME_NAME
            ),
            end_time_secs=time.time() + timeout_sec,
        )
        logger.info(f"Walking to ({x:.2f}, {y:.2f}, heading {math.degrees(heading):.0f}°)")
        started = time.time()
        while time.time() - started < timeout_sec:
            self.check_halt()
            cx, cy, cyaw = self.get_current_pose()
            yaw_err = abs(math.atan2(math.sin(heading - cyaw), math.cos(heading - cyaw)))
            if math.hypot(x - cx, y - cy) < tolerance_m and yaw_err < math.radians(10.0):
                return True
            time.sleep(0.2)
        logger.warning("Move to goal timed out.")
        return False
