import math
import os
import time
from typing import Self

import bosdyn.client
import cv2
import numpy as np
from bosdyn.api import geometry_pb2, manipulation_api_pb2
from bosdyn.api.graph_nav import graph_nav_pb2, nav_pb2
from bosdyn.client.frame_helpers import (
    BODY_FRAME_NAME,
    GRAV_ALIGNED_BODY_FRAME_NAME,
    ODOM_FRAME_NAME,
    VISION_FRAME_NAME,
    get_a_tform_b,
    math_helpers,
)
from bosdyn.client.graph_nav import GraphNavClient
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.recording import GraphNavRecordingServiceClient
from bosdyn.client.robot_command import (
    RobotCommandBuilder,
    RobotCommandClient,
    blocking_sit,
    blocking_stand,
)
from bosdyn.client.robot_state import RobotStateClient
from loguru import logger

from .config import load_robot_credentials
from .janitor import Janitor
from .spot_streamer import SpotStreamer
from .spot_tracker import SpotTracker


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
        stream: bool = False,
        navigation: bool = False,
    ) -> None:
        """Store connection parameters; the actual connection happens in __enter__."""
        creds = load_robot_credentials()
        self.hostname: str = creds["hostname"]
        self.username: str = creds["username"]
        self.password: str = creds["password"]
        self.stream_state = stream
        self.navigation_state = navigation
        self.client_name = client_name
        self.keep_alive_period_sec = keep_alive_period_sec
        self.force_lease = force_lease
        self.sdk: bosdyn.client.Sdk | None = None
        self.robot: bosdyn.client.Robot | None = None
        self.lease_client: LeaseClient | None = None
        self.cmd_client: RobotCommandClient | None = None
        self.img_client: ImageClient | None = None
        self.state_client: RobotStateClient | None = None
        self.graph_nav_client: GraphNavClient | None = None
        self.recording_client: GraphNavRecordingServiceClient | None = None
        self.manip_client: ManipulationApiClient | None = None
        self._lease_keepalive: LeaseKeepAlive | None = None

    @SpotTracker("Spot agent services startup", exit_on_fail=True)
    def __enter__(self) -> Self:
        """Authenticate, acquire lease, power on, stand, and optionally start streaming/navigation."""
        self._auto_login(self.username, self.password)
        self._get_lease(force=self.force_lease)
        _ = self.robot.ensure_client(RobotCommandClient.default_service_name)
        self._clear_behavior_faults()
        self._power_on_and_stand()
        with SpotTracker("Streaming startup", exit_on_fail=False):
            if self.stream_state:
                streamer = SpotStreamer(self.robot)
                streamer.start()
        with SpotTracker("Navigation startup", exit_on_fail=False):
            if self.navigation_state:
                self.graph = self.get_current_graph()
                if not self.graph:
                    map_dir = os.getenv("SPOT_MAP_DIR", "./runtime/maps/amp_3")
                    self.graph = self.upload_graph_and_snapshots(map_dir)
                if not self.graph:
                    logger.warning("No graph available. Check connection and graph path.")
                    return self
                if not self.initialize_graphnav_to_fiducial():
                    logger.warning("Failed to initialise GraphNav. Check fiducial visibility.")
                    return self
        return self

    def __exit__(self, *args: object) -> None:
        """Release lease and shut down on context exit."""
        self._shutdown()

    # ── Private: admin ────────────────────────────────────────────────────────

    @SpotTracker("Login and Time Sync")
    def _auto_login(self, username: str, password: str) -> None:
        """Create SDK, authenticate with the robot, sync time, and initialise all service clients."""
        self.sdk = bosdyn.client.create_standard_sdk(self.client_name)
        self.robot = self.sdk.create_robot(self.hostname)
        self.robot.authenticate(username, password)
        self.robot.time_sync.start()  # type: ignore[union-attr]
        sync_success = False
        for i in range(3):
            try:
                self.robot.time_sync.wait_for_sync(timeout_sec=5.0)  # type: ignore[union-attr]
                sync_success = True
                break
            except Exception:
                logger.warning(f"Time sync failed, attempt ({i + 1}/3)...")
                time.sleep(1)
        if not sync_success:
            raise RuntimeError("Cannot establish time sync with the robot.")
        self.lease_client = self.robot.ensure_client(LeaseClient.default_service_name)
        self.cmd_client = self.robot.ensure_client(RobotCommandClient.default_service_name)
        self.img_client = self.robot.ensure_client(ImageClient.default_service_name)
        self.state_client = self.robot.ensure_client(RobotStateClient.default_service_name)
        self.graph_nav_client = self.robot.ensure_client(GraphNavClient.default_service_name)
        self.recording_client = self.robot.ensure_client(
            GraphNavRecordingServiceClient.default_service_name
        )
        self.manip_client = self.robot.ensure_client(ManipulationApiClient.default_service_name)

    @SpotTracker("Take Spot lease", exit_on_fail=True)
    def _get_lease(self, force: bool = False) -> None:
        """Acquire the robot lease, optionally force-taking it from another holder."""
        if self._lease_keepalive:
            self._lease_keepalive.shutdown()
        if force:
            try:
                self.lease_client.take()  # type: ignore[union-attr]
            except Exception as e:
                logger.warning(f"Lease take failed: {e}")
        self._lease_keepalive = LeaseKeepAlive(
            self.lease_client, must_acquire=True, return_at_exit=True
        )

    @SpotTracker("Power on and Stand", exit_on_fail=True)
    def _power_on_and_stand(self, arm: bool = False) -> None:
        """Power on the robot and command it to stand; optionally extend the arm."""
        if not self.robot.is_powered_on():  # type: ignore[union-attr]
            self.robot.power_on(timeout_sec=20)  # type: ignore[union-attr]
        blocking_stand(self.cmd_client, timeout_sec=10)
        if arm:
            self._arm_out()

    @SpotTracker("Shutdown", exit_on_fail=False)
    def _shutdown(self) -> None:
        """Release the lease keepalive."""
        if self._lease_keepalive:
            self._lease_keepalive.shutdown()

    @SpotTracker("Sit down", exit_on_fail=False)
    def sit(self) -> None:
        """Command the robot to sit down."""
        if self.cmd_client is None:
            raise RuntimeError("Clients not initialised.")
        blocking_sit(self.cmd_client, timeout_sec=10.0)
        logger.info("Spot sat down.")

    @SpotTracker("Power off", exit_on_fail=False)
    def power_off(self) -> None:
        """Power off the robot."""
        if self.robot is None:
            raise RuntimeError("Robot not initialised.")
        if self.robot.is_powered_on():
            self.robot.power_off(cut_immediately=False, timeout_sec=20.0)
            logger.info("Spot powered off.")

    @SpotTracker("Clear Behavior Faults", exit_on_fail=False)
    def _clear_behavior_faults(self) -> bool:
        """Query and clear any active behavior faults before issuing commands."""
        if self.state_client is None or self.cmd_client is None:
            logger.warning("Clients not initialised — cannot clear faults.")
            return False
        try:
            state = self.state_client.get_robot_state()
            faults = state.behavior_fault_state.faults  # type: ignore[union-attr]
            if not faults:
                logger.info("No behavior faults. Spot is ready.")
                return True
            logger.info(f"{len(faults)} fault(s) detected, clearing...")
            for fault in faults:
                logger.info(f"Fault ID: {fault.behavior_fault_id} | Cause: {fault.cause}")
                self.cmd_client.clear_behavior_fault(
                    behavior_fault_id=fault.behavior_fault_id
                )
                time.sleep(0.5)
            time.sleep(1.0)
            remaining = self.state_client.get_robot_state().behavior_fault_state.faults  # type: ignore[union-attr]
            if not remaining:
                logger.info("All faults cleared.")
                return True
            logger.warning(f"Still {len(remaining)} fault(s) remaining — check with tablet.")
            return False
        except Exception as e:
            logger.error(f"Error clearing faults: {e}")
            return False

    # ── Private: arm ──────────────────────────────────────────────────────────

    @SpotTracker("Arm Out (Absolute)", exit_on_fail=False)
    def _arm_out(self) -> None:
        """Extend the arm forward and open the gripper ready for scanning."""
        if self.cmd_client is None:
            raise RuntimeError("Clients not initialised.")
        try:
            pitch_rad = 25.0 * math.pi / 180.0
            q = math_helpers.Quat.from_pitch(pitch_rad)
            arm_cmd = RobotCommandBuilder.arm_pose_command(
                0.80, 0.0, -0.15,
                q.w, q.x, q.y, q.z,
                GRAV_ALIGNED_BODY_FRAME_NAME,
                2.0,
            )
            self.cmd_client.robot_command(arm_cmd)
            time.sleep(2.0)
            self.cmd_client.robot_command(RobotCommandBuilder.claw_gripper_open_command())
            time.sleep(0.5)
            logger.info("Arm extended and gripper open.")
        except Exception as e:
            logger.error(f"Arm out failed: {e}")

    @SpotTracker("Arm Release and Stow", exit_on_fail=False)
    def _arm_release(self, bin: bool = False) -> None:
        """Lower the arm, open the gripper to drop the object, then stow."""
        if self.cmd_client is None:
            raise RuntimeError("Clients not initialised.")
        try:
            target_x = 0.80 if bin else 1.0
            target_z = 0 if bin else -0.20
            pitch_rad = 60.0 * math.pi / 180.0
            q = math_helpers.Quat.from_pitch(pitch_rad)
            lower_cmd = RobotCommandBuilder.arm_pose_command(
                target_x, 0.4, target_z,
                q.w, q.x, q.y, q.z,
                GRAV_ALIGNED_BODY_FRAME_NAME,
                1.0,
            )
            self.cmd_client.robot_command(lower_cmd)
            time.sleep(1.0)
            self.cmd_client.robot_command(RobotCommandBuilder.claw_gripper_open_command())
            time.sleep(1.0)
            self.cmd_client.robot_command(RobotCommandBuilder.arm_stow_command())
            time.sleep(0.5)
            logger.info("Object released and arm stowed.")
        except Exception as e:
            logger.error(f"Arm release failed: {e}")

    @SpotTracker("Arm In", exit_on_fail=False)
    def _arm_in(self) -> None:
        """Close the gripper and stow the arm."""
        if self.cmd_client is None:
            raise RuntimeError("Clients not initialised.")
        try:
            self.cmd_client.robot_command(RobotCommandBuilder.claw_gripper_close_command())
            self.cmd_client.robot_command(RobotCommandBuilder.arm_stow_command())
            logger.info("Arm stowing...")
        except Exception as e:
            logger.error(f"Arm stow failed: {e}")

    # ── Public: vision & manipulation ─────────────────────────────────────────

    @SpotTracker("Scanning", exit_on_fail=False)
    def find_target_yolo(self, yolo_detector: object) -> dict | None:
        """Capture the hand camera and return the highest-confidence YOLO target detection."""
        try:
            image_responses = self.img_client.get_image(  # type: ignore[union-attr]
                [build_image_request("hand_color_image")]
            )
        except Exception as e:
            logger.error(f"Failed to acquire images: {e}")
            return None
        images_to_detect: dict = {}
        for img_resp in image_responses:
            img_data = np.frombuffer(img_resp.shot.image.data, dtype=np.uint8)
            cv_img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            if cv_img is not None:
                images_to_detect[img_resp.source.name] = cv_img
        if not images_to_detect:
            return None
        detections = yolo_detector.detect_targets_in_batch(images_to_detect, conf=0.1)  # type: ignore[union-attr]
        if detections:
            top_hit = detections[0]
            logger.info(f"Best hit from [{top_hit['camera']}]: {top_hit['class']} ({top_hit['conf']:.2f})")
            return top_hit
        return None

    @SpotTracker("Quick scanning", exit_on_fail=False)
    def quick_detect(self, yolo_detector: object) -> dict | None:
        """Scan all six cameras, show OpenCV overlay of the top hit, and wait for keypress."""
        camera_sources = [
            "frontleft_fisheye_image",
            "frontright_fisheye_image",
            "left_fisheye_image",
            "right_fisheye_image",
            "back_fisheye_image",
            "hand_color_image",
        ]
        try:
            image_responses = self.img_client.get_image(  # type: ignore[union-attr]
                [build_image_request(src) for src in camera_sources]
            )
        except Exception as e:
            logger.error(f"Failed to acquire images: {e}")
            return None
        images_to_detect: dict = {}
        for img_resp in image_responses:
            img_data = np.frombuffer(img_resp.shot.image.data, dtype=np.uint8)
            cv_img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            if cv_img is not None:
                images_to_detect[img_resp.source.name] = cv_img
        if not images_to_detect:
            return None
        detections = yolo_detector.detect_targets_in_batch(images_to_detect, conf=0.1)  # type: ignore[union-attr]
        if detections:
            top_hit = detections[0]
            best_img = images_to_detect[top_hit["camera"]].copy()
            cx, cy = top_hit["cx"], top_hit["cy"]
            cv2.circle(best_img, (cx, cy), radius=8, color=(0, 0, 255), thickness=-1)
            cv2.circle(best_img, (cx, cy), radius=20, color=(0, 255, 0), thickness=2)
            cv2.putText(
                best_img, f"{top_hit['class']} {top_hit['conf']:.2f}",
                (cx + 25, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )
            window_name = f"Spot Detection: {top_hit['camera']}"
            cv2.imshow(window_name, best_img)
            logger.info(f"Click '{window_name}' to continue...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return top_hit
        return None

    @SpotTracker("Scanning", exit_on_fail=False)
    def find_target(self, yolo_detector: object) -> tuple | None:
        """Capture hand camera and return (image_response, cam, cx, cy, class) for the top hit."""
        try:
            image_responses = self.img_client.get_image(  # type: ignore[union-attr]
                [build_image_request("hand_color_image")]
            )
        except Exception as e:
            logger.error(f"Failed to acquire images: {e}")
            return None
        images_to_detect: dict = {}
        resp_map: dict = {}
        for img_resp in image_responses:
            cam_name = img_resp.source.name
            resp_map[cam_name] = img_resp
            img_data = np.frombuffer(img_resp.shot.image.data, dtype=np.uint8)
            cv_img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            if cv_img is not None:
                images_to_detect[cam_name] = cv_img
        if not images_to_detect:
            logger.warning("Failed to decode any images.")
            return None
        detections = yolo_detector.detect_targets_in_batch(images_to_detect, conf=0.1)  # type: ignore[union-attr]
        if not detections:
            logger.info("No targets found.")
            return None
        top_hit = detections[0]
        cam = top_hit["camera"]
        return resp_map[cam], cam, top_hit["cx"], top_hit["cy"], top_hit["class"]

    @SpotTracker("Grasping", exit_on_fail=False)
    def grasp_object(
        self,
        target_img_resp: object,
        cam_name: str,
        cx: int,
        cy: int,
        cls_name: str,
        timeout_sec: float = 30.0,
    ) -> bool:
        """Send a pixel-based grasp command and poll until success, failure, or timeout."""
        from bosdyn.client.frame_helpers import VISION_FRAME_NAME

        logger.info(f"Grasping '{cls_name}' at ({cx}, {cy}) via {cam_name}")
        pick_vec = geometry_pb2.Vec2(x=cx, y=cy)
        grasp = manipulation_api_pb2.PickObjectInImage(
            pixel_xy=pick_vec,
            transforms_snapshot_for_camera=target_img_resp.shot.transforms_snapshot,  # type: ignore[union-attr]
            frame_name_image_sensor=target_img_resp.shot.frame_name_image_sensor,  # type: ignore[union-attr]
            camera_model=target_img_resp.source.pinhole,  # type: ignore[union-attr]
        )
        grasp.grasp_params.grasp_params_frame_name = VISION_FRAME_NAME
        manip_req = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=grasp)
        try:
            cmd_response = self.manip_client.manipulation_api_command(  # type: ignore[union-attr]
                manipulation_api_request=manip_req
            )
            cmd_id = cmd_response.manipulation_cmd_id
            start_time = time.time()
            last_state_name = ""
            while True:
                if time.time() - start_time > timeout_sec:
                    logger.warning(f"Grasp timed out ({timeout_sec}s).")
                    self._recover_arm_safely()
                    return False
                feedback_req = manipulation_api_pb2.ManipulationApiFeedbackRequest(
                    manipulation_cmd_id=cmd_id
                )
                response = self.manip_client.manipulation_api_feedback_command(  # type: ignore[union-attr]
                    manipulation_api_feedback_request=feedback_req
                )
                state_name = manipulation_api_pb2.ManipulationFeedbackState.Name(
                    response.current_state
                )
                if state_name != last_state_name:
                    logger.info(f"Grasp state: {state_name}")
                    last_state_name = state_name
                if response.current_state in (
                    manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED,
                    manipulation_api_pb2.MANIP_STATE_DONE,
                ):
                    time.sleep(0.5)
                    gripper_pct = (
                        self.state_client.get_robot_state()  # type: ignore[union-attr]
                        .manipulator_state.gripper_open_percentage
                    )
                    if gripper_pct < 3.0:
                        logger.warning("Gripper fully closed — likely empty. Recovering.")
                        self._recover_arm_safely()
                        return False
                    logger.info("Grasp confirmed. Lifting to carry position.")
                    try:
                        self.cmd_client.robot_command(  # type: ignore[union-attr]
                            RobotCommandBuilder.synchro_stand_command()
                        )
                        time.sleep(0.2)
                        q = math_helpers.Quat.from_pitch(15.0 * math.pi / 180.0)
                        self.cmd_client.robot_command(  # type: ignore[union-attr]
                            RobotCommandBuilder.arm_pose_command(
                                0.75, 0.0, 0.35, q.w, q.x, q.y, q.z,
                                GRAV_ALIGNED_BODY_FRAME_NAME, 2.5,
                            )
                        )
                        time.sleep(0.2)
                        self.cmd_client.robot_command(  # type: ignore[union-attr]
                            RobotCommandBuilder.arm_carry_command()
                        )
                    except Exception as e:
                        logger.error(f"Post-grasp lift failed: {e}")
                    return True
                if (
                    response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_FAILED
                    or "FAILED" in state_name
                    or "NO_SOLUTION" in state_name
                ):
                    logger.warning(f"Grasp failed: {state_name}")
                    self._recover_arm_safely()
                    return False
                time.sleep(0.5)
        except Exception as e:
            logger.exception(f"Grasp error: {e}")
            self._recover_arm_safely()
            return False

    def _recover_arm_safely(self) -> None:
        """Open the gripper and stow the arm after a failed or timed-out grasp."""
        try:
            self.cmd_client.robot_command(RobotCommandBuilder.claw_gripper_open_command())  # type: ignore[union-attr]
            time.sleep(0.5)
            self.cmd_client.robot_command(RobotCommandBuilder.arm_stow_command())  # type: ignore[union-attr]
            logger.info("Arm recovered.")
        except Exception as e:
            logger.error(f"Arm recovery failed: {e}")

    # ── Public: navigation ────────────────────────────────────────────────────

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
        try:
            nav_cmd_id = self.graph_nav_client.navigate_to(  # type: ignore[union-attr]
                destination_waypoint_id=destination_wp_id,
                cmd_duration=timeout_sec,
            )
            start_time = time.time()
            while True:
                if time.time() - start_time > timeout_sec:
                    logger.warning(f"Navigation timeout ({timeout_sec}s).")
                    return False
                feedback = self.graph_nav_client.navigation_feedback(nav_cmd_id)  # type: ignore[union-attr]
                status = feedback.status
                if status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_REACHED_GOAL:
                    logger.info("Reached destination.")
                    return True
                elif status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_LOST:
                    logger.warning("Spot lost localisation.")
                    return False
                elif status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_STUCK:
                    logger.warning("Spot is stuck.")
                time.sleep(0.5)
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
                logger.info(f"Found '{target_name}' -> {wp.id[:6]}...")
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
        """Get current x, y, and yaw (in radians) of the body frame relative to vision or odom frame."""
        target_frame = VISION_FRAME_NAME if frame == "vision" else ODOM_FRAME_NAME
        state = self.state_client.get_robot_state()
        transforms = state.kinematic_state.transforms_snapshot
        tform = get_a_tform_b(transforms, target_frame, BODY_FRAME_NAME)
        q = tform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return tform.position.x, tform.position.y, yaw

    @SpotTracker("Move to coordinate goal", exit_on_fail=False)
    def move_to_goal(
        self,
        x: float,
        y: float,
        angle_deg: float,
        frame: str = "vision",
        use_local_origin: bool = False,
        timeout_sec: float = 15.0,
    ) -> bool:
        """Command the robot to move to a specific SE2 pose (x, y, yaw in degrees)."""
        goal_yaw = math.radians(angle_deg)
        target_frame = VISION_FRAME_NAME if frame == "vision" else ODOM_FRAME_NAME

        if use_local_origin:
            origin_x = getattr(self, "start_x", 0.0)
            origin_y = getattr(self, "start_y", 0.0)
            origin_yaw = getattr(self, "start_yaw", 0.0)
            final_x = origin_x + (x * math.cos(origin_yaw) - y * math.sin(origin_yaw))
            final_y = origin_y + (x * math.sin(origin_yaw) + y * math.cos(origin_yaw))
            final_yaw = origin_yaw + goal_yaw
        else:
            final_x, final_y, final_yaw = x, y, goal_yaw

        now = self.robot.time_sync.robot_timestamp_from_local_secs(time.time())
        end_time_sec = now.seconds + (now.nanos / 1e9) + timeout_sec
        cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=final_x, goal_y=final_y, goal_heading=final_yaw, frame_name=target_frame
        )
        self.cmd_client.robot_command(cmd, end_time_secs=end_time_sec)
        logger.info(f"Trajectory command sent. Target: ({final_x:.2f}, {final_y:.2f}, heading={math.degrees(final_yaw):.1f}°)")

        # Poll and check if reached
        start_time = time.time()
        while time.time() - start_time < timeout_sec:
            try:
                curr_x, curr_y, curr_yaw = self.get_current_pose(frame)
                dist = math.hypot(final_x - curr_x, final_y - curr_y)
                yaw_diff = abs(math.atan2(math.sin(final_yaw - curr_yaw), math.cos(final_yaw - curr_yaw)))
                if dist < 0.15 and yaw_diff < math.radians(10.0):
                    logger.info("Reached trajectory target successfully.")
                    return True
            except Exception as e:
                logger.warning(f"Error checking pose: {e}")
            time.sleep(0.2)
        logger.warning("Move to goal timed out.")
        return False

    @SpotTracker("Place Flat in Front", exit_on_fail=False)
    def place_flat_in_front(self) -> None:
        """Lower the arm close to the ground in front, open gripper to place object flat, then stow."""
        if self.cmd_client is None:
            raise RuntimeError("Clients not initialised.")
        try:
            logger.info("Placing object flat in front...")
            # Move arm down close to the ground, pointing forward-down
            pitch_rad = 20.0 * math.pi / 180.0
            q = math_helpers.Quat.from_pitch(pitch_rad)
            lower_cmd = RobotCommandBuilder.arm_pose_command(
                0.75, 0.0, -0.42, # x=0.75m forward, z=-0.42m below body (very close to ground)
                q.w, q.x, q.y, q.z,
                GRAV_ALIGNED_BODY_FRAME_NAME,
                2.0,
            )
            self.cmd_client.robot_command(lower_cmd)
            time.sleep(2.0)

            # Open gripper to release the bottle gently
            self.cmd_client.robot_command(RobotCommandBuilder.claw_gripper_open_command())
            time.sleep(1.5)

            # Move arm slightly up and back before stowing to avoid hitting the placed object
            raise_cmd = RobotCommandBuilder.arm_pose_command(
                0.60, 0.0, -0.20,
                q.w, q.x, q.y, q.z,
                GRAV_ALIGNED_BODY_FRAME_NAME,
                1.0,
            )
            self.cmd_client.robot_command(raise_cmd)
            time.sleep(1.0)

            # Stow the arm
            self.cmd_client.robot_command(RobotCommandBuilder.arm_stow_command())
            time.sleep(1.0)
            logger.info("Object placed flat and arm stowed.")
        except Exception as e:
            logger.error(f"Place flat in front failed: {e}")
