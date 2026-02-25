# ===== =================== =====
# ===== RMIT Spot control class =====
# ===== =================== =====
import os
import time
import math
import cv2
from typing import Optional
import numpy as np
# ===== Tracker and Streamer =====
from cls_rmit_spot_tracker import SpotTracker
from cls_rmit_spot_stream import SpotStreamer
# ===== BostonDynamic APIs =====
import bosdyn.client
from bosdyn.api import manipulation_api_pb2
from bosdyn.api import geometry_pb2
from bosdyn.api.graph_nav import graph_nav_pb2, nav_pb2, map_pb2
from bosdyn.client.graph_nav import GraphNavClient
from bosdyn.client.image import build_image_request, ImageClient
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.recording import GraphNavRecordingServiceClient
from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder, blocking_stand
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.frame_helpers import (
    GRAV_ALIGNED_BODY_FRAME_NAME,
    HAND_FRAME_NAME,
    ODOM_FRAME_NAME,     
    BODY_FRAME_NAME,   
    get_a_tform_b,
    get_se2_a_tform_b,
    math_helpers
)

class SpotAgent:

    # region  Private APIs: Initialisation

    @SpotTracker("Spot agent initialisation", exit_on_fail=True)
    def __init__(
        self,
        *,
        client_name: str = "SpotAgent",
        keep_alive_period_sec: float = 2.0,
        force_lease: bool = True,
        stream: bool = False,
        navigation: bool = False
    ):
        self.hostname = "192.168.80.3"
        self.username = "user"
        self.password = "myjujz7e2prj"
        self.stream_state = stream
        self.navigation_state = navigation
        self.client_name = client_name
        self.keep_alive_period_sec = keep_alive_period_sec
        self.sdk: Optional[bosdyn.client.Sdk] = None
        self.robot: Optional[bosdyn.client.Robot] = None
        self.lease_client: Optional[LeaseClient] = None
        self.cmd_client: Optional[RobotCommandClient] = None
        self.img_client: Optional[ImageClient] = None
        self.state_client: Optional[RobotStateClient] = None
        self.graph_nav_client: Optional[GraphNavClient] = None
        self.recording_client: Optional[GraphNavRecordingServiceClient] = None
        self._lease_keepalive: Optional[LeaseKeepAlive] = None
        self.force_lease = force_lease  

    @SpotTracker("Spot agent services startup", exit_on_fail=True)
    def __enter__(self):
        assert self.robot is not None, "Robot must be initialized"
        self._auto_login(self.username, self.password)
        self._get_lease(force=self.force_lease)
        _ = self.robot.ensure_client(RobotCommandClient.default_service_name)
        self._clear_behavior_faults()
        self._power_on_and_stand()
        with SpotTracker("Streming startup", exit_on_fail=False):
            if self.stream_state:
                streamer = SpotStreamer(self.robot)
                streamer.start()
        with SpotTracker("Navigation startup", exit_on_fail=False):  
            if self.navigation_state:
                graph = self.get_current_graph()
                if not graph:
                    graph = self.upload_graph_and_snapshots("./graph_nav_command_line/08_12_office")
                if not graph:
                    print("[No graph available and failed to upload. Check connection and graph path.")
                    return
                if not self.initialize_graphnav_to_fiducial():
                    print("Failed to initialize GraphNav. Check connection and fiducial setup.")
                    return
        return self

    def __exit__(self, *args):
        self._shutdown()

    # endregion

    # region  Private APIs: Spot admin

    @SpotTracker("Login and Time Sync")
    def _auto_login(self, username: str, password: str):
        self.sdk = bosdyn.client.create_standard_sdk(self.client_name)
        self.robot = self.sdk.create_robot(self.hostname)
        self.robot.authenticate(username, password)
        self.robot.time_sync.start() # type: ignore
        sync_success = False
        for i in range(3):
            try:
                self.robot.time_sync.wait_for_sync(timeout_sec=5.0) # type: ignore
                sync_success = True
                break
            except Exception:
                print(f"Time sync failed, attemp ({i+1}/3)...")
                time.sleep(1)
        if not sync_success:
            raise RuntimeError("Can not establish time sync with the robot. Please check the connection and try again.")
        self.lease_client = self.robot.ensure_client(LeaseClient.default_service_name)
        self.cmd_client = self.robot.ensure_client(RobotCommandClient.default_service_name)
        self.img_client = self.robot.ensure_client(ImageClient.default_service_name)
        self.state_client = self.robot.ensure_client(RobotStateClient.default_service_name)
        self.graph_nav_client = self.robot.ensure_client(GraphNavClient.default_service_name)
        self.recording_client = self.robot.ensure_client(GraphNavRecordingServiceClient.default_service_name)
        self.manip_client = self.robot.ensure_client(ManipulationApiClient.default_service_name)

    @SpotTracker("Take Spot lease", exit_on_fail=True)
    def _get_lease(self, force: bool = False):
        if self._lease_keepalive:
            self._lease_keepalive.shutdown()
        if force:
            try:
                self.lease_client.take() # type: ignore
            except Exception as e:
                print(f"[Lease] ⚠️ Fail to get the lease: {e}")
        self._lease_keepalive = LeaseKeepAlive(self.lease_client, must_acquire=True, return_at_exit=True)
    
    @SpotTracker("Power on and Stand", exit_on_fail=True)
    def _power_on_and_stand(self, arm = False):
        if not self.robot.is_powered_on(): # type: ignore
            self.robot.power_on(timeout_sec=20) # type: ignore
        blocking_stand(self.cmd_client, timeout_sec=10)
        if arm:
            self._arm_out()

    @SpotTracker("Shutdown", exit_on_fail=False)
    def _shutdown(self):
        if self._lease_keepalive:
            self._lease_keepalive.shutdown()
    
    @SpotTracker("Clear Behavior Faults", exit_on_fail=False)
    def _clear_behavior_faults(self) -> bool:
        if self.state_client is None or self.cmd_client is None:
            print("Clients not initialised, cannot check or clear faults.")
            return False
        try:
            state = self.state_client.get_robot_state()
            faults = state.behavior_fault_state.faults # type: ignore
            if not faults:
                print("No behavior faults detected. Spot is ready to go")
                return True
            print(f"{len(faults)} faults, attempting to clear...")
            for fault in faults:
                print(f"  -> 🛑 故障 ID: {fault.behavior_fault_id}")
                print(f"  -> 📝 故障原因: {fault.cause}")
                self.cmd_client.clear_behavior_fault(behavior_fault_id=fault.behavior_fault_id)
                time.sleep(0.5)
            time.sleep(1.0)
            new_state = self.state_client.get_robot_state()
            if not new_state.behavior_fault_state.faults: # type: ignore
                print(" 🎉 所有行为故障已成功清除！")
                return True
            else:
                print(f" There are still {len(new_state.behavior_fault_state.faults)} faults remaining.") # type: ignore
                print("Use tablet to investigate and clear them manually before retrying.")
                return False
        except Exception as e:
            print(f"Error: {e}")
            return False    

    # endregion
    @SpotTracker("Arm Out", exit_on_fail=False)
    def _arm_out(self):
        if self.cmd_client is None or self.img_client is None:
            raise RuntimeError("Clients not initialised.")
        try:
            snapshot = self.state_client.get_robot_state().kinematic_state.transforms_snapshot # type: ignore
            root_frame = GRAV_ALIGNED_BODY_FRAME_NAME
            root_T_hand = get_a_tform_b(snapshot, root_frame, HAND_FRAME_NAME)
            delta_hand = math_helpers.SE3Pose(
                x = 0.30,   # 在 X 轴方向（前方）移动 0.3 米
                y = 0.0,    # 左右不偏移
                z = -0.25,  # 在 Z 轴方向向下移动 0.25 米
                rot = math_helpers.Quat.from_pitch(15.0 * math.pi / 180.0) # 向下低头 30 度
            )
            root_T_target = root_T_hand * delta_hand # type: ignore
            q = root_T_target.rot # type: ignore
            arm_cmd = RobotCommandBuilder.arm_pose_command(
                root_T_target.x, root_T_target.y, root_T_target.z,
                q.w, q.x, q.y, q.z, root_frame, 1.2 # 1.2 秒内完成动作 # type: ignore
            )
            self.cmd_client.robot_command(arm_cmd)
            self.cmd_client.robot_command(RobotCommandBuilder.claw_gripper_open_command())
            time.sleep(0.4)
            print("Arm ready.")
        except Exception as e:
            print(f"Arm failed:{e}")
    
    @SpotTracker("Arm In", exit_on_fail=False)
    def _arm_in(self):
        if self.cmd_client is None:
            raise RuntimeError("Clients not initialised.")
        try:
            self.cmd_client.robot_command(RobotCommandBuilder.claw_gripper_close_command())
            stow_cmd = RobotCommandBuilder.arm_stow_command()
            self.cmd_client.robot_command(stow_cmd)
            print("Arm stowing...")
        except Exception as e:
            print(f"Arm stow failed: {e}")    

    # endregion
    
    # region Public APIs: Manipulation & Vision Logic

    @SpotTracker("Scanning", exit_on_fail=False)
    def find_targetyolo(self, yolo_detector):
        print("Scanning for targets")
        camera_sources = [
            "frontleft_fisheye_image", 
            "frontright_fisheye_image", 
            "left_fisheye_image", 
            "right_fisheye_image", 
            "back_fisheye_image",
            "hand_color_image"
        ]
        image_requests = [build_image_request(src) for src in camera_sources]
        try:
            image_responses = self.img_client.get_image(image_requests) # type: ignore
        except Exception as e:
            print(f"Failed to accquire images: {e}")
            return None
        images_to_detect = {}
        for img_resp in image_responses: # type: ignore
            cam_name = img_resp.source.name # type: ignore
            img_data = np.frombuffer(img_resp.shot.image.data, dtype=np.uint8) # type: ignore
            cv_img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            if cv_img is not None:
                images_to_detect[cam_name] = cv_img
        if not images_to_detect:
            return None
        detections = yolo_detector.detect_targets_in_batch(images_to_detect, conf=0.1)
        if detections:
            top_hit = detections[0]
            print(f" Best target is from [{top_hit['camera']}]: "
                  f"{top_hit['class']} (Conf: {top_hit['conf']:.2f})")
            return top_hit
        return None
    
    @SpotTracker("Scanning", exit_on_fail=False)
    def find_and_grasp_target(self, yolo_detector, timeout_sec=60.0):
        print("Scanning for targets")
        camera_sources = [
            "frontleft_fisheye_image", 
            "frontright_fisheye_image", 
            "hand_color_image"
        ]
        image_requests = [build_image_request(src) for src in camera_sources]
        try:
            image_responses = self.img_client.get_image(image_requests) # type: ignore
        except Exception as e:
            print(f"[Error] 获取多相机图像失败: {e}")
            return False
        images_to_detect = {}
        resp_map = {} 
        for img_resp in image_responses: # type: ignore
            cam_name = img_resp.source.name # type: ignore
            resp_map[cam_name] = img_resp # 保存原始响应，抓取时需要用到里面的相机内参
            img_data = np.frombuffer(img_resp.shot.image.data, dtype=np.uint8) # type: ignore
            cv_img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            if cv_img is not None:
                images_to_detect[cam_name] = cv_img
        if not images_to_detect:
            print("[Grasp] ❌ 图像解码失败！")
            return False
        print("[Grasp] 🧠 图像获取成功，开始批量 YOLO 识别...")
        detections = yolo_detector.detect_targets_in_batch(images_to_detect, conf=0.1)
        if not detections:
            print("[Grasp] ❌ 未能在当前视野中找到任何目标。")
            return False
        top_hit = detections[0]
        cam_name = top_hit["camera"]
        cx, cy = top_hit["cx"], top_hit["cy"]
        cls_name = top_hit["class"]
        print(f"Set target: {cls_name}, from {cam_name}, coordinates: ({cx}, {cy}), confidence: {top_hit['conf']:.2f}")
        target_img_resp = resp_map[cam_name]
        print("Sending grasp command to robot...")
        pick_vec = geometry_pb2.Vec2(x=cx, y=cy) # type: ignore
        grasp_request = manipulation_api_pb2.PickObjectInImage(
            pixel_xy=pick_vec, # type: ignore
            transforms_snapshot_for_camera=target_img_resp.shot.transforms_snapshot, # type: ignore
            frame_name_image_sensor=target_img_resp.shot.frame_name_image_sensor, # type: ignore
            camera_model=target_img_resp.source.pinhole # type: ignore
        )
        manip_req = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=grasp_request) # type: ignore
        try:
            cmd_response = self.manip_client.manipulation_api_command(manipulation_api_request=manip_req)
            cmd_id = cmd_response.manipulation_cmd_id
            print(f"Task send, ID: {cmd_id}")
            start_time = time.time()
            while True:
                if time.time() - start_time > timeout_sec:
                    print(f"Timeout ({timeout_sec} ), ababdoning grasp attempt.")
                    return False
                feedback_req = manipulation_api_pb2.ManipulationApiFeedbackRequest(manipulation_cmd_id=cmd_id) # type: ignore
                feedback_resp = self.manip_client.manipulation_api_feedback_command(manipulation_api_feedback_request=feedback_req)
                state_name = manipulation_api_pb2.ManipulationFeedbackState.Name(feedback_resp.current_state)
                print(f"{state_name}")
                if state_name in ['MANIP_STATE_DONE', 'MANIP_STATE_GRASP_SUCCEEDED']:
                    print("Grasp success")
                    try:
                        carry_cmd = RobotCommandBuilder.arm_carry_command()
                        self.cmd_client.robot_command(carry_cmd) # type: ignore
                        print("Switch to carry mode！")
                    except Exception as e:
                        print(f"Failed to switch to carry mode: {e}")
                    return True  
                elif 'FAILED' in state_name or 'NO_SOLUTION' in state_name:
                    print(f"[Failed to grasp: {state_name}")
                    try:
                        self.cmd_client.robot_command(RobotCommandBuilder.arm_stow_command()) # type: ignore
                        print("Stow arm after failed grasp.")
                    except Exception as e:
                        print(f"Stow arm failed: {e}")
                    return False 
                time.sleep(1.0) 
        except Exception as e:
            print(f"Error during grasp operation: {e}")
            return False

    # endregion

    # region Public APIs: Navigation Logic

    @SpotTracker("Navigation initialisation", exit_on_fail=False)
    def initialize_graphnav_to_fiducial(self, fiducial_id: Optional[int] = None):
        print("Determine initial position for GraphNav localization...")
        try:
            initial_guess = nav_pb2.Localization()
            if fiducial_id is not None:

                self.graph_nav_client.set_localization( # type: ignore
                    initial_guess_localization=initial_guess,  
                    fiducial_init=graph_nav_pb2.SetLocalizationRequest.FIDUCIAL_INIT_SPECIFIC, # type: ignore
                    use_fiducial_id=fiducial_id
                )
            else:
                self.graph_nav_client.set_localization( # type: ignore
                    initial_guess_localization=initial_guess, 
                    fiducial_init=graph_nav_pb2.SetLocalizationRequest.FIDUCIAL_INIT_NEAREST # type: ignore
                )
            state = self.graph_nav_client.get_localization_state() # type: ignore
            if not state.localization.waypoint_id: # type: ignore
                print("No QR code detected for localisation! Please make sure Spot can see the fiducials, and try again.")
                return False
            print(f"Localisation success: Spot is near {state.localization.waypoint_id[:6]}") # type: ignore
            return True
        except Exception as e:
            print(f"Error during localisation: {e}")
            return False
        
    @SpotTracker("Navigate to waypoint", exit_on_fail=False)
    def navigate_to_waypoint(self, destination_wp_id: str, timeout_sec: float = 60.0):
        print(f"Navigate to: {destination_wp_id[:6]}...")
        try:
            nav_cmd_id = self.graph_nav_client.navigate_to( # type: ignore
                destination_waypoint_id=destination_wp_id,
                cmd_duration=timeout_sec
            )
            start_time = time.time()
            while True:
                current_time = time.time()
                if current_time - start_time > timeout_sec:
                    print(f"Navigation timeout ({timeout_sec}s), abandoning attempt.")
                    return False
                feedback = self.graph_nav_client.navigation_feedback(nav_cmd_id) # type: ignore
                status = feedback.status # type: ignore
                if status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_REACHED_GOAL: # type: ignore
                    print("Reached destination! Navigation successful.")
                    return True
                elif status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_LOST: # type: ignore
                    print("Spot has lost localization. Attempting to re-localise...")
                    return False
                elif status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_STUCK: # type: ignore
                    print("Spot is stuck and cannot reach the destination. Please check for obstacles or narrow passages, and try again.")
                time.sleep(0.5)
        except Exception as e:
            print(f"Error: {e}")
            return False
        
    @SpotTracker("Loading graph", exit_on_fail=False)
    def get_current_graph(self):
        print("Reading graph from robot memory...")
        try:
            graph = self.graph_nav_client.download_graph() # type: ignore
            
            if not graph.waypoints: # type: ignore
                print("No waypoints found in the graph! Please make sure the robot has a valid map and try again.")
                return None
            print(f"Graph found. Current graph including {len(graph.waypoints)} warypoints, {len(graph.edges)} edges。") # type: ignore
            return graph
        except Exception as e:
            print(f"Error reading graph: {e}")
            return None
    
    def get_waypoint_id_by_name(self, graph, target_name: str) -> str:
        available_names = []
        for wp in graph.waypoints:
            wp_name = wp.annotations.name
            available_names.append(wp_name)
            if wp_name.lower() == target_name.lower():
                print(f"Found target '{target_name}'. Accquired ID:  {wp.id[:6]}...")
                return wp.id
        print(f"Can not find waypoint named '{target_name}'!")
        print(f"Available waypoints: {', '.join(available_names)}")
        return None # type: ignore
    
    @SpotTracker("Uploading graph and snapshots", exit_on_fail=False)
    def upload_graph_and_snapshots(self, save_dir: str): 
        print(f"Upload '{save_dir}' to spot")
        self.graph_nav_client.clear_graph() # type: ignore
        graph_path = os.path.join(save_dir, "graph")
        if not os.path.exists(graph_path):
            print(f"[GraphNav] ❌ 找不到地图文件: {graph_path}")
            return None
        with open(graph_path, "rb") as f:
            graph = map_pb2.Graph()
            graph.ParseFromString(f.read()) # type: ignore
        print("Uploading graph structure...")
        self.graph_nav_client.upload_graph(graph=graph, generate_new_anchoring=True) # type: ignore
        print("Uploading waypoint snapshots...")
        for wp in graph.waypoints: # type: ignore
            if wp.snapshot_id:
                wp_path = os.path.join(save_dir, f"wp_{wp.snapshot_id}")
                if os.path.exists(wp_path):
                    with open(wp_path, "rb") as f:
                        snap = map_pb2.WaypointSnapshot()
                        snap.ParseFromString(f.read())  # type: ignore
                        for img in snap.images:   # type: ignore
                            img.shot.image.data = b"" 
                        self.graph_nav_client.upload_waypoint_snapshot(snap) # type: ignore
                else:
                    print(f" Missing waypoint snapshot file: {wp_path}")
        print("Uploading edge snapshots...")
        for edge in graph.edges: # type: ignore
            if edge.snapshot_id:
                edge_path = os.path.join(save_dir, f"edge_{edge.snapshot_id}")
                if os.path.exists(edge_path):
                    with open(edge_path, "rb") as f:
                        snap = map_pb2.EdgeSnapshot()
                        snap.ParseFromString(f.read()) # type: ignore
                        self.graph_nav_client.upload_edge_snapshot(snap) # type: ignore
                else:
                    print(f" Missing edge snapshot file: {edge_path}")
        print("Graph upload complete.")
        return graph
