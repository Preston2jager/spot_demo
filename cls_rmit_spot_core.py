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
        #assert self.robot is not None, "Robot must be initialized"
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
                self.graph = self.get_current_graph()
                if not self.graph:
                    self.graph = self.upload_graph_and_snapshots("./Maps/amp2")
                    # List of all maps
                    # ./Maps/lv12_office 
                if not self.graph:
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
    
    # region Private APIs: Arm actions
    
    @SpotTracker("Arm Out (Absolute)", exit_on_fail=False)
    def _arm_out(self):
        if self.cmd_client is None:
            raise RuntimeError("Clients not initialised.")
        try:
            # 绝对坐标系：原点在机器人身体中心，X指向前方，Y指向左侧，Z指向上方
            root_frame = GRAV_ALIGNED_BODY_FRAME_NAME
            
            # 设置绝对目标位置：
            target_x = 0.80   # 距离机器人身体中心正前方 0.8 米
            target_y = 0.0    # 左右居中
            target_z = -0.15  # 距离身体中心高度向下 0.15 米
            
            # 设置绝对旋转：向下低头 25 度
            pitch_rad = 25.0 * math.pi / 180.0 
            q = math_helpers.Quat.from_pitch(pitch_rad)
            
            # 构建并发送手臂移动指令
            arm_cmd = RobotCommandBuilder.arm_pose_command(
                target_x, target_y, target_z,
                q.w, q.x, q.y, q.z, 
                root_frame, 
                2.0  # 给予 2.0 秒的时间让手臂平滑移动到指定位置
            )
            self.cmd_client.robot_command(arm_cmd)
            time.sleep(2.0) # 等待手臂到位
            
            # 打开抓手准备扫描
            self.cmd_client.robot_command(RobotCommandBuilder.claw_gripper_open_command())
            time.sleep(0.5)
            print("Arm is out and ready (Absolute pose).")
            
        except Exception as e:
            print(f"Arm out failed: {e}")

    @SpotTracker("Arm Release and Stow", exit_on_fail=False)
    def _arm_release(self, bin=False):
        """
        抓取成功后调用：将物品移动到低处 -> 松开抓手 -> 收起手臂
        """
        if self.cmd_client is None:
            raise RuntimeError("Clients not initialised.")
        try:
            root_frame = GRAV_ALIGNED_BODY_FRAME_NAME
            print("Moving arm to a lower position to release...")
            if bin:
                target_x = 0.8   #
                target_y = 0.0    # 左右居中
                target_z = 0.2  # 尽可能低的位置 (基于身体中心向下 0.4 米)
                pitch_rad = 60.0 * math.pi / 180.0 
            else:
                target_x = 1   # 身体正前方 0.85 米
                target_y = 0.0    # 左右居中
                target_z = -0.20  # 尽可能低的位置 (基于身体中心向下 0.4 米)
                pitch_rad = 60.0 * math.pi / 180.0 
            q = math_helpers.Quat.from_pitch(pitch_rad)
            lower_cmd = RobotCommandBuilder.arm_pose_command(
                target_x, target_y, target_z,
                q.w, q.x, q.y, q.z, 
                root_frame, 
                1.0  # 因为手上拿着东西，给 2.5 秒时间让下降动作更平滑
            )
            self.cmd_client.robot_command(lower_cmd)
            time.sleep(1.0) # 必须等待移动完成，否则可能会在半空中直接扔掉
            print("Releasing object...")
            self.cmd_client.robot_command(RobotCommandBuilder.claw_gripper_open_command())
            time.sleep(1.0) # 给物品掉落的时间
            print("Stowing arm...")
            stow_cmd = RobotCommandBuilder.arm_stow_command()
            self.cmd_client.robot_command(stow_cmd)
            time.sleep(0.5) # 等待 stow 指令开始执行
            print("Object released and arm stowed.")
        except Exception as e:
            print(f"Arm release failed: {e}")
    
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
    
    @SpotTracker("Quick scanning", exit_on_fail=False)
    def quick_detect(self, yolo_detector):
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
            
        # 调用你定义的检测器
        detections = yolo_detector.detect_targets_in_batch(images_to_detect, conf=0.1)
        
        if detections:
            top_hit = detections[0]
            print(f" Best target is from [{top_hit['camera']}]: "
                  f"{top_hit['class']} (Conf: {top_hit['conf']:.2f})")
            
            # ================= 新增：弹窗显示代码 =================
            # 1. 从字典中提取出对应相机的原图
            best_img = images_to_detect[top_hit['camera']].copy()
            
            # 2. 提取出你 YOLO 检测器返回的中心点坐标 cx, cy
            cx = top_hit['cx']
            cy = top_hit['cy']
            
            # 3. 在中心点画一个红色的实心圆点 (BGR 格式，所以 (0, 0, 255) 是红色)
            cv2.circle(best_img, (cx, cy), radius=8, color=(0, 0, 255), thickness=-1)
            # 可选：再画一个大一点的绿色空心圆圈，像一个“准星”
            cv2.circle(best_img, (cx, cy), radius=20, color=(0, 255, 0), thickness=2)
            
            # 4. 在准星旁边写上类别和置信度文本
            label = f"{top_hit['class']} {top_hit['conf']:.2f}"
            cv2.putText(best_img, label, (cx + 25, cy - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 5. 使用 OpenCV 弹窗显示图片
            window_name = f"Spot Detection: {top_hit['camera']}"
            cv2.imshow(window_name, best_img)
            
            print(f"👉 弹窗已开启，请在图片弹窗 '{window_name}' 上按任意键盘按键关闭并继续程序...")
            cv2.waitKey(0)           # 程序在此暂停，等待你在弹窗上按任意键
            cv2.destroyAllWindows()  # 按键后关闭所有 OpenCV 窗口
            # ======================================================
            
            return top_hit
            
        return None
    
    @SpotTracker("Scanning", exit_on_fail=False)
    def find_target(self, yolo_detector):
        print("Scanning for targets")
        #camera_sources = ["hand_color_image"]
        camera_sources = [
            'hand_color_image'
        ]
        image_requests = [build_image_request(src) for src in camera_sources]
        try:
            image_responses = self.img_client.get_image(image_requests) # type: ignore
        except Exception as e:
            print(f"Failed to acquire images: {e}")
            return None
        images_to_detect = {}
        resp_map = {} 
        for img_resp in image_responses: # type: ignore
            cam_name = img_resp.source.name # type: ignore
            resp_map[cam_name] = img_resp
            img_data = np.frombuffer(img_resp.shot.image.data, dtype=np.uint8) # type: ignore
            cv_img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            if cv_img is not None:
                images_to_detect[cam_name] = cv_img 
        if not images_to_detect:
            print("Failed to decode any images for detection.")
            return None
            
        print("[Grasp] 🧠 图像获取成功，开始 YOLO 识别...")
        detections = yolo_detector.detect_targets_in_batch(images_to_detect, conf=0.1)
        if not detections:
            print("[Grasp] ❌ 未能在当前视野中找到任何目标。")
            return None
            
        top_hit = detections[0]
        cam_name = top_hit["camera"]
        cx, cy = top_hit["cx"], top_hit["cy"]
        cls_name = top_hit["class"]
        
        # ⚠️ 关键修复：必须把当时的图像响应对象也返回，抓取时需要里面的相机内参和坐标系快照
        target_img_resp = resp_map[cam_name]
        return target_img_resp, cam_name, cx, cy, cls_name
    
    @SpotTracker("Scanning", exit_on_fail=False)
    def find_target_2(self, yolo_detector):
        print("Scanning for targets")
        
        # 1. 定义视觉相机与其对应的“对齐深度相机”映射关系
        VISUAL_TO_DEPTH_MAP = {
            'hand_color_image': 'hand_depth_in_hand_color_frame',
            'left_fisheye_image': 'left_depth_in_visual_frame',
            'right_fisheye_image': 'right_depth_in_visual_frame'
        }
        
        camera_sources = list(VISUAL_TO_DEPTH_MAP.keys())
        depth_sources = list(VISUAL_TO_DEPTH_MAP.values())
        
        # 将视觉图和深度图一并请求，保证快照时间戳一致
        all_sources = camera_sources + depth_sources
        image_requests = [build_image_request(src) for src in all_sources]
        
        try:
            image_responses = self.img_client.get_image(image_requests) # type: ignore
        except Exception as e:
            print(f"Failed to acquire images: {e}")
            return None
            
        images_to_detect = {}
        depth_images = {}
        resp_map = {} 
        
        # 2. 分类解析视觉图像和深度图像
        for img_resp in image_responses: # type: ignore
            cam_name = img_resp.source.name # type: ignore
            resp_map[cam_name] = img_resp
            
            # 解析深度图
            if cam_name in depth_sources:
                try:
                    # Spot 的深度图通常是 16-bit 格式，单位是毫米 (mm)
                    # 如果格式是 RAW，使用 numpy 解析；如果是压缩格式（如PNG），使用 OpenCV AnyDepth 解析
                    if img_resp.shot.image.format == img_resp.shot.image.FORMAT_RAW:
                        img_data = np.frombuffer(img_resp.shot.image.data, dtype=np.uint16)
                        cv_depth = img_data.reshape(img_resp.shot.image.rows, img_resp.shot.image.cols)
                    else:
                        img_data = np.frombuffer(img_resp.shot.image.data, dtype=np.uint8)
                        cv_depth = cv2.imdecode(img_data, cv2.IMREAD_ANYDEPTH)
                        
                    if cv_depth is not None:
                        depth_images[cam_name] = cv_depth
                except Exception as e:
                    print(f"Failed to decode depth image for {cam_name}: {e}")
                    
            # 解析彩色/视觉图
            else:
                img_data = np.frombuffer(img_resp.shot.image.data, dtype=np.uint8) # type: ignore
                cv_img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                if cv_img is not None:
                    images_to_detect[cam_name] = cv_img 
                    
        if not images_to_detect:
            print("Failed to decode any images for detection.")
            return None
            
        print("[Grasp] 🧠 图像获取成功，开始 YOLO 识别...")
        detections = yolo_detector.detect_targets_in_batch(images_to_detect, conf=0.1)
        if not detections:
            print("[Grasp] ❌ 未能在当前视野中找到任何目标。")
            return None
            
        # 3. 距离校验过滤 (最大距离 4 米)
        MAX_DISTANCE_MM = 4000 
        valid_detections = []
        
        for hit in detections:
            cam_name = hit["camera"]
            cx, cy = int(hit["cx"]), int(hit["cy"])
            depth_cam_name = VISUAL_TO_DEPTH_MAP.get(cam_name)
            
            if depth_cam_name and depth_cam_name in depth_images:
                depth_img = depth_images[depth_cam_name]
                
                # 确保提取坐标没有越界
                if 0 <= cy < depth_img.shape[0] and 0 <= cx < depth_img.shape[1]:
                    distance_mm = depth_img[cy, cx]
                    
                    # 距离过滤：>0 排除传感器盲区失效黑洞，<= 4000 确保在4米范围内
                    if 0 < distance_mm <= MAX_DISTANCE_MM:
                        hit["distance_mm"] = distance_mm # 记录距离备用
                        valid_detections.append(hit)
                        print(f"✔️ 目标有效: {hit['class']} 在 {distance_mm/1000.0:.2f}米处.")
                    else:
                        print(f"❌ 目标过滤: {hit['class']} 在 {distance_mm/1000.0:.2f}米处 (超出4米范围或无效).")
                else:
                    print(f"⚠️ 越界: ({cx}, {cy}) 不在深度图范围内.")
            else:
                # 容错处理：如果没有获取到深度图，默认先保留该目标（视你实际安全需求也可直接丢弃）
                print(f"⚠️ 缺少对应的深度数据，保守保留 {hit['class']} 目标.")
                valid_detections.append(hit)

        if not valid_detections:
            print("[Grasp] ❌ 视野内发现了目标，但均在4米以外或不可达范围，丢弃。")
            return None
            
        # 这里默认取了有效目标中的第一个，你也可以改为根据 distance_mm 排序取最近的
        # top_hit = sorted(valid_detections, key=lambda x: x.get("distance_mm", float('inf')))[0]
        top_hit = valid_detections[0]
        
        cam_name = top_hit["camera"]
        cx, cy = top_hit["cx"], top_hit["cy"]
        cls_name = top_hit["class"]
        
        # ⚠️ 关键修复：返回当时的图像响应对象，抓取时需要里面的相机内参和坐标系快照
        target_img_resp = resp_map[cam_name]
        return target_img_resp, cam_name, cx, cy, cls_name
    

    @SpotTracker("Grasping", exit_on_fail=False)
    def grasp_object(self, target_img_resp, cam_name, cx, cy, cls_name, timeout_sec=30.0):
        """
        执行抓取动作，包含状态监控、防呆验证和超时恢复。
        """
        print(f"Grasping object {cls_name} at ({cx}, {cy}) in camera {cam_name}")
        
        from bosdyn.client.frame_helpers import VISION_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME, math_helpers
        from bosdyn.client.robot_command import RobotCommandBuilder

        pick_vec = geometry_pb2.Vec2(x=cx, y=cy) # type: ignore
        grasp = manipulation_api_pb2.PickObjectInImage( # type: ignore
            pixel_xy=pick_vec, 
            transforms_snapshot_for_camera=target_img_resp.shot.transforms_snapshot, 
            frame_name_image_sensor=target_img_resp.shot.frame_name_image_sensor, 
            camera_model=target_img_resp.source.pinhole 
        )
        
        grasp.grasp_params.grasp_params_frame_name = VISION_FRAME_NAME
        
        #constraint_top_down = grasp.grasp_params.allowable_orientation.add()
        #constraint_top_down.vector_alignment_with_tolerance.axis_on_gripper_ewrt_gripper.CopyFrom(geometry_pb2.Vec3(x=1, y=0, z=0)) # type: ignore
        #constraint_top_down.vector_alignment_with_tolerance.axis_to_align_with_ewrt_frame.CopyFrom(geometry_pb2.Vec3(x=0, y=0, z=-1)) # type: ignore
        # 容差放宽到 0.25 (约 15度)，给机械臂留出足够的运动空间
        #constraint_top_down.vector_alignment_with_tolerance.threshold_radians = 0.78 
        # =================================================================

        manip_req = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=grasp) # type: ignore
        
        try:
            print("Sending grasp command to robot...")
            cmd_response = self.manip_client.manipulation_api_command(manipulation_api_request=manip_req) # type: ignore
            cmd_id = cmd_response.manipulation_cmd_id
            print(f"Task sent, ID: {cmd_id}")
            
            start_time = time.time()
            last_state_name = "" 
            
            while True:
                if time.time() - start_time > timeout_sec:
                    print(f"❌ 抓取超时 ({timeout_sec}s)，放弃尝试。")
                    self._recover_arm_safely()
                    return False
                    
                feedback_req = manipulation_api_pb2.ManipulationApiFeedbackRequest(manipulation_cmd_id=cmd_id) # type: ignore
                response = self.manip_client.manipulation_api_feedback_command(manipulation_api_feedback_request=feedback_req) # type: ignore
                state_name = manipulation_api_pb2.ManipulationFeedbackState.Name(response.current_state) # type: ignore
                
                if state_name != last_state_name:
                    print(f"🔄 机器人当前动作: {state_name}")
                    last_state_name = state_name
                
                # ⚠️ 修复卡死问题：同时监听 SUCCEEDED 和 DONE 两个状态
                if response.current_state in [manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED, manipulation_api_pb2.MANIP_STATE_DONE]: # type: ignore
                    print("✅ 收到抓取完成信号，正在物理验证是否抓空...")
                    time.sleep(0.5)  
                    
                    robot_state = self.state_client.get_robot_state()
                    gripper_open_pct = robot_state.manipulator_state.gripper_open_percentage
                    print(f"🔍 抓手当前张开比例: {gripper_open_pct:.2f}%")
                    
                    if gripper_open_pct < 3.0: 
                        print("❌ 警告：抓手完全闭合，说明抓空了！")
                        self._recover_arm_safely()
                        return False
                        
                    print("✅ 物理验证通过，确实抓到了物品！")
                    
                    try:
                        # 🌟 新增：让机器狗恢复默认站立高度
                        print("🐕 正在恢复标准站立姿态...")
                        stand_cmd = RobotCommandBuilder.synchro_stand_command()
                        # 发送站立指令（假设你的 cmd_client 已经初始化）
                        self.cmd_client.robot_command(stand_cmd)
                        time.sleep(0.2) # 给它一点时间站起来
                        
                        print("🔄 正在将物体举高到绝对位置...")
                        abs_x, abs_y, abs_z = 0.75, 0.0, 0.35
                        q = math_helpers.Quat.from_pitch(15.0 * math.pi / 180.0)
                        
                        lift_cmd = RobotCommandBuilder.arm_pose_command(
                            abs_x, abs_y, abs_z, q.w, q.x, q.y, q.z,
                            GRAV_ALIGNED_BODY_FRAME_NAME, 2.5 
                        )
                        self.cmd_client.robot_command(lift_cmd)
                        time.sleep(0.2) 
                        
                        carry_cmd = RobotCommandBuilder.arm_carry_command()
                        self.cmd_client.robot_command(carry_cmd) # type: ignore
                        print("✅ 已切换至 Carry 模式。")
                        
                    except Exception as e:
                        print(f"❌ 举高/站立动作失败: {e}")
                    return True
                    
                elif response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_FAILED or 'FAILED' in state_name or 'NO_SOLUTION' in state_name: # type: ignore
                    print(f"❌ 抓取失败/无运动解: {state_name}")
                    self._recover_arm_safely()
                    return False 
                time.sleep(0.5) 
        except Exception as e:
            print(f"❌ Error during grasp operation: {e}")
            import traceback
            traceback.print_exc()
            self._recover_arm_safely()
            return False
    
    def _recover_arm_safely(self):
        """抓取失败、抓空或超时时的安全回退逻辑"""
        try:
            self.cmd_client.robot_command(RobotCommandBuilder.claw_gripper_open_command())
            time.sleep(0.5)
            self.cmd_client.robot_command(RobotCommandBuilder.arm_stow_command())
            print("🔄 已张开抓手并收回机械臂。")
        except Exception as e:
            print(f"⚠️ 机械臂收回指令发送失败: {e}")

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
