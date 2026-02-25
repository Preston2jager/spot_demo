# ===== =================== =====
# ===== RMIT Spot control class =====
# ===== =================== =====
import os
import time
import math
import cv2
import sys
import threading
import traceback
import functools
from typing import Optional
import numpy as np
# ===== Tracker and Streamer =====
from cls_rmit_spot_tracker import SpotTracker
from cls_rmit_spot_stream import SpotStreamer
# ===== BostonDynamic APIs =====
from bosdyn.client.math_helpers import SE3Pose, Quat
import bosdyn.client
from bosdyn.api import image_pb2
from bosdyn.api.graph_nav import graph_nav_pb2, nav_pb2, map_pb2
from bosdyn.client.image import build_image_request, ImageClient
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
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

from bosdyn.client.graph_nav import GraphNavClient
from bosdyn.client.recording import GraphNavRecordingServiceClient
from bosdyn.api import manipulation_api_pb2
from bosdyn.api import geometry_pb2
from bosdyn.client.manipulation_api_client import ManipulationApiClient


class SpotAgent:

    # region  Private APIs: Initialisation

    def __init__(
        self,
        *,
        client_name: str = "SpotAgent",
        keep_alive_period_sec: float = 2.0,
        force_lease: bool = True,
    ):
        self.hostname = "192.168.80.3"
        self.username = "user"
        self.password = "myjujz7e2prj"
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
        self.default_hold = 0.9 
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.origin_yaw = 0.0
        self._latest_grid = None
        self._latest_objects = [] 
        self._auto_login(self.username, self.password)
        self._get_lease(force=force_lease)
        
    def __enter__(self):
        time.sleep(1)
        command_client = self.robot.ensure_client(RobotCommandClient.default_service_name)
        self._clear_behavior_faults()
        self._power_on_and_stand()
        streamer = SpotStreamer(self.robot)
        streamer.start()
        
        return self

    def __exit__(self, *args):
        self._shutdown()

    # endregion

    # region  Private APIs: Spot admin
    def _auto_login(self, username: str, password: str):
        self.sdk = bosdyn.client.create_standard_sdk(self.client_name)
        self.robot = self.sdk.create_robot(self.hostname)
        with SpotTracker("Login and Time Sync"):
            self.robot.authenticate(username, password)
            self.robot.time_sync.start()
            sync_success = False
            for i in range(3):
                try:
                    self.robot.time_sync.wait_for_sync(timeout_sec=5.0)
                    sync_success = True
                    break
                except Exception:
                    print(f"[System] ⚠️ 时间同步超时，正在重试 ({i+1}/3)...")
                    time.sleep(1)
        if not sync_success:
            raise RuntimeError("❌ Can not establish time sync with the robot. Please check the connection and try again.")
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
                self.lease_client.take() 
            except Exception as e:
                print(f"[Lease] ⚠️ Fail to get the lease: {e}")
        self._lease_keepalive = LeaseKeepAlive(self.lease_client, must_acquire=True, return_at_exit=True)
    
    @SpotTracker("Power on and Stand", exit_on_fail=True)
    def _power_on_and_stand(self, arm = False):
        if not self.robot.is_powered_on():
            self.robot.power_on(timeout_sec=20)
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
            print("客户端未初始化，无法检查故障。")
            return False
        try:
            state = self.state_client.get_robot_state()
            faults = state.behavior_fault_state.faults
            if not faults:
                print("当前无行为故障，运动系统正常。")
                return True
            print(f"⚠️ 发现 {len(faults)} 个行为故障，正在尝试清除...")
            for fault in faults:
                print(f"  -> 🛑 故障 ID: {fault.behavior_fault_id}")
                print(f"  -> 📝 故障原因: {fault.cause}")
                self.cmd_client.clear_behavior_fault(behavior_fault_id=fault.behavior_fault_id)
                time.sleep(0.5)
            time.sleep(1.0)
            new_state = self.state_client.get_robot_state()
            if not new_state.behavior_fault_state.faults:
                print(" 🎉 所有行为故障已成功清除！")
                return True
            else:
                print(f"❌ 仍有 {len(new_state.behavior_fault_state.faults)} 个故障未能消除！")
                print("💡 提示：某些严重故障（如急停拍下、严重跌倒）无法通过代码清除，请检查机器人本体或使用平板电脑操作。")
                return False
        except Exception as e:
            print(f"[Error] 检查或清除故障时发生异常: {e}")
            return False    

    # endregion
    @SpotTracker("Arm Out", exit_on_fail=False)
    def _arm_out(self):
        if self.cmd_client is None or self.img_client is None:
            raise RuntimeError("cmd_client/img_client 未初始化。")
        try:
            snapshot = self.state_client.get_robot_state().kinematic_state.transforms_snapshot
            root_frame = GRAV_ALIGNED_BODY_FRAME_NAME
            root_T_hand = get_a_tform_b(snapshot, root_frame, HAND_FRAME_NAME)
            delta_hand = math_helpers.SE3Pose(
                x = 0.30,   # 在 X 轴方向（前方）移动 0.3 米
                y = 0.0,    # 左右不偏移
                z = -0.25,  # 在 Z 轴方向向下移动 0.25 米
                rot = math_helpers.Quat.from_pitch(15.0 * math.pi / 180.0) # 向下低头 30 度
            )
            root_T_target = root_T_hand * delta_hand
            q = root_T_target.rot
            arm_cmd = RobotCommandBuilder.arm_pose_command(
                root_T_target.x, root_T_target.y, root_T_target.z,
                q.w, q.x, q.y, q.z, root_frame, 1.2 # 1.2 秒内完成动作
            )
            self.cmd_client.robot_command(arm_cmd)
            self.cmd_client.robot_command(RobotCommandBuilder.claw_gripper_open_command())
            time.sleep(0.4)
            print("[Arm] Arm ready.")
        except Exception as e:
            print(f"[Arm] Arm failed:{e}")
    
    @SpotTracker("Arm In", exit_on_fail=False)
    def _arm_in(self):
        if self.cmd_client is None:
            raise RuntimeError("cmd_client 未初始化。")
        try:
            self.cmd_client.robot_command(RobotCommandBuilder.claw_gripper_close_command())
            stow_cmd = RobotCommandBuilder.arm_stow_command()
            self.cmd_client.robot_command(stow_cmd)
            print("[Arm] Arm stowing...")
        except Exception as e:
            print(f"[Arm] Stow failed: {e}")    

    # endregion
    
    # region Manipulation & Vision Logic
    def find_targetyolo(self, yolo_detector):
        """
        批量获取所有相机图像并进行一次性 YOLO 检测，极大提高推理速度。
        """
        # 1. 定义要抓取的相机列表
        camera_sources = [
            "frontleft_fisheye_image", 
            "frontright_fisheye_image", 
            "left_fisheye_image", 
            "right_fisheye_image", 
            "back_fisheye_image",
            "hand_color_image"
        ]
        
        # 批量构建请求
        image_requests = [build_image_request(src) for src in camera_sources]
        
        try:
            # 一次性获取所有相机的响应
            image_responses = self.img_client.get_image(image_requests)
        except Exception as e:
            print(f"[Error] 获取多相机图像失败: {e}")
            return None

        # 2. 准备 Batch 数据字典
        images_to_detect = {}
        for img_resp in image_responses:
            cam_name = img_resp.source.name
            img_data = np.frombuffer(img_resp.shot.image.data, dtype=np.uint8)
            cv_img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            
            if cv_img is not None:
                images_to_detect[cam_name] = cv_img

        if not images_to_detect:
            return None

        # 3. 调用你提供的 batch 检测函数
        # 这里的 conf 设置为 0.1 左右比较稳妥
        detections = yolo_detector.detect_targets_in_batch(images_to_detect, conf=0.1)

        # 4. 处理结果
        if detections:
            # 因为 detect_targets_in_batch 已经按置信度排过序了
            # 我们直接拿最高的那一个
            top_hit = detections[0]
            print(f"🌟 [Batch Vision] 最佳目标来自相机 [{top_hit['camera']}]: "
                  f"{top_hit['class']} (Conf: {top_hit['conf']:.2f})")
            return top_hit

        return None
    
    def find_and_grasp_target(self, yolo_detector, timeout_sec=60.0):
        """
        使用前方和手部摄像头进行批量扫描识别，找到最佳目标后发起自由姿态抓取，
        并在抓取成功后将机械臂收回到 Carry (持物) 模式。
        """
        print("[Grasp] 📸 正在启动全景扫描寻找可抓取目标...")
        
        # 1. 仅保留前方和手部相机，避免侧面鱼眼畸变导致的定位狂奔
        camera_sources = [
            "frontleft_fisheye_image", 
            "frontright_fisheye_image", 
            "hand_color_image"
        ]
         # 先把手臂伸出来，增加抓取范围和稳定性
        # 批量获取图像
        image_requests = [build_image_request(src) for src in camera_sources]
        try:
            image_responses = self.img_client.get_image(image_requests)
        except Exception as e:
            print(f"[Error] 获取多相机图像失败: {e}")
            return False

        # 2. 准备批量检测数据，并建立映射以便提取指定的 protobuf 响应
        images_to_detect = {}
        resp_map = {} 
        for img_resp in image_responses:
            cam_name = img_resp.source.name
            resp_map[cam_name] = img_resp # 保存原始响应，抓取时需要用到里面的相机内参
            
            img_data = np.frombuffer(img_resp.shot.image.data, dtype=np.uint8)
            cv_img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            if cv_img is not None:
                images_to_detect[cam_name] = cv_img

        if not images_to_detect:
            print("[Grasp] ❌ 图像解码失败！")
            return False

        # 3. 执行批量检测
        print("[Grasp] 🧠 图像获取成功，开始批量 YOLO 识别...")
        detections = yolo_detector.detect_targets_in_batch(images_to_detect, conf=0.1)

        if not detections:
            print("[Grasp] ❌ 未能在当前视野中找到任何目标。")
            return False

        # 4. 提取最佳目标信息
        top_hit = detections[0]
        cam_name = top_hit["camera"]
        cx, cy = top_hit["cx"], top_hit["cy"]
        cls_name = top_hit["class"]
        
        print(f"[Grasp] 🎯 锁定目标: {cls_name}, 位于相机 [{cam_name}], 像素坐标: ({cx}, {cy}), 置信度: {top_hit['conf']:.2f}")

        # 提取目标所在相机的专属 protobuf 响应对象
        target_img_resp = resp_map[cam_name]

        # 5. 构建抓取请求 (已移除所有姿态限制，让机器人自由发挥)
        print("[Grasp] 🦾 正在向机械臂发送自动抓取指令 (自由姿态)...")
        pick_vec = geometry_pb2.Vec2(x=cx, y=cy)
        grasp_request = manipulation_api_pb2.PickObjectInImage(
            pixel_xy=pick_vec,
            transforms_snapshot_for_camera=target_img_resp.shot.transforms_snapshot,
            frame_name_image_sensor=target_img_resp.shot.frame_name_image_sensor,
            camera_model=target_img_resp.source.pinhole
        )
        
        manip_req = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=grasp_request)
        
        # 6. 发送指令并监控状态
        try:
            cmd_response = self.manip_client.manipulation_api_command(manipulation_api_request=manip_req)
            cmd_id = cmd_response.manipulation_cmd_id
            print(f"[Grasp] ✅ 抓取命令已发送，Task ID: {cmd_id}")
            
            start_time = time.time()
            while True:
                if time.time() - start_time > timeout_sec:
                    print(f"[Grasp] ⚠️ 抓取动作超时 ({timeout_sec}秒)，放弃等待。")
                    return False
                    
                feedback_req = manipulation_api_pb2.ManipulationApiFeedbackRequest(manipulation_cmd_id=cmd_id)
                feedback_resp = self.manip_client.manipulation_api_feedback_command(manipulation_api_feedback_request=feedback_req)
                
                state_name = manipulation_api_pb2.ManipulationFeedbackState.Name(feedback_resp.current_state)
                print(f"[Grasp] 🔄 当前状态: {state_name}")
                
                # 抓取成功判定
                if state_name in ['MANIP_STATE_DONE', 'MANIP_STATE_GRASP_SUCCEEDED']:
                    print("[Grasp] 🎉 抓取动作已顺利完成！准备收回机械臂...")
                    
                    try:
                        carry_cmd = RobotCommandBuilder.arm_carry_command()
                        self.cmd_client.robot_command(carry_cmd)
                        print("[Grasp] 🎒 机械臂已切换至 Carry 护送模式！")
                    except Exception as e:
                        print(f"[Grasp] ⚠️ 切换 Carry 模式失败: {e}")
                    
                    return True  
                    
                # ⭐️ 新增：抓取失败判定 (把 NO_SOLUTION 也加进来，防止死循环)
                elif 'FAILED' in state_name or 'NO_SOLUTION' in state_name:
                    print(f"[Grasp] ❌ 抓取终止：机械臂无法完成该动作，最终状态: {state_name}")
                    
                    # 抓取失败后，把手臂收起 (Stow)，避免伸着个胳膊到处跑
                    try:
                        self.cmd_client.robot_command(RobotCommandBuilder.arm_stow_command())
                        print("[Grasp] 🔄 机械臂已自动复位 (Stow)。")
                    except:
                        pass
                        
                    return False 
                
                time.sleep(1.0) 
                
        except Exception as e:
            print(f"[Error] 抓取调用或状态查询发生异常: {e}")
            return False

    # endregion
    
    # ==========================================================
    # 导航核心逻辑: 定位与移动
    # ==========================================================

    def initialize_graphnav_to_fiducial(self, fiducial_id: Optional[int] = None):
        """
        告诉 Spot：“看你眼前的二维码，确定你在地图里的位置！”
        """
        print("[GraphNav] 📍 正在尝试通过 QR 码初始化位置...")
        try:
            # ===== 关键修复：创建一个空的初始猜测对象 =====
            initial_guess = nav_pb2.Localization()
            # ==============================================

            # 1. 设定定位请求
            if fiducial_id is not None:
                # 找特定的码 (比如 101)
                self.graph_nav_client.set_localization(
                    initial_guess_localization=initial_guess,  # <--- 填入这里
                    fiducial_init=graph_nav_pb2.SetLocalizationRequest.FIDUCIAL_INIT_SPECIFIC,
                    use_fiducial_id=fiducial_id
                )
            else:
                # 找视野里最近的码
                self.graph_nav_client.set_localization(
                    initial_guess_localization=initial_guess,  # <--- 填入这里
                    fiducial_init=graph_nav_pb2.SetLocalizationRequest.FIDUCIAL_INIT_NEAREST
                )
            
            # 2. 验证定位是否成功
            state = self.graph_nav_client.get_localization_state()
            if not state.localization.waypoint_id:
                print("[GraphNav] ❌ 定位失败！Spot 没有在视野中找到有效的地图 QR 码。请确保相机正对着码。")
                return False
                
            print(f"[GraphNav] ✅ 定位成功！Spot 认为自己目前在路点: {state.localization.waypoint_id[:6]}... 附近")
            return True

        except Exception as e:
            print(f"[GraphNav] ❌ 初始化位置时发生异常: {e}")
            return False
    def navigate_to_waypoint(self, destination_wp_id: str, timeout_sec: float = 60.0):
        """
        向 Spot 下发自动导航指令，前往指定路点。如果传入了 detector，则在行进中每2秒扫描一次。
        """
        print(f"[GraphNav] 🚀 收到导航指令，目标路点: {destination_wp_id[:6]}...")
        try:
            nav_cmd_id = self.graph_nav_client.navigate_to(
                destination_waypoint_id=destination_wp_id,
                cmd_duration=timeout_sec
            )
            start_time = time.time()
            last_scan_time = time.time() # 记录上一次 YOLO 扫描的时间
            while True:
                current_time = time.time()
                if current_time - start_time > timeout_sec:
                    print(f"[GraphNav] ⚠️ 导航超时 ({timeout_sec}s)，放弃任务。")
                    return False
                feedback = self.graph_nav_client.navigation_feedback(nav_cmd_id)
                status = feedback.status
                if status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_REACHED_GOAL:
                    print("[GraphNav] 🎉 已成功抵达目标路点！")
                    return True
                elif status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_LOST:
                    print("[GraphNav] ❌ 导航失败：Spot 迷路了。")
                    return False
                elif status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_STUCK:
                    print("[GraphNav] ⚠️ 导航受阻：系统正在尝试绕行...")
                time.sleep(0.5)
        except Exception as e:
            print(f"[GraphNav] ❌ 导航过程中发生异常: {e}")
            return False
    
    def get_current_graph(self):
        """
        直接从机器人内存中下载当前的 Graph 拓扑结构 (不保存到本地)
        """
        print("[GraphNav] 📡 正在从机器人大脑读取当前地图...")
        try:
            # 直接调用 API 下载 graph
            graph = self.graph_nav_client.download_graph()
            
            if not graph.waypoints:
                print("[GraphNav] ⚠️ 机器人内存中的地图是空的！(可能还没录制，或者重启被清空了)")
                return None
                
            print(f"[GraphNav] ✅ 成功读取！当前地图包含 {len(graph.waypoints)} 个路点, {len(graph.edges)} 条边。")
            return graph
            
        except Exception as e:
            print(f"[GraphNav] ❌ 读取地图失败: {e}")
            return None
    
    def get_waypoint_id_by_name(self, graph, target_name: str) -> str:
        """
        通过路点的易读名称（如 "waypoint 32"）查找它的内部 ID。
        """
        available_names = []
        for wp in graph.waypoints:
            wp_name = wp.annotations.name
            available_names.append(wp_name)
            if wp_name.lower() == target_name.lower():
                print(f"[GraphNav] 🔍 找到目标 '{target_name}'，对应的 ID 为: {wp.id[:6]}...")
                return wp.id
        print(f"[GraphNav] ❌ 找不到名为 '{target_name}' 的路点！")
        print(f"💡 当前地图中所有可用的路点名称有: {', '.join(available_names)}")
        return None
    
    def upload_graph_and_snapshots(self, save_dir: str):
        """
        将本地保存的地图 (graph, wp_xxx, edge_xxx) 上传到 Spot 的大脑中。
        包含自动剔除庞大图像数据的瘦身逻辑，大幅提升通过 WiFi 的上传速度。
        """
        print(f"[GraphNav] 📂 准备从 '{save_dir}' 上传地图到机器人...")
        self.graph_nav_client.clear_graph()
        graph_path = os.path.join(save_dir, "graph")
        if not os.path.exists(graph_path):
            print(f"[GraphNav] ❌ 找不到地图文件: {graph_path}")
            return None
        with open(graph_path, "rb") as f:
            graph = map_pb2.Graph()
            graph.ParseFromString(f.read())
        print("[GraphNav] ⬆️ 正在上传基础 Graph 结构...")
        self.graph_nav_client.upload_graph(graph=graph, generate_new_anchoring=True)
        print("[GraphNav] ⬆️ 正在上传路点快照 (执行图像瘦身)...")
        for wp in graph.waypoints:
            if wp.snapshot_id:
                wp_path = os.path.join(save_dir, f"wp_{wp.snapshot_id}")
                if os.path.exists(wp_path):
                    with open(wp_path, "rb") as f:
                        snap = map_pb2.WaypointSnapshot()
                        snap.ParseFromString(f.read())
                        for img in snap.images: 
                            img.shot.image.data = b"" 
                        self.graph_nav_client.upload_waypoint_snapshot(snap)
                else:
                    print(f"[GraphNav] ⚠️ 警告: 缺失路点快照文件 {wp_path}")

        print("[GraphNav] ⬆️ 正在上传边缘快照...")
        for edge in graph.edges:
            if edge.snapshot_id:
                edge_path = os.path.join(save_dir, f"edge_{edge.snapshot_id}")
                if os.path.exists(edge_path):
                    with open(edge_path, "rb") as f:
                        snap = map_pb2.EdgeSnapshot()
                        snap.ParseFromString(f.read())
                        self.graph_nav_client.upload_edge_snapshot(snap)
                else:
                    print(f"[GraphNav] ⚠️ 警告: 缺失边缘快照文件 {edge_path}")

        print("[GraphNav] 🎉 地图及所有特征数据上传完毕！机器人的记忆已更新。")
        
        return graph
