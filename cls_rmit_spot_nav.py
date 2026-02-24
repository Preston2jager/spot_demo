# ===== =================== =====
# ===== RMIT Spot control class =====
# ===== =================== =====
import os
import time
import math
import cv2
import threading
from typing import Optional
import numpy as np

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

# 新增：GraphNav 与 Recording 客户端
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
        self._power_on_and_stand()
        return self

    def __exit__(self, *args):
        self._shutdown()

    # endregion

    # region  Private APIs: Spot admin

    def _auto_login(self, username: str, password: str):
        self.sdk = bosdyn.client.create_standard_sdk(self.client_name)
        self.robot = self.sdk.create_robot(self.hostname)
        self.robot.authenticate(username, password)
        print("[System] 正在与 Spot 进行时间同步...")
        self.robot.time_sync.start()
        sync_success = False
        for i in range(3):
            try:
                self.robot.time_sync.wait_for_sync(timeout_sec=5.0)
                sync_success = True
                print("[System] ✅ 时间同步成功！")
                break
            except Exception:
                print(f"[System] ⚠️ 时间同步超时，正在重试 ({i+1}/3)...")
                time.sleep(1)
        if not sync_success:
            raise RuntimeError("❌ 无法建立时间同步，请确认 Wi-Fi 连接。")
        self.lease_client = self.robot.ensure_client(LeaseClient.default_service_name)
        self.cmd_client = self.robot.ensure_client(RobotCommandClient.default_service_name)
        self.img_client = self.robot.ensure_client(ImageClient.default_service_name)
        self.state_client = self.robot.ensure_client(RobotStateClient.default_service_name)
        self.graph_nav_client = self.robot.ensure_client(GraphNavClient.default_service_name)
        self.recording_client = self.robot.ensure_client(GraphNavRecordingServiceClient.default_service_name)
        self.manip_client = self.robot.ensure_client(ManipulationApiClient.default_service_name)

    def _get_lease(self, force: bool = False):
        if self._lease_keepalive:
            self._lease_keepalive.shutdown()
        self._lease_keepalive = LeaseKeepAlive(self.lease_client, must_acquire=force, return_at_exit=True)

    def _power_on_and_stand(self):
        if not self.robot.is_powered_on():
            self.robot.power_on(timeout_sec=20)
        blocking_stand(self.cmd_client, timeout_sec=10)

    def _shutdown(self):
        if self._lease_keepalive:
            self._lease_keepalive.shutdown()

    # endregion
    
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


    # region  GraphNav & Navigate Logic
    
    def record_square_path(self, side_length: float = 2.0, save_dir: str = "square_map"):
        print(f"[GraphNav] 开始自动化正方形路径录制 (边长: {side_length}m)...")
        self.recording_client.start_recording()
        try:
            for i in range(4):
                print(f"  -> 正在行走第 {i+1} 条边...")
                self._move_relative(side_length, 0.0, 0.0)
                time.sleep(0.5) 
                print(f"  -> 正在左转 90 度...")
                self._move_relative(0.0, 0.0, math.radians(90))
                time.sleep(0.5)
            print("[GraphNav] 正方形路径完成，正在保存地图...")         
        except Exception as e:
            print(f"[Error] 自动行走过程中发生错误: {e}")        
        finally:
            try:
                self.recording_client.stop_recording()
                self._download_and_save_graph(save_dir)
                print(f"[GraphNav] 录制成功完成，地图已保存至: {save_dir}")
            except Exception as e:
                print(f"[Error] 停止录制失败: {e}")

    def upload_graph_and_snapshots(self, save_dir: str):
        print("[GraphNav] 准备上传地图(执行瘦身预处理)...")
        self.graph_nav_client.clear_graph()
        with open(os.path.join(save_dir, "graph"), "rb") as f:
            graph = map_pb2.Graph()
            graph.ParseFromString(f.read())
        self.graph_nav_client.upload_graph(graph=graph, generate_new_anchoring=True, rpc_timeout=15)
        for wp in graph.waypoints:
            if wp.snapshot_id:
                path = os.path.join(save_dir, f"wp_{wp.snapshot_id}")
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        snap = map_pb2.WaypointSnapshot()
                        snap.ParseFromString(f.read())
                        for img in snap.images: img.shot.image.data = b"" # 关键瘦身步骤
                        self.graph_nav_client.upload_waypoint_snapshot(snap, rpc_timeout=10)
        for edge in graph.edges:
            if edge.snapshot_id:
                path = os.path.join(save_dir, f"edge_{edge.snapshot_id}")
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        snap = map_pb2.EdgeSnapshot()
                        snap.ParseFromString(f.read())
                        self.graph_nav_client.upload_edge_snapshot(snap, rpc_timeout=10)
        return graph

    # endregion

    # region  Movement & Save Helpers

    def _download_and_save_graph(self, save_dir):
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        print("[GraphNav] 正在下载并持久化地图...")
        graph = self.graph_nav_client.download_graph()
        with open(os.path.join(save_dir, "graph"), "wb") as f:
            f.write(graph.SerializeToString())
        for wp in graph.waypoints:
            if wp.snapshot_id:
                snap = self.graph_nav_client.download_waypoint_snapshot(wp.snapshot_id)
                with open(os.path.join(save_dir, f"wp_{wp.snapshot_id}"), "wb") as f:
                    f.write(snap.SerializeToString())
        for edge in graph.edges:
            if edge.snapshot_id:
                snap = self.graph_nav_client.download_edge_snapshot(edge.snapshot_id)
                with open(os.path.join(save_dir, f"edge_{edge.snapshot_id}"), "wb") as f:
                    f.write(snap.SerializeToString())
        print(f"[GraphNav] 地图已成功导出至: {save_dir}")


    # endregion
    
    # region Manipulation & Vision Logic


    def find_and_grasp_target(self, yolo_detector, timeout_sec=45.0):
        """
        使用机械臂相机拍照，通过传入的 YOLO 实例进行识别，并自动发起抓取指令。
        该函数是阻塞的，会一直等待抓取动作完成、失败或超时后才返回。
        
        :param yolo_detector: 实例化的 YOLO 检测器
        :param timeout_sec: 抓取动作的最大等待时间（秒），默认 45 秒
        :return: 抓取成功返回 True，未发现目标或抓取失败返回 False
        """
        print("[Grasp] 📸 正在调用 hand_color_image 获取图像...")
        
        # 1. 获取 hand_color_image
        image_request = build_image_request("hand_color_image")
        try:
            image_responses = self.img_client.get_image([image_request])
        except Exception as e:
            print(f"[Error] 获取相机图像失败: {e}")
            return False
            
        if not image_responses:
            print("[Error] 相机返回图像为空！")
            return False
            
        img_resp = image_responses[0]
        
        # 2. 解码 protobuf 图像为 numpy array
        img_data = np.frombuffer(img_resp.shot.image.data, dtype=np.uint8)
        cv_img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        
        if cv_img is None:
            print("[Error] 图像解码失败！")
            return False
            
        print("[Grasp] 🧠 图像获取成功，开始 YOLO 识别...")
        detection = yolo_detector.detect_single_image(cv_img, conf=0.1)
        
        if not detection:
            print("[Grasp] ❌ 未能在当前视野中找到目标。")
            return False
            
        cx, cy = detection["cx"], detection["cy"]
        cls_name = detection["class"]
        print(f"[Grasp] 🎯 发现目标: {cls_name}, 像素坐标: ({cx}, {cy}), 置信度: {detection['conf']:.2f}")
        print("[Grasp] 🦾 正在向机械臂发送抓取指令...")
        
        # 4. 构建 Manipulation API 抓取请求 (PickObjectInImage)
        pick_vec = geometry_pb2.Vec2(x=cx, y=cy)
        grasp_request = manipulation_api_pb2.PickObjectInImage(
            pixel_xy=pick_vec,
            transforms_snapshot_for_camera=img_resp.shot.transforms_snapshot,
            frame_name_image_sensor=img_resp.shot.frame_name_image_sensor,
            camera_model=img_resp.source.pinhole
        )
        
        # =========================================================
        # [新增] 强制机械臂使用“顶部抓取 (Top-Down Grasp)”
        # =========================================================
        # 1. 指定夹爪的 X 轴 (即夹爪伸出的正方向)
        axis_on_gripper = geometry_pb2.Vec3(x=1, y=0, z=0)
        # 2. 指定参考坐标系中，垂直朝下的方向 (Z轴负方向)
        axis_to_align_with = geometry_pb2.Vec3(x=0, y=0, z=-1)
        
        # 3. 添加姿态约束到抓取请求中
        constraint = grasp_request.grasp_params.allowable_orientation.add()
        constraint.vector_alignment_with_tolerance.axis_on_gripper_ewrt_gripper.CopyFrom(axis_on_gripper)
        constraint.vector_alignment_with_tolerance.axis_to_align_with_ewrt_frame.CopyFrom(axis_to_align_with)
        
        # 4. 设置容差: 0.25 弧度 (约 15度)，机械臂可以为了避障稍微倾斜一点点
        constraint.vector_alignment_with_tolerance.threshold_radians = 0.25
        
        # 5. 明确告诉 Spot 这个方向是基于全局的 "vision" 坐标系（非常关键）
        grasp_request.grasp_params.grasp_params_frame_name = "vision"
        # =========================================================
        
        manip_req = manipulation_api_pb2.ManipulationApiRequest(
            pick_object_in_image=grasp_request
        )
        
        try:
            cmd_response = self.manip_client.manipulation_api_command(
                manipulation_api_request=manip_req
            )
            cmd_id = cmd_response.manipulation_cmd_id
            print(f"[Grasp] ✅ 抓取命令已发送 (已开启顶部抓取限制)，Task ID: {cmd_id}")
            
            print("[Grasp] ⏳ 正在等待机械臂完成抓取动作...")
            start_time = time.time()
            while True:
                if time.time() - start_time > timeout_sec:
                    print(f"[Grasp] ⚠️ 抓取动作超时 ({timeout_sec}秒)，放弃等待。")
                    return False
                    
                feedback_req = manipulation_api_pb2.ManipulationApiFeedbackRequest(
                    manipulation_cmd_id=cmd_id
                )
                feedback_resp = self.manip_client.manipulation_api_feedback_command(
                    manipulation_api_feedback_request=feedback_req
                )
                
                state = feedback_resp.current_state 
                # 获取状态的文本名称
                state_name = manipulation_api_pb2.ManipulationFeedbackState.Name(state)
                
                # 【新增】把机器人的实时状态打印出来，方便监控它到底在干嘛
                print(f"[Grasp] 🔄 当前状态: {state_name}")
                
                # --- 使用字符串匹配来判断状态，兼容性最强 ---
                
                # 如果状态是 DONE (完成) 或者 GRASP_SUCCEEDED (抓取成功)
                if state_name in ['MANIP_STATE_DONE', 'MANIP_STATE_GRASP_SUCCEEDED']:
                    print("[Grasp] 🎉 抓取动作已顺利完成！")
                    return True  
                    
                # 如果状态包含 FAILED (失败)
                elif 'FAILED' in state_name:
                    print(f"[Grasp] ❌ 抓取动作失败，最终状态: {state_name}")
                    return False 
                    
                # ------------------------------------------------
                
                time.sleep(1.0) # 把检测频率改成1秒一次，减少刷屏
                
        except Exception as e:
            print(f"[Error] 抓取调用或状态查询发生异常: {e}")
            return False

    # endregion


