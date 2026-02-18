import time
import math
import os
import sys
import shutil
import threading
import numpy as np
import cv2

# Boston Dynamics SDK
import bosdyn.client
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder, blocking_stand
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.estop import EstopClient
from bosdyn.client.graph_nav import GraphNavClient
from bosdyn.client.recording import GraphNavRecordingServiceClient
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.frame_helpers import get_a_tform_b, GRAV_ALIGNED_BODY_FRAME_NAME, get_vision_tform_body
from bosdyn.api import basic_command_pb2, image_pb2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2

# 容错导入 ODOM_FRAME_NAME
try:
    from bosdyn.client.frame_helpers import ODOM_FRAME_NAME
except ImportError:
    ODOM_FRAME_NAME = "odom"

class SpotContinuousMapper:
    def __init__(self, hostname, username, password):
        print(f"[Init] Connecting to {hostname}...")
        try:
            self.sdk = bosdyn.client.create_standard_sdk("SpotImageMap")
            self.robot = self.sdk.create_robot(hostname)
            self.robot.authenticate(username, password)
            self.robot.time_sync.wait_for_sync()
        except Exception as e:
            print(f"[Init] Failed: {e}")
            sys.exit(1)
        
        # 初始化客户端
        self.lease_client = self.robot.ensure_client(LeaseClient.default_service_name)
        self.cmd_client = self.robot.ensure_client(RobotCommandClient.default_service_name)
        self.state_client = self.robot.ensure_client(RobotStateClient.default_service_name)
        self.estop_client = self.robot.ensure_client(EstopClient.default_service_name)
        self.rec_client = self.robot.ensure_client(GraphNavRecordingServiceClient.default_service_name)
        self.graph_client = self.robot.ensure_client(GraphNavClient.default_service_name)
        self.image_client = self.robot.ensure_client(ImageClient.default_service_name) # 关键替换
        
        self.lease_keepalive = None
        self.output_dir = "./continuous_map_data"
        
        # 建图相关
        self._collecting = False
        self._collect_thread = None
        self.global_point_cloud = [] 
        self.points_lock = threading.Lock()

    def _clear_output_directory(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)

    # --- 核心算法：深度图 -> 3D点云 ---
    def depth_image_to_pointcloud(self, image_response):
        """
        将 Spot 的深度图响应解算为 Odom 坐标系下的 XYZ 点云
        """
        try:
            # 1. 解析图像数据 (Uint16 -> Millimeters)
            dtype = np.uint16
            img = np.frombuffer(image_response.shot.image.data, dtype=dtype)
            width = image_response.shot.image.cols
            height = image_response.shot.image.rows
            img = img.reshape(height, width)

            # 2. 获取相机内参
            source = image_response.source
            intrinsics = source.pinhole.intrinsics
            fx = intrinsics.focal_length.x
            fy = intrinsics.focal_length.y
            cx = intrinsics.principal_point.x
            cy = intrinsics.principal_point.y

            # 3. 深度过滤 (Spot 有效深度 0.3m - 4.0m)
            # 缩放因子：Depth scale (1/1000 to meters)
            depth_scale = 1.0 / 1000.0
            depth_img = img.astype(np.float32) * depth_scale
            
            # 创建网格
            # 稀疏化：这里直接在像素层级做降采样 (每4个点取1个)，极大提升速度
            step = 6 
            v_idx, u_idx = np.indices((height, width))
            v_idx = v_idx[::step, ::step].flatten()
            u_idx = u_idx[::step, ::step].flatten()
            z_vals = depth_img[::step, ::step].flatten()

            # 过滤无效深度
            valid = (z_vals > 0.3) & (z_vals < 4.5)
            z_vals = z_vals[valid]
            u_idx = u_idx[valid]
            v_idx = v_idx[valid]

            if len(z_vals) == 0: return None

            # 4. 反投影 (Reprojection) -> 相机坐标系
            x_vals = (u_idx - cx) * z_vals / fx
            y_vals = (v_idx - cy) * z_vals / fy
            # Spot 相机系：Z向前，X向右，Y向下 (通常)
            # 组成 (N, 3)
            points_sensor = np.stack((x_vals, y_vals, z_vals), axis=1)

            # 5. 坐标变换：Sensor -> Odom
            # 获取 Sensor 在那一刻相对于 Odom 的位姿
            snapshot = image_response.shot.transforms_snapshot
            frame_sensor = image_response.shot.frame_name_image_sensor
            odom_t_sensor = get_a_tform_b(snapshot, ODOM_FRAME_NAME, frame_sensor)
            
            # 批量变换
            ones = np.ones((len(points_sensor), 1))
            points_homo = np.hstack((points_sensor, ones))
            t_matrix = odom_t_sensor.to_matrix()
            points_odom = (t_matrix @ points_homo.T).T
            
            return points_odom[:, :3] # 返回 XYZ

        except Exception as e:
            print(f"Reprojection error: {e}")
            return None

    def _collection_loop(self, interval):
        # 使用深度源 (不含 hand，因为 hand 需要额外展开)
        sources = [
            "frontleft_depth", "frontright_depth",
            "left_depth", "right_depth", "back_depth"
        ]
        
        while self._collecting:
            start_t = time.time()
            try:
                # 请求深度图 (Format: RAW, Pixel: DepthU16)
                reqs = [build_image_request(s, pixel_format=image_pb2.Image.PIXEL_FORMAT_DEPTH_U16) for s in sources]
                image_responses = self.image_client.get_image(reqs)
                
                new_points_batch = []
                
                for resp in image_responses:
                    pts = self.depth_image_to_pointcloud(resp)
                    if pts is not None:
                        new_points_batch.append(pts)
                
                if new_points_batch:
                    merged = np.vstack(new_points_batch)
                    with self.points_lock:
                        self.global_point_cloud.extend(merged.tolist())
                    print(f"[Scan] +{len(merged)} pts (Total: {len(self.global_point_cloud)})")

            except Exception as e:
                print(f"[Scan] Capture error: {e}")
            
            elapsed = time.time() - start_t
            time.sleep(max(0, interval - elapsed))

    def start_background_collection(self, interval=2.0):
        if self._collecting: return
        self._collecting = True
        self._collect_thread = threading.Thread(target=self._collection_loop, args=(interval,), daemon=True)
        self._collect_thread.start()
        print(f"[Scan] Background Image-to-Cloud started (Interval: {interval}s)")

    def stop_background_collection(self):
        if self._collecting:
            self._collecting = False
            if self._collect_thread:
                self._collect_thread.join(timeout=3.0)

    def save_final_merged_map(self):
        self.stop_background_collection()
        if not self.global_point_cloud:
            print("[Scan] No points collected.")
            return

        filename = os.path.join(self.output_dir, "final_merged.ply")
        print(f"[Scan] Saving {len(self.global_point_cloud)} points to {filename}...")
        
        with open(filename, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(self.global_point_cloud)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            for p in self.global_point_cloud:
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
        print("[Scan] Save Complete.")

    # --- 控制部分 (保持不变) ---
    def check_estop(self):
        try: return self.estop_client.get_status().stop_level in [3, 4]
        except: return True

    def acquire_lease(self):
        try:
            self.lease_client.take()
            self.lease_keepalive = LeaseKeepAlive(self.lease_client, must_acquire=True, return_at_exit=True)
            return True
        except: return False

    def power_on_and_stand(self):
        if not self.check_estop(): return
        if not self.acquire_lease(): return
        if not self.robot.is_powered_on(): self.robot.power_on(timeout_sec=20)
        blocking_stand(self.cmd_client, timeout_sec=10)

    def _get_odom_pose(self):
        state = self.state_client.get_robot_state()
        snapshot = state.kinematic_state.transforms_snapshot
        odom_t_body = get_a_tform_b(snapshot, ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)
        q = odom_t_body.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return odom_t_body.position.x, odom_t_body.position.y, yaw

    def move_relative(self, fwd=0.0, turn_deg=0.0):
        cx, cy, cyaw = self._get_odom_pose()
        target_yaw = cyaw + math.radians(turn_deg)
        tx, ty = cx + fwd * math.cos(cyaw), cy + fwd * math.sin(cyaw)
        
        print(f"[Cmd] Plan: ({tx:.2f}, {ty:.2f})")
        params = spot_command_pb2.MobilityParams(obstacle_params=spot_command_pb2.ObstacleParams(obstacle_avoidance_padding=0.1))
        cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(goal_x=tx, goal_y=ty, goal_heading=target_yaw, frame_name=ODOM_FRAME_NAME, params=params)
        cmd_id = self.cmd_client.robot_command(cmd, end_time_secs=time.time() + 10.0)

        start_t = time.time()
        timeout = abs(fwd)/0.2 + abs(math.radians(turn_deg))/0.5 + 5.0
        while time.time() - start_t < timeout:
            fb = self.cmd_client.robot_command_feedback(cmd_id)
            if fb.feedback.HasField("synchronized_feedback"):
                status_code = fb.feedback.synchronized_feedback.mobility_command_feedback.se2_trajectory_feedback.status
                try: status_str = basic_command_pb2.SE2TrajectoryCommand.Feedback.Status.Name(status_code)
                except: status_str = str(status_code)
                if status_str == "STATUS_AT_GOAL":
                    print("[Done] Reached.")
                    return
            time.sleep(0.2)

    def run(self):
        self._clear_output_directory()
        try:
            self.power_on_and_stand()
            if input("\nStart Continuous Mapping? (y/n): ").strip().lower() == 'y':
                self.start_background_collection(interval=1.5)

            print("\n=== COMMANDS: 1(Fwd), 2(Bck), 3(L), 4(R), 9(Save&Quit) ===")
            while True:
                raw = input(">> ").strip()
                if not raw: continue
                parts = raw.split()
                code = parts[0]
                if code == '9': break
                if code == '5': self.power_on_and_stand()
                if code == '6': self.cmd_client.robot_command(RobotCommandBuilder.synchro_sit_command())
                if len(parts) >= 2:
                    val = float(parts[1])
                    if code == '1': self.move_relative(fwd=val)
                    elif code == '2': self.move_relative(fwd=-val)
                    elif code == '3': self.move_relative(turn_deg=val)
                    elif code == '4': self.move_relative(turn_deg=-val)
        finally:
            self.save_final_merged_map()
            print("Program Finished.")

if __name__ == "__main__":
    agent = SpotContinuousMapper("192.168.80.3", "user", "myjujz7e2prj")
    agent.run()