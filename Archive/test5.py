import time
import math
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
import sys
import shutil
import threading
import numpy as np

# Boston Dynamics SDK
import bosdyn.client
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder, blocking_stand
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.estop import EstopClient
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.frame_helpers import get_a_tform_b, GRAV_ALIGNED_BODY_FRAME_NAME
from bosdyn.api import basic_command_pb2, image_pb2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2

try:
    from bosdyn.client.frame_helpers import ODOM_FRAME_NAME
except ImportError:
    ODOM_FRAME_NAME = "odom"

class SpotLiveMapper:
    def __init__(self, hostname, username, password):
        print(f"[Init] Connecting to {hostname}...")
        try:
            self.sdk = bosdyn.client.create_standard_sdk("SpotLiveMap")
            self.robot = self.sdk.create_robot(hostname)
            self.robot.authenticate(username, password)
            self.robot.time_sync.wait_for_sync()
        except Exception as e:
            print(f"[Init] Failed: {e}")
            sys.exit(1)
        
        self.lease_client = self.robot.ensure_client(LeaseClient.default_service_name)
        self.cmd_client = self.robot.ensure_client(RobotCommandClient.default_service_name)
        self.state_client = self.robot.ensure_client(RobotStateClient.default_service_name)
        self.estop_client = self.robot.ensure_client(EstopClient.default_service_name)
        self.image_client = self.robot.ensure_client(ImageClient.default_service_name)
        
        self.lease_keepalive = None
        self.output_dir = os.path.abspath("./live_map_data") # 绝对路径更安全
        
        self._collecting = False
        self._collect_thread = None

    def _clear_output_directory(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"[Storage] Output directory ready: {self.output_dir}")

    def depth_image_to_pointcloud(self, image_response):
        try:
            dtype = np.uint16
            img = np.frombuffer(image_response.shot.image.data, dtype=dtype)
            width = image_response.shot.image.cols
            height = image_response.shot.image.rows
            img = img.reshape(height, width)

            source = image_response.source
            intrinsics = source.pinhole.intrinsics
            fx, fy = intrinsics.focal_length.x, intrinsics.focal_length.y
            cx, cy = intrinsics.principal_point.x, intrinsics.principal_point.y

            depth_scale = 1.0 / 1000.0
            depth_img = img.astype(np.float32) * depth_scale
            
            # 降采样 (Step=6)
            step = 6 
            z_vals = depth_img[::step, ::step].flatten()
            v_idx, u_idx = np.indices((height, width))
            v_idx = v_idx[::step, ::step].flatten()
            u_idx = u_idx[::step, ::step].flatten()

            valid = (z_vals > 0.3) & (z_vals < 4.5)
            z_vals = z_vals[valid]
            u_idx = u_idx[valid]
            v_idx = v_idx[valid]

            if len(z_vals) == 0: return None

            x_vals = (u_idx - cx) * z_vals / fx
            y_vals = (v_idx - cy) * z_vals / fy
            points_sensor = np.stack((x_vals, y_vals, z_vals), axis=1)

            snapshot = image_response.shot.transforms_snapshot
            frame_sensor = image_response.shot.frame_name_image_sensor
            odom_t_sensor = get_a_tform_b(snapshot, ODOM_FRAME_NAME, frame_sensor)
            
            ones = np.ones((len(points_sensor), 1))
            points_homo = np.hstack((points_sensor, ones))
            t_matrix = odom_t_sensor.to_matrix()
            points_odom = (t_matrix @ points_homo.T).T
            
            return points_odom[:, :3]
        except Exception:
            return None

    def save_chunk_to_ply(self, points_np, filename):
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(points_np)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("end_header\n")
            for p in points_np:
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")

    def _collection_loop(self, interval):
        sources = ["frontleft_depth", "frontright_depth", "left_depth", "right_depth", "back_depth"]
        while self._collecting:
            start_t = time.time()
            try:
                reqs = [build_image_request(s, pixel_format=image_pb2.Image.PIXEL_FORMAT_DEPTH_U16) for s in sources]
                image_responses = self.image_client.get_image(reqs)
                new_points_batch = []
                for resp in image_responses:
                    pts = self.depth_image_to_pointcloud(resp)
                    if pts is not None:
                        new_points_batch.append(pts)
                
                if new_points_batch:
                    merged_chunk = np.vstack(new_points_batch)
                    # 使用纳秒时间戳作为文件名，确保前端按顺序加载
                    timestamp = time.time_ns()
                    filename = f"cloud_{timestamp}.ply"
                    self.save_chunk_to_ply(merged_chunk, filename)
                    # 静默模式，只打印一个点表示在工作
                    print(".", end="", flush=True)

            except Exception as e:
                print(f"[Error] Collection failed: {e}")
            
            elapsed = time.time() - start_t
            time.sleep(max(0, interval - elapsed))

    def start_background_collection(self, interval=1.0):
        if self._collecting: return
        self._collecting = True
        self._collect_thread = threading.Thread(target=self._collection_loop, args=(interval,), daemon=True)
        self._collect_thread.start()
        print(f"\n[Map] Live streaming to disk STARTED (Interval: {interval}s)...")

    def stop_background_collection(self):
        if self._collecting:
            self._collecting = False
            if self._collect_thread:
                self._collect_thread.join(timeout=3.0)
            print("\n[Map] Live streaming STOPPED.")

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
        
        print(f"\n[Cmd] Plan: ({cx:.2f}, {cy:.2f}) -> ({tx:.2f}, {ty:.2f})")
        params = spot_command_pb2.MobilityParams(
            obstacle_params=spot_command_pb2.ObstacleParams(obstacle_avoidance_padding=0.1)
        )
        cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=tx, goal_y=ty, goal_heading=target_yaw, frame_name=ODOM_FRAME_NAME, params=params
        )
        cmd_id = self.cmd_client.robot_command(cmd, end_time_secs=time.time() + 10.0)

        start_t = time.time()
        timeout = abs(fwd)/0.2 + abs(math.radians(turn_deg))/0.5 + 5.0
        last_status = ""
        while time.time() - start_t < timeout:
            fb = self.cmd_client.robot_command_feedback(cmd_id)
            if fb.feedback.HasField("synchronized_feedback"):
                status_code = fb.feedback.synchronized_feedback.mobility_command_feedback.se2_trajectory_feedback.status
                try: status_str = basic_command_pb2.SE2TrajectoryCommand.Feedback.Status.Name(status_code)
                except: status_str = f"CODE_{status_code}"
                
                if status_str != last_status and status_str != "STATUS_GOING_TO_GOAL":
                    print(f"   [Status] {status_str}")
                    last_status = status_str

                if status_str == "STATUS_AT_GOAL":
                    print("[Done] Goal Reached.")
                    return
                elif status_str in ["STATUS_STALLED", "STATUS_STOPPED", "STATUS_NEAR_GOAL"]:
                    print(f"[Stop] Stopped: {status_str}")
                    return
            time.sleep(0.2)
        print("[Warn] Move timed out.")

    def run(self):
        self._clear_output_directory()
        try:
            self.power_on_and_stand()
            if input("\nStart Live Mapping? (y/n): ").strip().lower() == 'y':
                self.start_background_collection(interval=1.0) # 每秒一次

            print("\n=== COMMANDS: 1(Fwd), 2(Bck), 3(L), 4(R), 9(Quit) ===")
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
            self.stop_background_collection()
            print("Program Finished. Data in './live_map_data'")

if __name__ == "__main__":
    agent = SpotLiveMapper("192.168.80.3", "user", "myjujz7e2prj")
    agent.run()