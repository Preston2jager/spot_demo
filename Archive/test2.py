import time
import math
import os
import sys
import threading

# Boston Dynamics SDK
import bosdyn.client
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder, blocking_stand
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.estop import EstopClient
from bosdyn.client.graph_nav import GraphNavClient
from bosdyn.client.recording import GraphNavRecordingServiceClient
from bosdyn.client.frame_helpers import get_a_tform_b, GRAV_ALIGNED_BODY_FRAME_NAME
from bosdyn.api import basic_command_pb2, estop_pb2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2

# 容错导入 ODOM_FRAME_NAME
try:
    from bosdyn.client.frame_helpers import ODOM_FRAME_NAME
except ImportError:
    ODOM_FRAME_NAME = "odom"

class SpotMapCommander:
    def __init__(self, hostname, username, password):
        print(f"[Init] Connecting to {hostname}...")
        try:
            self.sdk = bosdyn.client.create_standard_sdk("SpotMapCmd")
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
        
        self.lease_keepalive = None
        self._recording = False

    def check_estop(self):
        """检查急停，允许状态码 4"""
        try:
            status = self.estop_client.get_status()
            code = status.stop_level
            if code == 3 or code == 4: # NONE or SPECIAL_GO
                return True
            print(f"[!!!] E-Stop Active (Code {code}). Please Release via Tablet.")
            return False
        except:
            return True # 忽略查询错误以防万一

    def acquire_lease(self):
        try:
            self.lease_client.take()
            self.lease_keepalive = LeaseKeepAlive(self.lease_client, must_acquire=True, return_at_exit=True)
            return True
        except:
            print("[!!!] Failed to acquire Lease. (Check tablet connection)")
            return False

    def power_on_and_stand(self):
        if not self.check_estop(): return
        if not self.acquire_lease(): return

        if not self.robot.is_powered_on():
            print("[Action] Powering on...")
            self.robot.power_on(timeout_sec=20)
        
        print("[Action] Standing up...")
        blocking_stand(self.cmd_client, timeout_sec=10)

    def start_recording(self):
        try:
            print("[Map] Starting recording service...")
            try: self.rec_client.stop_recording()
            except: pass 
            
            self.rec_client.start_recording()
            self._recording = True
            print("[Map] Recording STARTED (Lidar/Vision Fusion Active).")
        except Exception as e:
            print(f"[Map] Start failed: {e}")

    def stop_and_download_map(self):
        if not self._recording: return
        print("\n[Map] Stopping recording...")
        try: self.rec_client.stop_recording()
        except: pass
        
        path = "./final_map_data"
        if not os.path.exists(path): os.makedirs(path)
        
        print(f"[Map] Downloading map to '{path}'...")
        try:
            graph = self.graph_client.download_graph()
            with open(os.path.join(path, "graph"), "wb") as f:
                f.write(graph.SerializeToString())
            
            print(f" > Downloading {len(graph.waypoints)} waypoints...")
            for wp in graph.waypoints:
                d = self.graph_client.download_waypoint_snapshot(wp.snapshot_id)
                with open(os.path.join(path, f"{wp.id}"), "wb") as f:
                    f.write(d.SerializeToString())
            
            print(f" > Downloading {len(graph.edges)} edges...")
            for edge in graph.edges:
                d = self.graph_client.download_edge_snapshot(edge.snapshot_id)
                with open(os.path.join(path, f"{edge.id}"), "wb") as f:
                    f.write(d.SerializeToString())
            
            print("[Map] Download Complete.")
        except Exception as e:
            print(f"[Map] Save failed: {e}")

    def _get_odom_pose(self):
        state = self.state_client.get_robot_state()
        snapshot = state.kinematic_state.transforms_snapshot
        odom_t_body = get_a_tform_b(snapshot, ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)
        x = odom_t_body.position.x
        y = odom_t_body.position.y
        q = odom_t_body.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return x, y, yaw

    def move_relative(self, fwd=0.0, turn_deg=0.0):
        # 1. 计算目标点
        cur_x, cur_y, cur_yaw = self._get_odom_pose()
        turn_rad = math.radians(turn_deg)
        target_yaw = cur_yaw + turn_rad
        
        delta_x = fwd * math.cos(cur_yaw)
        delta_y = fwd * math.sin(cur_yaw)
        
        target_x = cur_x + delta_x
        target_y = cur_y + delta_y
        
        print(f"[Cmd] Plan: ({cur_x:.2f}, {cur_y:.2f}) -> ({target_x:.2f}, {target_y:.2f})")

        # 2. 注入避障参数 (Padding=0.1m)
        mobility_params = spot_command_pb2.MobilityParams(
            obstacle_params=spot_command_pb2.ObstacleParams(
                obstacle_avoidance_padding=0.1 
            )
        )

        # 3. 发送指令
        cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=target_x,
            goal_y=target_y,
            goal_heading=target_yaw,
            frame_name=ODOM_FRAME_NAME,
            params=mobility_params
        )
        
        cmd_id = self.cmd_client.robot_command(cmd, end_time_secs=time.time() + 10.0)

        # 4. 监控执行 (安全版)
        timeout = abs(fwd)/0.2 + abs(turn_rad)/0.5 + 5.0
        start_t = time.time()
        last_status = ""
        
        while time.time() - start_t < timeout:
            feedback = self.cmd_client.robot_command_feedback(cmd_id)
            if feedback.feedback.HasField("synchronized_feedback"):
                mob_fb = feedback.feedback.synchronized_feedback.mobility_command_feedback
                status_code = mob_fb.se2_trajectory_feedback.status
                
                # === 关键修改：使用字符串转换，避免 AttributeError ===
                try:
                    status_str = basic_command_pb2.SE2TrajectoryCommand.Feedback.Status.Name(status_code)
                except ValueError:
                    status_str = f"UNKNOWN_CODE_{status_code}"
                
                # 打印状态变化
                if status_str != last_status:
                    # 过滤掉刷屏的 GOING_TO_GOAL
                    if status_str != "STATUS_GOING_TO_GOAL":
                        print(f"   [Status] {status_str}")
                    last_status = status_str

                # 判断逻辑
                if status_str == "STATUS_AT_GOAL":
                    print("[Done] Goal Reached.")
                    return
                elif status_str in ["STATUS_STALLED", "STATUS_STOPPED", "STATUS_NEAR_GOAL"]:
                     # 遇到障碍物卡住，不算程序崩溃，只是任务提前结束
                     print(f"[Stop] Movement stopped: {status_str} (Obstacle or limit reached)")
                     return
            
            time.sleep(0.2)
        
        print("[Warn] Action timed out.")

    def run(self):
        try:
            self.power_on_and_stand()
            
            if input("\nStart Recording Map? (y/n): ").strip().lower() == 'y':
                self.start_recording()

            print("\n" + "="*40)
            print("      SPOT COMMANDER (V2 Stable)")
            print("="*40)
            print(" 1 <dist> : Move Forward  (e.g. 1 1.0)")
            print(" 2 <dist> : Move Backward (e.g. 2 0.5)")
            print(" 3 <deg>  : Turn Left     (e.g. 3 90)")
            print(" 4 <deg>  : Turn Right    (e.g. 4 90)")
            print(" 5        : Stand Up (Reset)")
            print(" 6        : Sit Down")
            print(" 9        : Save Map & Quit")
            print("="*40)

            while True:
                raw = input(">> ").strip()
                if not raw: continue
                parts = raw.split()
                code = parts[0]

                if code == '9': break
                
                if code == '5':
                    self.power_on_and_stand()
                    continue
                if code == '6':
                    self.cmd_client.robot_command(RobotCommandBuilder.synchro_sit_command())
                    continue

                if len(parts) >= 2:
                    try:
                        val = float(parts[1])
                        if code == '1': self.move_relative(fwd=val)
                        elif code == '2': self.move_relative(fwd=-val)
                        elif code == '3': self.move_relative(turn_deg=val)
                        elif code == '4': self.move_relative(turn_deg=-val)
                    except ValueError:
                        print("Invalid number.")
                else:
                    if code in ['1','2','3','4']:
                        print("Need value (e.g. '1 0.5')")

        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"\nCRITICAL ERROR: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop_and_download_map()
            print("Program Finished.")

if __name__ == "__main__":
    agent = SpotMapCommander("192.168.80.3", "user", "myjujz7e2prj")
    agent.run()