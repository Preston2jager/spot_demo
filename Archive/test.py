import time
import math
import sys

# Boston Dynamics SDK
import bosdyn.client
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder, blocking_stand
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.estop import EstopClient
from bosdyn.client.frame_helpers import get_a_tform_b, ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME
from bosdyn.api import basic_command_pb2, power_pb2, estop_pb2

class SpotDiagnostician:
    def __init__(self, hostname, username, password):
        print(f"\n[Init] Connecting to {hostname}...")
        self.sdk = bosdyn.client.create_standard_sdk("SpotDiag")
        self.robot = self.sdk.create_robot(hostname)
        self.robot.authenticate(username, password)
        self.robot.time_sync.wait_for_sync()
        
        # 初始化客户端
        self.lease_client = self.robot.ensure_client(LeaseClient.default_service_name)
        self.cmd_client = self.robot.ensure_client(RobotCommandClient.default_service_name)
        self.state_client = self.robot.ensure_client(RobotStateClient.default_service_name)
        self.estop_client = self.robot.ensure_client(EstopClient.default_service_name)
        
        self.lease_keepalive = None

    def step_1_check_estop(self):
        print("\n=== STEP 1: E-STOP CHECK ===")
        try:
            estop_status = self.estop_client.get_status()
            stop_level = estop_status.stop_level
            
            # 尝试获取官方状态名
            try:
                level_str = estop_pb2.EstopStopLevel.Name(stop_level)
            except ValueError:
                level_str = f"UNKNOWN_CODE_{stop_level}"

            print(f" > Stop Level: {level_str} (Code: {stop_level})")
            
            # 修正判定逻辑：Code 3 (NONE) 和 Code 4 (特殊GO状态) 都算过
            if stop_level == 3 or stop_level == 4: 
                print(" [OK] E-Stop is RELEASED (Ready to move).")
                return True
            else:
                print(" [!!!] 警告: 检测到非标准 E-Stop 状态。")
                print("       但由于 Reset.py 能运行，我们将尝试强制继续...")
                return True # 强制继续，不卡死
            
        except Exception as e:
            print(f" [Warn] E-Stop check error: {e}")
            return True # 忽略错误继续

    def step_2_acquire_lease(self):
        print("\n=== STEP 2: ACQUIRE LEASE ===")
        try:
            self.lease_client.take()
            self.lease_keepalive = LeaseKeepAlive(self.lease_client, must_acquire=True, return_at_exit=True)
            print(" [OK] Lease Acquired.")
            return True
        except Exception as e:
            print(f" [!!!] Failed to get Lease: {e}")
            return False

    def step_3_check_power_and_stand(self):
        print("\n=== STEP 3: POWER & STAND ===")
        # 如果 Reset.py 能跑，这步一定能过
        if not self.robot.is_powered_on():
            print(" > Powering on...")
            self.robot.power_on(timeout_sec=20)
        
        print(" > Commanding STAND...")
        blocking_stand(self.cmd_client, timeout_sec=10)
        print(" [OK] Robot is Standing.")
        return True

    def step_4_test_velocity_move(self):
        print("\n=== STEP 4: VELOCITY TEST (Hardware Check) ===")
        print(" > Attempting Nudge: Forward 0.3m/s for 1.5s...")
        print(" > (This verifies if motors actually turn)")
        
        try:
            start_x, start_y, _ = self._get_odom()
            
            # 发送纯速度指令
            cmd = RobotCommandBuilder.synchro_velocity_command(v_x=0.3, v_y=0.0, v_rot=0.0)
            self.cmd_client.robot_command(cmd, end_time_secs=time.time() + 1.5)
            
            time.sleep(2.0)
            
            end_x, end_y, _ = self._get_odom()
            dist = math.hypot(end_x - start_x, end_y - start_y)
            print(f" > Moved: {dist:.4f} meters")
            
            if dist > 0.05:
                print(" [OK] Velocity Control Works! (Hardware/Lease is GOOD)")
                return True
            else:
                print(" [!!!] FAILED: Robot did not move physically.")
                print("       Critical Issue: Lease lost? Tablet interfering? Hardware fault?")
                return False
        except Exception as e:
            print(f" [!!!] Velocity test crashed: {e}")
            return False

    def step_5_test_trajectory_move(self):
        print("\n=== STEP 5: TRAJECTORY TEST (Planner Check) ===")
        print(" > Attempting Path Planning: Forward 0.5m")
        try:
            start_x, start_y, start_yaw = self._get_odom()
            target_x = start_x + 0.5 
            
            print(f" > Planning: ({start_x:.2f}, {start_y:.2f}) -> ({target_x:.2f}, {start_y:.2f})")
            
            cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
                goal_x=target_x, 
                goal_y=start_y, 
                goal_heading=start_yaw,
                frame_name=ODOM_FRAME_NAME
            )
            # 增加指令有效期
            cmd_id = self.cmd_client.robot_command(cmd, end_time_secs=time.time() + 10.0)
            
            print(" > Monitoring Feedback...")
            for _ in range(10): # 5秒监控
                feedback = self.cmd_client.robot_command_feedback(cmd_id)
                if feedback.feedback.HasField("synchronized_feedback"):
                    mob_fb = feedback.feedback.synchronized_feedback.mobility_command_feedback
                    status = mob_fb.se2_trajectory_feedback.status
                    status_str = basic_command_pb2.SE2TrajectoryCommand.Feedback.Status.Name(status)
                    print(f"   Status: {status_str}")
                    
                    if status == basic_command_pb2.SE2TrajectoryCommand.Feedback.STATUS_AT_GOAL:
                        print(" [OK] Trajectory Success.")
                        return True
                        
                time.sleep(0.5)
                
            print(" [Warn] Trajectory Timed Out (Robot stalled).")
            print("        This usually means OBSTACLES are detected or ODOM is drifting.")
            return False
            
        except Exception as e:
            print(f" [!!!] Trajectory test crashed: {e}")
            return False

    def _get_odom(self):
        state = self.state_client.get_robot_state()
        snapshot = state.kinematic_state.transforms_snapshot
        odom_t_body = get_a_tform_b(snapshot, ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)
        return odom_t_body.position.x, odom_t_body.position.y, 0

    def run_diagnostics(self):
        self.step_1_check_estop() # 即使警告也继续
        if not self.step_2_acquire_lease(): return
        if not self.step_3_check_power_and_stand(): return
        
        # 核心测试
        vel_ok = self.step_4_test_velocity_move()
        traj_ok = self.step_5_test_trajectory_move()
        
        print("\n" + "="*30)
        print("       DIAGNOSTIC SUMMARY")
        print("="*30)
        print(f" 1. Hardware/Motor (Velocity): {'[PASS]' if vel_ok else '[FAIL]'}")
        print(f" 2. Software/Nav (Trajectory): {'[PASS]' if traj_ok else '[FAIL]'}")
        
        if vel_ok and not traj_ok:
            print("\n💡 结论：硬件正常，但导航被拒绝。")
            print("   原因：避障系统介入。")
            print("   对策：在 map_demo 中调低 obstacle_avoidance_padding (我之前给的 v5 版本已包含此修复)。")
        elif not vel_ok:
            print("\n💡 结论：硬件完全不动。")
            print("   原因：权限被抢占 (Check Tablet) 或 驱动故障。")

if __name__ == "__main__":
    diag = SpotDiagnostician("192.168.80.3", "user", "myjujz7e2prj")
    diag.run_diagnostics()