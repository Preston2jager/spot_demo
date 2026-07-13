import time
import math
import threading
import bosdyn.client
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder, blocking_stand, blocking_sit
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.image import ImageClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.api import manipulation_api_pb2
from bosdyn.api import image_pb2 
from bosdyn.client.frame_helpers import get_a_tform_b, ODOM_FRAME_NAME, BODY_FRAME_NAME
from bosdyn.client.math_helpers import SE3Pose, SE2Pose
import cv2
import numpy as np

class SpotAgent:
    def __init__(self, hostname: str, username: str, password: str, force_lease: bool = True):
        self.robot = None
        self.lease_client = None
        self.cmd_client = None
        self._lease_keepalive = None
        
        self._streaming = False
        self._stream_thread = None
        
        try:
            sdk = bosdyn.client.create_standard_sdk("SpotAgent")
            self.robot = sdk.create_robot(hostname)
            self.robot.authenticate(username, password)
            self.robot.time_sync.wait_for_sync() 
            
            self.lease_client = self.robot.ensure_client(LeaseClient.default_service_name)
            self.cmd_client = self.robot.ensure_client(RobotCommandClient.default_service_name)
            self.robot.ensure_client(RobotStateClient.default_service_name)
            if force_lease:
                self.lease_client.take()
            else:
                self.lease_client.acquire()
            self._lease_keepalive = LeaseKeepAlive(self.lease_client, return_at_exit=True)
            print("[Init] Success.")
        except Exception as e:
            print(f"[Init Err] {e}")

    def get_ready(self):
        print("[Cmd] Power On & Stand...")
        try:
            if not self.robot.is_powered_on():
                self.robot.power_on()
            blocking_stand(self.cmd_client, timeout_sec=10.0)
            stow = RobotCommandBuilder.arm_stow_command()
            self.cmd_client.robot_command(stow)
        except Exception as e:
            print(f"[Ready Err] {e}")

    def rest_down(self):
        print("[Cmd] Sit & Power Off...")
        self.stop_stream()
        try:
            blocking_sit(self.cmd_client, timeout_sec=10.0)
            self.robot.power_off(cut_immediately=False)
        except Exception as e:
            print(f"[Sit Err] {e}")

    # --- 监控逻辑 ---
    def start_stream(self):
        if self._streaming:
            print("⚠️ 监控已经在运行中。")
            return
        print("[Stream] 正在启动 5路 摄像头监控窗口...")
        self._streaming = True
        self._stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._stream_thread.start()

    def stop_stream(self):
        if self._streaming:
            print("[Stream] 正在关闭监控...")
            self._streaming = False
            if self._stream_thread:
                self._stream_thread.join(timeout=2.0)
            cv2.destroyAllWindows()

    def _stream_loop(self):
        image_client = self.robot.ensure_client(ImageClient.default_service_name)
        sources = [
            'left_fisheye_image', 'frontleft_fisheye_image', 'frontright_fisheye_image', 'right_fisheye_image', 
            'back_fisheye_image'
        ]
        W, H = 320, 240 

        while self._streaming:
            try:
                responses = image_client.get_image_from_sources(sources)
                img_map = {}
                for res in responses:
                    if res.status == res.STATUS_OK:
                        arr = np.frombuffer(res.shot.image.data, dtype=np.uint8)
                        decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        img_map[res.source.name] = cv2.resize(decoded, (W, H))
                    else:
                        img_map[res.source.name] = np.zeros((H, W, 3), dtype=np.uint8)

                empty_block = np.zeros((H, W, 3), dtype=np.uint8)
                row1 = np.hstack([
                    img_map.get('left_fisheye_image', empty_block),
                    img_map.get('frontleft_fisheye_image', empty_block),
                    img_map.get('frontright_fisheye_image', empty_block)
                ])
                row2 = np.hstack([
                    img_map.get('right_fisheye_image', empty_block),
                    img_map.get('back_fisheye_image', empty_block),
                    empty_block 
                ])
                grid = np.vstack([row1, row2])
                
                cv2.imshow("Spot 360 View (Press 'cam' in terminal to stop)", grid)
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    self._streaming = False
                    break
            except Exception as e:
                print(f"[Stream Err] {e}")
                time.sleep(1.0)
        cv2.destroyAllWindows()

    # --- 移动逻辑 ---
    def move_relative(self, fwd=0.0, left=0.0, rot_deg=0.0):
        LINEAR_SPEED = 0.5  
        ANGULAR_SPEED = 0.8 
        duration = 0.0
        v_x, v_y, v_rot = 0.0, 0.0, 0.0

        if fwd != 0:
            duration = abs(fwd) / LINEAR_SPEED
            v_x = LINEAR_SPEED if fwd > 0 else -LINEAR_SPEED
        elif left != 0:
            duration = abs(left) / LINEAR_SPEED
            v_y = LINEAR_SPEED if left > 0 else -LINEAR_SPEED
        elif rot_deg != 0:
            rot_rad = math.radians(rot_deg)
            duration = abs(rot_rad) / ANGULAR_SPEED
            v_rot = ANGULAR_SPEED if rot_rad > 0 else -ANGULAR_SPEED

        print(f"[Move] Activating motors for {duration:.2f}s...")
        start_time = time.time()
        while (time.time() - start_time) < duration:
            cmd = RobotCommandBuilder.synchro_velocity_command(v_x=v_x, v_y=v_y, v_rot=v_rot)
            self.cmd_client.robot_command(
                cmd, 
                end_time_secs=time.time() + 0.6,
                timesync_endpoint=self.robot.time_sync.endpoint
            )
            time.sleep(0.1)

        stop_cmd = RobotCommandBuilder.stop_command()
        self.cmd_client.robot_command(
            stop_cmd,
            end_time_secs=time.time() + 0.6,
            timesync_endpoint=self.robot.time_sync.endpoint
        )
        print("[Move] Done.")

    def record_location(self) -> dict:
        try:
            state_client = self.robot.ensure_client(RobotStateClient.default_service_name)
            robot_state = state_client.get_robot_state()
            odom_tform_body = get_a_tform_b(
                robot_state.kinematic_state.transforms_snapshot,
                ODOM_FRAME_NAME,
                BODY_FRAME_NAME
            )
            if odom_tform_body:
                pose = SE3Pose.from_obj(odom_tform_body)
                return {"x": pose.x, "y": pose.y, "z": pose.z, "yaw": pose.rot.to_yaw()}
            return None
        except Exception:
            return None

    def search_once(self) -> dict:
        image_client = self.robot.ensure_client(ImageClient.default_service_name)
        sources = ['frontleft_fisheye_image', 'frontright_fisheye_image', 
                   'left_fisheye_image', 'right_fisheye_image', 'back_fisheye_image']
        responses = image_client.get_image_from_sources(sources)
        result = {}
        for res in responses:
            if res.status == res.STATUS_OK:
                img = np.frombuffer(res.shot.image.data, dtype=np.uint8)
                result[res.source.name] = {
                    "cv2_img": cv2.imdecode(img, cv2.IMREAD_COLOR),
                    "raw_response": res
                }
        return result

    # --- 修复后的 grasp_target ---
    def grasp_target(self, raw_image_response, pixel_x: int, pixel_y: int):
        """
        参考 cls_spot_demo 逻辑：使用带反馈的抓取流程，并在成功后进入 Carry 姿态
        """
        manip_client = self.robot.ensure_client(ManipulationApiClient.default_service_name)
        
        # 提取相机模型
        image = raw_image_response
        cam_model = getattr(image.source, "pinhole", None) or \
                    getattr(image.source, "fisheye", None) or \
                    image.source.pinhole

        # 构建更精确的 PickObjectInImage 请求
        pick = manipulation_api_pb2.PickObjectInImage(
            pixel_xy=geometry_pb2.Vec2(x=int(pixel_x), y=int(pixel_y)),
            transforms_snapshot_for_camera=image.shot.transforms_snapshot,
            frame_name_image_sensor=image.shot.frame_name_image_sensor,
            camera_model=cam_model,
        )
        
        print(f"[grasp] 发送抓取指令: ({pixel_x}, {pixel_y})")
        req = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=pick)
        rsp = manip_client.manipulation_api_command(manipulation_api_request=req)

        # 轮询抓取状态反馈
        deadline = time.time() + 30.0
        succeeded = False
        while time.time() < deadline:
            fb = manip_client.manipulation_api_feedback_command(
                manipulation_api_pb2.ManipulationApiFeedbackRequest(
                    manipulation_cmd_id=rsp.manipulation_cmd_id
                )
            )
            state = fb.current_state
            if state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED:
                succeeded = True
                break
            elif state == manipulation_api_pb2.MANIP_STATE_GRASP_FAILED:
                succeeded = False
                break
            time.sleep(0.25)

        if succeeded:
            print("[grasp] 抓取成功，进入搬运姿态 (Carry)")
            # 抓取成功后务必抬起手臂，防止瓶子磕碰地面
            cid = self.cmd_client.robot_command(RobotCommandBuilder.arm_carry_command())
            block_until_arm_arrives(self.cmd_client, cid, timeout_sec=6.0)
        else:
            print("[grasp] 抓取失败或超时")
        
        return succeeded

    def return_and_drop(self):
        """
        导航回起点 (0,0,0) 并放下目标
        """
        print("[return] 开始返航至起点 (0,0,0)")
        
        # 1. 导航回起点
        zero_pose = SE2Pose(0.0, 0.0, 0.0)
        go_home_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=zero_pose.x,
            goal_y=zero_pose.y,
            goal_heading=zero_pose.angle,
            frame_name=ODOM_FRAME_NAME
        )
        self.cmd_client.robot_command(go_home_cmd)
        time.sleep(12.0)  # 根据距离调整等待时间

        # 2. 放置动作：参考 cls_spot_demo 将手臂下放到地面高度
        print("[drop] 将物体放置到地面")
        # 这里调用之前定义的 arm_place_down_at_body 逻辑（如果存在）或者直接移动 arm
        try:
            # 简单的下放动作
            stow_cmd = RobotCommandBuilder.arm_stow_command() # 如果没有专门的place动作，先stow也会触发释放位置
            # 或者使用更稳妥的释放位姿
            time.sleep(1.0)
        except Exception as e:
            print(f"[drop] 放置位姿异常: {e}")

        # 3. 释放抓手
        print("[drop] 释放抓手")
        self.cmd_client.robot_command(RobotCommandBuilder.claw_gripper_open_command())
        time.sleep(2.0)
        
        # 4. 手臂收回
        self.cmd_client.robot_comm