# ===== =================== =====
# ===== RMIT Spot control class =====
# ===== =================== =====
import time
import math
import cv2
import threading
import sys
import select
from typing import Optional
import numpy as np
from flask import Flask, Response, render_template_string
# ===== BostonDynamic APIs =====
import bosdyn.client
from bosdyn.api import image_pb2
from bosdyn.client.image import build_image_request, ImageClient
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder, blocking_stand
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn.client.frame_helpers import (
    ODOM_FRAME_NAME,     
    VISION_FRAME_NAME,
    BODY_FRAME_NAME,   
    get_a_tform_b
)

class SpotAgent:
    def __init__(
        self,
        hostname: str,
        username: str,
        password: str,
        *,
        client_name: str = "SpotAgent",
        keep_alive_period_sec: float = 2.0,
        force_lease: bool = True,
    ):
        self.hostname = hostname
        self.client_name = client_name
        self.keep_alive_period_sec = keep_alive_period_sec
        self.sdk: Optional[bosdyn.client.Sdk] = None
        self.robot: Optional[bosdyn.client.Robot] = None
        self.lease_client: Optional[LeaseClient] = None
        self.cmd_client: Optional[RobotCommandClient] = None
        self.img_client: Optional[ImageClient] = None
        self.state_client: Optional[RobotStateClient] = None
        self._lease_keepalive: Optional[LeaseKeepAlive] = None
        self.default_hold = 0.9 
        self._auto_login(username, password)
        self._get_lease(force=force_lease)
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.origin_yaw = 0.0
        self.home_x = 0.0
        self.home_y = 0.0
        self.home_yaw = 0.0
        self.guard_x = 0.0
        self.guard_y = 0.0
        self.guard_yaw = 0.0
        self._latest_grid = None
        self._streaming = True
        # Note: start streaming at init.
        threading.Thread(target=self._stream_loop, daemon=True).start()
        self._start_web_server(host="0.0.0.0", port=5555)

    # region  Private APIs: Spot admin

    def _auto_login(
            self, 
            username: str, 
            password: str
        ):
        self.sdk = bosdyn.client.create_standard_sdk(self.client_name)
        self.robot = self.sdk.create_robot(self.hostname)
        self.robot.authenticate(username, password)
        try:
            self.robot.time_sync.wait_for_sync()
        except Exception:
            self.robot.time_sync.start()
            self.robot.time_sync.wait_for_sync()
        self.lease_client = self.robot.ensure_client(LeaseClient.default_service_name)
        self.cmd_client = self.robot.ensure_client(RobotCommandClient.default_service_name)
        self.img_client = self.robot.ensure_client(ImageClient.default_service_name)
        self.state_client = self.robot.ensure_client(RobotStateClient.default_service_name)

    def _make_keepalive(
            self, 
            *, 
            must_acquire: bool, 
            return_at_exit: bool
        ) -> LeaseKeepAlive:
        try:
            return LeaseKeepAlive(
                self.lease_client,
                must_acquire=must_acquire,
                return_at_exit=return_at_exit,
                period_sec=self.keep_alive_period_sec,
            )
        except TypeError:
            return LeaseKeepAlive(
                self.lease_client,
                must_acquire=must_acquire,
                return_at_exit=return_at_exit,
            )

    def _get_lease(
            self, 
            force: bool = False
        ) -> LeaseKeepAlive:
        if self.lease_client is None:
            raise RuntimeError("lease_client not yet initialised.")
        if self._lease_keepalive is not None:
            try:
                self._lease_keepalive.shutdown()
            except Exception:
                pass
            self._lease_keepalive = None
        if force:
            try:
                self.lease_client.take()
            except Exception:
                self.lease_client.acquire()
            self._lease_keepalive = self._make_keepalive(must_acquire=False, return_at_exit=True)
        else:
            self._lease_keepalive = self._make_keepalive(must_acquire=True, return_at_exit=True)
        return self._lease_keepalive
    
    # endregion

    # region  Private APIs: Calulation tools

    @staticmethod
    def _yaw_from_quat(q) -> float:
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                          1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    
    # endregion

    # region  Public APIs: Basic actions

    def power_on_and_stand(
            self, 
            timeout_sec: float = 20.0, 
            stand_timeout_sec: float = 10.0
        ):
        if self.robot is None or self.cmd_client is None:
            raise RuntimeError("Require login")
        if not self.robot.is_powered_on():
            print("[robot] Power up...")
            self.robot.power_on(timeout_sec=timeout_sec)
        print("[robot] Standing up...")
        blocking_stand(self.cmd_client, timeout_sec=stand_timeout_sec)

    def shutdown(
            self, 
            power_off: bool = False
            ):
        if self._lease_keepalive is not None:
            try:
                self._lease_keepalive.shutdown()
            except Exception:
                pass
            self._lease_keepalive = None
        if power_off and self.robot is not None:
            try:
                self.robot.power_off(cut_immediately=False)
            except Exception:
                pass

    def move_to_goal(
            self, 
            x, 
            y, 
            angle_deg, 
            frame="vision", 
            use_local_origin=True
        ):
        goal_yaw = math.radians(angle_deg)
        target_frame = VISION_FRAME_NAME if frame == "vision" else ODOM_FRAME_NAME
        if use_local_origin:
            final_x = self.origin_x + (x * math.cos(self.origin_yaw) - y * math.sin(self.origin_yaw))
            final_y = self.origin_y + (x * math.sin(self.origin_yaw) + y * math.cos(self.origin_yaw))
            final_yaw = self.origin_yaw + goal_yaw
        else:
            final_x, final_y, final_yaw = x, y, goal_yaw
        now = self.robot.time_sync.robot_timestamp_from_local_secs(time.time())
        end_time_sec = now.seconds + (now.nanos / 1e9) + 10.0
        cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=final_x, goal_y=final_y, goal_heading=final_yaw, frame_name=target_frame
        )
        self.cmd_client.robot_command(cmd, end_time_secs=end_time_sec)
        print(f"[GoTo] Logic Target: ({x}, {y}, {angle_deg}°) -> SDK Target: ({final_x:.2f}, {final_y:.2f})")

    def guard(
            self, 
            detector
        ) -> dict:
        print("\n[Guard] 正在扫描全向环境 (6路相机)...")
        sources = [
            'hand_color_image', 
            'left_fisheye_image', 'right_fisheye_image', 
            'frontleft_fisheye_image', 'frontright_fisheye_image', 
            'back_fisheye_image'
        ]
        image_client = self.robot.ensure_client('image')
        reqs = [build_image_request(src, quality_percent=70) for src in sources]
        try:
            responses = image_client.get_image(reqs)
        except Exception as e:
            print(f"[Guard] 获取图像失败: {e}")
            return None
        images_dict = {}
        for res in responses:
            if res.status == image_pb2.ImageResponse.STATUS_OK:
                img_np = np.frombuffer(res.shot.image.data, dtype=np.uint8)
                img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                if img is not None:
                    images_dict[res.source.name] = img
            else:
                pass 
        if not images_dict:
            print("[Guard] 未获取到有效图像，扫描结束。")
            return None
        results = detector.detect_bottles_in_batch(images_dict, conf=0.25)
        if results:
            best = results[0]
            print(f"[Guard] 锁定最佳目标: {best['camera']} (置信度: {best['conf']:.2f})")
            return best
        else:
            print("[Guard] 未发现目标。")
            return None

    def grab_bottle(
            self, 
            detector
        ) -> bool:
        """
        [Grab] 只进行抓取，不回家，不松手。
        复用 cls_spot_demo.py 中的 handcam_detect_and_grab_once_sameframe。
        """
        # FIXME
        print("\n[Grab] 开始执行单纯抓取...")
        
        # 调用现有的同帧检测抓取逻辑
        # carry_on_success=True: 抓到后把手提起来，方便移动
        # open_on_success=False: 抓到后不要松开
        # stow_on_finish=False: 不要收纳手臂，保持持握姿态
        success = self.handcam_detect_and_grab_once_sameframe(
            detector,
            source="hand_color_image",
            interval=0.5,
            timeout=30.0,
            carry_on_success=True, 
            open_on_success=False,
            stow_on_finish=False 
        )
        if success:
            print("[Grab] 抓取成功，保持持握姿态。")
        else:
            print("[Grab] 抓取失败。")
        return success
    
    def release_object(self) -> bool:
        print("\n[Release] 执行释放...")
        # FIXME
        if not self.cmd_client:
            self.cmd_client = self.robot.ensure_client(RobotCommandBuilder.default_service_name)
        try:
            cmd = RobotCommandBuilder.claw_gripper_open_fraction_command(1.0)
            self.cmd_client.robot_command(cmd)
            time.sleep(1.5)
            print("[Release] 夹爪已打开，物体释放完毕。")
            return True
        except Exception as e:
            print(f"[Release] 释放失败: {e}")
            return False
    
    # endregion

    # region  Public APIs: Navigation functions
    
    def reset_local_origin(self, frame: str = "vision"):
        from bosdyn.client.frame_helpers import ODOM_FRAME_NAME, VISION_FRAME_NAME, BODY_FRAME_NAME
        state = self.state_client.get_robot_state()
        target_frame = VISION_FRAME_NAME if frame == "vision" else ODOM_FRAME_NAME
        tform = get_a_tform_b(state.kinematic_state.transforms_snapshot, target_frame, BODY_FRAME_NAME)
        self.origin_x = tform.position.x
        self.origin_y = tform.position.y
        self.origin_yaw = self._yaw_from_quat(tform.rotation)
        print(f"[Origin] Local (0,0,0) set to current {frame} pose.")

    def get_home_location(self):
        state = self.state_client.get_robot_state()
        transforms = state.kinematic_state.transforms_snapshot
        vision_tform = get_a_tform_b(transforms, VISION_FRAME_NAME, BODY_FRAME_NAME)
        self.home_x = vision_tform.x
        self.home_y = vision_tform.y
        self.home_yaw = math.degrees(self._yaw_from_quat(vision_tform.rotation))

    def get_guard_location(self):
        state = self.state_client.get_robot_state()
        transforms = state.kinematic_state.transforms_snapshot
        vision_tform = get_a_tform_b(transforms, VISION_FRAME_NAME, BODY_FRAME_NAME)
        self.guard_x = vision_tform.x
        self.guard_y = vision_tform.y
        self.guard_yaw = math.degrees(self._yaw_from_quat(vision_tform.rotation))

    # endregion

    # region  Private APIs: Web streaming

    def _start_web_server(self, host="0.0.0.0", port=5555):
        app = Flask(__name__)
        @app.route('/')
        def index():
            # 极简的 HTML 页面，显示流画面
            return render_template_string("""
                <html>
                <head><title>Spot 360 View</title></head>
                <body style="background: #111; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0;">
                    <img src="/video_feed" style="width: 80%; border: 2px solid #555;">
                </body>
                </html>
            """)
        def gen_frames():
            while True:
                if self._latest_grid is not None:
                    # 将 OpenCV 图像转为 JPG 格式
                    ret, buffer = cv2.imencode('.jpg', self._latest_grid)
                    frame = buffer.tobytes()
                    # 使用 MJPEG 格式拼接
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                time.sleep(0.05) # 限制 20fps 左右，节省 CPU
        @app.route('/video_feed')
        def video_feed():
            return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
        threading.Thread(target=lambda: app.run(host=host, port=port, debug=False, use_reloader=False), daemon=True).start()
        print(f"[WebUI] Server started at http://{host}:{port}")

    def _stream_loop(self):
        image_client = self.robot.ensure_client("image")
        source_names = [
            'left_fisheye_image', 'frontleft_fisheye_image', 'frontright_fisheye_image', 
            'right_fisheye_image', 'hand_color_image', 'back_fisheye_image'
        ]
        display_names = {
            'left_fisheye_image': 'Left',
            'frontleft_fisheye_image': 'Front-L',
            'frontright_fisheye_image': 'Front-R',
            'right_fisheye_image': 'Right',
            'hand_color_image': 'Hand',
            'back_fisheye_image': 'Back'
        }
        rotate_180_sources = ['right_fisheye_image'] 
        W, H = 320, 240
        reqs = [
            build_image_request(src, pixel_format=image_pb2.Image.PIXEL_FORMAT_RGB_U8, quality_percent=70) 
            for src in source_names
        ]
        while self._streaming:
            try:
                responses = image_client.get_image(reqs)
                img_map = {}
                empty_block = np.zeros((H, W, 3), dtype=np.uint8)
                for res in responses:
                    source_name = res.source.name
                    if res.status == image_pb2.ImageResponse.STATUS_OK:
                        arr = np.frombuffer(res.shot.image.data, dtype=np.uint8)
                        decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if decoded is not None:
                            resized = cv2.resize(decoded, (W, H))
                            if source_name in rotate_180_sources:
                                resized = cv2.rotate(resized, cv2.ROTATE_180)
                            label = display_names.get(source_name, source_name)
                            cv2.putText(resized, label, (10, 30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            img_map[source_name] = resized
                        else:
                            img_map[source_name] = empty_block
                    else:
                        img_map[source_name] = empty_block
                row1 = np.hstack([
                    img_map.get('frontleft_fisheye_image', empty_block),
                    img_map.get('hand_color_image', empty_block),
                    img_map.get('frontright_fisheye_image', empty_block)
                ])
                row2 = np.hstack([
                    img_map.get('left_fisheye_image', empty_block),
                    img_map.get('back_fisheye_image', empty_block),
                    img_map.get('right_fisheye_image', empty_block)
                ])
                self._latest_grid = np.vstack([row1, row2])
            except Exception as e:
                print(f"[Stream Err] {e}")
                time.sleep(0.5)
    # endregion

    # region  Publick APIs: Debug tools

    def debug_pose(self):
        state = self.state_client.get_robot_state()
        transforms = state.kinematic_state.transforms_snapshot
        vision_tform = get_a_tform_b(transforms, VISION_FRAME_NAME, BODY_FRAME_NAME)
        print(f"\n--- SDK [VISION] Frame ---")
        print(f"X: {vision_tform.x:.4f}, Y: {vision_tform.y:.4f}")
        print(f"Yaw: {math.degrees(self._yaw_from_quat(vision_tform.rotation)):.2f}°")
        print("="*30 + "\n")   

    # endregion 
