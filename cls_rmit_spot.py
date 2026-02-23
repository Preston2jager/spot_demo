# ===== =================== =====
# ===== RMIT Spot control class =====
# ===== =================== =====
import time
import math
import cv2
import threading
from typing import Optional
import numpy as np
# ===== For web ui =====
from flask import Flask, Response, render_template_string
# ===== BostonDynamic APIs =====
import bosdyn.client
from bosdyn.api import image_pb2
from bosdyn.client.image import build_image_request, ImageClient
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder, blocking_stand, block_until_arm_arrives
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn.client.frame_helpers import (
    GRAV_ALIGNED_BODY_FRAME_NAME,
    HAND_FRAME_NAME,
    ODOM_FRAME_NAME,     
    VISION_FRAME_NAME,
    BODY_FRAME_NAME,   
    get_a_tform_b,
    math_helpers
)
from bosdyn.client.math_helpers import SE3Pose, Quat
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.api import geometry_pb2, manipulation_api_pb2

class SpotAgent:

    # region  Private APIs: Initialisation

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
        
    def __enter__(self):
        threading.Thread(target=self._stream_loop, daemon=True).start()
        self._start_web_server(host="0.0.0.0", port=5555)
        time.sleep(1)
        self._power_on_and_stand()
        self._arm_out()
        return self

    def __exit__(self, *args):
        self._shutdown()

    # endregion

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
    
    def _power_on_and_stand(
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
    
    def _shutdown(
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
    
    
    # endregion

    # region  Private APIs: Tools

    @staticmethod
    def _yaw_from_quat(q) -> float:
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                          1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    
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
            'hand_color_image',       
            'left_fisheye_image',     
            'right_fisheye_image',    
            'back_fisheye_image'      
        ]
        display_names = {
            'hand_color_image': 'Front (Hand)',
            'left_fisheye_image': 'Left',
            'right_fisheye_image': 'Right',
            'back_fisheye_image': 'Back'
        }
        camera_rotations = {
            'hand_color_image': 0,       
            'left_fisheye_image': 0,
            'right_fisheye_image': 180,  
            'back_fisheye_image': 0
        }
        W_STD, H_STD = 320, 240
        W_WIDE, H_WIDE = 640, 480
        reqs = [
            build_image_request(src, pixel_format=image_pb2.Image.PIXEL_FORMAT_RGB_U8, quality_percent=70) 
            for src in source_names
        ]
        while self._streaming:
            try:
                responses = image_client.get_image(reqs)
                img_map = {}
                empty_std = np.zeros((H_STD, W_STD, 3), dtype=np.uint8)
                empty_wide = np.zeros((H_WIDE, W_WIDE, 3), dtype=np.uint8)
                for res in responses:
                    source_name = res.source.name
                    if res.status == image_pb2.ImageResponse.STATUS_OK:
                        arr = np.frombuffer(res.shot.image.data, dtype=np.uint8)
                        decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if decoded is not None:
                            angle = camera_rotations.get(source_name, 0) % 360
                            rotated = decoded
                            if angle == 90: rotated = cv2.rotate(decoded, cv2.ROTATE_90_COUNTERCLOCKWISE)
                            elif angle == 180: rotated = cv2.rotate(decoded, cv2.ROTATE_180)
                            elif angle == 270: rotated = cv2.rotate(decoded, cv2.ROTATE_90_CLOCKWISE)
                            elif angle != 0:
                                h_o, w_o = decoded.shape[:2]
                                M = cv2.getRotationMatrix2D((w_o//2, h_o//2), angle, 1.0)
                                rotated = cv2.warpAffine(decoded, M, (w_o, h_o))
                            if source_name in ['hand_color_image', 'back_fisheye_image']:
                                target_w, target_h = W_WIDE, H_WIDE
                                font_scale = 1.0
                            else:
                                target_w, target_h = W_STD, H_STD
                                font_scale = 0.7
                            final_img = cv2.resize(rotated, (target_w, target_h))
                            label = display_names.get(source_name, source_name)
                            cv2.putText(final_img, label, (10, 40), 
                                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
                            img_map[source_name] = final_img
                # ==========================================
                # 1. 第一行和第二行 (原相机画面)
                # ==========================================
                row1_front = img_map.get('hand_color_image', empty_wide)
                row2_side = np.hstack([
                    img_map.get('left_fisheye_image', empty_std),
                    img_map.get('right_fisheye_image', empty_std)
                ])
                
                # ==========================================
                # 2. 第三行：生成 2D 俯视平面图 (宽640 高480)
                # ==========================================
                map_img = np.zeros((H_WIDE, W_WIDE, 3), dtype=np.uint8)
                
                # 画背景网格 (增强视觉效果)
                grid_size = 40
                for i in range(0, W_WIDE, grid_size):
                    cv2.line(map_img, (i, 0), (i, H_WIDE), (30, 30, 30), 1)
                for i in range(0, H_WIDE, grid_size):
                    cv2.line(map_img, (0, i), (W_WIDE, i), (30, 30, 30), 1)
                    
                # 定义地图中心(即原点)
                cx, cy = W_WIDE // 2, H_WIDE // 2
                cv2.drawMarker(map_img, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)
                cv2.putText(map_img, "Origin", (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                
                # 比例尺: 1米 = 100个像素
                scale = 100 
                
                # 渲染扫描到的 Bottle / Can 坐标
                if hasattr(self, '_latest_objects') and self._latest_objects:
                    for obj in self._latest_objects:
                        obj_x = obj['x']
                        obj_y = obj['y']
                        cam = obj['camera_name'].split('_')[0] # 取出 hand/left 等前缀
                        
                        # 物理坐标系映射至像素系 (Spot vision 坐标中，X前，Y左)
                        # OpenCV图像中，X向右，Y向下
                        # 所以我们让地图的 Y轴(垂直)对应物理 X轴，地图的 X轴(水平)对应物理 Y轴的负方向
                        px = int(cx - obj_y * scale)
                        py = int(cy - obj_x * scale)
                        
                        # 若点在画布内，则画出红点与外圈
                        if 0 <= px < W_WIDE and 0 <= py < H_WIDE:
                            cv2.circle(map_img, (px, py), 8, (0, 0, 255), -1)    # 内部红色实体
                            cv2.circle(map_img, (px, py), 12, (0, 255, 255), 2)  # 外围黄色警告圈
                            cv2.putText(map_img, f"Target({cam})", (px + 15, py + 5), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                cv2.putText(map_img, "2D Target Plan (1m=100px)", (15, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                # 垂直堆叠三行，最终画面将变成 640x1200
                self._latest_grid = np.vstack([row1_front, row2_side, map_img])
                
            except Exception as e:
                print(f"[Stream Err] {e}")
                time.sleep(0.5)
    # endregion

    # region  Private APIs: Basic actions
    
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

    # endregion

    # region  Public APIs: Basic actions
    
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
 
    def scan(self, detector) -> list:
        print("\n[Guard] 正在扫描全向环境寻找所有目标...")
        sources = [
            'hand_color_image', 
            'left_fisheye_image', 
            'right_fisheye_image', 
            'back_fisheye_image'
        ]
        camera_rotations = {
            'left_fisheye_image': 0,
            'right_fisheye_image': 180,  
            'hand_color_image': 0,       
            'back_fisheye_image': 0
        }
        image_client = self.robot.ensure_client('image')
        reqs = [build_image_request(src, quality_percent=70) for src in sources]
        try:
            responses = image_client.get_image(reqs)
        except Exception as e:
            print(f"[Guard] 获取图像失败: {e}")
            return []
        raw_responses = {}
        images_dict = {}
        cam_meta = {} 
        for res in responses:
            if res.status == image_pb2.ImageResponse.STATUS_OK:
                cam_name = res.source.name
                raw_responses[cam_name] = res
                img_np = np.frombuffer(res.shot.image.data, dtype=np.uint8)
                img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                if img is not None:
                    orig_h, orig_w = img.shape[:2]
                    angle = camera_rotations.get(cam_name, 0) % 360
                    rotated_img = img
                    if angle == 180:
                        rotated_img = cv2.rotate(img, cv2.ROTATE_180)
                    images_dict[cam_name] = rotated_img
                    cam_meta[cam_name] = {"orig_w": orig_w, "orig_h": orig_h, "angle": angle}
        if not images_dict:
            return []
        results = detector.detect_targets_in_batch(images_dict, conf=0.05)
        detection_list = []
        if results:
            print(f"[Guard] 发现 {len(results)} 个潜在目标，正在反推原始坐标...")
            for best in results:
                cam_name = best['camera']
                cx, cy = best.get('cx', 0), best.get('cy', 0)
                meta = cam_meta[cam_name]
                orig_w, orig_h, angle = meta['orig_w'], meta['orig_h'], meta['angle']
                raw_pixel_x, raw_pixel_y = cx, cy
                if angle == 180:
                    raw_pixel_x = orig_w - cx - 1
                    raw_pixel_y = orig_h - cy - 1
                print(f"        -> [{cam_name}] 原始像素坐标: ({raw_pixel_x:.1f}, {raw_pixel_y:.1f})")
                detection_list.append((raw_responses[cam_name], raw_pixel_x, raw_pixel_y))
        else:
            print("[Guard] 未发现目标。")
        return detection_list

    def point_arm_to_pixel(self, detection_result: tuple, assumed_dist: float = 1.0) -> bool:
        """
        接收视觉返回值，计算 3D 坐标：
        1. 控制机器狗底盘原地旋转，面朝目标。
        2. 将手臂向前伸出，并调整手爪姿态指向目标。
        """
        import math
        import time
        from bosdyn.client.frame_helpers import (
            VISION_FRAME_NAME, HAND_FRAME_NAME, BODY_FRAME_NAME, 
            get_a_tform_b, math_helpers
        )
        from bosdyn.client.robot_command import RobotCommandBuilder

        if not detection_result:
            print("[Arm] 没有收到有效目标，保持原地待命。")
            return False

        image_response, pixel_x, pixel_y = detection_result

        if getattr(self, "cmd_client", None) is None or getattr(self, "state_client", None) is None:
            raise RuntimeError("cmd_client/state_client 未初始化。")

        try:
            print(f"\n[Arm] 正在计算 3D 坐标...")
            
            # --- 1. 提取内参 ---
            source = image_response.source
            if source.HasField('pinhole'):
                intrinsics = source.pinhole.intrinsics
            elif source.HasField('fisheye'):
                intrinsics = source.fisheye.intrinsics
            else:
                print("[Arm] 不支持的相机模型，无法转换坐标。")
                return False
                
            fx, fy = intrinsics.focal_length.x, intrinsics.focal_length.y
            cx, cy = intrinsics.principal_point.x, intrinsics.principal_point.y
            
            # --- 2. 像素坐标 -> 局部 3D 射线坐标 ---
            x_cam = (pixel_x - cx) / fx
            y_cam = (pixel_y - cy) / fy
            z_cam = 1.0 
            
            length = math.sqrt(x_cam**2 + y_cam**2 + z_cam**2)
            # 在相机坐标系下，根据假设距离推算物体的确切点
            target_cam = math_helpers.SE3Pose(
                x=(x_cam/length)*assumed_dist, 
                y=(y_cam/length)*assumed_dist, 
                z=(z_cam/length)*assumed_dist, 
                rot=math_helpers.Quat()
            )
            
            # --- 3. 转换到绝对世界坐标系 (VISION) ---
            root_frame = VISION_FRAME_NAME
            cam_frame = image_response.shot.frame_name_image_sensor
            camera_snapshot = image_response.shot.transforms_snapshot
            
            world_T_cam = get_a_tform_b(camera_snapshot, root_frame, cam_frame)
            if world_T_cam is None:
                print(f"[Arm] 致命错误：无法提取 {root_frame} 到相机的转换。")
                return False
                
            # 获取目标在世界中的绝对 3D 坐标 (这个坐标是固定的，即使狗动了也不会变)
            target_world = world_T_cam * target_cam
            
            # =========================================================
            # 第一阶段：转动狗身 (Base Control)
            # =========================================================
            print("[Base] 准备转动底盘面朝目标...")
            current_state = self.state_client.get_robot_state()
            world_T_body = get_a_tform_b(current_state.kinematic_state.transforms_snapshot, root_frame, BODY_FRAME_NAME)
            
            # 计算狗身到目标的朝向 (Yaw)
            dx_body = target_world.x - world_T_body.x
            dy_body = target_world.y - world_T_body.y
            body_yaw = math.atan2(dy_body, dx_body)
            
            # 发送底盘移动指令：保持 X, Y 不变，仅原地旋转到目标 Yaw
            turn_cmd = RobotCommandBuilder.synchro_se2_trajectory_command(
                goal_x=world_T_body.x,
                goal_y=world_T_body.y,
                goal_heading=body_yaw,
                frame_name=root_frame
            )
            self.cmd_client.robot_command(turn_cmd)
            
            # 等待底盘转动到位 (视具体需求可改为轮询 feedback，这里简单用延时)
            time.sleep(3.0) 
            print(f"[Base] 底盘已转到位 (目标 Yaw: {math.degrees(body_yaw):.1f}°)")

            # =========================================================
            # 第二阶段：伸出手臂并指向目标 (Arm Control)
            # =========================================================
            print("[Arm] 准备伸出手臂指向目标...")
            # 获取转动身体后的最新状态
            new_state = self.state_client.get_robot_state()
            new_world_T_body = get_a_tform_b(new_state.kinematic_state.transforms_snapshot, root_frame, BODY_FRAME_NAME)
            
            # 定义一个相对于“新身体位置”向前伸出的手部预期位置
            # X: 向前伸 0.7m, Y: 居中 0.0m, Z: 抬高 0.3m (相对 body 坐标系)
            body_T_extended_hand = math_helpers.SE3Pose(x=0.7, y=0.0, z=0.3, rot=math_helpers.Quat())
            
            # 将这个预期的伸出手部位置转换回 VISION 世界坐标系
            world_T_hand_target = new_world_T_body * body_T_extended_hand
            
            # 计算从这个新伸出的手部位置，指向目标的 Pitch 和 Yaw
            dx_hand = target_world.x - world_T_hand_target.x
            dy_hand = target_world.y - world_T_hand_target.y
            dz_hand = target_world.z - world_T_hand_target.z
            
            arm_yaw = math.atan2(dy_hand, dx_hand)
            dist_xy_hand = math.sqrt(dx_hand**2 + dy_hand**2)
            arm_pitch = math.atan2(-dz_hand, dist_xy_hand) # 向下低头 Pitch 为正
            
            target_rot = math_helpers.Quat.from_yaw(arm_yaw) * math_helpers.Quat.from_pitch(arm_pitch)
            
            # 发送手臂动作指令
            arm_cmd = RobotCommandBuilder.arm_pose_command(
                world_T_hand_target.x, world_T_hand_target.y, world_T_hand_target.z, # 伸出的新 XYZ 坐标
                target_rot.w, target_rot.x, target_rot.y, target_rot.z, # 指向目标的四元数旋转
                root_frame, 
                2.0 # 2.0 秒内伸出并转到位
            )
            
            self.cmd_client.robot_command(arm_cmd)
            time.sleep(2.5) # 等待手臂动作完成
            
            print("[Arm] 手臂已伸出并指向目标！")
            time.sleep(3)
            return True
            
        except Exception as e:
            print(f"[Arm] 指向目标失败: {e}")
            return False
    
    def object_register(self, detection_list: list, assumed_dist: float = 1.0) -> list:
        if not detection_list:
            print("[Vision] 未收到有效的检测结果，清除平面图目标。")
            self._latest_objects = [] # 清除过期目标
            return []
        registered_objects = []
        print(f"\n[Vision] 开始批量解算 {len(detection_list)} 个目标的 3D 坐标...")
        for idx, detection in enumerate(detection_list):
            image_response, pixel_x, pixel_y = detection
            try:
                source = image_response.source
                if source.HasField('pinhole'):
                    intrinsics = source.pinhole.intrinsics
                elif source.HasField('fisheye'):
                    intrinsics = source.fisheye.intrinsics
                else:
                    continue
                fx, fy = intrinsics.focal_length.x, intrinsics.focal_length.y
                cx, cy = intrinsics.principal_point.x, intrinsics.principal_point.y
                x_cam = (pixel_x - cx) / fx
                y_cam = (pixel_y - cy) / fy
                z_cam = 1.0 
                length = math.sqrt(x_cam**2 + y_cam**2 + z_cam**2)
                target_cam = math_helpers.SE3Pose(
                    x=(x_cam/length)*assumed_dist, 
                    y=(y_cam/length)*assumed_dist, 
                    z=(z_cam/length)*assumed_dist, 
                    rot=math_helpers.Quat()
                )
                root_frame = "body"
                cam_frame = image_response.shot.frame_name_image_sensor
                camera_snapshot = image_response.shot.transforms_snapshot
                world_T_cam = get_a_tform_b(camera_snapshot, root_frame, cam_frame)
                if world_T_cam is None:
                    continue
                target_world = world_T_cam * target_cam
                print(f"         [目标 {idx+1}] 坐标系: {root_frame} | 位置: X={target_world.x:.3f}, Y={target_world.y:.3f}, Z={target_world.z:.3f}")
                registered_objects.append({
                    "frame": root_frame,
                    "x": target_world.x,
                    "y": target_world.y,
                    "z": target_world.z,
                    "camera_name": source.name
                })
            except Exception as e:
                print(f"[Vision] ❌ 目标 3D 注册异常: {e}")
        self._latest_objects = registered_objects
        return registered_objects
        
    def grab_first_target(self, detection_list: list) -> bool:
        """
        从 scan 返回的检测列表中提取第一个目标，自动走过去并抓取。
        内置了底层 Manipulation API 的完整抓取与反馈轮询逻辑。
        """
        if not detection_list:
            print("\n[Grab] ⚠️ 检测列表为空，没有找到可以抓取的目标。")
            return False
        first_target = detection_list[0]
        if isinstance(first_target, dict):
            print("\n[Grab] ❌ 数据格式错误！请确保传入的是 agent.scan() 返回的原始列表。")
            return False
        try:
            image_response, pixel_x, pixel_y = first_target[:3]
        except Exception as e:
            print(f"\n[Grab] ❌ 解析目标数据失败: {e}")
            return False
        cam_name = image_response.source.name
        print(f"\n[Grab] 🎯 锁定首个目标！")
        print(f"       发现位置: {cam_name}")
        print(f"       像素坐标: ({pixel_x:.1f}, {pixel_y:.1f})")
        print("[Grab] 🐕 正在移交底层 Manipulation API，Spot 将自动接近并尝试抓取...")

        # =========================================================
        # 核心抓取与寻路逻辑 (自动融合相机快照与 3D 逆解算)
        # =========================================================
        if getattr(self, "robot", None) is None:
            print("[Grab] ❌ 机器人未连接或未初始化。")
            return False

        # 初始化抓取和指令客户端
        manip_client = self.robot.ensure_client(ManipulationApiClient.default_service_name)
        cmd_client = getattr(self, "cmd_client", None) or self.robot.ensure_client(RobotCommandClient.default_service_name)
        self.cmd_client = cmd_client

        # 1. 提取相机模型参数 (Pinhole 或 Fisheye)
        cam_model = getattr(image_response.source, "pinhole", None) or \
                    getattr(image_response.source, "fisheye", None) or \
                    image_response.source.pinhole

        # 2. 构建 3D 抓取请求 (PickObjectInImage)
        pick = manipulation_api_pb2.PickObjectInImage(
            pixel_xy=geometry_pb2.Vec2(x=int(pixel_x), y=int(pixel_y)),
            transforms_snapshot_for_camera=image_response.shot.transforms_snapshot,
            frame_name_image_sensor=image_response.shot.frame_name_image_sensor,
            camera_model=cam_model,
        )
        req = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=pick)
        
        # 3. 发送指令 (此刻机器狗会自己开始算路径、转身并走过去)
        print("[Grab] 📡 抓取请求已发送，等待机器狗规划路径及执行动作...")
        rsp = manip_client.manipulation_api_command(manipulation_api_request=req)

        # 4. 开启轮询，监控 Spot 的执行状态
        feedback_timeout_sec = 60.0  # 留足转身、走路和抓取的时间
        feedback_interval_sec = 0.5
        deadline = time.time() + feedback_timeout_sec
        succeeded = False
        last_name = ""
        
        while time.time() < deadline:
            fb = manip_client.manipulation_api_feedback_command(
                manipulation_api_pb2.ManipulationApiFeedbackRequest(
                    manipulation_cmd_id=rsp.manipulation_cmd_id
                )
            )
            state = fb.current_state
            name = manipulation_api_pb2.ManipulationFeedbackState.Name(state)
            
            # 状态一旦发生变化，打印出来以便调试
            if name != last_name:
                print(f"[Grab] 🔄 状态更新: {name}")
                last_name = name
                
            if state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED:
                succeeded = True
                break
            if state == manipulation_api_pb2.MANIP_STATE_GRASP_FAILED:
                succeeded = False
                break
                
            time.sleep(feedback_interval_sec)

        # =========================================================
        # 抓取完成后的动作后处理
        # =========================================================
        if succeeded:
            print("\n[Grab] ✅ 抓取大成功！准备进入携带 (Carry) 姿态...")
            try:
                # 抓紧目标后，把手臂抬起收拢到胸前，防止走路时碰到
                cid = cmd_client.robot_command(RobotCommandBuilder.arm_carry_command())
                block_until_arm_arrives(cmd_client, cid, timeout_sec=6.0)
                time.sleep(0.5)
                print("[Grab] 🦾 已稳稳拿住目标！处于 Carry 姿态原地待命。")
            except Exception as e:
                print(f"[Grab] ⚠️ 转换为 Carry 姿态失败: {e}")
        else:
            print("\n[Grab] ❌ 抓取失败 (可能因为目标超出物理可达范围、目标移动，或防撞机制触发)。")
        return succeeded
    
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

    # region  Public APIs: Debug tools

    def debug_pose(self):
        state = self.state_client.get_robot_state()
        transforms = state.kinematic_state.transforms_snapshot
        vision_tform = get_a_tform_b(transforms, VISION_FRAME_NAME, BODY_FRAME_NAME)
        print(f"\n--- SDK [VISION] Frame ---")
        print(f"X: {vision_tform.x:.4f}, Y: {vision_tform.y:.4f}")
        print(f"Yaw: {math.degrees(self._yaw_from_quat(vision_tform.rotation)):.2f}°")
        print("="*30 + "\n")   

    # endregion 
    
    def grab_target_with_nav(self, detector, detection_list: list) -> bool:
        """
        引入两段式抓取逻辑：
        1. 粗定位：解算目标的绝对 3D 坐标
        2. 寻路对齐：计算预抓取点（距离目标前方约 0.65 米），调用 move_to_goal 走过去
        3. 姿态准备：控制机械臂前伸并低头，俯视目标可能存在的区域 (新增)
        4. 精确定位：使用手部相机重新扫描，确保在正前方无死角
        5. 出爪：基于新画面执行精准抓取
        """
        import time
        import math
        import numpy as np
        import cv2
        from bosdyn.client.frame_helpers import get_a_tform_b, VISION_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME, math_helpers
        from bosdyn.client.robot_state import RobotStateClient
        from bosdyn.client.image import build_image_request
        from bosdyn.client.math_helpers import SE3Pose, Quat
        from bosdyn.client.robot_command import RobotCommandBuilder

        if not detection_list:
            print("\n[NavGrab] ⚠️ 检测列表为空。")
            return False

        first_target = detection_list[0]
        try:
            image_response, pixel_x, pixel_y = first_target[:3]
        except Exception as e:
            print(f"\n[NavGrab] ❌ 数据解析失败: {e}")
            return False

        # ==========================================
        # 1. 粗定位：解算大致的绝对 3D 世界坐标
        # ==========================================
        source = image_response.source
        cam_model = getattr(source, "pinhole", None) or getattr(source, "fisheye", None) or source.pinhole
        intrinsics = cam_model.intrinsics
        
        fx, fy = intrinsics.focal_length.x, intrinsics.focal_length.y
        cx, cy = intrinsics.principal_point.x, intrinsics.principal_point.y
        
        x_cam = (pixel_x - cx) / fx
        y_cam = (pixel_y - cy) / fy
        z_cam = 1.0 
        length = math.sqrt(x_cam**2 + y_cam**2 + z_cam**2)
        
        # 安全设定：预估目标距离设为 0.8 米
        assumed_dist = 0.8  
        target_cam = SE3Pose(
            x=(x_cam/length)*assumed_dist, 
            y=(y_cam/length)*assumed_dist, 
            z=(z_cam/length)*assumed_dist, 
            rot=Quat()
        )
                             
        root_frame = VISION_FRAME_NAME
        cam_frame = image_response.shot.frame_name_image_sensor
        camera_snapshot = image_response.shot.transforms_snapshot
        
        world_T_cam = get_a_tform_b(camera_snapshot, root_frame, cam_frame)
        target_world = world_T_cam * target_cam
        obj_x, obj_y = target_world.x, target_world.y

        # ==========================================
        # 2. 寻路对齐：计算最佳抓取身位并走过去
        # ==========================================
        state_client = self.robot.ensure_client(RobotStateClient.default_service_name)
        rs = state_client.get_robot_state()
        tf = rs.kinematic_state.transforms_snapshot
        world_T_body = get_a_tform_b(tf, root_frame, "body")
        rob_x, rob_y = world_T_body.position.x, world_T_body.position.y
        
        dx = obj_x - rob_x
        dy = obj_y - rob_y
        dist = math.hypot(dx, dy)
        angle = math.atan2(dy, dx)
        
        standoff_dist = 0.65  # 机器狗停在距离目标 0.65 米的地方出爪最舒服
        
        if dist > standoff_dist:
            nav_x = obj_x - standoff_dist * math.cos(angle)
            nav_y = obj_y - standoff_dist * math.sin(angle)
        else:
            nav_x, nav_y = rob_x, rob_y
            
        nav_yaw_deg = math.degrees(angle)
        
        print(f"\n[NavGrab] 🧭 目标大致位置推算: X={obj_x:.2f}, Y={obj_y:.2f}")
        print(f"[NavGrab] 🚶 正在前往预备抓取点: X={nav_x:.2f}, Y={nav_y:.2f}, 转身对齐角度={nav_yaw_deg:.1f}°")
        
        self.move_to_goal(x=nav_x, y=nav_y, angle_deg=nav_yaw_deg, frame="vision", use_local_origin=False)
        
        print("[NavGrab] ⏳ 等待移动到位 (8秒)...")
        time.sleep(8.0)

        # ==========================================
        # 3. 姿态准备：控制机械臂低头看向抓取点 (新增逻辑)
        # ==========================================
        print("\n[NavGrab] 🦾 正在调整机械臂姿态，低头俯视预抓取区域...")
        try:
            # 手部放在身体正前方 0.35米，高度 0.1米，并向下低头 45 度
            pitch_deg = 45.0
            q_pitch = math_helpers.Quat.from_pitch(math.radians(pitch_deg))
            
            look_down_cmd = RobotCommandBuilder.arm_pose_command(
                x=0.35, y=0.0, z=0.1, 
                qw=q_pitch.w, qx=q_pitch.x, qy=q_pitch.y, qz=q_pitch.z, 
                frame_name=GRAV_ALIGNED_BODY_FRAME_NAME, 
                seconds=2.0
            )
            self.cmd_client.robot_command(look_down_cmd)
            time.sleep(2.5) # 等待手臂移动平稳，防止画面模糊
        except Exception as e:
            print(f"[NavGrab] ⚠️ 调整机械臂姿态失败: {e}")

        # ==========================================
        # 4. 精确定位：移动和低头后，重新拍照获取最新快照
        # ==========================================
        print("\n[NavGrab] 📸 姿态调整完毕，正在使用手部(正前)相机重新进行精确锁定...")
        image_client = self.robot.ensure_client('image')
        req = build_image_request('hand_color_image', quality_percent=70)
        
        try:
            res = image_client.get_image([req])[0]
            if res.status != image_pb2.ImageResponse.STATUS_OK:
                print("[NavGrab] ❌ 获取手部图像失败。")
                return False
                
            img_np = np.frombuffer(res.shot.image.data, dtype=np.uint8)
            img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            
            # 使用传进来的 YOLO 检测器重新扫一眼这张新图
            results = detector.detect_targets_in_batch({'hand_color_image': img}, conf=0.05)
            
            if not results:
                print("[NavGrab] ❌ 走近低头后丢失目标！(可能被踢飞或在视野边缘)")
                return False
                
            best = results[0]
            new_x, new_y = best['cx'], best['cy']
            print(f"[NavGrab] 🎯 完美！已在正前方重新锁定目标，新像素: ({new_x}, {new_y})")
            
            # 5. 基于全新的照片和像素出爪！
            return self._execute_grasp(res, new_x, new_y)
            
        except Exception as e:
            print(f"[NavGrab] ⚠️ 重定位阶段发生异常: {e}")
            return False

    def _execute_grasp(self, image_response, pixel_x, pixel_y) -> bool:
        """ 内部专用函数：仅负责发送 Manipulation 抓取请求并轮询状态 """
        import time
        from bosdyn.api import geometry_pb2, manipulation_api_pb2
        from bosdyn.client.manipulation_api_client import ManipulationApiClient
        from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder, block_until_arm_arrives

        print("[Grab] 🐕 移交底层机械臂 API，开始计算 IK 并抓取...")
        manip_client = self.robot.ensure_client(ManipulationApiClient.default_service_name)
        cmd_client = getattr(self, "cmd_client", None) or self.robot.ensure_client(RobotCommandClient.default_service_name)
        self.cmd_client = cmd_client

        cam_model = getattr(image_response.source, "pinhole", None) or \
                    getattr(image_response.source, "fisheye", None) or \
                    image_response.source.pinhole

        pick = manipulation_api_pb2.PickObjectInImage(
            pixel_xy=geometry_pb2.Vec2(x=int(pixel_x), y=int(pixel_y)),
            transforms_snapshot_for_camera=image_response.shot.transforms_snapshot,
            frame_name_image_sensor=image_response.shot.frame_name_image_sensor,
            camera_model=cam_model,
        )
        req = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=pick)
        rsp = manip_client.manipulation_api_command(manipulation_api_request=req)

        deadline = time.time() + 30.0
        succeeded = False
        last_name = ""
        
        while time.time() < deadline:
            fb = manip_client.manipulation_api_feedback_command(
                manipulation_api_pb2.ManipulationApiFeedbackRequest(
                    manipulation_cmd_id=rsp.manipulation_cmd_id
                )
            )
            state = fb.current_state
            name = manipulation_api_pb2.ManipulationFeedbackState.Name(state)
            if name != last_name:
                print(f"       🔄 动作状态: {name}")
                last_name = name
                
            if state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED:
                succeeded = True
                break
            if state == manipulation_api_pb2.MANIP_STATE_GRASP_FAILED:
                succeeded = False
                break
            time.sleep(0.5)

        if succeeded:
            print("\n[Grab] ✅ 抓取大成功！收拢手臂进入 Carry 姿态...")
            try:
                cid = cmd_client.robot_command(RobotCommandBuilder.arm_carry_command())
                block_until_arm_arrives(cmd_client, cid, timeout_sec=6.0)
                time.sleep(0.5)
            except Exception:
                pass
        else:
            print("\n[Grab] ❌ 抓取失败。")

        return succeeded
