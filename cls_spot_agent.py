import time
import math
import cv2
import curses

from typing import Optional, Iterator, Tuple, Callable
import numpy as np

from bosdyn.api import image_pb2
from bosdyn.client.image import build_image_request, ImageClient
import bosdyn.client
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder, blocking_stand
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.frame_helpers import (
            HAND_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME,
            get_a_tform_b, math_helpers
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
        self.default_hold = 0.9  # send_velocity 的 end_time_secs 持续窗口

        self._auto_login(username, password)
        self._get_lease(force=force_lease)

    # ========== Admin ==========

    def _auto_login(self, username: str, password: str):
        self.sdk = bosdyn.client.create_standard_sdk(self.client_name)
        self.robot = self.sdk.create_robot(self.hostname)
        self.robot.authenticate(username, password)
        # 时间同步
        try:
            self.robot.time_sync.wait_for_sync()
        except Exception:
            self.robot.time_sync.start()
            self.robot.time_sync.wait_for_sync()
        # clients
        self.lease_client = self.robot.ensure_client(LeaseClient.default_service_name)
        self.cmd_client = self.robot.ensure_client(RobotCommandClient.default_service_name)
        self.img_client = self.robot.ensure_client(ImageClient.default_service_name)
        self.state_client = self.robot.ensure_client(RobotStateClient.default_service_name)

    def _make_keepalive(self, *, must_acquire: bool, return_at_exit: bool) -> LeaseKeepAlive:
        try:
            return LeaseKeepAlive(
                self.lease_client,
                must_acquire=must_acquire,
                return_at_exit=return_at_exit,
                period_sec=self.keep_alive_period_sec,
            )
        except TypeError:
            # 老版本无 period_sec
            return LeaseKeepAlive(
                self.lease_client,
                must_acquire=must_acquire,
                return_at_exit=return_at_exit,
            )

    def _get_lease(self, force: bool = False) -> LeaseKeepAlive:
        if self.lease_client is None:
            raise RuntimeError("lease_client 尚未初始化。")
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

    def _read_char(self):
        import sys, select
        if not select.select([sys.stdin], [], [], 0)[0]:
            return None
        return sys.stdin.read(1)

    # ========== Tools ==========

    @staticmethod
    def _yaw_from_quat(q) -> float:
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                          1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    @staticmethod
    def _wrap_pi(a: float) -> float:
        return (a + math.pi) % (2 * math.pi) - math.pi

    @staticmethod
    def _resize_for_display(frame, target_w: int, target_h: int, method: str = "lanczos"):
        interp = {
            "nearest": cv2.INTER_NEAREST,
            "linear":  cv2.INTER_LINEAR,
            "cubic":   cv2.INTER_CUBIC,
            "lanczos": cv2.INTER_LANCZOS4,
            "area":    cv2.INTER_AREA,
        }.get(method, cv2.INTER_LANCZOS4)
        return cv2.resize(frame, (target_w, target_h), interpolation=interp)
    
    # ========== Basic Functions ==========

    def power_on_and_stand(self, timeout_sec: float = 20.0, stand_timeout_sec: float = 10.0):
        if self.robot is None or self.cmd_client is None:
            raise RuntimeError("Require login")
        if not self.robot.is_powered_on():
            print("[robot] Power up...")
            self.robot.power_on(timeout_sec=timeout_sec)
        print("[robot] Standing up...")
        blocking_stand(self.cmd_client, timeout_sec=stand_timeout_sec)

    def shutdown(self, power_off: bool = False):
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

    # ========== Basic function ==========

    def send_velocity(self, v_x: float, v_y: float, v_rot: float, hold: Optional[float] = None):
        """BODY 系速度命令；end_time_secs = now + hold。"""
        if self.cmd_client is None:
            raise RuntimeError("cmd_client require init")
        hold = self.default_hold if hold is None else hold
        cmd = RobotCommandBuilder.synchro_velocity_command(v_x=v_x, v_y=v_y, v_rot=v_rot)
        self.cmd_client.robot_command(cmd, end_time_secs=time.time() + hold)

    # ========== keyboard Moving==========

    def move_spot(self, *, lin_speed: float = 0.6, ang_speed: float = 1.0, fps: float = 30.0) -> None:
        if not hasattr(self, "send_velocity"):
            raise RuntimeError("Not find self.send_velocity(vx, vy, vrot)。")
        period = 1.0 / max(1.0, fps)
        def _loop(stdscr):
            stdscr.nodelay(True)
            stdscr.keypad(True)
            curses.noecho()
            stdscr.addstr(0, 0, "W/S Forward and Back, A/D turning,Q exit")
            stdscr.refresh()
            while True:
                t0 = time.time()
                vx, vrot = 0.0, 0.0
                ch = stdscr.getch()
                last = ch
                while ch != -1:
                    last = ch
                    ch = stdscr.getch()
                if last != -1:
                    if   last in (ord('w'), ord('W'), curses.KEY_UP):    vx, vrot = +lin_speed, 0.0
                    elif last in (ord('s'), ord('S'), curses.KEY_DOWN):  vx, vrot = -lin_speed, 0.0
                    elif last in (ord('a'), ord('A'), curses.KEY_LEFT):  vx, vrot = 0.0, +ang_speed
                    elif last in (ord('d'), ord('D'), curses.KEY_RIGHT): vx, vrot = 0.0, -ang_speed
                    elif last in (ord('q'), ord('Q')):                   raise KeyboardInterrupt
                try:
                    self.send_velocity(vx, 0.0, vrot)
                except Exception:
                    pass
                stdscr.addstr(1, 0, f"vx={vx:+.2f} m/s   yaw_rate={vrot:+.2f} rad/s   ")
                stdscr.clrtoeol()
                stdscr.refresh()
                dt = time.time() - t0
                if dt < period:
                    time.sleep(period - dt)
        try:
            curses.wrapper(_loop)
        except KeyboardInterrupt:
            pass
        finally:
            try:
                self.send_velocity(0.0, 0.0, 0.0)
            except Exception:
                pass

    # ========== Arm_cam ==========

    def handcam_on(
        self,
        window_title: str = "Spot HandCam",
        *,
        width: int = 1280,
        height: int = 960,
        fps: float = 15.0,
        jpeg_quality: int = 70,
        source: str = "hand_color_image",
    ) -> None:
        if self.cmd_client is None or self.img_client is None:
            raise RuntimeError("cmd_client/img_client 未初始化。")
        try:
            self.cmd_client.robot_command(RobotCommandBuilder.claw_gripper_open_command())
            time.sleep(0.4)
            print("[gripper] Opening。")
        except Exception as e:
            print(f"[gripper] Opening failed:{e}")
        try:
            snapshot = self.state_client.get_robot_state().kinematic_state.transforms_snapshot
            root_frame = GRAV_ALIGNED_BODY_FRAME_NAME
            root_T_hand = get_a_tform_b(snapshot, root_frame, HAND_FRAME_NAME)

            delta_hand = math_helpers.SE3Pose(
                x=0.30, y=0.0, z=-0.25,
                rot=math_helpers.Quat.from_pitch(30.0 * math.pi / 180.0)
            )
            root_T_target = root_T_hand * delta_hand
            q = root_T_target.rot

            arm_cmd = RobotCommandBuilder.arm_pose_command(
                root_T_target.x, root_T_target.y, root_T_target.z,
                q.w, q.x, q.y, q.z, root_frame, 1.2
            )
            self.cmd_client.robot_command(arm_cmd)
            time.sleep(1.2)
            print("[arm] Locked。")
        except Exception as e:
            print(f"[arm] Locking failed:{e}")
        req = [build_image_request(
            source,
            image_format=image_pb2.Image.FORMAT_JPEG,
            quality_percent=int(jpeg_quality),
        )]
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
        cv2.resizeWindow(window_title, width, height)
        period = 1.0 / max(1.0, fps)
        running = True
        while running:
            t0 = time.time()
            try:
                resp = self.img_client.get_image(req)[0]
                buf = np.frombuffer(resp.shot.image.data, dtype=np.uint8)
                frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                frame = self._resize_for_display(frame, width, height, method="lanczos")
            except Exception as e:
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.putText(frame, f"Image error: {e}", (20, height//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            hud = f"{source}  (q/Esc to quit)"
            cv2.putText(frame, hud, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.imshow(window_title, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                running = False
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)
        try:
            cv2.destroyWindow(window_title)
        except Exception:
            pass

    def handcam_detect_bottle_stream_with_image(
        self,
        detector,
        *,
        source: str = "hand_color_image",
        interval: float = 1.0,
        jpeg_quality: int = 70,
        max_frames: Optional[int] = None,
        timeout: Optional[float] = None,
    ):
        if not str(source).startswith("hand_"):
            print(f"\x1b[33m[warn] 当前 source='{source}' 不是 hand_* 相机；仍将使用它进行检测。\x1b[0m")
        img_client = getattr(self, "img_client", None)
        if img_client is None:
            if getattr(self, "robot", None) is None:
                raise RuntimeError("SpotAgent 未登录/未持有 robot。")
            img_client = self.robot.ensure_client(ImageClient.default_service_name)
            self.img_client = img_client
        req = [build_image_request(
            source,
            image_format=image_pb2.Image.FORMAT_JPEG,
            quality_percent=int(jpeg_quality),
        )]
        period = max(0.1, float(interval))
        start = time.time()
        n = 0
        while True:
            if timeout is not None and (time.time() - start) >= timeout:
                print("[detect] 超时退出。")
                return
            if max_frames is not None and n >= max_frames:
                print("[detect] 达到最大帧数，退出。")
                return
            t0 = time.time()
            try:
                resp = img_client.get_image(req)[0]
                buf = np.frombuffer(resp.shot.image.data, dtype=np.uint8)
                frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)  # BGR
                xy = detector.detect_first_bottle_xy(frame)

                if xy is None:
                    print(f"[detect] Result: No Hit (source={source})")
                    yield (None, None)
                else:
                    xy_int = tuple(map(int, xy))
                    print(f"[detect] Result: Hit  ~ xy={xy_int} (source={source})")
                    yield (xy_int, resp)

            except Exception as e:
                print(f"[handcam] Failed:{e}")
                yield (None, None)
            n += 1
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)




