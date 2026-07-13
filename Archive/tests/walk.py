#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Spot 键盘遥控 + hand_color_image 实时直播（q 退出）
- 先 acquire lease，再 power_on / stand / move（避免 NoSuchLease: body）
- 速度命令使用 end_time_secs=now+0.9s 并以 ~10Hz 刷新（避免 ExpiredError）
- hand_color_image 强制请求 JPEG，OpenCV 稳定显示
"""

import argparse
import getpass
import sys
import time
from typing import Optional, Set

import cv2
import numpy as np

import bosdyn.client
from bosdyn.api import image_pb2
import bosdyn.client.util
from bosdyn.client.frame_helpers import VISION_FRAME_NAME
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.math_helpers import SE3Pose, Quat
from bosdyn.client.robot_command import (
    RobotCommandBuilder,
    RobotCommandClient,
    blocking_stand,
    block_until_arm_arrives,
)


# ----------- 默认参数 -----------
DEFAULT_SPEED = 0.6       # m/s
DEFAULT_ROT   = 1.2       # rad/s
CONTROL_HZ    = 10.0      # 发送速度命令频率
HOLD_SECONDS  = 0.9       # 每次速度命令的有效期（避免 ExpiredError）
WIN_NAME      = "Spot HandCam - hand_color_image"


def decode_image_to_cv2(resp) -> Optional[np.ndarray]:
    """将 ImageResponse 解码为 OpenCV BGR 图像（本脚本实际强制请求 JPEG）。"""
    img = resp.shot.image
    if img.format in (image_pb2.Image.FORMAT_JPEG, image_pb2.Image.FORMAT_PNG):
        arr = np.frombuffer(img.data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame
    elif img.format == image_pb2.Image.FORMAT_RAW:
        h, w = img.rows, img.cols
        if img.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGB_U8:
            arr = np.frombuffer(img.data, dtype=np.uint8).reshape(h, w, 3)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        elif img.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8:
            arr = np.frombuffer(img.data, dtype=np.uint8).reshape(h, w)
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    return None


def send_velocity(cmd_client: RobotCommandClient, v_x: float, v_y: float, v_rot: float, hold: float = HOLD_SECONDS):
    """发送一次短时效速度命令，并由上层定时循环刷新。"""
    cmd = RobotCommandBuilder.synchro_velocity_command(v_x=v_x, v_y=v_y, v_rot=v_rot)
    cmd_client.robot_command(cmd, end_time_secs=time.time() + hold)


def set_arm_named_pose(cmd_client: RobotCommandClient, name: str):
    if name == "ready":
        arm_cmd = RobotCommandBuilder.arm_ready_command()
    elif name == "carry":
        arm_cmd = RobotCommandBuilder.arm_carry_command()
    elif name == "stow":
        arm_cmd = RobotCommandBuilder.arm_stow_command()
    else:
        return
    cmd_id = cmd_client.robot_command(arm_cmd)
    block_until_arm_arrives(cmd_client, cmd_id, timeout_sec=6.0)


def raise_arm_headish(cmd_client: RobotCommandClient):
    """
    将手端抬到机身前上方（类似“抬头”），使用正确的 from_pose 接口。
    坐标系：VISION_FRAME
    """
    hand_pose = SE3Pose(x=0.55, y=0.0, z=0.45, rot=Quat())  # 单位四元数：不改变朝向
    arm_cmd = RobotCommandBuilder.arm_pose_command_from_pose(
        hand_pose.to_proto(), VISION_FRAME_NAME, seconds=2.5
    )
    cmd_id = cmd_client.robot_command(arm_cmd)
    block_until_arm_arrives(cmd_client, cmd_id, timeout_sec=5.0)


def build_keysets() -> dict:
    """兼容 OpenCV 的方向键键值（不同平台会有差异）。"""
    return {
        "UP":    {82, 2490368},   # ↑
        "DOWN":  {84, 2621440},   # ↓
        "LEFT":  {81, 2424832},   # ←
        "RIGHT": {83, 2555904},   # →
    }


def main():
    parser = argparse.ArgumentParser(description="Spot 键盘遥控 + hand_color_image 实时直播（q 退出）")
    parser.add_argument("hostname", help="Spot 的 IP 或主机名，例如 192.168.80.3")
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED, help="线速度 m/s")
    parser.add_argument("--rot-speed", type=float, default=DEFAULT_ROT, help="角速度 rad/s")
    parser.add_argument("--fps", type=float, default=15.0, help="图像刷新节流（不影响控制频率）")
    args = parser.parse_args()

    username = input("Username: ")
    password = getpass.getpass(f"[{username}] Password: ")

    # --- 连接、认证、时钟同步 ---
    sdk = bosdyn.client.create_standard_sdk("SpotWalkTeleop")
    robot = sdk.create_robot(args.hostname)
    robot.authenticate(username, password)
    robot.time_sync.wait_for_sync()

    # --- 客户端 ---
    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    img_client   = robot.ensure_client(ImageClient.default_service_name)
    cmd_client   = robot.ensure_client(RobotCommandClient.default_service_name)

    # --- 先拿租约，再上电 / 站立 / 控制 ---
    with LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        print("[robot] 上电中...")
        robot.power_on(timeout_sec=20)

        print("[robot] 站立中...")
        blocking_stand(cmd_client, timeout_sec=10)

        # 手臂展开 + 抬到“头部”位姿
        print("[arm] Arm Ready ...")
        set_arm_named_pose(cmd_client, "ready")
        try:
            print("[arm] Raise head-ish ...")
            raise_arm_headish(cmd_client)
        except Exception as e:
            print(f"[arm] Raise 失败（忽略继续）：{e}")

        # 相机请求：强制 JPEG，避免 PNG 兼容问题
        img_req = [build_image_request(
            "hand_color_image",
            image_format=image_pb2.Image.FORMAT_JPEG,
            quality_percent=70
        )]

        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_NAME, 960, 720)
        print("[control] ↑↓←→ / WASD 控制；Space 急停；E/R/C/T 切换手臂姿态；Q 退出。")

        # 控制循环
        keysets = build_keysets()
        period = 1.0 / CONTROL_HZ
        last_send = 0.0
        running = True

        v_x = v_y = v_rot = 0.0

        while running:
            # --- 图像获取（独立于控制频率，仅做简单节流） ---
            try:
                resp_list = img_client.get_image(img_req)
                if resp_list:
                    frame = decode_image_to_cv2(resp_list[0])
                    if frame is not None:
                        hud = [
                            "Arrows/WASD drive  Space: stop  Q: quit",
                            "E: arm_ready  R: raise(head-ish)  C: carry  T: stow",
                            f"v_x={v_x:+.2f}  v_y={v_y:+.2f}  v_rot={v_rot:+.2f}  src=hand_color_image",
                        ]
                        y = 28
                        for text in hud:
                            cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            y += 28
                        cv2.imshow(WIN_NAME, frame)
            except Exception as e:
                # 摄像头暂不可用时也不阻塞控制
                black = np.zeros((720, 960, 3), np.uint8)
                cv2.putText(black, f"Image error: {e}", (20, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow(WIN_NAME, black)

            # --- 键盘读取（按住即动，松开即停） ---
            v_x = v_y = v_rot = 0.0
            k = cv2.waitKeyEx(1) & 0xFFFFFFFF

            if k in (ord("q"), ord("Q")):
                running = False
            elif k in keysets["UP"] or k in (ord("w"), ord("W")):
                v_x = +args.speed
            elif k in keysets["DOWN"] or k in (ord("s"), ord("S")):
                v_x = -args.speed
            elif k in keysets["LEFT"]:
                v_rot = +args.rot_speed
            elif k in keysets["RIGHT"]:
                v_rot = -args.rot_speed
            elif k in (ord("a"), ord("A")):
                v_y = +0.6 * args.speed  # 侧移（正 y 左移）
            elif k in (ord("d"), ord("D")):
                v_y = -0.6 * args.speed  # 侧移（负 y 右移）
            elif k == ord(" "):
                v_x = v_y = v_rot = 0.0
            elif k in (ord("e"), ord("E")):
                set_arm_named_pose(cmd_client, "ready")
            elif k in (ord("c"), ord("C")):
                set_arm_named_pose(cmd_client, "carry")
            elif k in (ord("t"), ord("T")):
                set_arm_named_pose(cmd_client, "stow")
            elif k in (ord("r"), ord("R")):
                try:
                    raise_arm_headish(cmd_client)
                except Exception as e:
                    print(f"[arm] Raise 失败：{e}")

            # --- 定频发送速度命令，end_time 给足 ---
            now = time.time()
            if (now - last_send) >= period:
                try:
                    send_velocity(cmd_client, v_x, v_y, v_rot, hold=HOLD_SECONDS)
                except Exception as e:
                    print(f"[command] 发送失败：{e}")
                last_send = now

        # 退出：停稳并收臂
        try:
            send_velocity(cmd_client, 0.0, 0.0, 0.0, hold=0.5)
        except Exception:
            pass
        try:
            set_arm_named_pose(cmd_client, "stow")
        except Exception:
            pass
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[exit] Ctrl-C")
        sys.exit(0)
