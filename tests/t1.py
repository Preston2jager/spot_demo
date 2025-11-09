#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import threading
import time
from typing import List

import cv2
import numpy as np

import bosdyn.client
import bosdyn.client.util
from bosdyn.client.lease import LeaseKeepAlive, LeaseClient
from bosdyn.client.robot_command import (
    RobotCommandClient,
    RobotCommandBuilder,
    blocking_stand,
)
from bosdyn.client.image import ImageClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.api import image_pb2


def decode_image_to_cv2(img_resp: image_pb2.ImageResponse):
    """将 ImageResponse 解码为 OpenCV 图像 (BGR)."""
    shot = img_resp.shot.image
    if shot.format == image_pb2.Image.FORMAT_JPEG or shot.format == image_pb2.Image.FORMAT_PNG:
        data = np.frombuffer(shot.data, dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        # SDK 的 JPEG 通常为灰度或RGB，这里统一转 BGR 以便显示
        if frame is None:
            return None
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame
    else:
        # RAW / RLE：根据像素格式重构矩阵
        h, w = shot.rows, shot.cols
        if shot.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGB_U8:
            arr = np.frombuffer(shot.data, dtype=np.uint8).reshape((h, w, 3))
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        elif shot.pixel_format in (
            image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8,
            image_pb2.Image.PIXEL_FORMAT_DEPTH_U16,
            image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U16,
        ):
            dtype = np.uint16 if "U16" in image_pb2.Image.PixelFormat.Name(shot.pixel_format) else np.uint8
            arr = np.frombuffer(shot.data, dtype=dtype).reshape((h, w))
            if arr.dtype != np.uint8:
                # 简单归一化到可视范围（仅预览用途）
                arr = (255.0 * (arr - arr.min()) / max(1, arr.ptp())).astype(np.uint8)
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        else:
            return None


def camera_stream_loop(image_client: ImageClient, sources: List[str], stop_evt: threading.Event, fps: float = 10.0):
    """相机采集线程：循环抓取并显示多个源。按 'q' 退出。"""
    win_names = [f"Spot Camera - {s}" for s in sources]
    try:
        while not stop_evt.is_set():
            try:
                resps = image_client.get_image_from_sources(sources)
            except Exception as e:
                print(f"[image] 获取图像失败：{e}")
                time.sleep(0.5)
                continue

            for resp, win in zip(resps, win_names):
                frame = decode_image_to_cv2(resp)
                if frame is not None:
                    cv2.imshow(win, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                stop_evt.set()
                break

            time.sleep(max(0.0, 1.0 / fps))
    finally:
        for win in win_names:
            try:
                cv2.destroyWindow(win)
            except:
                pass


def velocity_for(command_client: RobotCommandClient, v_x: float, v_y: float, v_rot: float, duration: float):
    """以给定速度运行 duration 秒（SDK 速度/轨迹命令需要 end_time_secs）。"""
    cmd = RobotCommandBuilder.synchro_velocity_command(v_x=v_x, v_y=v_y, v_rot=v_rot)
    end_time = time.time() + duration
    command_client.robot_command(cmd, end_time_secs=end_time)  # 关键：设置 end_time_secs
    # 让命令自然到时结束
    time.sleep(duration)
    # 发送 0 速度以确保停稳
    stop_cmd = RobotCommandBuilder.synchro_velocity_command(v_x=0.0, v_y=0.0, v_rot=0.0)
    command_client.robot_command(stop_cmd, end_time_secs=time.time() + 0.2)


def walk_back_and_forth(command_client: RobotCommandClient, distance_m: float, speed_mps: float,
                        rot_speed_rps: float, repeats: int):
    """执行来回走动（前进 -> 原地180° -> 再前进 -> 原地180°），重复 repeats 趟。"""
    forward_time = abs(distance_m) / max(0.05, abs(speed_mps))
    turn_time = np.pi / max(0.05, abs(rot_speed_rps))

    for i in range(repeats):
        print(f"[motion] 第 {i+1}/{repeats} 趟：直行 {distance_m} m")
        velocity_for(command_client, v_x=np.sign(distance_m) * abs(speed_mps), v_y=0.0, v_rot=0.0, duration=forward_time)

        print("[motion] 原地旋转 180°")
        velocity_for(command_client, v_x=0.0, v_y=0.0, v_rot=rot_speed_rps, duration=turn_time)

        print(f"[motion] 再直行 {distance_m} m（返回）")
        velocity_for(command_client, v_x=np.sign(distance_m) * abs(speed_mps), v_y=0.0, v_rot=0.0, duration=forward_time)

        print("[motion] 再旋转回 180°（恢复原朝向）")
        velocity_for(command_client, v_x=0.0, v_y=0.0, v_rot=rot_speed_rps, duration=turn_time)


def main():
    parser = argparse.ArgumentParser(description="Spot 前后行走 + 相机投屏（OpenCV）")
    bosdyn.client.util.add_base_arguments(parser)  # --username/--password 等
    parser.add_argument("hostname", help="Spot 机器人 IP / 主机名")
    parser.add_argument("--distance", type=float, default=1.0, help="每段直行距离（米）")
    parser.add_argument("--speed", type=float, default=0.5, help="直行速度（米/秒）")
    parser.add_argument("--rot-speed", type=float, default=1.0, help="原地旋转角速度（弧度/秒）")
    parser.add_argument("--repeats", type=int, default=1, help="来回趟数")
    parser.add_argument("--sources", nargs="+", default=["frontleft_fisheye_image"], help="要显示的相机源名")
    parser.add_argument("--list-sources", action="store_true", help="仅列出可用相机源并退出")
    args = parser.parse_args()

    # 初始化 SDK / 连接 / 鉴权
    sdk = bosdyn.client.create_standard_sdk("WalkAndStreamClient")
    robot = sdk.create_robot(args.hostname)
    bosdyn.client.util.authenticate(robot)  # 会根据 add_base_arguments 处理用户名密码
    robot.time_sync.wait_for_sync()

    # 可选：列出相机源
    image_client = robot.ensure_client(ImageClient.default_service_name)
    if args.list_sources:
        sources = image_client.list_image_sources()
        print("可用相机源：")
        for s in sources:
            print(" -", s.name)
        return 0

    # 租约 & 上电 & 站立
    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    with LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        if not robot.is_powered_on():
            print("[robot] 上电中...")
            robot.power_on(timeout_sec=20)
        cmd_client = robot.ensure_client(RobotCommandClient.default_service_name)

        print("[robot] 站立中...")
        blocking_stand(cmd_client, timeout_sec=10)

        # 启动相机线程
        stop_evt = threading.Event()
        cam_thread = threading.Thread(
            target=camera_stream_loop, args=(image_client, args.sources, stop_evt), daemon=True
        )
        cam_thread.start()

        try:
            # 执行来回走动
            walk_back_and_forth(
                cmd_client,
                distance_m=args.distance,
                speed_mps=args.speed,
                rot_speed_rps=args.rot_speed,
                repeats=args.repeats,
            )
        finally:
            # 停止相机
            stop_evt.set()
            cam_thread.join(timeout=2.0)
            # 站立保持/可按需坐下或断电
            print("[robot] 行走结束，保持站立。按需手动 power_off。")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[exit] 用户中断")
        sys.exit(0)
