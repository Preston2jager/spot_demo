# Copyright (c) 2023 Boston Dynamics, Inc.
# Tutorial extended: YOLO auto-pick bottle nearest to image center, then place near ground before opening.

import sys
import time
from typing import Optional, Tuple

import cv2
import numpy as np

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.api import estop_pb2, geometry_pb2, image_pb2, manipulation_api_pb2
from bosdyn.client.estop import EstopClient
from bosdyn.client.frame_helpers import (
    VISION_FRAME_NAME,
    BODY_FRAME_NAME,
    get_vision_tform_body,
    math_helpers,
)
from bosdyn.client.image import ImageClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import (
    RobotCommandClient,
    blocking_stand,
    RobotCommandBuilder,
    block_until_arm_arrives,
)
from bosdyn.client.robot_state import RobotStateClient

# ----------------------- 固定参数（按需修改） -----------------------
HOSTNAME = "192.168.80.3"             # Spot 的 IP
IMAGE_SOURCE = "hand_color_image"     # 建议手腕相机：hand_color_image / hand_depth_in_hand_color_frame
USE_YOLO = True
YOLO_WEIGHTS = "yolov8n.pt"           # 也可用 yolo11n.pt 等
YOLO_CONF = 0.25

# 抓取姿态约束（只选一个 True；否则全 False）
FORCE_TOP_DOWN_GRASP = False
FORCE_HORIZONTAL_GRASP = False
FORCE_45_ANGLE_GRASP = False
FORCE_SQUEEZE_GRASP = False

# 放置点（以 BODY 坐标系），以及移动时长
PLACE_X = 0.60    # 前方 0.6 m
PLACE_Y = 0.00    # 机器人正前方
PLACE_Z = 0.05    # 距地面 ~5 cm（按需调高/调低）
PLACE_DURATION = 3.0  # 手到放置点的插补时间（秒）

SHOW_DEBUG_WINDOWS = True  # 是否弹窗显示调试图
# ---------------------------------------------------------------

# YOLO（Ultralytics）
try:
    from ultralytics import YOLO
    _UL_OK = True
except Exception:
    YOLO = None
    _UL_OK = False

g_image_click = None
g_image_display = None


def verify_estop(robot):
    client = robot.ensure_client(EstopClient.default_service_name)
    if client.get_status().stop_level != estop_pb2.ESTOP_LEVEL_NONE:
        error_message = (
            "Robot is estopped. Use an external E-Stop client to configure E-Stop."
        )
        robot.logger.error(error_message)
        raise Exception(error_message)


def run_yolo_select_bottle_center(
    img_bgr: np.ndarray, model, conf_thres: float
) -> Optional[Tuple[int, int]]:
    """返回最靠近图像中心的 bottle 框中心像素 (x,y)，未检测到返回 None。"""
    if img_bgr is None or img_bgr.size == 0:
        return None
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

    h, w = img_bgr.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    results = model.predict(source=img_bgr, verbose=False, conf=conf_thres)
    if not results:
        return None
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None

    names = r.names if hasattr(r, "names") else {}
    best_xy, best_d2 = None, None

    xyxy = r.boxes.xyxy.cpu().numpy()
    cls_ids = r.boxes.cls.cpu().numpy().astype(int)

    for (x1, y1, x2, y2), cid in zip(xyxy, cls_ids):
        name = names.get(cid, str(cid)).lower()
        if name != "bottle":
            continue
        mx, my = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        d2 = (mx - cx) ** 2 + (my - cy) ** 2
        if best_d2 is None or d2 < best_d2:
            best_d2, best_xy = d2, (int(round(mx)), int(round(my)))

    # 叠加可视化
    if SHOW_DEBUG_WINDOWS:
        global g_image_display
        g_image_display = img_bgr.copy()
        cv2.circle(g_image_display, (int(cx), int(cy)), 10, (255, 255, 255), 2)
        if best_xy is not None:
            cv2.drawMarker(g_image_display, best_xy, (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            cv2.putText(g_image_display, "YOLO bottle target", (best_xy[0] + 8, best_xy[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return best_xy


def move_hand_to_pose(command_client: RobotCommandClient,
                      hand_pose: geometry_pb2.SE3Pose,
                      root_frame: str,
                      seconds: float,
                      timeout_pad: float = 7.0):
    """兼容多版本 SDK 的 hand pose 发送：优先 arm_pose_command，失败则走轨迹构造。"""
    cmd_id = None
    try:
        # 可能是 arm_pose_command(pose, root_frame, seconds)
        cmd = RobotCommandBuilder.arm_pose_command(hand_pose, root_frame, seconds)
        cmd_id = command_client.robot_command(cmd)
    except TypeError:
        try:
            # 或者使用具名参数版本
            cmd = RobotCommandBuilder.arm_pose_command(
                hand_pose, root_frame_name=root_frame, duration=seconds
            )
            cmd_id = command_client.robot_command(cmd)
        except Exception:
            # 兜底：用轨迹接口
            tp = RobotCommandBuilder.create_arm_pose_trajectory_point(hand_pose, seconds)
            arm_traj = RobotCommandBuilder.create_arm_pose_command([tp], root_frame)
            cmd = RobotCommandBuilder.build_arm_command(arm_traj)
            cmd_id = command_client.robot_command(cmd)

    block_until_arm_arrives(
        command_client, cmd_id, timeout_sec=int(seconds + timeout_pad)
    )


def cv_mouse_callback(event, x, y, flags, param):
    global g_image_click, g_image_display
    clone = g_image_display.copy()
    if event == cv2.EVENT_LBUTTONUP:
        g_image_click = (x, y)
    else:
        color = (30, 30, 30)
        thickness = 2
        title = "Click to grasp"
        h, w = clone.shape[:2]
        cv2.line(clone, (0, y), (w, y), color, thickness)
        cv2.line(clone, (x, 0), (x, h), color, thickness)
        cv2.imshow(title, clone)


def add_grasp_constraint(cfg, grasp, robot_state_client):
    use_vector = FORCE_TOP_DOWN_GRASP or FORCE_HORIZONTAL_GRASP
    grasp.grasp_params.grasp_params_frame_name = VISION_FRAME_NAME

    if use_vector:
        if FORCE_TOP_DOWN_GRASP:
            axis_on_gripper = geometry_pb2.Vec3(x=1, y=0, z=0)
            axis_to_align = geometry_pb2.Vec3(x=0, y=0, z=-1)
        if FORCE_HORIZONTAL_GRASP:
            axis_on_gripper = geometry_pb2.Vec3(x=0, y=1, z=0)
            axis_to_align = geometry_pb2.Vec3(x=0, y=0, z=1)

        c = grasp.grasp_params.allowable_orientation.add()
        c.vector_alignment_with_tolerance.axis_on_gripper_ewrt_gripper.CopyFrom(axis_on_gripper)
        c.vector_alignment_with_tolerance.axis_to_align_with_ewrt_frame.CopyFrom(axis_to_align)
        c.vector_alignment_with_tolerance.threshold_radians = 0.17

    elif FORCE_45_ANGLE_GRASP:
        robot_state = robot_state_client.get_robot_state()
        vision_T_body = get_vision_tform_body(robot_state.kinematic_state.transforms_snapshot)
        body_Q_grasp = math_helpers.Quat.from_pitch(0.785398)  # 45°
        vision_Q_grasp = vision_T_body.rotation * body_Q_grasp
        c = grasp.grasp_params.allowable_orientation.add()
        c.rotation_with_tolerance.rotation_ewrt_frame.CopyFrom(vision_Q_grasp.to_proto())
        c.rotation_with_tolerance.threshold_radians = 0.17

    elif FORCE_SQUEEZE_GRASP:
        c = grasp.grasp_params.allowable_orientation.add()
        c.squeeze_grasp.SetInParent()


def arm_object_grasp():
    bosdyn.client.util.setup_logging(False)

    sdk = bosdyn.client.create_standard_sdk("ArmObjectGraspClient")
    robot = sdk.create_robot(HOSTNAME)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    assert robot.has_arm(), "Robot requires an arm."
    verify_estop(robot)

    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    image_client = robot.ensure_client(ImageClient.default_service_name)
    manipulation_api_client = robot.ensure_client(ManipulationApiClient.default_service_name)

    # YOLO
    yolo_model = None
    if USE_YOLO and _UL_OK:
        yolo_model = YOLO(YOLO_WEIGHTS)

    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        robot.logger.info("Powering on robot...")
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), "Robot power on failed."
        robot.logger.info("Robot powered on.")

        robot.logger.info("Standing...")
        command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        blocking_stand(command_client, timeout_sec=10)
        robot.logger.info("Robot standing.")

        # 拿一帧图
        robot.logger.info("Getting an image from: %s", IMAGE_SOURCE)
        image_responses = image_client.get_image_from_sources([IMAGE_SOURCE])
        if len(image_responses) != 1:
            raise RuntimeError(f"Invalid camera response count: {len(image_responses)}")

        image = image_responses[0]
        dtype = np.uint16 if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16 else np.uint8
        img = np.frombuffer(image.shot.image.data, dtype=dtype)
        if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
            img = img.reshape(image.shot.image.rows, image.shot.image.cols)
            if img.ndim == 2 and dtype == np.uint8:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img = cv2.imdecode(img, -1)

        # YOLO 自动选点
        target_xy = None
        if yolo_model is not None and image.shot.image.pixel_format != image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
            robot.logger.info("Running YOLO for 'bottle'...")
            target_xy = run_yolo_select_bottle_center(img, yolo_model, YOLO_CONF)
            if target_xy is not None:
                robot.logger.info(f"YOLO picked ({target_xy[0]}, {target_xy[1]})")
                if SHOW_DEBUG_WINDOWS and g_image_display is not None:
                    try:
                        cv2.imshow("YOLO target", g_image_display)
                        cv2.waitKey(500)
                        cv2.destroyWindow("YOLO target")
                    except Exception:
                        pass
            else:
                robot.logger.info("YOLO found no bottle; fall back to click.")

        # 若 YOLO 没选到，点击选点
        global g_image_click, g_image_display
        if target_xy is None:
            if SHOW_DEBUG_WINDOWS:
                image_title = "Click to grasp"
                cv2.namedWindow(image_title)
                cv2.setMouseCallback(image_title, cv_mouse_callback)
                g_image_display = img
                cv2.imshow(image_title, g_image_display)
                while g_image_click is None:
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), ord("Q")):
                        print('"q" pressed, exiting.')
                        return
                target_xy = g_image_click
            else:
                raise RuntimeError("No YOLO target and UI disabled. Enable SHOW_DEBUG_WINDOWS or add target source.")

        robot.logger.info(f"Picking at ({target_xy[0]}, {target_xy[1]})")

        pick_vec = geometry_pb2.Vec2(x=target_xy[0], y=target_xy[1])
        grasp = manipulation_api_pb2.PickObjectInImage(
            pixel_xy=pick_vec,
            transforms_snapshot_for_camera=image.shot.transforms_snapshot,
            frame_name_image_sensor=image.shot.frame_name_image_sensor,
            camera_model=image.source.pinhole,
        )
        add_grasp_constraint(None, grasp, robot_state_client)

        grasp_request = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=grasp)
        cmd_response = manipulation_api_client.manipulation_api_command(
            manipulation_api_request=grasp_request
        )

        # 轮询抓取结果
        grasp_ok = False
        while True:
            fb_req = manipulation_api_pb2.ManipulationApiFeedbackRequest(
                manipulation_cmd_id=cmd_response.manipulation_cmd_id
            )
            fb = manipulation_api_client.manipulation_api_feedback_command(
                manipulation_api_feedback_request=fb_req
            )
            print("Current state:",
                  manipulation_api_pb2.ManipulationFeedbackState.Name(fb.current_state))
            if fb.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED:
                grasp_ok = True
                break
            if fb.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_FAILED:
                break
            time.sleep(0.25)

        robot.logger.info("Grasp finished.")
        time.sleep(0.4)

        if grasp_ok:
            try:
                # 抬起（carry）
                robot.logger.info("Arm carry pose...")
                carry_cmd = RobotCommandBuilder.arm_carry_command()
                carry_id = command_client.robot_command(carry_cmd)
                block_until_arm_arrives(command_client, carry_id, timeout_sec=10)
                time.sleep(0.8)

                # ↓↓↓ 新增：移动到地面附近的放置点，再开爪
                robot.logger.info("Move hand to near-ground place pose...")
                place_pose = geometry_pb2.SE3Pose(
                    position=geometry_pb2.Vec3(x=PLACE_X, y=PLACE_Y, z=PLACE_Z),
                    rotation=math_helpers.Quat.from_yaw(0.0).to_proto(),  # 简单朝向
                )
                move_hand_to_pose(
                    command_client, place_pose, BODY_FRAME_NAME, seconds=PLACE_DURATION
                )

                robot.logger.info("Open gripper to release...")
                open_cmd = RobotCommandBuilder.claw_gripper_open_fraction_command(1.0)
                command_client.robot_command(open_cmd)
                time.sleep(1.2)

                robot.logger.info("Stow arm...")
                stow_cmd = RobotCommandBuilder.arm_stow_command()
                stow_id = command_client.robot_command(stow_cmd)
                block_until_arm_arrives(command_client, stow_id, timeout_sec=10)
            except Exception as e:
                robot.logger.error(f"Place sequence failed: {e}")
        else:
            robot.logger.info("Grasp failed, skip place.")

        time.sleep(0.8)
        robot.logger.info("Powering off (safe)...")
        robot.power_off(cut_immediately=False, timeout_sec=20)
        assert not robot.is_powered_on(), "Power off failed."
        robot.logger.info("Robot safely powered off.")


def main():
    try:
        arm_object_grasp()
        return True
    except Exception:
        logger = bosdyn.client.util.get_logger()
        logger.exception("Threw an exception")
        return False


if __name__ == "__main__":
    if not main():
        sys.exit(1)
