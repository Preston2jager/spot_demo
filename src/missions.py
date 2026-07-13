import time
import functools
import cv2
import questionary
import math
import numpy as np
from loguru import logger
from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn.client.frame_helpers import (
    get_a_tform_b,
    VISION_FRAME_NAME,
    GRAV_ALIGNED_BODY_FRAME_NAME,
    math_helpers,
)
from bosdyn.client.math_helpers import SE3Pose, Quat
from bosdyn.client.image import build_image_request
from bosdyn.api import image_pb2

from .spot_agent import SpotAgent
from .spot_detector_openvino import SpotDetector


def safety_fallback_decorator(func):
    @functools.wraps(func)
    def wrapper(agent: SpotAgent, *args, **kwargs):
        try:
            return func(agent, *args, **kwargs)
        except KeyboardInterrupt:
            # Let KeyboardInterrupt propagate directly
            raise
        except Exception as e:
            logger.error(f"Safety Fallback: Error encountered: {e}")
            try:
                # 1. Close gripper and stow arm
                logger.info("Safety Fallback: Retracting arm to stow...")
                agent._arm_in()
                time.sleep(1.0)
                # 2. Return to nominal stand
                logger.info("Safety Fallback: Resetting robot to nominal stand...")
                cmd = RobotCommandBuilder.synchro_stand_command()
                agent.cmd_client.robot_command(cmd)
                time.sleep(1.5)
            except Exception as inner_err:
                logger.error(f"Safety Fallback: Failed to retract arm/stand: {inner_err}")
            
            # Ask the user if they want to stop/exit
            try:
                confirm = questionary.confirm("An error occurred. Do you want to stop/exit?").ask()
                if confirm:
                    logger.info("Safety Fallback: Sitting down...")
                    agent.sit()
                    logger.info("Safety Fallback: Powering off...")
                    agent.power_off()
            except Exception as confirm_err:
                logger.error(f"Safety Fallback: Failed during confirmation or sit: {confirm_err}")
    return wrapper


@safety_fallback_decorator
def run_go_home(agent: SpotAgent, waypoint_name: str) -> None:
    wp_id = agent.get_waypoint_id_by_name(agent.graph, waypoint_name)
    if not wp_id:
        logger.error(f"Waypoint not found: {waypoint_name}")
        return
    agent.navigate_to_waypoint(wp_id)


@safety_fallback_decorator
def run_guard_loop(agent: SpotAgent, guard_point: str, home_point: str) -> None:
    detector = SpotDetector()
    guard_id = agent.get_waypoint_id_by_name(agent.graph, guard_point)
    home_id = agent.get_waypoint_id_by_name(agent.graph, home_point)
    if not guard_id or not home_id:
        logger.error("Required waypoints not found — aborting mission.")
        return
    logger.info("Starting guard loop. Ctrl+C to stop.")
    try:
        while True:
            logger.info(f"[Phase 1] Navigating to guard point: {guard_point}")
            if not agent.navigate_to_waypoint(guard_id):
                logger.warning("Failed to reach guard point. Retrying in 3 s...")
                time.sleep(3.0)
                continue
            logger.info("At guard point — searching for target...")
            target_grasped = False
            while not target_grasped:
                agent._arm_out()
                time.sleep(2.0)
                detected = agent.find_target(detector)
                if detected is None:
                    logger.info("No target found. Rescanning in 3 s...")
                    time.sleep(3.0)
                    continue
                target_img_resp, cam_name, cx, cy, cls_name = detected
                if agent.grasp_object(target_img_resp, cam_name, cx, cy, cls_name):
                    logger.info("Target grasped.")
                    target_grasped = True
                    time.sleep(0.2)
                else:
                    logger.warning("Grasp failed — returning to guard point...")
                    agent.navigate_to_waypoint(guard_id)
                    time.sleep(2.0)
            logger.info(f"[Phase 2] Returning home: {home_point}")
            if agent.navigate_to_waypoint(home_id):
                logger.info("At home — releasing target.")
                time.sleep(0.2)
                agent._arm_release(bin=True)
            else:
                logger.error("Failed to reach home — stopping mission.")
                break
            logger.info("Cycle complete. Starting next cycle...")
            time.sleep(2.0)
    except KeyboardInterrupt:
        logger.info("Guard loop interrupted by user.")


@safety_fallback_decorator
def run_guard_first(
    agent: SpotAgent,
    guard_point: str,
    home_point: str,
    initial_route: list[str],
    standby_point: str,
) -> None:
    detector = SpotDetector()
    guard_id = agent.get_waypoint_id_by_name(agent.graph, guard_point)
    home_id = agent.get_waypoint_id_by_name(agent.graph, home_point)
    standby_id = agent.get_waypoint_id_by_name(agent.graph, standby_point)
    route_ids = [agent.get_waypoint_id_by_name(agent.graph, wp) for wp in initial_route]
    if not all([guard_id, home_id, standby_id] + route_ids):
        logger.error("Required waypoints not found — aborting mission.")
        return
    logger.info("Running initial patrol route...")
    for i, wp_id in enumerate(route_ids):
        agent.navigate_to_waypoint(wp_id)
        if i == 1:
            time.sleep(5)
    logger.info(f"Navigating to guard point: {guard_point}")
    while not agent.navigate_to_waypoint(guard_id):
        logger.warning("Failed to reach guard point. Retrying in 3 s...")
        time.sleep(3.0)
    logger.info("At guard point — searching for target...")
    target_grasped = False
    while not target_grasped:
        agent._arm_out()
        time.sleep(2.0)
        detected = agent.find_target(detector)
        if detected is None:
            logger.info("No object detected. Waiting 3 s...")
            time.sleep(3.0)
            continue
        target_img_resp, cam_name, cx, cy, cls_name = detected
        if agent.grasp_object(target_img_resp, cam_name, cx, cy, cls_name):
            logger.info("Target grasped.")
            target_grasped = True
            time.sleep(0.2)
        else:
            logger.warning("Grasp failed — returning to guard point...")
            agent.navigate_to_waypoint(guard_id)
            time.sleep(2.0)
    logger.info(f"Returning home: {home_point}")
    if agent.navigate_to_waypoint(home_id):
        logger.info("At home — releasing target.")
        time.sleep(0.2)
        agent._arm_release(bin=True)
        time.sleep(1.2)
        logger.info(f"Moving to standby: {standby_point}")
        agent.navigate_to_waypoint(standby_id)
        logger.info("Mission complete.")
    else:
        logger.error("Failed to reach home — check manually.")


@safety_fallback_decorator
def run_guard_no_graphnav(agent: SpotAgent) -> None:
    detector = SpotDetector()
    
    # 1. 启动时记录起始位置
    start_x, start_y, start_yaw = agent.get_current_pose("vision")
    agent.start_x = start_x
    agent.start_y = start_y
    agent.start_yaw = start_yaw
    logger.info(f"Recorded starting position: X={start_x:.2f}, Y={start_y:.2f}, Yaw={math.degrees(start_yaw):.1f}°")

    # 2. 伸臂，寻找面前的瓶子 (尝试 5 次)
    logger.info("Looking for a bottle in front...")
    detected = None
    for attempt in range(5):
        agent._arm_out()
        time.sleep(2.0)
        detected = agent.find_target(detector)
        if detected is not None:
            break
        logger.info(f"Attempt {attempt + 1}: No bottle detected. Retrying in 2 s...")
        time.sleep(2.0)

    if detected is None:
        logger.error("No bottle detected in front. Aborting mission.")
        agent._arm_in()
        return

    target_img_resp, cam_name, cx, cy, cls_name = detected
    logger.info(f"Detected target: {cls_name} at pixel ({cx}, {cy}) using camera {cam_name}")

    # 3. 走近瓶子
    source = target_img_resp.source
    cam_model = getattr(source, "pinhole", None) or getattr(source, "fisheye", None) or source.pinhole
    intrinsics = cam_model.intrinsics
    
    fx, fy = intrinsics.focal_length.x, intrinsics.focal_length.y
    cx_cam, cy_cam = intrinsics.principal_point.x, intrinsics.principal_point.y
    
    x_cam = (cx - cx_cam) / fx
    y_cam = (cy - cy_cam) / fy
    z_cam = 1.0 
    length = math.sqrt(x_cam**2 + y_cam**2 + z_cam**2)
    
    assumed_dist = 0.8  
    target_cam = SE3Pose(
        x=(x_cam/length)*assumed_dist, 
        y=(y_cam/length)*assumed_dist, 
        z=(z_cam/length)*assumed_dist, 
        rot=Quat()
    )
                         
    root_frame = VISION_FRAME_NAME
    cam_frame = target_img_resp.shot.frame_name_image_sensor
    camera_snapshot = target_img_resp.shot.transforms_snapshot
    
    world_T_cam = get_a_tform_b(camera_snapshot, root_frame, cam_frame)
    target_world = world_T_cam * target_cam
    obj_x, obj_y = target_world.x, target_world.y

    state = agent.state_client.get_robot_state()
    tf = state.kinematic_state.transforms_snapshot
    world_T_body = get_a_tform_b(tf, root_frame, "body")
    rob_x, rob_y = world_T_body.position.x, world_T_body.position.y
    
    dx = obj_x - rob_x
    dy = obj_y - rob_y
    angle = math.atan2(dy, dx)
    
    # 采用 0.95 米的最佳 standoff 距离，保证机械臂向前伸时（0.8m）手部相机刚好在瓶子前方约 15cm 处
    standoff_dist = 0.95
    nav_x = obj_x - standoff_dist * math.cos(angle)
    nav_y = obj_y - standoff_dist * math.sin(angle)
    nav_yaw_deg = math.degrees(angle)
    
    logger.info(f"Target estimated coordinates: X={obj_x:.2f}, Y={obj_y:.2f}")
    logger.info(f"Walking to prep stance (standoff={standoff_dist}m): X={nav_x:.2f}, Y={nav_y:.2f}, Yaw={nav_yaw_deg:.1f}°")
    
    # 移动到对齐抓取的位置
    if not agent.move_to_goal(x=nav_x, y=nav_y, angle_deg=nav_yaw_deg, frame="vision", use_local_origin=False, timeout_sec=12.0):
        logger.warning("Failed to reach prep stance. Attempting grasp anyway.")

    # 4. 精确定位和出抓
    # 保持 _arm_out 姿态重新对齐扫描（此时手部相机在瓶子前方 15cm）
    logger.info("Extending arm forward to scan for target...")
    agent._arm_out()
    time.sleep(2.0)

    logger.info("Scanning for precise target locking...")
    target_grasped = False
    try:
        image_responses = agent.img_client.get_image([build_image_request("hand_color_image", quality_percent=70)])
        if image_responses:
            res = image_responses[0]
            img_np = np.frombuffer(res.shot.image.data, dtype=np.uint8)
            img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            
            results = detector.detect_targets_in_batch({'hand_color_image': img}, conf=0.05)
            if results:
                best = results[0]
                new_x, new_y = best['cx'], best['cy']
                logger.info(f"Target relocalized at pixel: ({new_x}, {new_y}). Executing grasp...")
                if agent.grasp_object(res, "hand_color_image", new_x, new_y, best['class']):
                    logger.info("Target grasped successfully.")
                    target_grasped = True
            else:
                logger.warning("Lost target after walking closer. Trying fallback direct grasp using original detection...")
                # 备用方案：由于在极近距离下有些瓶子轮廓可能被切割从而无法在第二帧检测中识别，
                # 我们可以直接向之前解析的、已经非常准确的绝对空间点发送抓取指令。
                if agent.grasp_object(target_img_resp, cam_name, cx, cy, cls_name):
                    logger.info("Fallback grasp succeeded.")
                    target_grasped = True
    except Exception as e:
        logger.error(f"Precise localization/grasp failed: {e}")

    if not target_grasped:
        logger.error("Grasp failed. Returning to start position...")
        agent._recover_arm_safely()
        agent.move_to_goal(x=start_x, y=start_y, angle_deg=math.degrees(start_yaw), frame="vision", use_local_origin=False, timeout_sec=15.0)
        return

    # 5. 走回到启动点，平放到面前
    logger.info("Walking back to the starting point...")
    time.sleep(0.5)
    if not agent.move_to_goal(x=start_x, y=start_y, angle_deg=math.degrees(start_yaw), frame="vision", use_local_origin=False, timeout_sec=15.0):
        logger.warning("Timed out or failed walking back to start. Placing object here.")

    agent.place_flat_in_front()
    logger.info("Mission complete.")


@safety_fallback_decorator
def run_patrol(agent: SpotAgent, route: list[str]) -> None:
    ids = []
    for wp_name in route:
        wp_id = agent.get_waypoint_id_by_name(agent.graph, wp_name)
        if wp_id:
            ids.append(wp_id)
        else:
            logger.warning(f"Waypoint not found: {wp_name}")
    for wp_id in ids:
        agent.navigate_to_waypoint(wp_id)
    time.sleep(1.0)


@safety_fallback_decorator
def run_reset(agent: SpotAgent) -> None:
    logger.info("Reset: Spot is loaded and standing. Querying battery info...")

    # 1. Query battery state and display
    try:
        state = agent.state_client.get_robot_state()
        if state.battery_states:
            for idx, battery_state in enumerate(state.battery_states):
                pct = battery_state.charge_percentage.value
                status = battery_state.Status.Name(battery_state.status)
                runtime_secs = battery_state.estimated_runtime.seconds
                logger.info(f"Battery {idx}: Status={status}, Charge={pct:.1f}%, Est. Runtime={runtime_secs // 60}m {runtime_secs % 60}s")
        else:
            logger.warning("No battery states reported.")
    except Exception as e:
        logger.error(f"Failed to query battery state: {e}")

    logger.info("Reset: Holding nominal stand for 2 s...")
    time.sleep(2.0)
    logger.info("Reset: Spot is sitting down...")
    agent.sit()
    logger.info("Reset: Spot is powering off...")
    agent.power_off()


@safety_fallback_decorator
def run_stretch(agent: SpotAgent) -> None:
    logger.info("Stretch: Spot is loaded and standing. Performing a deep stretch...")
    from bosdyn.geometry import EulerZXY
    from bosdyn.client.math_helpers import SE3Pose
    from bosdyn.api import trajectory_pb2
    from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2

    try:
        # Deep stretch: butt backward, head down, front legs extended as much as possible
        # x_offset = -0.12 (butt backward), z_offset = -0.06 (lower body), pitch = -0.35 (head down)
        logger.info("Stretch: stretching (head down, butt back)...")

        quat = EulerZXY(pitch=-0.35).to_quaternion()
        pose = SE3Pose(x=-0.12, y=0.0, z=-0.06, rot=quat).to_proto()
        point = trajectory_pb2.SE3TrajectoryPoint(pose=pose)
        traj = trajectory_pb2.SE3Trajectory(points=[point])
        body_control = spot_command_pb2.BodyControlParams(base_offset_rt_footprint=traj)
        params = spot_command_pb2.MobilityParams(body_control=body_control)

        cmd = RobotCommandBuilder.synchro_stand_command(params=params)
        agent.cmd_client.robot_command(cmd)
        time.sleep(4.0)

        # Return to nominal stand
        logger.info("Stretch: Returning to nominal stance...")
        cmd_nominal = RobotCommandBuilder.synchro_stand_command()
        agent.cmd_client.robot_command(cmd_nominal)
        time.sleep(2.0)
    except Exception as e:
        logger.error(f"Failed to stretch: {e}")

    logger.info("Stretch: Spot is sitting down...")
    agent.sit()
    logger.info("Stretch: Spot is powering off...")
    agent.power_off()


@safety_fallback_decorator
def run_stepping_in_place(agent: SpotAgent) -> None:
    logger.info("Stepping: Spot is standing. Preparing to step in place...")
    from bosdyn.client import frame_helpers, math_helpers
    from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2

    try:
        # Get current body pose to construct relative stance targets
        state = agent.state_client.get_robot_state()
        vo_T_body = frame_helpers.get_se2_a_tform_b(
            state.kinematic_state.transforms_snapshot,
            frame_helpers.VISION_FRAME_NAME,
            frame_helpers.GRAV_ALIGNED_BODY_FRAME_NAME
        )

        if vo_T_body is None:
            logger.error("Stepping: Could not get transform from body to vision frame.")
            return

        x_offset = 0.30
        y_offset = 0.22
        step_dist = 0.08  # 8cm step distance
        step_duration = 0.6  # time to step
        cycles = 3

        # Configure swing height to HIGH for a very clear leg lifting action
        params = spot_command_pb2.MobilityParams(
            swing_height=spot_command_pb2.SWING_HEIGHT_HIGH
        )

        # Define nominal (rest) positions for all 4 feet
        fl_nom = vo_T_body * math_helpers.SE2Pose(x_offset, y_offset, 0)
        fr_nom = vo_T_body * math_helpers.SE2Pose(x_offset, -y_offset, 0)
        hl_nom = vo_T_body * math_helpers.SE2Pose(-x_offset, y_offset, 0)
        hr_nom = vo_T_body * math_helpers.SE2Pose(-x_offset, -y_offset, 0)

        # Define stepped positions (shift forward by step_dist)
        fl_step = vo_T_body * math_helpers.SE2Pose(x_offset + step_dist, y_offset, 0)
        fr_step = vo_T_body * math_helpers.SE2Pose(x_offset + step_dist, -y_offset, 0)
        hl_step = vo_T_body * math_helpers.SE2Pose(-x_offset + step_dist, y_offset, 0)
        hr_step = vo_T_body * math_helpers.SE2Pose(-x_offset + step_dist, -y_offset, 0)

        def send_stance(fl, fr, hl, hr):
            stance_cmd = RobotCommandBuilder.stance_command(
                frame_helpers.VISION_FRAME_NAME,
                fl.position, fr.position, hl.position, hr.position,
                params=params
            )
            # Stance commands require a short-term end_time
            end_time_secs = time.time() + step_duration + 2.0
            stance_cmd.synchronized_command.mobility_command.stance_request.end_time.CopyFrom(
                agent.robot.time_sync.robot_timestamp_from_local_secs(end_time_secs)
            )
            agent.cmd_client.robot_command(stance_cmd)
            time.sleep(step_duration)

        logger.info("Stepping: Starting clockwise step sequence (FL -> FR -> HR -> HL)...")
        for cycle in range(cycles):
            logger.info(f"Stepping Cycle {cycle + 1}/{cycles}...")

            # 1. Front Left (FL)
            logger.info("Stepping FL...")
            send_stance(fl_step, fr_nom, hl_nom, hr_nom)
            send_stance(fl_nom, fr_nom, hl_nom, hr_nom)

            # 2. Front Right (FR)
            logger.info("Stepping FR...")
            send_stance(fl_nom, fr_step, hl_nom, hr_nom)
            send_stance(fl_nom, fr_nom, hl_nom, hr_nom)

            # 3. Hind Right (HR)
            logger.info("Stepping HR...")
            send_stance(fl_nom, fr_nom, hl_nom, hr_step)
            send_stance(fl_nom, fr_nom, hl_nom, hr_nom)

            # 4. Hind Left (HL)
            logger.info("Stepping HL...")
            send_stance(fl_nom, fr_nom, hl_step, hr_nom)
            send_stance(fl_nom, fr_nom, hl_nom, hr_nom)

        logger.info("Stepping: sequence completed.")
    except Exception as e:
        logger.error(f"Failed to step in place: {e}")

    logger.info("Stepping: Spot is sitting down...")
    agent.sit()
    logger.info("Stepping: Spot is powering off...")
    agent.power_off()


@safety_fallback_decorator
def run_wasd_teleop(agent: SpotAgent) -> None:
    import cv2
    import numpy as np
    from bosdyn.client.image import build_image_request
    from bosdyn.api import image_pb2
    from bosdyn.client.math_helpers import Quat
    from bosdyn.client.frame_helpers import BODY_FRAME_NAME

    logger.info("WASD Teleop: Pointing arm straight ahead and opening gripper...")

    # Point arm straight ahead
    try:
        q = Quat()  # identity quaternion
        arm_cmd = RobotCommandBuilder.arm_pose_command(
            0.80, 0.0, 0.20,
            q.w, q.x, q.y, q.z,
            BODY_FRAME_NAME,
            2.0
        )
        agent.cmd_client.robot_command(arm_cmd)
        time.sleep(1.5)
        # Open gripper
        agent.cmd_client.robot_command(RobotCommandBuilder.claw_gripper_open_command())
        time.sleep(0.5)
    except Exception as e:
        logger.error(f"Failed to position arm: {e}")

    logger.info("WASD Teleop: Starting video stream window. Focus the OpenCV window to drive!")

    img_req = [build_image_request(
        "hand_color_image",
        image_format=image_pb2.Image.FORMAT_JPEG,
        quality_percent=70
    )]

    WIN_NAME = "Spot WASD Video Stream"
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, 960, 720)

    try:
        while True:
            v_x = v_y = v_rot = 0.0
            desc = ""

            # 1. Grab image and display
            try:
                resp_list = agent.img_client.get_image(img_req)
                if resp_list:
                    img_resp = resp_list[0]
                    img_data = np.frombuffer(img_resp.shot.image.data, dtype=np.uint8)
                    frame = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

                    if frame is not None:
                        hud = [
                            "WASD: Drive  Q/E: Turn",
                            "X or ESC: Exit & Sit Down",
                        ]
                        y = 30
                        for text in hud:
                            cv2.putText(frame, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            y += 25
                        cv2.imshow(WIN_NAME, frame)
            except Exception as e:
                # If image fetch fails, show a black screen in OpenCV
                black = np.zeros((720, 960, 3), np.uint8)
                cv2.putText(black, f"Image error: {e}", (20, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow(WIN_NAME, black)

            # Check if OpenCV window was closed
            if cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
                logger.info("WASD: Video window closed by user.")
                break

            # 2. Non-blocking key check (wait 30ms)
            k = cv2.waitKey(30) & 0xFF

            if k == ord('x') or k == ord('X') or k == 27:  # 'x' or ESC
                logger.info("WASD: Exiting teleop...")
                break

            if k == ord('w') or k == ord('W'):
                v_x = 0.5
                desc = "Forward"
            elif k == ord('s') or k == ord('S'):
                v_x = -0.5
                desc = "Backward"
            elif k == ord('a') or k == ord('A'):
                v_y = 0.5
                desc = "Strafe Left"
            elif k == ord('d') or k == ord('D'):
                v_y = -0.5
                desc = "Strafe Right"
            elif k == ord('q') or k == ord('Q'):
                v_rot = 0.8
                desc = "Turn Left"
            elif k == ord('e') or k == ord('E'):
                v_rot = -0.8
                desc = "Turn Right"

            # 3. Send command to Spot
            if v_x != 0.0 or v_y != 0.0 or v_rot != 0.0:
                logger.info(f"WASD Command: {desc}")
                try:
                    cmd = RobotCommandBuilder.synchro_velocity_command(v_x=v_x, v_y=v_y, v_rot=v_rot)
                    agent.cmd_client.robot_command(cmd, end_time_secs=time.time() + 0.6)
                except Exception as e:
                    logger.error(f"Failed to send movement command: {e}")

    finally:
        cv2.destroyAllWindows()

    # Sit and power off on exit
    logger.info("WASD: commanding robot to sit down...")
    agent.sit()
    logger.info("WASD: powering off...")
    agent.power_off()


@safety_fallback_decorator
def run_stand_and_view(agent: SpotAgent) -> None:
    logger.info("Stand: Spot is standing. Extending arm and opening hand camera...")
    # 1. 伸展手臂
    agent._arm_out()
    
    # 2. 打开 OpenCV 窗口展示手部相机画面
    WIN_NAME = "Spot Hand Camera Stream"
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, 960, 720)
    
    img_req = [build_image_request(
        "hand_color_image",
        image_format=image_pb2.Image.FORMAT_JPEG,
        quality_percent=70
    )]
    
    logger.info("Press ESC or 'q' in the OpenCV window to exit and stow arm.")
    try:
        while True:
            try:
                resp_list = agent.img_client.get_image(img_req)
                if resp_list:
                    img_resp = resp_list[0]
                    img_data = np.frombuffer(img_resp.shot.image.data, dtype=np.uint8)
                    frame = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                    if frame is not None:
                        cv2.putText(frame, "Press ESC or 'q' to exit", (15, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.imshow(WIN_NAME, frame)
            except Exception as e:
                logger.warning(f"Failed to fetch image: {e}")
                
            if cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
                
            k = cv2.waitKey(30) & 0xFF
            if k == 27 or k == ord('q') or k == ord('Q'):
                break
    finally:
        cv2.destroyAllWindows()
        logger.info("Stowing arm...")
        agent._arm_in()
        time.sleep(1.0)


@safety_fallback_decorator
def run_arm_height_view(agent: SpotAgent) -> None:
    import random
    logger.info("Arm 1m Height & View: Spot is standing. Positioning arm to 1.0m height, facing straight ahead...")
    
    # 1. Get current body height from ground (odom frame) to compute target z
    try:
        state = agent.state_client.get_robot_state()
        transforms = state.kinematic_state.transforms_snapshot
        from bosdyn.client.frame_helpers import ODOM_FRAME_NAME, BODY_FRAME_NAME
        tform = get_a_tform_b(transforms, ODOM_FRAME_NAME, BODY_FRAME_NAME)
        body_height = tform.position.z
        logger.info(f"Current body height from ground: {body_height:.2f} m")
    except Exception as e:
        logger.warning(f"Could not retrieve body height, defaulting to 0.5m: {e}")
        body_height = 0.5

    # 1.0m height from ground -> target_z relative to body frame
    target_z = 1.0 - body_height
    logger.info(f"Commanding arm to x=0.75m, z={target_z:.2f}m in body frame...")

    try:
        q = Quat()  # Identity rotation (w=1, x=0, y=0, z=0): points gripper straight forward
        arm_cmd = RobotCommandBuilder.arm_pose_command(
            0.75, 0.0, target_z,
            q.w, q.x, q.y, q.z,
            GRAV_ALIGNED_BODY_FRAME_NAME,
            2.0
        )
        agent.cmd_client.robot_command(arm_cmd)
        time.sleep(2.0)
        
        # Open gripper
        agent.cmd_client.robot_command(RobotCommandBuilder.claw_gripper_open_command())
        time.sleep(0.5)
    except Exception as e:
        logger.error(f"Failed to position arm: {e}")

    # 2. Open OpenCV Window and show video stream at highest quality
    WIN_NAME = "Spot Hand Camera - 1m Height (Press ESC or Q to Exit)"
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, 1280, 720)

    img_req = [build_image_request(
        "hand_color_image",
        quality_percent=95,
        image_format=image_pb2.Image.FORMAT_JPEG
    )]

    logger.info("Starting video stream. Focus the OpenCV window and press ESC or 'q' to exit.")
    
    start_time = time.time()
    last_cycle_idx = -1
    angle_deg = 0.0

    try:
        while True:
            # Gripper sweep logic
            current_time = time.time()
            elapsed = current_time - start_time
            cycle_idx = int(elapsed / 12.0)
            
            if cycle_idx != last_cycle_idx:
                last_cycle_idx = cycle_idx
                # Random angle between -60 and 60 degrees
                angle_deg = random.uniform(-60.0, 60.0)
                angle_rad = math.radians(angle_deg)
                logger.info(f"Sweep Cycle {cycle_idx}: rotating to {angle_deg:.1f}° over 6 seconds...")
                try:
                    q_target = Quat.from_yaw(angle_rad)
                    arm_cmd = RobotCommandBuilder.arm_pose_command(
                        0.75, 0.0, target_z,
                        q_target.w, q_target.x, q_target.y, q_target.z,
                        GRAV_ALIGNED_BODY_FRAME_NAME,
                        6.0  # 6 seconds to rotate
                    )
                    agent.cmd_client.robot_command(arm_cmd)
                except Exception as e:
                    logger.error(f"Failed to command arm rotation: {e}")

            try:
                resp_list = agent.img_client.get_image(img_req)
                if resp_list:
                    img_resp = resp_list[0]
                    img_data = np.frombuffer(img_resp.shot.image.data, dtype=np.uint8)
                    frame = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                    if frame is not None:
                        h, w = frame.shape[:2]
                        # Upscale low-res frames to fill the window
                        if w < 1280:
                            scale = 1280.0 / w
                            new_w = 1280
                            new_h = int(h * scale)
                            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                            h, w = frame.shape[:2]

                        # Put resolution and exit helper text on HUD
                        cv2.putText(
                            frame,
                            f"Res: {w}x{h} | ESC/Q: Exit & Sit/PowerOff",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )
                        
                        # Add current angle/status to HUD
                        action_str = "Rotating to" if (elapsed % 12.0 < 6.0) else "Holding at"
                        cv2.putText(
                            frame,
                            f"Angle: {angle_deg:.1f} deg ({action_str})",
                            (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        
                        cv2.imshow(WIN_NAME, frame)
            except Exception as e:
                logger.warning(f"Failed to fetch image: {e}")

            # Check if OpenCV window was closed
            if cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
                logger.info("Video window closed by user.")
                break

            k = cv2.waitKey(30) & 0xFF
            if k == 27 or k == ord('q') or k == ord('Q'):
                logger.info("User requested exit.")
                break
    finally:
        cv2.destroyAllWindows()
        logger.info("Stowing arm...")
        agent._arm_in()
        time.sleep(1.0)
        logger.info("Commanding robot to sit down...")
        agent.sit()
        logger.info("Powering off...")
        agent.power_off()


@safety_fallback_decorator
def run_sit_down_rest(agent: SpotAgent) -> None:
    logger.info("Sit down / Rest: Stowing arm...")
    agent._arm_in()
    time.sleep(1.5)
    logger.info("Sitting down...")
    agent.sit()
    logger.info("Powering off...")
    agent.power_off()


@safety_fallback_decorator
def run_fetch_one_point(agent: SpotAgent, guard_point: str, home_point: str) -> None:
    detector = SpotDetector()
    guard_id = agent.get_waypoint_id_by_name(agent.graph, guard_point)
    home_id = agent.get_waypoint_id_by_name(agent.graph, home_point)
    if not guard_id or not home_id:
        logger.error("Required waypoints not found — aborting mission.")
        return
    logger.info(f"Navigating to target point: {guard_point}")
    while not agent.navigate_to_waypoint(guard_id):
        logger.warning("Failed to reach target point. Retrying in 3 s...")
        time.sleep(3.0)
    logger.info("At target point — searching for target...")
    target_grasped = False
    while not target_grasped:
        agent._arm_out()
        time.sleep(2.0)
        detected = agent.find_target(detector)
        if detected is None:
            logger.info("No object detected. Waiting 3 s...")
            time.sleep(3.0)
            continue
        target_img_resp, cam_name, cx, cy, cls_name = detected
        if agent.grasp_object(target_img_resp, cam_name, cx, cy, cls_name):
            logger.info("Target grasped.")
            target_grasped = True
            time.sleep(0.2)
        else:
            logger.warning("Grasp failed — returning to target point...")
            agent.navigate_to_waypoint(guard_id)
            time.sleep(2.0)
    logger.info(f"Returning home: {home_point}")
    if agent.navigate_to_waypoint(home_id):
        logger.info("At home — releasing target.")
        time.sleep(0.2)
        agent._arm_release(bin=True)
        time.sleep(1.2)
        logger.info("Mission complete.")
    else:
        logger.error("Failed to reach home — check manually.")


@safety_fallback_decorator
def run_fetch_one_point_loop(agent: SpotAgent, guard_point: str, home_point: str) -> None:
    return run_guard_loop(agent, guard_point, home_point)
