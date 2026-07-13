from cls_spot_agent import SpotAgent

import time
import math
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.robot_state import RobotStateClient


class SpotAgent_demo(SpotAgent):

    def detect_grab_return_and_release(
        self,
        detector,
        *,
        source: str = "hand_color_image",
        interval: float = 1.0,
        jpeg_quality: int = 70,
        timeout: float = 25.0,
        home_frame: str = "odom",
        open_fraction: float = 1.0,
        stow_on_finish: bool = False,
        home_tolerance: float = 0.5,
    ) -> bool:
        """
        Detect, grab, return to home, and release object flow.
        After grasping, carries arm up for safe transport.
        """
        ok = self.handcam_detect_and_grab_once_sameframe(
            detector,
            source=source,
            interval=interval,
            jpeg_quality=jpeg_quality,
            timeout=timeout,
            carry_on_success=True,
            open_on_success=False,
            stow_on_finish=False,
        )
        if not ok:
            print("[flow] Grasp failed, flow terminated.")
            return False

        # Brief wait after carry
        print("[flow] Grasp succeeded and arm in carry position, ready to return home.")
        time.sleep(0.5)

        # Return home and release
        reached = self.go_home_and_release(
            open_fraction=open_fraction,
            stow_after=stow_on_finish,
            home_tolerance=home_tolerance,
        )
        print(f"[flow] Return home and release: {'Reached' if reached else 'Not reached (still executed placement and release)'}")
        return True

    def mark_home(self, frame_name: str = "odom") -> None:
        """
        Record the starting pose to self._home_{frame, x, y, yaw}.
        Default uses ODOM (continuous, no jumps), suitable for "return to start" semantics.
        """
        from bosdyn.client.frame_helpers import get_a_tform_b

        if getattr(self, "robot", None) is None:
            raise RuntimeError("SpotAgent not logged in or does not hold robot.")

        state_client: RobotStateClient = self.robot.ensure_client(RobotStateClient.default_service_name)
        robot_state = state_client.get_robot_state()
        tf_snapshot = robot_state.kinematic_state.transforms_snapshot

        a_tform_b = get_a_tform_b(tf_snapshot, frame_name, "body")
        x, y = a_tform_b.position.x, a_tform_b.position.y
        yaw = self._yaw_from_quat(a_tform_b.rotation)

        self._home_frame = frame_name
        self._home_x = float(x)
        self._home_y = float(y)
        self._home_yaw = float(yaw)
        print(f"[home] Marked home: frame={frame_name}  pose=({self._home_x:.3f},{self._home_y:.3f},{self._home_yaw:.3f} rad)")

    def handcam_detect_and_grab_once_sameframe(
        self,
        detector,
        *,
        source: str = "hand_color_image",
        interval: float = 1.0,
        jpeg_quality: int = 70,
        timeout: float = 30.0,
        carry_on_success: bool = True,
        open_on_success: bool = False,
        stow_on_finish: bool = False,
    ) -> bool:
        """
        One-step detection + grasp (same frame): when detected, use the same frame's 
        image_response for grasping to avoid inter-frame jitter deviation.
        """
        for xy, img_resp in self.handcam_detect_bottle_stream_with_image(
            detector,
            source=source,
            interval=interval,
            jpeg_quality=jpeg_quality,
            timeout=timeout,
        ):
            if xy is None or img_resp is None:
                continue
            x, y = int(xy[0]), int(xy[1])
            print(f"[bottle] Detected bottle at pixel: ({x}, {y}) → same-frame grasp")
            ok = self.grasp_from_known_image_pixel(
                img_resp, x, y,
                feedback_timeout_sec=30.0,
                carry_on_success=carry_on_success,
                open_on_success=open_on_success,
                stow_on_finish=stow_on_finish,
            )
            print(f"[bottle] Grasp result: {'SUCCESS' if ok else 'FAIL'}")
            return ok

        print("[bottle] Timeout without detection, terminating")
        return False

    def go_home_and_release(
        self,
        *,
        home_tolerance: float = 0.5,
        yaw_tolerance: float = 0.30,
        max_secs: float = 90.0,
        open_fraction: float = 1.0,
        stow_after: bool = False,
    ) -> bool:
        """
        Return to mark_home() start point → place arm to ground pose → 
        open gripper to release → retract arm to intermediate pose → optional stow.
        Uses velocity closed-loop navigation.
        """
        from bosdyn.client.frame_helpers import get_a_tform_b

        if not all(hasattr(self, a) for a in ("_home_frame", "_home_x", "_home_y", "_home_yaw")):
            raise RuntimeError("mark_home() has not been called to record starting point.")
        if getattr(self, "robot", None) is None:
            raise RuntimeError("SpotAgent not logged in or does not hold robot.")

        cmd_client: RobotCommandClient = getattr(self, "cmd_client", None) or \
            self.robot.ensure_client(RobotCommandClient.default_service_name)
        self.cmd_client = cmd_client
        state_client: RobotStateClient = self.robot.ensure_client(RobotStateClient.default_service_name)

        frame = self._home_frame
        gx, gy, gyaw = self._home_x, self._home_y, self._home_yaw

        # Clear motion state before returning home
        try:
            stand_cmd = RobotCommandBuilder.synchro_stand_command()
            cmd_client.robot_command(stand_cmd)
            time.sleep(0.5)
            print("[home] Cleared motion state, preparing to return home")
        except Exception as e:
            print(f"[home] Failed to clear state (continuing): {e}")

        # Check if already close to home
        rs = state_client.get_robot_state()
        tf = rs.kinematic_state.transforms_snapshot
        aTb = get_a_tform_b(tf, frame, "body")
        cx, cy = float(aTb.position.x), float(aTb.position.y)
        cyaw = self._yaw_from_quat(aTb.rotation)
        
        initial_dist = math.hypot(gx - cx, gy - cy)
        initial_yaw_err = abs(self._wrap_pi(gyaw - cyaw))
        
        print(f"[home] Current position: ({cx:.3f},{cy:.3f},{cyaw:.3f})")
        print(f"[home] Target position: ({gx:.3f},{gy:.3f},{gyaw:.3f})")
        print(f"[home] Distance to home: {initial_dist:.3f}m, yaw error: {initial_yaw_err:.3f}rad")
        
        if initial_dist <= home_tolerance:
            print(f"[home] Already within home tolerance ({home_tolerance}m), skipping navigation")
            reached = True
        else:
            # Use velocity closed-loop to return home
            print(f"[home] Using velocity closed-loop to return home (tolerance={home_tolerance}m)...")
            reached = self._drive_to_se2_with_velocity(
                frame, gx, gy, gyaw, 
                xy_tol=home_tolerance,
                yaw_tol=yaw_tolerance, 
                max_secs=max_secs
            )

        if not reached:
            print("[home] Did not reach home area within time limit (still continuing placement and release).")

        # Place arm to ground pose (arm is already in carry, just move it down)
        print("[place] Moving arm from carry to ground placement pose...")
        placed = self.arm_place_down_at_body(
            state_client=state_client,
            forward_m=0.45, 
            down_m=0.40, 
            pitch_deg=40.0, 
            move_seconds=1.8
        )
        if not placed:
            print("[place] Placement pose failed (still attempting gripper release).")
        else:
            print("[place] Placement pose completed.")
        
        # Wait for arm to reach position
        time.sleep(0.8)

        # Open gripper to release
        try:
            print(f"[home] Opening gripper to release (fraction={open_fraction})...")
            open_cmd = RobotCommandBuilder.claw_gripper_open_fraction_command(float(open_fraction))
            cmd_client.robot_command(open_cmd)
            time.sleep(2.0)
            print("[home] Gripper opened.")
        except Exception as e:
            print(f"[home] Gripper release failed: {e}")

        # Retract arm to intermediate pose before stowing
        print("[retract] Retracting arm to intermediate pose...")
        retracted = self.arm_retract_to_intermediate_pose(
            state_client=state_client,
            target_x=0.30,
            target_y=0.0,
            target_z=0.25,
            pitch_deg=30.0,
            move_seconds=1.5
        )
        if retracted:
            print("[retract] Arm retracted to intermediate pose.")
        else:
            print("[retract] Retraction failed (continuing).")

        # Optional stow
        if stow_after:
            time.sleep(0.5)
            try:
                print("[home] Stowing arm...")
                stow_cmd = RobotCommandBuilder.arm_stow_command()
                sid = cmd_client.robot_command(stow_cmd)
                # Wait for stow to complete
                from bosdyn.client.robot_command import block_until_arm_arrives
                block_until_arm_arrives(cmd_client, sid, timeout_sec=6.0)
                time.sleep(1.0)
                print("[home] Arm stowed.")
            except Exception as e:
                print(f"[home] Stow failed: {e}")

        return reached

    def grasp_from_known_image_pixel(
        self,
        image_response,
        x: int,
        y: int,
        *,
        feedback_timeout_sec: float = 30.0,
        feedback_interval_sec: float = 0.25,
        carry_on_success: bool = True,
        open_on_success: bool = False,
        stow_on_finish: bool = False,
    ) -> bool:
        """
        Similar to grasp_from_image_pixel, but uses "passed-in image_response" for grasping,
        ensuring pixel consistency with camera intrinsics/snapshot, avoiding detection frame 
        vs grasp frame mismatch.
        """
        from bosdyn.api import geometry_pb2, manipulation_api_pb2
        from bosdyn.client.manipulation_api_client import ManipulationApiClient

        if getattr(self, "robot", None) is None:
            raise RuntimeError("SpotAgent not logged in or does not hold robot.")

        manip_client: ManipulationApiClient = self.robot.ensure_client(ManipulationApiClient.default_service_name)
        cmd_client: RobotCommandClient = getattr(self, "cmd_client", None) or \
            self.robot.ensure_client(RobotCommandClient.default_service_name)
        self.cmd_client = cmd_client

        image = image_response
        # Camera model: prefer pinhole; fallback to fisheye; last resort pinhole
        cam_model = getattr(image.source, "pinhole", None) or getattr(image.source, "fisheye", None) or image.source.pinhole

        pick = manipulation_api_pb2.PickObjectInImage(
            pixel_xy=geometry_pb2.Vec2(x=int(x), y=int(y)),
            transforms_snapshot_for_camera=image.shot.transforms_snapshot,
            frame_name_image_sensor=image.shot.frame_name_image_sensor,
            camera_model=cam_model,
        )
        req = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=pick)
        rsp = manip_client.manipulation_api_command(manipulation_api_request=req)

        # Poll feedback
        deadline = time.time() + float(feedback_timeout_sec)
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
                print(f"[grasp] state: {name}")
                last_name = name
            if state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED:
                succeeded = True
                break
            if state == manipulation_api_pb2.MANIP_STATE_GRASP_FAILED:
                succeeded = False
                break
            time.sleep(float(feedback_interval_sec))

        # Post-success handling
        try:
            if succeeded and carry_on_success:
                print("[grasp] Moving to carry position...")
                cid = cmd_client.robot_command(RobotCommandBuilder.arm_carry_command())
                from bosdyn.client.robot_command import block_until_arm_arrives
                block_until_arm_arrives(cmd_client, cid, timeout_sec=6.0)
                time.sleep(0.5)
                print("[grasp] Carry position reached.")
                
            if succeeded and open_on_success:
                print("[grasp] Opening gripper...")
                cmd_client.robot_command(RobotCommandBuilder.claw_gripper_open_fraction_command(1.0))
                time.sleep(1.0)
                print("[grasp] Gripper opened.")
                
            if stow_on_finish:
                print("[grasp] Stowing arm...")
                sid = cmd_client.robot_command(RobotCommandBuilder.arm_stow_command())
                from bosdyn.client.robot_command import block_until_arm_arrives
                block_until_arm_arrives(cmd_client, sid, timeout_sec=8.0)
                time.sleep(1.0)
                print("[grasp] Arm stowed.")
                
        except Exception as e:
            print(f"[grasp] Post-processing failed: {e}")

        return succeeded

    def _drive_to_se2_with_velocity(
        self,
        frame: str,
        gx: float, gy: float, gyaw: float,
        *,
        xy_tol: float = 0.5,
        yaw_tol: float = 0.30,
        max_secs: float = 120.0,
    ) -> bool:
        """
        Velocity closed-loop return to (gx,gy,gyaw):
        Strategy: 3-phase approach with relaxed tolerances for narrow spaces
        1) First rotate to face target position (skip if already facing)
        2) Drive straight to target position (accept home_tolerance range)
        3) Finally rotate to target heading (relaxed tolerance)
        """
        from bosdyn.client.frame_helpers import get_a_tform_b

        state_client: RobotStateClient = self.robot.ensure_client(RobotStateClient.default_service_name)

        MAX_V = 0.6      # m/s
        MAX_W = 1.0      # rad/s
        KP_POS = 0.8     # P gain (position)
        KP_YAW = 1.0     # P gain (angle)

        period = 0.1
        t_end = time.time() + float(max_secs)
        
        # Phase 1: Rotate to face target position (if needed)
        print("[velocity] Phase 1: Checking orientation to target...")
        rs = state_client.get_robot_state()
        tf = rs.kinematic_state.transforms_snapshot
        aTb = get_a_tform_b(tf, frame, "body")
        cx, cy = float(aTb.position.x), float(aTb.position.y)
        cyaw = self._yaw_from_quat(aTb.rotation)
        
        dx, dy = gx - cx, gy - cy
        angle_to_target = math.atan2(dy, dx)
        initial_angle_error = self._wrap_pi(angle_to_target - cyaw)
        
        if abs(initial_angle_error) > 0.35:  # Only rotate if > ~20 degrees off
            print(f"[velocity] Rotating to face target (error={abs(initial_angle_error):.3f}rad)...")
            phase1_deadline = time.time() + 30.0
            while time.time() < phase1_deadline and time.time() < t_end:
                rs = state_client.get_robot_state()
                tf = rs.kinematic_state.transforms_snapshot
                aTb = get_a_tform_b(tf, frame, "body")
                cx, cy = float(aTb.position.x), float(aTb.position.y)
                cyaw = self._yaw_from_quat(aTb.rotation)

                dx, dy = gx - cx, gy - cy
                angle_to_target = math.atan2(dy, dx)
                angle_error = self._wrap_pi(angle_to_target - cyaw)
                
                if abs(angle_error) < 0.26:  # ~15 degrees
                    print(f"[velocity] Phase 1 complete (angle_error={abs(angle_error):.3f}rad)")
                    break
                
                vrot = max(-MAX_W, min(MAX_W, KP_YAW * angle_error))
                self.send_velocity(0.0, 0.0, vrot)
                time.sleep(period)
        else:
            print(f"[velocity] Already facing target (error={abs(initial_angle_error):.3f}rad), skipping Phase 1")
        
        # Phase 2: Drive to target position
        print(f"[velocity] Phase 2: Driving to target (tolerance={xy_tol:.2f}m)...")
        phase2_deadline = time.time() + 60.0
        while time.time() < phase2_deadline and time.time() < t_end:
            rs = state_client.get_robot_state()
            tf = rs.kinematic_state.transforms_snapshot
            aTb = get_a_tform_b(tf, frame, "body")
            cx, cy = float(aTb.position.x), float(aTb.position.y)
            cyaw = self._yaw_from_quat(aTb.rotation)

            dx, dy = gx - cx, gy - cy
            dist = math.hypot(dx, dy)
            
            # If within tolerance, done with Phase 2
            if dist <= xy_tol:
                print(f"[velocity] Phase 2 complete (dist={dist:.3f}m, within tolerance={xy_tol:.2f}m)")
                break
            
            # Calculate forward velocity and gentle heading correction
            angle_to_target = math.atan2(dy, dx)
            angle_error = self._wrap_pi(angle_to_target - cyaw)
            
            # Forward velocity based on distance
            vx = max(-MAX_V, min(MAX_V, KP_POS * dist))
            
            # Gentle rotation to keep facing target
            if abs(angle_error) > 0.35:  # ~20 degrees
                vrot = max(-MAX_W * 0.5, min(MAX_W * 0.5, KP_YAW * 0.5 * angle_error))
            else:
                vrot = 0.0
            
            self.send_velocity(vx, 0.0, vrot)
            time.sleep(period)
        
        # Phase 3: Rotate to target heading (only if needed and not too far from target)
        rs = state_client.get_robot_state()
        tf = rs.kinematic_state.transforms_snapshot
        aTb = get_a_tform_b(tf, frame, "body")
        cx, cy = float(aTb.position.x), float(aTb.position.y)
        cyaw = self._yaw_from_quat(aTb.rotation)
        
        final_dist = math.hypot(gx - cx, gy - cy)
        dyaw = self._wrap_pi(gyaw - cyaw)
        
        # Only do precise yaw alignment if we're close to home
        if final_dist <= xy_tol * 1.5 and abs(dyaw) > yaw_tol:
            print(f"[velocity] Phase 3: Adjusting heading (error={abs(dyaw):.3f}rad)...")
            phase3_deadline = time.time() + 30.0
            while time.time() < phase3_deadline and time.time() < t_end:
                rs = state_client.get_robot_state()
                tf = rs.kinematic_state.transforms_snapshot
                aTb = get_a_tform_b(tf, frame, "body")
                cyaw = self._yaw_from_quat(aTb.rotation)

                dyaw = self._wrap_pi(gyaw - cyaw)
                
                if abs(dyaw) <= yaw_tol:
                    self.send_velocity(0.0, 0.0, 0.0)
                    print(f"[velocity] Phase 3 complete (yaw_error={abs(dyaw):.3f}rad)")
                    break
                
                vrot = max(-MAX_W, min(MAX_W, KP_YAW * dyaw))
                self.send_velocity(0.0, 0.0, vrot)
                time.sleep(period)
        else:
            print(f"[velocity] Phase 3: Skipping precise heading (dist={final_dist:.3f}m, yaw_error={abs(dyaw):.3f}rad)")

        # Stop robot
        try:
            self.send_velocity(0.0, 0.0, 0.0)
        except Exception:
            pass
        
        # Check final position
        rs = state_client.get_robot_state()
        tf = rs.kinematic_state.transforms_snapshot
        aTb = get_a_tform_b(tf, frame, "body")
        cx, cy = float(aTb.position.x), float(aTb.position.y)
        cyaw = self._yaw_from_quat(aTb.rotation)
        
        final_dist = math.hypot(gx - cx, gy - cy)
        final_yaw_err = abs(self._wrap_pi(gyaw - cyaw))
        
        success = (final_dist <= xy_tol and final_yaw_err <= yaw_tol)
        print(f"[velocity] Navigation complete: dist={final_dist:.3f}m (tol={xy_tol:.2f}m), yaw_err={final_yaw_err:.3f}rad (tol={yaw_tol:.2f}rad), success={success}")
        return success

    def arm_place_down_at_body(
        self,
        state_client: RobotStateClient,
        *,
        forward_m: float = 0.45,
        down_m: float = 0.40,
        pitch_deg: float = 40.0,
        move_seconds: float = 1.8,
    ) -> bool:
        """
        In "body frame (GRAV_ALIGNED_BODY)", have arm reach forward and sink to near-ground position,
        slight downward pitch, then hold (don't immediately stow). Used for "put hand on ground 
        after returning home before releasing".
        """
        from bosdyn.client.frame_helpers import (
            HAND_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME, get_a_tform_b, math_helpers
        )

        try:
            snapshot = state_client.get_robot_state().kinematic_state.transforms_snapshot
            root_frame = GRAV_ALIGNED_BODY_FRAME_NAME
            root_T_hand = get_a_tform_b(snapshot, root_frame, HAND_FRAME_NAME)

            # Target: forward_m in front of body frame, down_m downward, wrist pitched down pitch_deg
            delta = math_helpers.SE3Pose(
                x=float(forward_m), y=0.0, z=-float(down_m),
                rot=math_helpers.Quat.from_pitch(pitch_deg * math.pi / 180.0)
            )
            root_T_target = root_T_hand * delta
            q = root_T_target.rot

            arm_cmd = RobotCommandBuilder.arm_pose_command(
                root_T_target.x, root_T_target.y, root_T_target.z,
                q.w, q.x, q.y, q.z, root_frame, float(move_seconds)
            )
            self.cmd_client.robot_command(arm_cmd)
            time.sleep(float(move_seconds))
            return True
        except Exception as e:
            print(f"[place] Failed to issue placement pose: {e}")
            return False

    def arm_retract_to_intermediate_pose(
        self,
        state_client: RobotStateClient,
        *,
        target_x: float = 0.30,
        target_y: float = 0.0,
        target_z: float = 0.25,
        pitch_deg: float = 30.0,
        move_seconds: float = 1.5,
    ) -> bool:
        """
        Retract arm to an absolute intermediate pose after releasing object.
        This moves the arm directly to a specified position in the body frame,
        not relative to current hand position.
        """
        from bosdyn.client.frame_helpers import (
            GRAV_ALIGNED_BODY_FRAME_NAME, math_helpers
        )

        try:
            root_frame = GRAV_ALIGNED_BODY_FRAME_NAME
            
            # Set absolute target position in body frame
            # x: forward, y: left/right, z: up/down (positive is up)
            target_quat = math_helpers.Quat.from_pitch(pitch_deg * math.pi / 180.0)

            arm_cmd = RobotCommandBuilder.arm_pose_command(
                float(target_x), 
                float(target_y), 
                float(target_z),
                target_quat.w, 
                target_quat.x, 
                target_quat.y, 
                target_quat.z, 
                root_frame, 
                float(move_seconds)
            )
            self.cmd_client.robot_command(arm_cmd)
            time.sleep(float(move_seconds))
            return True
        except Exception as e:
            print(f"[retract] Failed to retract arm: {e}")
            return False

    # ========== 趣味动作：挥手舞（兼容多版本 SDK） ==========
    def arm_wave_overhead_simple(self) -> bool:
        """
        固定流程（无参数）：
        1) 打开抓手（可选）
        2) 把手臂抬到头顶上方
        3) 在头顶左右来回摆动若干次
        结束后保持手臂在最后姿态（不收纳）
        """
        import time, math
        from bosdyn.client.robot_command import RobotCommandBuilder
        from bosdyn.client.frame_helpers import (
            HAND_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME, get_a_tform_b, math_helpers
        )

        if self.cmd_client is None or self.state_client is None:
            raise RuntimeError("cmd_client/state_client 未初始化。")

        root_frame = GRAV_ALIGNED_BODY_FRAME_NAME

        def send_pose(x, y, z, quat, sec):
            # 统一用位置参数形式，避免关键字差异
            arm_cmd = RobotCommandBuilder.arm_pose_command(
                x, y, z, quat.w, quat.x, quat.y, quat.z, root_frame, sec
            )
            self.cmd_client.robot_command(arm_cmd)
            time.sleep(sec)

        try:
            # 0) 可选：打开抓手，避免自碰
            try:
                self.cmd_client.robot_command(RobotCommandBuilder.claw_gripper_open_command())
                time.sleep(0.3)
            except Exception as e:
                print(f"[gripper] open fail (ignored): {e}")

            # 1) 读当前手部姿态
            snapshot = self.state_client.get_robot_state().kinematic_state.transforms_snapshot
            root_T_hand = get_a_tform_b(snapshot, root_frame, HAND_FRAME_NAME)
            q0 = root_T_hand.rot  # 保持当前手腕朝向，减少突兀旋转

            # 2) 抬到“头顶”上方的安全基准位姿（x 前 0.20m，z 上 0.80m）
            base_x = 0.20
            base_z = 0.80
            base_y = 0.00
            print("[arm] raise overhead...")
            send_pose(base_x, base_y, base_z, q0, 1.0)

            # 3) 左右摆动（幅度和次数都写死）
            swings = 16           # 往返 8 次（总共 16 个边）
            amp = 0.28           # 左右幅度（米）
            dwell = 0.45         # 每次到达目标的时间（秒）
            print("[arm] swing left/right...")

            # 先到左，再到右，循环
            side = +1
            for i in range(swings):
                y_target = side * amp
                send_pose(base_x, y_target, base_z, q0, dwell)
                side *= -1

            # 4) 回到正中
            send_pose(base_x, 0.0, base_z, q0, 0.45)

            print("[arm] wave done.")
            return True

        except Exception as e:
            print(f"[arm] wave failed: {e}")
            return False


