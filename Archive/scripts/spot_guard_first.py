import os
import time

os.environ["QT_QPA_PLATFORM"] = "xcb"

from src.spot_agent import SpotAgent
from src.spot_detector_openvino import SpotDetector


def main() -> None:
    """Single-run guard mission: patrol a short route, grasp one object, return home."""
    guard_point_name = "waypoint_89"
    home_point_name = "default"

    try:
        with SpotAgent(stream=True, navigation=True) as agent:
            detector = SpotDetector()
            guard_id = agent.get_waypoint_id_by_name(agent.graph, guard_point_name)
            home_id = agent.get_waypoint_id_by_name(agent.graph, home_point_name)
            wp_122_id = agent.get_waypoint_id_by_name(agent.graph, "waypoint_122")
            wp_124_id = agent.get_waypoint_id_by_name(agent.graph, "waypoint_124")
            wp_40_id = agent.get_waypoint_id_by_name(agent.graph, "waypoint_40")

            if not all([guard_id, home_id, wp_122_id, wp_124_id, wp_40_id]):
                print("Failed to find required waypoints. Check graph content.")
                return

            print("Starting initial route...")
            agent.navigate_to_waypoint(wp_122_id)
            agent.navigate_to_waypoint(wp_124_id)
            time.sleep(5)
            agent.navigate_to_waypoint(wp_122_id)

            print(f"Navigating to guard point: {guard_point_name}")
            while not agent.navigate_to_waypoint(guard_id):
                print("Failed to reach guard point. Retrying in 3 seconds...")
                time.sleep(3.0)

            print("Arrived at guard point. Starting search phase...")
            target_grasped = False

            while not target_grasped:
                agent._arm_out()
                time.sleep(2.0)
                detected = agent.find_target(detector)
                if detected is None:
                    print("No object detected. Waiting 3 seconds...")
                    time.sleep(3.0)
                    continue
                target_img_resp, cam_name, cx, cy, cls_name = detected
                if agent.grasp_object(target_img_resp, cam_name, cx, cy, cls_name):
                    print("Target grasped successfully.")
                    target_grasped = True
                    time.sleep(0.2)
                else:
                    print("Grasp failed. Returning to guard point...")
                    agent.navigate_to_waypoint(guard_id)
                    time.sleep(2.0)

            print(f"\nReturning home: {home_point_name}")
            if agent.navigate_to_waypoint(home_id):
                print("Reached home. Releasing target...")
                time.sleep(0.2)
                agent._arm_release(bin=True)
                time.sleep(1.2)
                print("Moving to standby: waypoint_40")
                agent.navigate_to_waypoint(wp_40_id)
                print("Task complete.")
            else:
                print("Failed to return home. Check manually.")

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt — exiting.")


if __name__ == "__main__":
    main()
