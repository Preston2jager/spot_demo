import os
import time

os.environ["QT_QPA_PLATFORM"] = "xcb"

from src.spot_agent import SpotAgent
from src.spot_detector_openvino import SpotDetector


def main() -> None:
    """Continuous guard loop from a different starting waypoint (waypoint_22)."""
    guard_point_name = "waypoint_22"
    home_point_name = "default"

    with SpotAgent(stream=True, navigation=True) as agent:
        detector = SpotDetector()
        guard_id = agent.get_waypoint_id_by_name(agent.graph, guard_point_name)
        home_id = agent.get_waypoint_id_by_name(agent.graph, home_point_name)

        if not guard_id or not home_id:
            print("Failed to find required waypoints. Check graph content.")
            return

        print("Starting mission loop. Press Ctrl+C to exit.")
        try:
            while True:
                print(f"\n[Phase 1] Navigating to guard point: {guard_point_name}")
                if not agent.navigate_to_waypoint(guard_id):
                    print("Failed to reach guard point. Retrying in 3 seconds...")
                    time.sleep(3.0)
                    continue

                print("Arrived at guard point. Starting search phase...")
                target_grasped = False

                while not target_grasped:
                    agent._arm_out()
                    time.sleep(2.0)
                    detected = agent.find_target(detector)
                    if detected is None:
                        print("No target found. Rescanning in 3 seconds...")
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

                print(f"\n[Phase 2] Returning home: {home_point_name}")
                if agent.navigate_to_waypoint(home_id):
                    print("Reached home. Releasing target...")
                    time.sleep(0.2)
                    agent._arm_release(bin=True)
                else:
                    print("Failed to return home. Check manually.")
                    break

                print("Mission cycle complete. Preparing next cycle...")
                time.sleep(2.0)

        except KeyboardInterrupt:
            print("\nKeyboardInterrupt — exiting mission loop.")


if __name__ == "__main__":
    main()
