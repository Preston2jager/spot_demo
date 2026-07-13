import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

import time
from cls_rmit_spot_core import SpotAgent
from cls_rmit_spot_detector import SpotDetector 

def main():

    guard_point_name = "waypoint_189"
    home_point_name = "default"

    with SpotAgent(
        stream=True,
        navigation=True
        ) as agent:
        detector = SpotDetector()
        guard_id = agent.get_waypoint_id_by_name(agent.graph, guard_point_name)
        home_id = agent.get_waypoint_id_by_name(agent.graph, home_point_name)
        
        if not guard_id or not home_id:
            print("Failed to find required waypoints in the graph. Check graph content.")
            return
        print("Starting mission loop. Press Ctrl+C to exit.")
        try:
            while True:
                print(f"\n-> Go to guard point: {guard_point_name}")
                if not agent.navigate_to_waypoint(guard_id):
                    print("Failed to reach guard point. Retrying...")
                    break 
                print("-> Start scanning for targets...")
                agent._arm_out()
                time.sleep(3.0)
                target_grasped = False
                while not target_grasped:
                    success = agent.quick_detect(detector)
                    if success:
                        print("Target grasped successfully.")
                        target_grasped = True
                        agent._arm_in()
                        time.sleep(3.0)
                    else:
                        time.sleep(5.0)
                print(f"\n-> Go back to home: {home_point_name}")
                if agent.navigate_to_waypoint(home_id):
                    print("-> Reached home point. Releasing target...")
                    time.sleep(1.0) 
                    # 提示：在这里你可能需要写一行让机器狗松开夹爪扔下物体的代码
                    agent._arm_in()
                else:
                    print("Failed to return to home point. Check connection and try again.")
                    break
        except KeyboardInterrupt:
            print("Exiting mission loop. Goodbye!")

if __name__ == "__main__":
    main()