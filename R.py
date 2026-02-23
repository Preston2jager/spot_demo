from cls_rmit_spot import SpotAgent
from cls_yolo_2 import YoloTargetDetector
import time

if __name__ == "__main__":
    detector = YoloTargetDetector()
    with SpotAgent(
        hostname="192.168.80.3",
        username="user",
        password="myjujz7e2prj",
        force_lease=True,
        client_name="MySpotClient",
    ) as agent:
        time.sleep(2)
        agent.move_to_goal(0,0,0)
        