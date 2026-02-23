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
        #agent._arm_out()
        time.sleep(2)
        raw_detection_list = agent.scan(detector)
        if raw_detection_list:
            agent.object_register(raw_detection_list)
            agent.grab_target_with_nav(detector,raw_detection_list)
        else:
            # 如果没找到，也要通知 register 清空雷达图上的旧标记
            agent.object_register([])
            print("未发现目标，继续巡逻...")
        time.sleep(3)
        agent.move_to_goal(0,0,0)
       