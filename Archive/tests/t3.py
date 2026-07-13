#!/usr/bin/env python3
import time, threading
from cls_spot_agent import SpotAgent

if __name__ == "__main__":
    agent = SpotAgent(
        hostname="192.168.80.3",
        username="user",
        password="myjujz7e2prj",
        force_lease=True,
        client_name="MySpotClient",
    )
    agent.power_on_and_stand()

    # 预览放到后台线程（不会阻塞后面的运动代码）
    t = threading.Thread(
        target=agent.arm_headcam_preview,
        kwargs={"window_title":"Spot HandCam", "width":1600, "height":1200, "fps":15},
        daemon=True,
    )
    t.start()

    # 现在会一边显示相机，一边走
    t0 = time.time()
    while time.time() - t0 < 6.0:
        agent.send_velocity(0.2, 0.0, 0.0)  # 速度稍提到 0.3 更明显
        time.sleep(0.1)

    agent.send_velocity(0.0, 0.0, 0.0, hold=0.5)
    agent.shutdown(power_off=False)
