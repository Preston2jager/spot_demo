import threading
import time
import math

from cls_spot_agent import SpotAgent
from cls_yolo import YoloBottleDetector

def stop():
    for _ in range(2):
        agent.send_velocity(0.0, 0.0, 0.0)
        time.sleep(0.12)


if __name__ == "__main__":

    agent = SpotAgent(
        hostname="192.168.80.3",
        username="user",
        password="myjujz7e2prj",
        force_lease=True,
        client_name="MySpotClient",
    )
    agent.power_on_and_stand()
    
    t = threading.Thread(
        target=agent.handcam_on,
        kwargs={"window_title":"Spot HandCam", "width":1600, "height":1200, "fps":15},
        daemon=True,
    )
    t.start()

    speed = 0.4            # 直线速度 m/s（如发现反向，就改成 -0.4）
    dist  = 5.0            # 单程 3 m
    ctrl_hz = 10.0         # 控制频率（10 Hz）
    tick = 1.0 / ctrl_hz
    turn_speed = 0.9       # 原地角速度 rad/s
    forward_time = dist / abs(speed)
    turn_time = math.pi / turn_speed  # 180°


    t0 = time.time()
    while time.time() - t0 < forward_time:
        agent.send_velocity(speed, 0.0, 0.0)   # 直线
        time.sleep(tick)
    stop()

    # —— 原地转身 180° ——
    t0 = time.time()
    while time.time() - t0 < turn_time:
        agent.send_velocity(0.0, 0.0, turn_speed)  # 只给角速度
        time.sleep(tick)
    stop()

    # —— 直线返回 3 m ——
    t0 = time.time()
    while time.time() - t0 < forward_time:
        agent.send_velocity(-speed, 0.0, 0.0)  # 反向直线
        time.sleep(tick)
    stop()

    # —— 再转回 180°（可选） ——
    t0 = time.time()
    while time.time() - t0 < turn_time:
        agent.send_velocity(0.0, 0.0, turn_speed)  # 或 -turn_speed，取决于你想面向哪边
        time.sleep(tick)
    stop()

    detector = YoloBottleDetector()
    ok = agent.patrol_autograb(
        detector,
        distance_m=3.0,
        speed_mps=0.5,
        detect_period=0.6,      # 0.5~1.0 之间都行
        source="hand_color_image",
        jpeg_quality=70,
        total_timeout=60.0,
        carry_on_success=True,
        open_on_success=False,
        stow_on_finish=True,
        turn_speed_rad=0.9,     # 越大转得越快，但别太夸张
        ctrl_hz=10.0,           # 指令刷新频率，8~12Hz 均可；环境抖就再高一点
    )


    agent.shutdown(power_off=False)