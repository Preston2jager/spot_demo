#!/usr/bin/env python3
import time
from cls_spot_agent import SpotAgent

if __name__ == "__main__":
    # 初始化时显式传入  hostname / username / password，且 force_lease=True
    agent = SpotAgent(
        hostname="192.168.80.3",
        username="user",
        password="myjujz7e2prj",
        force_lease=True,                 # 需要“抢租约”时开这个
        client_name="MySpotClient",
    )

    # 上电并站立
    agent.power_on_and_stand()

    # 走两秒
    t0 = time.time()
    while time.time() - t0 < 2.0:
        agent.send_velocity(-0.8, 0.0, 0.0)  # 0.4 m/s 前进
        time.sleep(0.1)

    # 停稳
    agent.send_velocity(0.0, 0.0, 0.0, hold=0.5)

    # 释放租约（可选断电）
    agent.shutdown(power_off=False)
