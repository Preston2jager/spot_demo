# t4.py
import threading, time
from cls_spot_agent import SpotAgent
from cls_patrol import PatrolManager

def main():
    agent = SpotAgent(
        hostname="192.168.80.3",
        username="user",
        password="myjujz7e2prj",
        force_lease=True,
        client_name="MySpotClient",
    )
    agent.power_on_and_stand()
    time.sleep(0.5)  # 给状态服务一点点时间

    # 1) 巡逻线程（后台跑）
    patrol = PatrolManager(agent.cmd_client, agent.state_client, distance_m=5.0)
    thr_patrol = threading.Thread(target=patrol.start, name="patrol", daemon=True)
    thr_patrol.start()

    try:
        while True:
            # 简单心跳，确保主线程不退出
            time.sleep(0.5)
    except KeyboardInterrupt:
        patrol.stop()
    finally:
        agent.shutdown(power_off=False)

if __name__ == "__main__":
    main()
