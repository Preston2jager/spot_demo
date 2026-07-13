from cls_rmit_spot import SpotAgent
import time

if __name__ == "__main__":
    agent = SpotAgent(
        hostname="192.168.80.3",
        username="user",
        password="myjujz7e2prj",
        force_lease=True,
        client_name="MySpotClient",
    )
    agent.power_on_and_stand()

    agent.get_home_location()
    agent.move_to_goal(2, 2, 180)
    time.sleep(10)
    
    agent.get_guard_location()
    agent.move_to_goal(agent.home_x, agent.home_y, agent.home_yaw)
    time.sleep(10)
    
    agent.move_to_goal(agent.guard_x, agent.guard_y, agent.guard_yaw)
    time.sleep(10)
    
    agent.debug_pose()

    # --- 关键修改：动作跑完了，但不退出 ---
    print("\n[INFO] 所有动作指令已完成。")
    print("[INFO] 推流和 Web Server 仍在后台运行...")
    print("[INFO] 按 Ctrl+C 可以手动结束程序。\n")

    try:
        # 使用死循环阻塞主线程，让后台的 Stream 线程继续工作
        while True:
            time.sleep(1) 
    except KeyboardInterrupt:
        print("\n[INFO] 正在关闭机器人连接并退出...")
        # 这里可以加入 agent.power_off() 之类的收尾动作