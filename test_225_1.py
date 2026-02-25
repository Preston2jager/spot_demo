import time
from cls_rmit_spot_nav import SpotAgent
from cls_rmit_spot_detector import SpotDetector 

def main():
    with SpotAgent() as agent:
        detector = SpotDetector()
        
        # 1. 地图与定位
        graph = agent.get_current_graph()
        if not graph:
            graph = agent.upload_graph_and_snapshots("./graph_nav_command_line/08_12_office")
            if not graph:
                print("[Error] 地图为空，终止。")
                return
                
        if not agent.initialize_graphnav_to_fiducial():
            print("[Error] 定位失败，终止。")
            return
            
        # 2. 验证路点
        guard_point_name = "waypoint_26"
        home_point_name = "default"
        guard_id = agent.get_waypoint_id_by_name(graph, guard_point_name)
        home_id = agent.get_waypoint_id_by_name(graph, home_point_name)
        
        if not guard_id or not home_id:
            print("[Error] 路点名称验证失败，终止。")
            return

        print("\n--- 哨兵模式已启动 (按 Ctrl+C 退出) ---")
        

        # 3. 核心状态机
        try:
            while True:
                # 阶段 1: 前往防守点
                print(f"\n-> 出击: {guard_point_name}")
                if not agent.navigate_to_waypoint(guard_id):
                    print("[Error] 前往防守点失败。")
                    break 
                
                print("-> 就位，扫描中...")
                time.sleep(1.0)
                agent._arm_out()
                
                # 阶段 2: 驻守与抓取
                target_grasped = False
                while not target_grasped:
                    # 调用抓取函数，返回 True 表示成功抓起
                    success = agent.find_and_grasp_target(detector)
                    
                    if success:
                        print("🚨 成功锁定并抓取目标！")
                        target_grasped = True
                    else:
                        time.sleep(1.0)
                
                # 阶段 3: 带着物体回城
                print(f"\n-> 撤退: {home_point_name}")
                if agent.navigate_to_waypoint(home_id):
                    print("-> 抵达基地，休整3秒...")
                    time.sleep(3.0) 
                    # 提示：在这里你可能需要写一行让机器狗松开夹爪扔下物体的代码
                else:
                    print("[Error] 返回基地失败。")
                    break

        except KeyboardInterrupt:
            print("\n-> 收到退出指令，结束任务。")

if __name__ == "__main__":
    main()