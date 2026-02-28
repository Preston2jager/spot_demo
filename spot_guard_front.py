import os
import time

# 环境变量设置
os.environ["QT_QPA_PLATFORM"] = "xcb"

from cls_rmit_spot_core import SpotAgent
from cls_rmit_spot_detector_ov import SpotDetector 

def main():
    guard_point_name = "waypoint_25"
    home_point_name = "default"

    # 添加 try 块以配合最后的 KeyboardInterrupt 捕获
    try:
        with SpotAgent(stream=True, navigation=True) as agent:
            detector = SpotDetector()
            
            # 统一获取并检查所有所需的导航点 ID
            guard_id = agent.get_waypoint_id_by_name(agent.graph, guard_point_name)
            home_id = agent.get_waypoint_id_by_name(agent.graph, home_point_name)
            wp_122_id = agent.get_waypoint_id_by_name(agent.graph, "waypoint_122")
            wp_124_id = agent.get_waypoint_id_by_name(agent.graph, "waypoint_124")
            wp_65_id = agent.get_waypoint_id_by_name(agent.graph, "waypoint_65")
            
            if not all([guard_id, home_id, wp_122_id, wp_124_id, wp_65_id]):
                print("❌ Failed to find required waypoints in the graph. Check graph content.")
                return
            
            
            # ----------------- 前往观测点 -----------------
            print(f"🚶 Navigating to guard point: {guard_point_name}")
            # 使用 while 循环来实现失败重试
            while not agent.navigate_to_waypoint(guard_id):
                print("⚠️ Failed to reach guard point. Retrying in 3 seconds...")
                time.sleep(3.0)
                        
            print("🎯 Arrived at guard point. Starting search phase...")
            
            # ----------------- 目标搜索与抓取阶段 -----------------
            target_grasped = False
                    
            while not target_grasped:
                # 每次尝试寻找前，确保手臂伸出（因为抓取失败后手臂会默认收起）
                agent._arm_out()
                time.sleep(2.0)
                        
                object_detected = agent.find_target(detector)
                
                if object_detected is None:
                    print("👀 未发现目标，等待 3 秒后重试扫描...")
                    time.sleep(3.0)
                    continue  # 直接进入下一次 while 循环重新扫描
                
                # 发现目标：正确解包元组中的 5 个参数
                target_img_resp, cam_name, cx, cy, cls_name = object_detected
                
                # 将 5 个参数分别传入抓取函数
                success = agent.grasp_object(target_img_resp, cam_name, cx, cy, cls_name)
                
                if success:
                    print("✅ Target grasped successfully!")
                    target_grasped = True
                    time.sleep(0.2)
                else:
                    print("❌ 抓取失败或抓空。重新退回观察点调整姿态...")
                    # 抓取失败时，_recover_arm_safely 已将手臂收起，这里只需走回原位重试
                    agent.navigate_to_waypoint(guard_id)
                    time.sleep(2.0)
                            
            # ----------------- 抓取成功，开始返航阶段 -----------------
            # 退出 while 循环说明抓取成功，进行 Phase 2
            print(f"\n🏠 [Phase 2] Go back to home: {home_point_name}")
            
            if agent.navigate_to_waypoint(home_id):
                print("🏁 Reached home point. Releasing target...")
                time.sleep(0.2) 
                agent._arm_release(bin=True)  # 放在 bin 里
                time.sleep(1.2) 
                
                # ----------------- 最终待命点 -----------------
                print("🚶 Moving to final standby point: waypoint_65")
                agent.navigate_to_waypoint(wp_65_id)  
                print("🎉 Task complete! Spot is resting at waypoint_65.")
            else:
                print("⚠️ Failed to return to home point! Please check manually.")
                   
    except KeyboardInterrupt:
        print("\n🛑 KeyboardInterrupt detected. Exiting mission loop. Goodbye!")

if __name__ == "__main__":
    main()