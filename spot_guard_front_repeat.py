import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

import time
from cls_rmit_spot_core import SpotAgent
from cls_rmit_spot_detector_ov import SpotDetector 

def main():
    guard_point_name = "waypoint_25"
    home_point_name = "default"

    with SpotAgent(stream=True, navigation=True) as agent:
        detector = SpotDetector()
        guard_id = agent.get_waypoint_id_by_name(agent.graph, guard_point_name)
        home_id = agent.get_waypoint_id_by_name(agent.graph, home_point_name)
        
        if not guard_id or not home_id:
            print("❌ Failed to find required waypoints in the graph. Check graph content.")
            return
            
        print("\n🚀 Starting mission loop. Press Ctrl+C to exit.")
        try:
            while True:
                print("\n" + "="*50)
                print(f"📍 [Phase 1] Navigating to guard point: {guard_point_name}")
                
                if not agent.navigate_to_waypoint(guard_id):
                    print("⚠️ Failed to reach guard point. Retrying in 3 seconds...")
                    time.sleep(3.0)
                    continue # 失败了不要退出，等一会再试
                    
                print("🎯 Arrived at guard point. Starting search phase...")
                target_grasped = False
                
                while not target_grasped:
                    # ⚠️ 关键修改：每次尝试寻找前，确保手臂伸出（因为抓取失败后手臂会默认收起）
                    agent._arm_out()
                    time.sleep(2.0)
                    
                    # 1. 寻找目标
                    object_detected = agent.find_target(detector)
                    
                    if object_detected is None:
                        print("👀 未发现目标，等待 3 秒后重试扫描...")
                        time.sleep(3.0)
                    else:
                        # 2. ⚠️ 关键修改：正确解包元组中的 5 个参数
                        target_img_resp, cam_name, cx, cy, cls_name = object_detected
                        
                        # 3. 将 5 个参数分别传入抓取函数
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
                            
                # ----------------- 抓取成功，开始返航 -----------------
                print(f"\n🏠 [Phase 2] Go back to home: {home_point_name}")
                #agent.navigate_to_waypoint(agent.get_waypoint_id_by_name(agent.graph, "waypoint_173"))
                if agent.navigate_to_waypoint(home_id):
                    print("🏁 Reached home point. Releasing target...")
                    time.sleep(0.2) 
                    agent._arm_release(bin=True)  # 放在 bin 里
                else:
                    print("⚠️ Failed to return to home point! Please check manually.")
                    # 这里可以选择 break，或者原地放下物品
                    break 
                    
                print("♻️ Mission cycle completed. Preparing for next cycle...")
                time.sleep(2.0)
                
        except KeyboardInterrupt:
            print("\n🛑 KeyboardInterrupt detected. Exiting mission loop. Goodbye!")

if __name__ == "__main__":
    main()