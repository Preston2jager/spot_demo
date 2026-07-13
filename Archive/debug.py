import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

import time
from cls_rmit_spot_core import SpotAgent
from cls_rmit_spot_detector import SpotDetector 
from cls_rmit_spot_tracker import SpotTracker

def main():
    guard_point_name = "waypoint_189"
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
                    object_detected = agent.quick_detect(detector)
                    
                    if object_detected is None:
                        print("👀 未发现目标，等待 3 秒后重试扫描...")
                        time.sleep(3.0)
                    else:
                        pass
                
        except KeyboardInterrupt:
            print("\n🛑 KeyboardInterrupt detected. Exiting mission loop. Goodbye!")

if __name__ == "__main__":
    main()