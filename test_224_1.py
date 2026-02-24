from cls_rmit_spot_nav import SpotAgent
from cls_yolo_2 import YoloTargetDetector
import time

if __name__ == "__main__":
    detector = YoloTargetDetector()
    with SpotAgent() as agent:
        agent._arm_out()
        t = 10
        
        for i in range(t):
            print(f"[Main] 正在进行第 {i+1} 次尝试...")
            
            # 因为 find_and_grasp_target 内部会阻塞等待
            # 如果它返回 True，说明：1. 找到了；2. 发送指令了；3. 且动作执行完了。
            if agent.find_and_grasp_target(detector):
                print("[Main] 检测并抓取成功，准备结束任务。")
                break  # 成功后直接跳出循环
            else:
                # 返回 False 可能是没找到，也可能是抓取失败
                print("[Main] 本轮未成功（未发现目标或抓取失败），等待 1 秒后重试...")
                time.sleep(1)
        
        # 无论是成功 break，还是 10 次耗尽，统一收起机械臂
        print("[Main] 正在收回机械臂...")
        agent._arm_in()
        
    print("[Main] 程序已安全退出。")