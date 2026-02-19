import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

import sys
import time
from cls_spot import SpotAgent
from cls_yolo_2 import YoloBottleDetector

def print_help():
    print("\n" + "="*30)
    print("🐕 Spot 命令行控制器")
    print("="*30)
    print("f [米]   : 前进 (例如 f 1.0)")
    print("b [米]   : 后退 (例如 b 0.5)")
    print("l [度]   : 左转 (例如 l 90)")
    print("r [度]   : 右转 (例如 r 45)")
    print("cam      : 开启/关闭 5路实时监控窗口 (New!)")
    print("g        : 启动一次自动寻物 (Guard)")
    print("sit      : 趴下并安全退出")
    print("help     : 显示此菜单")
    print("="*30 + "\n")

def main():
    print("[System] Loading YOLO...")
    yolo = YoloBottleDetector("yolov8m.pt")
    
    print("[System] Connecting to Spot...")
    # 请替换为你的真实IP和密码
    spot = SpotAgent("192.168.80.3", "user", "myjujz7e2prj")
    
    spot.get_ready()
    print_help()

    while True:
        try:
            cmd_str = input("Spot> ").strip().lower()
            if not cmd_str: continue

            parts = cmd_str.split()
            op = parts[0]

            if op == 'sit' or op == 'exit':
                print("停止中...")
                spot.rest_down()
                break

            elif op == 'help':
                print_help()

            # --- 新增：监控开关 ---
            elif op == 'cam':
                if spot._streaming:
                    spot.stop_stream()
                else:
                    spot.start_stream()

            # --- 移动指令 ---
            elif op == 'f': 
                dist = float(parts[1]) if len(parts) > 1 else 1.0
                spot.move_relative(fwd=dist)

            elif op == 'b': 
                dist = float(parts[1]) if len(parts) > 1 else 1.0
                spot.move_relative(fwd=-dist)

            elif op == 'l': 
                deg = float(parts[1]) if len(parts) > 1 else 90.0
                spot.move_relative(rot_deg=deg)

            elif op == 'r': 
                deg = float(parts[1]) if len(parts) > 1 else 90.0
                spot.move_relative(rot_deg=-deg)

            elif op == 'ml': 
                dist = float(parts[1]) if len(parts) > 1 else 0.5
                spot.move_relative(left=dist)

            elif op == 'mr': 
                dist = float(parts[1]) if len(parts) > 1 else 0.5
                spot.move_relative(left=-dist)

            # --- 自动任务 ---
            elif op == 'g': 
                print("[Task] 执行一次环视搜索...")
                data = spot.search_once()
                cv2_imgs = {k: v['cv2_img'] for k, v in data.items()}
                
                bottles = yolo.detect_bottles_in_batch(cv2_imgs)
                
                if bottles:
                    best = bottles[0]
                    print(f"✅ 发现瓶子! 相机: {best['camera']}, 置信度: {best['conf']:.2f}")
                    spot.grasp_target(data[best['camera']]['raw_response'], best['cx'], best['cy'])
                    spot.return_and_drop()
                    print("[Task] 任务完成。")
                else:
                    print("⚠️ 未发现目标。")

            else:
                print(f"未知指令: {op}")

        except ValueError:
            print("❌ 参数错误，请输入数字。")
        except KeyboardInterrupt:
            print("\n强制退出...")
            spot.rest_down()
            sys.exit(0)
        except Exception as e:
            print(f"❌ 执行错误: {e}")

if __name__ == "__main__":
    main()