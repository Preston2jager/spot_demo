import time
from pynput import keyboard

from cls_spot import SpotAgent
from cls_yolo import YoloBottleDetector

current_state = "MANUAL" 
active_keys = set() # 新增：用于记录当前按下的键，防止 Linux 连击机制干扰

def main():
    global current_state, active_keys
    
    print("[系统] 正在初始化 Spot 和 YOLO，请稍候...")
    yolo = YoloBottleDetector("yolov8m.pt")
    
    # 替换为你刚才成功站立的实际 IP 和密码
    spot = SpotAgent("192.168.80.3", "user", "myjujz7e2prj") 
    spot.get_ready()
    
    print("\n" + "="*40)
    print("🎮 控制指南 (已开启键盘调试):")
    print("W/S : 前进 / 后退")
    print("A/D : 左侧平移 / 右侧平移")
    print("Q/E : 原地左转 / 原地右转")
    print(" G  : 第一次按 -> 启动 GUARD 模式; 第二次按 -> 趴下关机")
    print("ESC : 紧急退出并趴下")
    print("="*40 + "\n")

    def on_press(key):
        global current_state, active_keys
        try:
            char = key.char.lower()
            
            # --- 防连击过滤：如果这个键已经是按下状态，忽略操作 ---
            if char in active_keys:
                return 
            active_keys.add(char)
            print(f"[Debug] ⬇️ 检测到按键按下: {char}")
            
            if char == 'g':
                if current_state == "MANUAL":
                    current_state = "GUARD"
                    print("\n[状态切换] 🐶 进入 GUARD(自动寻物) 模式！")
                    spot.stop_movement()
                elif current_state == "GUARD":
                    current_state = "EXIT"
                    print("\n[状态切换] 🛑 收到退出指令，准备趴下并关机...")
                    
            if current_state == "MANUAL":
                spot.update_movement_state(char, is_pressed=True)
                
        except AttributeError:
            if key == keyboard.Key.esc:
                current_state = "EXIT"
                print("\n[状态切换] 🛑 按下ESC，准备退出...")

    def on_release(key):
        global current_state, active_keys
        try:
            char = key.char.lower()
            if char in active_keys:
                active_keys.remove(char)
                print(f"[Debug] ⬆️ 检测到按键松开: {char}")
                
            if current_state == "MANUAL":
                spot.update_movement_state(char, is_pressed=False)
        except AttributeError:
            pass

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    try:
        while current_state != "EXIT":
            if current_state == "MANUAL":
                spot.step_movement()
                
            elif current_state == "GUARD":
                print("[Guard] 📸 正在环视扫描目标...")
                camera_data = spot.search_once()
                cv2_images = {cam: data['cv2_img'] for cam, data in camera_data.items()}
                bottles = yolo.detect_bottles_in_batch(cv2_images)
                
                if bottles:
                    best = bottles[0]
                    print(f"🎯 发现目标！在相机 {best['camera']} 中，置信度 {best['conf']:.2f}")
                    
                    raw_resp = camera_data[best['camera']]['raw_response']
                    spot.grasp_target(raw_resp, best['cx'], best['cy'])
                    spot.return_and_drop()
                    
                    print("[Guard] ✅ 本次搬运完成，继续寻找下一个目标...")
                    time.sleep(1.0) 
                else:
                    print("[Guard] 视野内未发现瓶子，2秒后重新扫描 (按 'g' 可结束)...")
                    time.sleep(2.0)

    except KeyboardInterrupt:
        print("\n[系统] 收到 Ctrl+C，准备退出...")
        current_state = "EXIT"

    print("\n[系统] 执行最终休眠程序...")
    listener.stop()
    spot.stop_movement()
    spot.rest_down()
    print("👋 Spot 已安全关机。")

if __name__ == "__main__":
    main()