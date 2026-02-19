import time
from typing import Optional, Tuple, Union, Dict, List
import numpy as np

class YoloBottleDetector:
    """
    YOLOv8 诊断版检测器：
    - 会打印出所有检测到的物体，帮助分析为什么漏检。
    - 依然只返回 'bottle' 给机器狗去抓，防止抓错。
    """
    def __init__(self, weights: str = "yolov8x.pt", device: Optional[str] = None):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.names = self.model.names 
        # 找到 bottle 的 ID
        self.bottle_id = next((i for i, n in self.names.items() if n == "bottle"), 39)
        self.device = device

    def detect_bottles_in_batch(
        self,
        images_dict: Dict[str, np.ndarray],
        conf: float = 0.25, # 这是决定是否去抓的“严谨阈值”
        iou: float = 0.45
    ) -> List[Dict]:
        
        print("\n" + "="*20 + " 🔍 YOLO 视觉诊断报告 " + "="*20)
        results_list = []

        for cam_name, image in images_dict.items():
            # 1. 诊断模式：检测所有物体 (classes=None)，且阈值极低 (0.1)
            # 这样我们能看到到底识别成了什么，或者是不是置信度太低
            results = self.model(
                source=image,
                conf=0.1, 
                iou=iou,
                classes=None, # 不限制类别，看它到底认成啥了
                verbose=False
            )

            if not results or len(results) == 0 or not results[0].boxes:
                # print(f"[{cam_name}] ... (画面太暗或无物体)")
                continue

            r = results[0]
            boxes = r.boxes
            
            # 2. 打印该相机看到的所有东西
            print(f"📸 [{cam_name}] 发现:")
            
            found_target_in_this_cam = False
            best_conf_in_this_cam = -1.0
            best_box = None

            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = self.names[cls_id]
                conf_val = float(box.conf[0])
                
                # 打印调试信息
                print(f"   -> 📦 物体: {cls_name:<10} | 置信度: {conf_val:.2f}")

                # 3. 筛选逻辑：只有真的是 bottle 且 置信度 > 原定阈值(0.25) 才算数
                # (如果你发现它总是把瓶子认成 cup，可以在这里加 or cls_name == 'cup')
                if cls_id == self.bottle_id:
                    if conf_val >= conf:
                        # 找到有效目标
                        if conf_val > best_conf_in_this_cam:
                            best_conf_in_this_cam = conf_val
                            best_box = box
                            found_target_in_this_cam = True
                    else:
                        print(f"      ⚠️ 是瓶子，但置信度 {conf_val:.2f} 低于阈值 {conf}，被忽略。")

            # 4. 如果这张图里有合格的瓶子，加入返回列表
            if found_target_in_this_cam and best_box is not None:
                xyxy = best_box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                
                results_list.append({
                    "camera": cam_name,
                    "cx": cx,
                    "cy": cy,
                    "conf": best_conf_in_this_cam
                })

        print("="*60 + "\n")
        
        # 排序返回
        results_list.sort(key=lambda x: x['conf'], reverse=True)
        return results_list