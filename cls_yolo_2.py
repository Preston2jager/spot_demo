# ===== =================== =====
# ===== RMIT Spot yolo class =====
# ===== =================== =====
from typing import Optional, Dict, List
import numpy as np
from ultralytics import YOLO

# yolov8x-worldv2 for matching plastic water bottle and aluminum soda can.
# yolov8x for bottle only.

class YoloTargetDetector:
    
    def __init__(self, weights: str = "yolov8x-worldv2.pt", device: Optional[str] = None):
        self.model = YOLO(weights)
        self.names = self.model.names 
        self.device = device
        self.target_ids = set()
        if self.model is not None and hasattr(self.model, 'model') and self.model.model is not None:
            if self.model == "yolov8x-worldv2.pt":
                self.TARGET_CLASSES = ["plastic water bottle", "aluminum soda can"]
            else:
                self.TARGET_CLASSES = ["bottle"]
        else:
            print("⚠️ 无法获取模型名称信息")

    def detect_targets_in_batch(
        self,
        images_dict: Dict[str, np.ndarray],
        conf: float = 0.05, 
        iou: float = 0.45
        ) -> List[Dict]:
        print("\n" + "="*20 + " 🔍 YOLO 视觉诊断报告 " + "="*20)
        results_list = []
        for cam_name, image in images_dict.items():
            results = self.model(
                source=image,
                conf=0.1, 
                iou=iou,
                classes=None, 
                verbose=False
            )
            if not results or len(results) == 0 or not results[0].boxes:
                continue
            r = results[0]
            boxes = r.boxes
            print(f"📸 [{cam_name}] 发现:")
            found_target_in_this_cam = False
            best_conf_in_this_cam = -1.0
            best_box = None
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = self.names.get(cls_id, "unknown")
                conf_val = float(box.conf[0])
                print(f"   -> 📦 物体: {cls_name:<10} | 置信度: {conf_val:.2f}")
                if cls_id in self.target_ids:
                    if conf_val >= conf:
                        if conf_val > best_conf_in_this_cam:
                            best_conf_in_this_cam = conf_val
                            best_box = box
                            found_target_in_this_cam = True
                    else:
                        print(f"      ⚠️ 是目标 ({cls_name})，但置信度 {conf_val:.2f} 低于阈值 {conf}，被忽略。")
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
        results_list.sort(key=lambda x: x['conf'], reverse=True)
        return results_list