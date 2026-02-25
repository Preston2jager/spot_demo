# ===== =================== =====
# ===== RMIT Spot yolo class =====
# ===== =================== =====
from typing import Optional, Dict, List
import numpy as np
import cv2
from ultralytics import YOLO

class SpotDetector:
    my_target_class = [
        "plastic water bottle", 
        "aluminum soda can"
    ]
    
    def __init__(self, weights: str = "yolo11x.pt", device: Optional[str] = None):
        self.model = YOLO(weights)
        self.device = device
        if "world" in weights.lower():
            self.TARGET_CLASSES = self.my_target_class
            self.model.set_classes(self.TARGET_CLASSES)
        else:
            self.TARGET_CLASSES = ["bottle"]
            
        self.names = self.model.names 
        self.target_ids = set()
        
        for cls_id, cls_name in self.names.items():
            if cls_name in self.TARGET_CLASSES:
                self.target_ids.add(cls_id)
                
        if not self.target_ids:
            print(f"⚠️ Target class not found in model {self.TARGET_CLASSES}")
        else:
            print(f"Targets: {self.target_ids} -> {self.TARGET_CLASSES}")

    def detect_targets_in_batch(
        self,
        images_dict: Dict[str, np.ndarray],
        conf: float = 0.05, 
        iou: float = 0.45
    ) -> List[Dict]:
        
        print("\n" + "="*5 + "Results" + "="*5)
        results_list = []
        
        if not images_dict:
            return results_list

        cam_names = []
        color_images = []

        # 1. 预处理：确保所有图像都是 3 通道 BGR 彩色图像
        for cam_name, image in images_dict.items():
            if len(image.shape) == 2:  # 单通道灰度图 (H, W)
                color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 1:  # (H, W, 1)
                color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 4:  # 包含透明通道的 RGBA
                color_img = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            else:
                color_img = image  # 已经是标准 3 通道
                
            cam_names.append(cam_name)
            color_images.append(color_img)

        # 2. 真正的 Batch 推理：一次性传入图像列表，大幅提升速度！
        results = self.model(
            source=color_images,
            conf=0.05, # 底层放宽一点，后面再精确过滤
            iou=iou,
            classes=None, 
            verbose=False
        )
        
        # 3. 结果解析
        for i, r in enumerate(results):
            cam_name = cam_names[i]
            boxes = r.boxes
            
            if boxes is None or len(boxes) == 0:
                continue
                
            print(f"📸 [{cam_name}] 发现:")
            
            found_target_in_this_cam = False
            best_conf_in_this_cam = -1.0
            best_box = None
            
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = self.names.get(cls_id, "unknown")
                conf_val = float(box.conf[0])
                
                print(f"   -> 📦 物体: {cls_name:<20} | 置信度: {conf_val:.2f}")
                
                # 检查是否是我们关心的目标，并且大于要求的阈值
                if cls_id in self.target_ids:
                    if conf_val >= conf:
                        if conf_val > best_conf_in_this_cam:
                            best_conf_in_this_cam = conf_val
                            best_box = box
                            found_target_in_this_cam = True
                    else:
                        print(f"      ⚠️ 是目标 ({cls_name})，但置信度 {conf_val:.2f} 低于阈值 {conf}，忽略。")
                        
            if found_target_in_this_cam and best_box is not None:
                xyxy = best_box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                
                results_list.append({
                    "camera": cam_name,
                    "cx": cx,
                    "cy": cy,
                    "conf": best_conf_in_this_cam,
                    "class": self.names.get(int(best_box.cls[0]))
                })
                
        print("="*60 + "\n")
        # 按照置信度从高到低排序，确保返回的第一个总是最优目标
        results_list.sort(key=lambda x: x['conf'], reverse=True)
        return results_list

    def detect_single_image(
        self,
        image: np.ndarray,
        conf: float = 0.05, 
        iou: float = 0.45
    ) -> Optional[Dict]:
        """
        处理单张图片，返回视野中置信度最高的目标信息供抓取使用。
        """
        # 单张图片同样需要色彩保护
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        results = self.model(
            source=image,
            conf=conf, 
            iou=iou,
            classes=None, 
            verbose=False
        )
        
        if not results or len(results) == 0 or not results[0].boxes:
            return None 
            
        boxes = results[0].boxes
        best_conf = -1.0
        best_box = None
        
        for box in boxes:
            cls_id = int(box.cls[0])
            conf_val = float(box.conf[0])
            if cls_id in self.target_ids and conf_val >= conf:
                if conf_val > best_conf:
                    best_conf = conf_val
                    best_box = box  
                    
        if best_box is not None:
            xyxy = best_box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = xyxy
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            return {
                "cx": cx,
                "cy": cy,
                "conf": best_conf,
                "class": self.names.get(int(best_box.cls[0]), "unknown")
            }
        return None