from typing import Optional, Dict, List
import numpy as np
import cv2
from ultralytics import YOLO

class SpotDetector:
    
    my_target_class = ["bottle"]
    
    def __init__(self, model_dir: str = "yolov8m_rmit_openvino_model/", device: str = "intel:npu"):
        """
        :param model_dir: OpenVINO 导出的模型文件夹路径
        :param device: 'NPU', 'CPU', 或 'GPU' (Intel 核显)
        """
        self.model = YOLO(model_dir, task='detect')
        self.device = device 
        self.TARGET_CLASSES = self.my_target_class if "world" in model_dir.lower() else ["bottle"]
        self.names = self.model.names 
        self.target_ids = set()
        for cls_id, cls_name in self.names.items():
            if cls_name in self.TARGET_CLASSES:
                self.target_ids.add(cls_id)         
        if not self.target_ids:
            print(f"⚠️ Target class not found in model {self.TARGET_CLASSES}")
        else:
            print(f"✅ Targets loaded: {self.target_ids} -> {self.TARGET_CLASSES}")
            print(f"🚀 Using OpenVINO Backend on Device: {self.device}")
            
    def detect_targets_in_batch(
        self,
        images_dict: Dict[str, np.ndarray],
        conf: float = 0.05, 
        iou: float = 0.45
    ) -> List[Dict]:
        print("\n" + "="*5 + " OpenVINO NPU Results " + "="*5)
        results_list = []
        if not images_dict:
            return results_list
        cam_names = []
        color_images = []
        for cam_name, image in images_dict.items():
            if len(image.shape) == 2:
                color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 1:
                color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 4:
                color_img = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            else:
                color_img = image 
            cam_names.append(cam_name)
            color_images.append(color_img)
        results = self.model(
            source=color_images,
            device=self.device,
            conf=0.05, 
            iou=iou,
            classes=None, 
            verbose=False
        )
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
        results_list.sort(key=lambda x: x['conf'], reverse=True)
        return results_list