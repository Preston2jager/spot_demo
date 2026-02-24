# ===== =================== =====
# ===== RMIT Spot yolo class =====
# ===== =================== =====
from typing import Optional, Dict, List
import numpy as np
from ultralytics import YOLO

class YoloTargetDetector:
    
    def __init__(self, weights: str = "yolov8x-worldv2.pt", device: Optional[str] = None):
        # 1. 初始化模型
        self.model = YOLO(weights)
        self.device = device
        
        # 2. 判断是否为 World 模型，并设置对应的目标类别
        if "world" in weights.lower():
            self.TARGET_CLASSES = ["plastic water bottle", "aluminum soda can"]
            # 必须告诉 World 模型你要寻找哪些文字标签
            self.model.set_classes(self.TARGET_CLASSES)
        else:
            self.TARGET_CLASSES = ["bottle"]
            
        # 3. 获取模型当前的类别映射表
        self.names = self.model.names 
        
        # 4. 动态填充 target_ids (非常重要，否则检测时找不到 id)
        self.target_ids = set()
        for cls_id, cls_name in self.names.items():
            if cls_name in self.TARGET_CLASSES:
                self.target_ids.add(cls_id)
                
        if not self.target_ids:
            print(f"⚠️ 警告: 模型中没有找到目标类别 {self.TARGET_CLASSES}")
        else:
            print(f"✅ 成功加载模型! 监控目标类别 IDs: {self.target_ids} -> {self.TARGET_CLASSES}")

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
                conf=0.1,  # 注意：这里固定成了0.1，如果你想用传进来的 conf，可以改为 conf=conf
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
                
                print(f"   -> 📦 物体: {cls_name:<20} | 置信度: {conf_val:.2f}")
                
                # 检查是否是我们关心的目标
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
                    "conf": best_conf_in_this_cam,
                    "class": self.names.get(int(best_box.cls[0])) # 顺便记录一下具体是哪种物品
                })
                
        print("="*60 + "\n")
        # 按照置信度从高到低排序
        results_list.sort(key=lambda x: x['conf'], reverse=True)
        return results_list
    
    # 将此方法添加到 cls_yolo_2.py 的 YoloTargetDetector 类中

    def detect_single_image(
        self,
        image: np.ndarray,
        conf: float = 0.05, 
        iou: float = 0.45
    ) -> Optional[Dict]:
        """
        处理单张图片，返回视野中置信度最高的目标信息供抓取使用。
        返回格式: {"cx": int, "cy": int, "conf": float, "class": str} 或 None
        """
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
            
            # 只提取我们在 target_ids 中定义的物体
            if cls_id in self.target_ids and conf_val >= conf:
                if conf_val > best_conf:
                    best_conf = conf_val
                    best_box = box
                    
        if best_box is not None:
            xyxy = best_box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = xyxy
            # 计算边界框中心点坐标
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            return {
                "cx": cx,
                "cy": cy,
                "conf": best_conf,
                "class": self.names.get(int(best_box.cls[0]), "unknown")
            }
            
        return None