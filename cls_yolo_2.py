import time
from typing import Optional, Tuple, Union, Dict, List
import numpy as np

class YoloTargetDetector:
    # 🎯 在这里硬编码你需要抓取的类别，支持随时增删
    TARGET_CLASSES = ["bottle", "can"] 

    def __init__(self, weights: str = "yolov8x.pt", device: Optional[str] = None):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.names = self.model.names 
        self.device = device
        
        # 自动将名称映射为模型内部的 ID 集合
        self.target_ids = set()
        for cls_name in self.TARGET_CLASSES:
            # 遍历 names 字典查找对应的 ID，找不到则忽略并警告
            cls_id = next((i for i, n in self.names.items() if n == cls_name), None)
            if cls_id is not None:
                self.target_ids.add(cls_id)
            else:
                print(f"⚠️ 警告: 模型字典中未找到类别 '{cls_name}'")

    def detect_targets_in_batch(
        self,
        images_dict: Dict[str, np.ndarray],
        conf: float = 0.05, # 这是决定是否去抓的“严谨阈值”
        iou: float = 0.45
    ) -> List[Dict]:
        print("\n" + "="*20 + " 🔍 YOLO 视觉诊断报告 " + "="*20)
        results_list = []
        
        for cam_name, image in images_dict.items():
            # 1. 诊断模式：检测所有物体，且阈值极低 (0.1)
            results = self.model(
                source=image,
                conf=0.1, 
                iou=iou,
                classes=None, # 不限制类别，看它到底认成啥了
                verbose=False
            )

            if not results or len(results) == 0 or not results[0].boxes:
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
                cls_name = self.names.get(cls_id, "unknown")
                conf_val = float(box.conf[0])
                
                # 打印调试信息
                print(f"   -> 📦 物体: {cls_name:<10} | 置信度: {conf_val:.2f}")

                # 3. 筛选逻辑：只要是在 TARGET_CLASSES 里的类别，且 置信度 >= conf 阈值就算数
                if cls_id in self.target_ids:
                    if conf_val >= conf:
                        # 找到有效目标，保留该相机画面下置信度最高的那一个
                        if conf_val > best_conf_in_this_cam:
                            best_conf_in_this_cam = conf_val
                            best_box = box
                            found_target_in_this_cam = True
                    else:
                        print(f"      ⚠️ 是目标 ({cls_name})，但置信度 {conf_val:.2f} 低于阈值 {conf}，被忽略。")

            # 4. 如果这张图里有合格的目标，加入返回列表
            if found_target_in_this_cam and best_box is not None:
                xyxy = best_box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                
                # 严格保持原有的输出格式不变
                results_list.append({
                    "camera": cam_name,
                    "cx": cx,
                    "cy": cy,
                    "conf": best_conf_in_this_cam
                })

        print("="*60 + "\n")
        
        # 按置信度排序返回
        results_list.sort(key=lambda x: x['conf'], reverse=True)
        return results_list
    
    def fast_detect(
        self,
        images_dict: Dict[str, np.ndarray],
        conf: float = 0.05,
        iou: float = 0.45
    ) -> Optional[str]:
        """
        ⚡ 快速检测模式：只要发现任何一个目标，立即返回对应的相机名称。
        专为速度优化，直接限制检测类别，减少后处理计算。
        """
        print("\n" + "="*20 + " ⚡ YOLO 快速检测 " + "="*20)
        
        # 将 target_ids 转为列表，传入模型以加速推理，过滤掉不相干的类别
        target_classes_list = list(self.target_ids) if self.target_ids else None

        for cam_name, image in images_dict.items():
            # 推理时直接通过 classes 参数限制只看指定的 ID
            results = self.model(
                source=image,
                conf=conf, 
                iou=iou,
                classes=target_classes_list, 
                verbose=False
            )

            if not results or len(results) == 0 or not results[0].boxes:
                continue

            # 因为前面已经过滤了 classes，只要这里有框，且置信度达标，就是我们要找的
            for box in results[0].boxes:
                conf_val = float(box.conf[0])
                if conf_val >= conf:
                    cls_id = int(box.cls[0])
                    cls_name = self.names.get(cls_id, "unknown")
                    print(f"⚡ 警报! 在 [{cam_name}] 快速锁定 {cls_name}！置信度: {conf_val:.2f}")
                    print("="*58 + "\n")
                    return cam_name # 找到就立刻返回相机名称，不再检查其他图片

        print("⚡ 未发现目标。")
        print("="*58 + "\n")
        return None