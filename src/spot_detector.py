from typing import Dict, List, Optional

import cv2
import numpy as np
from ultralytics import YOLO


class SpotDetector:
    """YOLOv8 detector (PyTorch backend) targeting bottles and cans."""

    my_target_class = ["plastic water bottle", "aluminum soda can"]

    def __init__(self, weights: str = "yolov8m_rmit.pt", device: Optional[str] = None) -> None:
        """Load YOLO model from weights path and resolve target class IDs."""
        self.model = YOLO(weights)
        self.device = device
        if "world" in weights.lower():
            self.TARGET_CLASSES = self.my_target_class
            self.model.set_classes(self.TARGET_CLASSES)
        else:
            self.TARGET_CLASSES = ["bottle"]
        self.names: dict = self.model.names
        self.target_ids: set = set()
        for cls_id, cls_name in self.names.items():
            if cls_name in self.TARGET_CLASSES:
                self.target_ids.add(cls_id)
        if not self.target_ids:
            print(f"Warning: target classes not found in model: {self.TARGET_CLASSES}")
        else:
            print(f"Targets: {self.target_ids} -> {self.TARGET_CLASSES}")

    def detect_targets_in_batch(
        self,
        images_dict: Dict[str, np.ndarray],
        conf: float = 0.05,
        iou: float = 0.45,
    ) -> List[Dict]:
        """Run YOLO on a dict of named BGR images; return target detections sorted by confidence."""
        print("\n" + "=" * 5 + " Results " + "=" * 5)
        results_list: List[Dict] = []
        if not images_dict:
            return results_list
        cam_names: List[str] = []
        color_images: List[np.ndarray] = []
        for cam_name, image in images_dict.items():
            cam_names.append(cam_name)
            color_images.append(self._to_bgr(image))
        results = self.model(source=color_images, conf=0.05, iou=iou, classes=None, verbose=False)
        for i, r in enumerate(results):
            cam_name = cam_names[i]
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                continue
            print(f"[{cam_name}] Found: {len(boxes)} objects")
            best_conf = -1.0
            best_box = None
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = self.names.get(cls_id, "unknown")
                conf_val = float(box.conf[0])
                print(f"  {cls_name:<20} | Confidence: {conf_val:.2f}")
                if cls_id in self.target_ids and conf_val >= conf and conf_val > best_conf:
                    best_conf = conf_val
                    best_box = box
            if best_box is not None:
                x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
                results_list.append({
                    "camera": cam_name,
                    "cx": int((x1 + x2) / 2),
                    "cy": int((y1 + y2) / 2),
                    "conf": best_conf,
                    "class": self.names.get(int(best_box.cls[0])),
                })
        print("=" * 60 + "\n")
        results_list.sort(key=lambda x: x["conf"], reverse=True)
        return results_list

    def detect_single_image(
        self,
        image: np.ndarray,
        conf: float = 0.05,
        iou: float = 0.45,
    ) -> Optional[Dict]:
        """Run YOLO on a single image and return the best target detection or None."""
        image = self._to_bgr(image)
        results = self.model(source=image, conf=conf, iou=iou, classes=None, verbose=False)
        if not results or not results[0].boxes:
            return None
        best_conf = -1.0
        best_box = None
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf_val = float(box.conf[0])
            if cls_id in self.target_ids and conf_val >= conf and conf_val > best_conf:
                best_conf = conf_val
                best_box = box
        if best_box is None:
            return None
        x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
        return {
            "cx": int((x1 + x2) / 2),
            "cy": int((y1 + y2) / 2),
            "conf": best_conf,
            "class": self.names.get(int(best_box.cls[0]), "unknown"),
        }

    @staticmethod
    def _to_bgr(image: np.ndarray) -> np.ndarray:
        """Convert grayscale or BGRA images to BGR for YOLO compatibility."""
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 1:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image
