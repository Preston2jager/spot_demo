import time

from typing import Optional, Tuple, Union
import numpy as np

class YoloBottleDetector:
    """
    YOLOv8 瓶子检测器：
    - 支持输入文件路径或 np.ndarray(BGR/RGB)。
    - 返回第一个 bottle 的中心点像素坐标 (x, y)(int)，左上角为 (0, 0)。
    - 若未检出，返回 None。
    """
    def __init__(self, weights: str = "yolov8m.pt", device: Optional[str] = None):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        # 解析 COCO 类别 id（更稳，不硬编码 39）
        names = self.model.names  # dict: {id: name}
        self.bottle_id = next((i for i, n in names.items() if n == "bottle"), 39)
        self.device = device
        if device is not None:
            # ultralytics 会在 predict 时自己选设备，这里只是提示/兼容
            pass

    def detect_first_bottle_xy(
        self,
        image: Union[str, np.ndarray],
        conf: float = 0.25,
        iou: float = 0.45,
        prefer_largest: bool = False,
    ) -> Optional[Tuple[int, int]]:
        """
        Args:
            image: 图像路径或 np.ndarray (H,W,3)。BGR/RGB 都可，YOLOv8 会自动处理。
            conf: 置信度阈值
            iou: NMS 阈值
            prefer_largest: 若为 True，返回面积最大的 bottle；否则返回最高置信度的第一个

        Returns:
            (x, y) 整数像素坐标，未检出则 None
        """
        # 推理（只保留 bottle 类）
        results = self.model(
            source=image,
            conf=conf,
            iou=iou,
            classes=[self.bottle_id],
            verbose=False
        )

        if not results or len(results) == 0:
            return None

        r = results[0]
        if r.boxes is None or r.boxes.xyxy is None or len(r.boxes) == 0:
            return None

        boxes = r.boxes  # ultralytics.engine.results.Boxes
        xyxy = boxes.xyxy.cpu().numpy()     # (N, 4) -> x1,y1,x2,y2
        confs = boxes.conf.cpu().numpy()    # (N,)

        if xyxy.shape[0] == 0:
            return None

        if prefer_largest:
            areas = (xyxy[:,2] - xyxy[:,0]) * (xyxy[:,3] - xyxy[:,1])
            idx = int(np.argmax(areas))
        else:
            idx = int(np.argmax(confs))

        x1, y1, x2, y2 = xyxy[idx].tolist()
        cx = int(round((x1 + x2) / 2.0))
        cy = int(round((y1 + y2) / 2.0))
        time.sleep(1)
        return (cx, cy)

