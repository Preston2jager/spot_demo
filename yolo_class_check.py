from cls_yolo_2 import YoloTargetDetector # 假设你已经实例化了

detector = YoloTargetDetector()

# 打印所有 80 个类别
print(detector.model.names)

# 如果想看得很整齐：
import json
print(json.dumps(detector.model.names, indent=4))