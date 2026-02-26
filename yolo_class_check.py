from cls_rmit_spot_detector_ov import SpotDetector # 假设你已经实例化了

detector = SpotDetector()

# 打印所有 80 个类别
print(detector.model.names)

# 如果想看得很整齐：
import json
print(json.dumps(detector.model.names, indent=4))