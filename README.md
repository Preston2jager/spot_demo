# Steps
1. Run the ```graph_nav_command_line.py``` in Maps to recording a map.
2. Make sure to have a fiducial in view when start the recording.
3. Download the map to local.
4. Use ```check_map.py``` in Maps to get the waypoint by order.
5. Use the waypoint to for navigation programming.

# Convert yolo model
```bash
yolo export model=yolov8m_rmit.pt format=openvino half=True imgsz=640 batch=3
```


==============================
📊 性能测试报告 (CPU)
==============================
总计测试次数: 50
多图 Batch Size: 3 (模拟 3 个摄像头同步推理)
Batch 平均推理耗时: 601.42 毫秒
相当于 Batch FPS: 1.66 帧/秒
单张图平均耗时约: 200.47 毫秒
==============================

==============================
📊 性能测试报告 (Intel:npu)
==============================
总计测试次数: 50
多图 Batch Size: 3 (模拟 3 个摄像头同步推理)
Batch 平均推理耗时: 184.97 毫秒
相当于 Batch FPS: 5.41 帧/秒
单张图平均耗时约: 61.66 毫秒
==============================

==============================
📊 性能测试报告 (Intel:GPU)
==============================
总计测试次数: 50
多图 Batch Size: 3 (模拟 3 个摄像头同步推理)
Batch 平均推理耗时: 84.95 毫秒
相当于 Batch FPS: 11.77 帧/秒
单张图平均耗时约: 28.32 毫秒
==============================