# Steps
1. Run the ```graph_nav_command_line.py``` in Maps to recording a map.
2. Make sure to have a fiducial in view when start the recording.
3. Download the map to local.
4. Use ```check_map.py``` in Maps to get the waypoint by order.
5. Use the waypoint to for navigation programming.

# Yolo model 
## Convert yolo model to openvino
```bash
yolo export model=yolov8m_rmit.pt format=openvino half=True imgsz=640 batch=3
```
## Benchmarking for Ultra 125h
Intel:CPU 200.47 ms/f
Intel:NPU 61.66 ms/f
Intel:GPU 28.32 ms/f

