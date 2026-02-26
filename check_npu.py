from openvino.runtime import Core
core = Core()
print("🎉 OpenVINO 当前可用设备:", core.available_devices)