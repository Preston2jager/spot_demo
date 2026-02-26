import time
import numpy as np
import cv2

# 导入刚刚我们写的 OpenVINO 检测类
# 请根据你的实际文件名修改导入路径
from cls_rmit_spot_detector_ov import SpotDetector

def run_npu_test():
    print("🚀 [Step 1] 开始加载模型到 NPU...")
    start_time = time.time()
    
    # 这里的 model_dir 替换为你导出的 OpenVINO 模型文件夹路径
    # 如果导出的是 YOLO-World，注意文件夹名字
    detector = SpotDetector(
        model_dir="yolov8m_rmit_openvino_model/", 
        #device="intel:npu"
        device="intel:gpu"# 如果 NPU 报错，可以临时改成 "CPU" 排除故障
    )
    
    load_time = time.time() - start_time
    print(f"✅ 模型加载耗时: {load_time:.2f} 秒\n")

    # --- 准备模拟数据 ---
    print("📦 [Step 2] 准备模拟多摄像头数据...")
    # 模拟 3 个摄像头的 640x640 画面 (随机生成彩色噪点图)
    dummy_images = {
        "cam_left": np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8),
        "cam_right": np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8),
        "cam_top": np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    }

    # --- 1. 预热 (Warm-up) ---
    print("🔥 [Step 3] 执行 NPU 预热 (首次推理会很慢，因为 NPU 在编译计算图)...")
    warmup_start = time.time()
    _ = detector.detect_targets_in_batch(dummy_images, conf=0.01)
    warmup_time = time.time() - warmup_start
    print(f"⚠️ 预热/首次推理耗时: {warmup_time:.2f} 秒\n")

    # --- 2. 性能测试 (Benchmark) ---
    print("⚡ [Step 4] 开始真实性能循环测试 (50 次)...")
    test_iterations = 50
    infer_start = time.time()
    
    for i in range(test_iterations):
        _ = detector.detect_targets_in_batch(dummy_images, conf=0.01)
        
    total_infer_time = time.time() - infer_start
    avg_time_per_batch = total_infer_time / test_iterations
    fps = 1.0 / avg_time_per_batch
    
    print("\n" + "="*30)
    print("📊 性能测试报告 (NPU)")
    print("="*30)
    print(f"总计测试次数: {test_iterations}")
    print(f"多图 Batch Size: {len(dummy_images)} (模拟 {len(dummy_images)} 个摄像头同步推理)")
    print(f"Batch 平均推理耗时: {avg_time_per_batch * 1000:.2f} 毫秒")
    print(f"相当于 Batch FPS: {fps:.2f} 帧/秒")
    print(f"单张图平均耗时约: {(avg_time_per_batch / len(dummy_images)) * 1000:.2f} 毫秒")
    print("="*30)

if __name__ == "__main__":
    run_npu_test()