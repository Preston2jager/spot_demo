import os
from datasets import load_dataset
from ultralytics import YOLO

# 1. 极速获取数据
print("正在下载数据集...")
ds = load_dataset("cuongdtdg/plastic_bottle")

# 2. 准备 YOLO 格式目录
base_path = "datasets/plastic_bottle"
os.makedirs(f"{base_path}/train/images", exist_ok=True)
os.makedirs(f"{base_path}/train/labels", exist_ok=True)

# 转换函数：将 HF 数据写入本地文件
print("正在转换格式以适配 CPU 训练...")
for i, item in enumerate(ds['train']):
    # 保存图片
    img_path = f"{base_path}/train/images/img_{i}.jpg"
    item['image'].convert("RGB").save(img_path)
    
    # 写入标签 (假设数据集 objects 包含 bbox 和 category)
    # 注意：YOLO 格式为 <category> <x_center> <y_center> <width> <height> (归一化)
    with open(f"{base_path}/train/labels/img_{i}.txt", "w") as f:
        # 如果数据集中没有 objects 字段，请根据实际 print(ds['train'][0]) 的结果调整
        for obj in item.get('objects', []):
            box = obj['bbox'] # 需确认是否已归一化
            cat = obj['category']
            f.write(f"{cat} {box[0]} {box[1]} {box[2]} {box[3]}\n")

# 3. 创建配置文件
yaml_str = f"""
path: {os.path.abspath(base_path)}
train: train/images
val: train/images
names:
  0: plastic_bottle
"""
with open("plastic_cpu.yaml", "w") as f:
    f.write(yaml_str)

# 4. 启动 CPU 优化训练
print("启动 YOLOv8x CPU 强化训练...")
model = YOLO("yolov8x.pt") # 加载官方预训练权重

model.train(
    data="plastic_cpu.yaml",
    epochs=10,         # CPU 训练慢，建议先跑 5-10 轮看效果
    imgsz=320,         # 【核心优化】缩小图片尺寸 (从640降到320) 可大幅提升 CPU 速度
    batch=1,           # CPU 训练建议 batch 设为 1 或 2，防止内存溢出
    device='cpu',      # 【强制指定】使用 CPU
    amp=False,         # CPU 必须关闭混合精度训练
    workers=1          # 减少线程，防止 CPU 负载过高导致卡死
)