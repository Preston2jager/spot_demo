import open3d as o3d
import os
import sys
import numpy as np

def view_ply(file_path):
    # 1. 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"❌ 错误: 找不到文件: {file_path}")
        print("请检查路径是否正确，或确保你已经运行过建图脚本。")
        return

    print(f"正在加载点云: {file_path} ...")
    
    # 2. 读取点云
    try:
        pcd = o3d.io.read_point_cloud(file_path)
    except Exception as e:
        print(f"读取失败: {e}")
        return

    if pcd.is_empty():
        print("⚠️ 警告: 点云是空的！请检查建图过程是否采集到了数据。")
        return

    # 打印基本信息
    points = np.asarray(pcd.points)
    print(f"✅ 加载成功!")
    print(f"   点数: {len(points)}")
    print(f"   范围: Min{points.min(axis=0)} / Max{points.max(axis=0)}")

    # 3. 创建坐标轴辅助 (X=红, Y=绿, Z=蓝)
    # size=0.5 表示坐标轴长度为 0.5 米
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])

    # 4. 启动可视化窗口
    print("\n操作指南:")
    print("  [鼠标左键] 旋转")
    print("  [鼠标右键] 平移")
    print("  [滚轮]     缩放")
    print("  [+/-]      增大/减小点的大小")
    print("  [Q]        退出")
    
    o3d.visualization.draw_geometries(
        [pcd, axis],
        window_name="Spot Map Viewer",
        width=2048,
        height=1536,
        left=50,
        top=50
    )

if __name__ == "__main__":
    # 默认路径：你之前脚本生成的默认位置
    default_path = "./final_map_result/final_pointcloud.ply"
    
    # 如果命令行带了参数，就用参数里的路径
    target_file = sys.argv[1] if len(sys.argv) > 1 else default_path
    
    view_ply(target_file)