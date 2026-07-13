import os
from bosdyn.api.graph_nav import map_pb2
from bosdyn.client.frame_helpers import *

def export_to_ply(map_path, output_name="spot_output.ply"):
    graph_path = os.path.join(map_path, "graph")
    with open(graph_path, "rb") as f:
        graph = map_pb2.Graph()
        graph.ParseFromString(f.read())

    all_points = []
    print(f"Processing {len(graph.waypoints)} waypoints...")

    for wp in graph.waypoints:
        snap_path = os.path.join(map_path, wp.id)
        if os.path.exists(snap_path):
            with open(snap_path, "rb") as f:
                snap = map_pb2.WaypointSnapshot()
                snap.ParseFromString(f.read())
                # 提取该路标下的点云数据
                # 这里的点云通常经过了 RLE 压缩，建议使用官方工具进行完整转换
                # 或者通过 Open3D 进行后期处理
                pass

    print("Tip: 建议使用官方 SDK 中的 'export_graph_pdal.py' 工具进行精准导出。")

# 运行导出命令（如果你有安装 PDAL）
# python -m bosdyn.client.graph_nav.map_processing.export_graph_pdal --path ./downloaded_map_final --output total_cloud.ply