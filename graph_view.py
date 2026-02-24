import os
import matplotlib.pyplot as plt
from bosdyn.api.graph_nav import map_pb2

# 修复：补充导入 Quat 四元数类
from bosdyn.client.math_helpers import SE3Pose, Quat

def view_saved_graph(save_dir="./spot_recorded_map"):
    graph_path = os.path.join(save_dir, "graph")
    if not os.path.exists(graph_path):
        print(f"找不到地图文件: {graph_path}")
        return

    # 1. 读取 Protobuf 格式的 Graph 文件
    with open(graph_path, "rb") as f:
        graph = map_pb2.Graph()
        graph.ParseFromString(f.read())

    if not graph.waypoints:
        print("地图中没有找到路点(Waypoints)！")
        return

    print(f"成功读取地图，共包含 {len(graph.waypoints)} 个路点，{len(graph.edges)} 条边。")

    # 2. 建立邻接表，存储相对位移
    adj = {wp.id: [] for wp in graph.waypoints}
    for edge in graph.edges:
        id0 = edge.id.from_waypoint
        id1 = edge.id.to_waypoint
        
        # 修复：使用 from_proto 替代已被弃用的 from_obj
        tform = SE3Pose.from_proto(edge.from_tform_to)
        
        if id0 in adj and id1 in adj:
            adj[id0].append((id1, tform))
            adj[id1].append((id0, tform.inverse())) 

    # 3. 使用广度优先搜索 (BFS) 推算所有点的“伪绝对坐标”
    start_id = graph.waypoints[0].id
    
    # 修复：正确初始化 SE3Pose，传入 x, y, z 和 Quat(w, x, y, z)
    identity_pose = SE3Pose(x=0.0, y=0.0, z=0.0, rot=Quat(w=1.0, x=0.0, y=0.0, z=0.0))
    wp_poses = {start_id: identity_pose} 
    
    queue = [start_id]

    while queue:
        curr_id = queue.pop(0)
        curr_pose = wp_poses[curr_id]

        for neighbor_id, tform in adj[curr_id]:
            if neighbor_id not in wp_poses:
                wp_poses[neighbor_id] = curr_pose * tform
                queue.append(neighbor_id)

    # 4. 准备绘图画布
    plt.figure(figsize=(8, 8))
    plt.title("Spot GraphNav 2D Path (Relative Estimation)")
    plt.xlabel("X (meters)")
    plt.ylabel("Y (meters)")
    plt.grid(True, linestyle='--', alpha=0.7)

    # 5. 绘制所有的 Waypoints (红点)
    xs = [pose.x for pose in wp_poses.values()]
    ys = [pose.y for pose in wp_poses.values()]
    plt.scatter(xs, ys, c='red', s=50, label="Waypoints", zorder=3)

    # 标注起点
    plt.scatter([wp_poses[start_id].x], [wp_poses[start_id].y], c='green', s=100, marker='*', label="Start Point", zorder=4)

    # 在点旁边标注路点 ID 的前4个字符
    for wp_id, pose in wp_poses.items():
        plt.text(pose.x, pose.y + 0.05, wp_id[:4], fontsize=8, ha='center')

    # 6. 绘制所有的 Edges (连接路点的蓝线)
    for edge in graph.edges:
        id0 = edge.id.from_waypoint
        id1 = edge.id.to_waypoint
        if id0 in wp_poses and id1 in wp_poses:
            x_vals = [wp_poses[id0].x, wp_poses[id1].x]
            y_vals = [wp_poses[id0].y, wp_poses[id1].y]
            plt.plot(x_vals, y_vals, c='blue', linewidth=2, zorder=2, alpha=0.6)

    # 显示图像
    plt.legend()
    plt.axis('equal') 
    plt.show()

if __name__ == "__main__":
    view_saved_graph("./spot_recorded_map")