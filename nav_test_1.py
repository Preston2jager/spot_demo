import os
import time
from cls_rmit_spot import SpotAgent
from bosdyn.client.graph_nav import GraphNavClient
from bosdyn.client.recording import GraphNavRecordingServiceClient
from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder
from bosdyn.client.frame_helpers import ODOM_FRAME_NAME

if __name__ == "__main__":
    with SpotAgent(
        hostname="192.168.80.3",
        username="user",
        password="myjujz7e2prj",
        force_lease=True,
        client_name="MySpotClient",
    ) as agent:
        
        # 1. 初始化所需的所有客户端
        graph_nav_client = agent.robot.ensure_client(GraphNavClient.default_service_name)
        recording_client = agent.robot.ensure_client(GraphNavRecordingServiceClient.default_service_name)
        command_client = agent.robot.ensure_client(RobotCommandClient.default_service_name)
        
        # (假设 SpotAgent 已经帮你完成了站立动作 robot_command.RobotCommandBuilder.synchro_stand_command)
        
        # 2. 清理旧地图并开始录制
        print("清理旧地图...")
        graph_nav_client.clear_graph()
        print("开始 GraphNav 录制...")
        recording_client.start_recording()
        
        # 3. 通过代码发送移动指令 (代替平板遥控)
        print("指令发送：向前直线移动 2.0 米...")
        
        # 构建一个相对于机器人当前机身坐标系的移动指令 (X正方向为正前方)
        # 注意：这里使用的是简单的 body 坐标系移动，如果你需要更复杂的路径，需要计算 ODOM 坐标
        move_cmd = RobotCommandBuilder.synchro_trajectory_command_in_body_frame(
            goal_x_rt_body=2.0,  # 向前走 2 米
            goal_y_rt_body=0.0,  # 左右不偏移
            goal_heading_rt_body=0.0, # 不转弯
            frame_name=bosdyn.client.frame_helpers.BODY_FRAME_NAME,
            params=None
        )
        
        # 发送指令并等待完成
        command_id = command_client.robot_command(move_cmd)
        
        # 简单的等待逻辑：给它 10 秒钟时间走完这 2 米
        # (在高级应用中，应该轮询 command_client.robot_command_feedback 来判断是否到达终点)
        time.sleep(10) 
        print("移动指令执行完毕。")
        
        # 4. 停止录制
        print("停止录制...")
        recording_client.stop_recording()
        
        # 5. 下载地图 (与之前的代码完全一致)
        print("正在下载测试地图...")
        graph = graph_nav_client.download_graph()
        download_dir = "test_auto_map"
        os.makedirs(download_dir, exist_ok=True)
        os.makedirs(os.path.join(download_dir, "waypoint_snapshots"), exist_ok=True)
        os.makedirs(os.path.join(download_dir, "edge_snapshots"), exist_ok=True)
        
        with open(os.path.join(download_dir, "graph"), "wb") as f:
            f.write(graph.SerializeToString())
            
        for waypoint in graph.waypoints:
            snapshot = graph_nav_client.download_waypoint_snapshot(waypoint.snapshot_id)
            with open(os.path.join(download_dir, "waypoint_snapshots", waypoint.snapshot_id), "wb") as f:
                f.write(snapshot.SerializeToString())
                
        for edge in graph.edges:
            if edge.snapshot_id:
                snapshot = graph_nav_client.download_edge_snapshot(edge.snapshot_id)
                with open(os.path.join(download_dir, "edge_snapshots", edge.snapshot_id), "wb") as f:
                    f.write(snapshot.SerializeToString())
                    
        print(f"测试地图已保存至 '{download_dir}' 目录！")