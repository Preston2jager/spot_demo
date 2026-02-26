import time
from cls_rmit_spot_core import SpotAgent

def main():
    with SpotAgent(
        stream=True,
        navigation=True
        ) as agent:
        patrol_route = ["waypoint_175", "default"]
        patrol_ids = []
        for wp_name in patrol_route:
            wp_id = agent.get_waypoint_id_by_name(agent.graph, wp_name)
            if wp_id is not None:
                patrol_ids.append(wp_id)
            else:
                print(f"Failed to find waypoint: {wp_name}")
        for wp_id in patrol_ids:
            agent.navigate_to_waypoint(wp_id)
        time.sleep(1.0)  # 等待一段时间以确保到达最后一个巡逻点
if __name__ == "__main__":
    main()