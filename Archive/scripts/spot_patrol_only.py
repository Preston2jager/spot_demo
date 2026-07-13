import time

from src.spot_agent import SpotAgent


def main() -> None:
    """Navigate Spot through a fixed patrol route once and exit."""
    with SpotAgent(stream=True, navigation=True) as agent:
        patrol_route = ["waypoint_175", "default"]
        patrol_ids = []
        for wp_name in patrol_route:
            wp_id = agent.get_waypoint_id_by_name(agent.graph, wp_name)
            if wp_id is not None:
                patrol_ids.append(wp_id)
            else:
                print(f"Waypoint not found: {wp_name}")
        for wp_id in patrol_ids:
            agent.navigate_to_waypoint(wp_id)
        time.sleep(1.0)


if __name__ == "__main__":
    main()
