from src.spot_agent import SpotAgent


def main() -> None:
    """Navigate Spot to a single named waypoint and exit."""
    wp_name = "waypoint_59"

    with SpotAgent(stream=True, navigation=True) as agent:
        agent.navigate_to_waypoint(agent.get_waypoint_id_by_name(agent.graph, wp_name))


if __name__ == "__main__":
    main()
