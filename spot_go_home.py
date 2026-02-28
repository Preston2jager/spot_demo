from cls_rmit_spot_core import SpotAgent
def main():
    with SpotAgent(
        stream=True,
        navigation=True
        ) as agent:
        wp_name = "waypoint_0"
        agent.navigate_to_waypoint(agent.get_waypoint_id_by_name(agent.graph, wp_name))
if __name__ == "__main__":
    main()