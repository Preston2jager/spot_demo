from cls_spot_agent_test import SpotAgent
import time

if __name__ == "__main__":
    agent = SpotAgent(
        hostname="192.168.80.3",
        username="user",
        password="myjujz7e2prj",
        force_lease=True,
        client_name="MySpotClient",
    )
    time.sleep(3)
    agent.power_on_and_stand()
    #agent.move_to_goal(0,0,0)
    agent.guard()