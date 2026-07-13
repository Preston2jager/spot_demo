import threading
from cls_spot_demo import SpotAgent_demo


if __name__ == "__main__":

    agent = SpotAgent_demo(
        hostname="192.168.80.3",
        username="user",
        password="myjujz7e2prj",
        force_lease=True,
        client_name="MySpotClient",
    )
    agent.power_on_and_stand()

    agent.arm_wave_overhead_simple()

    agent.shutdown(power_off=False)