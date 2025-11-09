from cls_spot_agent import SpotAgent
import threading


if __name__ == "__main__":

    agent = SpotAgent(
        hostname="192.168.80.3",
        username="user",
        password="myjujz7e2prj",
        force_lease=True,
        client_name="MySpotClient",
    )

    agent.power_on_and_stand()

    t = threading.Thread(
        target=agent.handcam_on,
        kwargs={"window_title":"Spot HandCam", "width":1600, "height":1200, "fps":15},
        daemon=True,
    )
    t.start()

    agent.move_spot()

    agent.shutdown(power_off=False)
