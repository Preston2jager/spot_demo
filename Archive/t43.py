import threading, time
from cls_spot_agent import SpotAgent
from cls_yolo import YoloBottleDetector


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


    detector = YoloBottleDetector("yolov8m.pt")
    agent.detect_grab_return_and_release(
        detector,
        source="hand_color_image",
        home_frame="odom",        # 也可用 "vision"（相对短程、但可能跳变）
        open_fraction=1.0,
        stow_on_finish=True,
    )

    agent.shutdown(power_off=False)