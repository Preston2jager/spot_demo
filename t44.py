import threading, time
from cls_spot_demo import SpotAgent_demo
from cls_yolo import YoloBottleDetector


if __name__ == "__main__":

    agent = SpotAgent_demo(
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
    agent.mark_home()
    time.sleep(2)

    t0 = time.time()
    while time.time() - t0 < 8.0:
        agent.send_velocity(0.25, 0.0, 0.0)  # 0.4 m/s 前进
        time.sleep(0.1)

    time.sleep(1)
    detector = YoloBottleDetector("yolov8m.pt")
    agent.detect_grab_return_and_release(
        detector,
        source="hand_color_image",
        home_frame="odom",        # 也可用 "vision"（相对短程、但可能跳变）
        open_fraction=1.0,
        stow_on_finish=True,
    )

    agent.shutdown(power_off=False)