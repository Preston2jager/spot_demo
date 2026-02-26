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

    detector = YoloBottleDetector()

    ok = agent.handcam_detect_and_grab_once(
        detector,
        source="hand_color_image",
        interval=1.0,
        jpeg_quality=70,
        timeout=25.0,
        carry_on_success=True,   # 成功后抱持
        open_on_success=False,   # 演示时想立刻放下可改 True
        stow_on_finish=True,    # 结束是否收臂
    )
    print("最终：", "抓到啦" if ok else "没抓到")


    agent.shutdown(power_off=False)