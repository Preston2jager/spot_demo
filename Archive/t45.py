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

    try:
        agent.power_on_and_stand()

        # 开启手摄窗口线程
        t = threading.Thread(
            target=agent.handcam_on,
            kwargs={"window_title":"Spot HandCam", "width":1600, "height":1200, "fps":15},
            daemon=True,
        )
        t.start()

        agent.mark_home()
        time.sleep(2)

        detector = YoloBottleDetector("yolov8m.pt")

        # === 每秒一次循环触发 ===
        while True:
            t0 = time.monotonic()

            ok = agent.detect_grab_return_and_release(
                detector,
                source="hand_color_image",
                home_frame="odom",   # 也可用 "vision"（相对短程、但可能跳变）
                open_fraction=1.0,
                stow_on_finish=True,
            )
            # 可选：根据结果决定是否短暂退避，避免立即进入下一轮
            # if not ok:
            #     time.sleep(0.5)

            # 1Hz：考虑到本轮执行耗时，补足到整秒
            elapsed = time.monotonic() - t0
            remain = 1.0 - elapsed
            if remain > 0:
                time.sleep(remain)

    except KeyboardInterrupt:
        print("\n[main] 用户中断，准备退出…")
    finally:
        # 不中断上电（按你原本的设定），仅清理会话
        agent.shutdown(power_off=False)
