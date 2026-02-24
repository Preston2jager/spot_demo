from cls_rmit_spot_nav import SpotAgent

if __name__ == "__main__":
    with SpotAgent() as agent:
        agent.record_square_path(side_length=1.0, save_dir="./spot_recorded_map")

    # 如果你的 SpotWebStreamer 有停止或清理方法，可以在这里调用，例如：
    # steamer.stop()
    print("[Main] 程序已安全退出。")