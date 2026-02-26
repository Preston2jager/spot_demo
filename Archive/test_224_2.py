import time
from cls_rmit_spot_core import SpotAgent
from cls_rmit_spot_stream import SpotStreamer
from cls_rmit_spot_detector import SpotDetector



if __name__ == "__main__":             
    with SpotAgent(force_lease=True) as agent:
        agent.recording()
    print("[Main] 程序已安全退出。")