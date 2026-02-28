from cls_rmit_spot_core import SpotAgent
import time

if __name__ == "__main__":
    with SpotAgent(
        stream=True,
        navigation=True
        ) as agent:
        time.sleep(2.0)
