import time

from src.spot_agent import SpotAgent


def main() -> None:
    """Boot Spot, stand it up, and hold for 2 seconds — used to clear faults and reset posture."""
    with SpotAgent(stream=True, navigation=True) as agent:
        time.sleep(2.0)


if __name__ == "__main__":
    main()
