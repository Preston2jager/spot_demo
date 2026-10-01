"""Spot Robot Control — entry point.

Starts the browser console, one page on one port:

    python start.py        # http://localhost:8080 — the console
                           # http://localhost:8081 — Dockge on the CoreIO, forwarded over SSH
    WEB_PORT=9000 python start.py

Cameras (with the AI layer), the RealSense, the map, missions, teleop, Stop, the grasp
confirmation and the log are all on that page.

One `SpotAgent` is built here and handed to the console, so the login, time sync and
speaker SSH session are paid for once for the whole session rather than per mission.

Starting the console is watching, not driving: the cameras, battery, map and RealSense come
up straight away, but the body lease is left with whoever has it — the ROS 2 driver, the
tablet — until the first command that needs it. Connecting does not move the robot either:
every mission stands it up itself.
"""
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from src import setup_logging

load_dotenv(Path(__file__).parent / ".env")
setup_logging()

_BANNER = """\033[1;31m
 ██████╗ ███╗   ███╗██╗████████╗
 ██╔══██╗████╗ ████║██║╚══██╔══╝
 ██████╔╝██╔████╔██║██║   ██║
 ██╔══██╗██║╚██╔╝██║██║   ██║
 ██║  ██║██║ ╚═╝ ██║██║   ██║
 ╚═╝  ╚═╝╚═╝     ╚═╝╚═╝   ╚═╝

  Centre for Future Construction\033[0m
"""


def main() -> None:
    from src.coreio_forward import start_dockge_forward
    from src.spot_agent import SpotAgent
    from src.web import SpotWebConsole

    print(_BANNER)
    start_dockge_forward()
    with SpotAgent(navigation=False, lease=False) as agent:
        console = SpotWebConsole(agent)
        try:
            console.serve()
        except KeyboardInterrupt:
            logger.info("Console stopped.")


if __name__ == "__main__":
    main()
