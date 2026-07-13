"""
Spot Robot Control — interactive entry point.

Run: python start.py
"""
import os
import time
import threading
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from src.spot_tracker import SpotTracker
import questionary
from questionary import Choice, Separator

load_dotenv(Path(__file__).parent / ".env")
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")


def _clear_screen() -> None:
    os.system('cls' if os.name == 'nt' else 'clear')


class BatteryMonitor:
    """Real-time battery monitor: Background thread runs persistent connection, outputs battery level in top-left or logs based on status."""
    def __init__(self):
        self.battery_status = "N/A"
        self.stop_event = threading.Event()
        self.thread = None
        self.task_running = False

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=1.0)

    def _run(self):
        from src.config import load_robot_credentials
        import bosdyn.client
        from bosdyn.client.robot_state import RobotStateClient

        try:
            creds = load_robot_credentials()
            sdk = bosdyn.client.create_standard_sdk("MenuBatteryMonitor")
            robot = sdk.create_robot(creds["hostname"])
            robot.authenticate(creds["username"], creds["password"])
            state_client = robot.ensure_client(RobotStateClient.default_service_name)
        except Exception as e:
            self.battery_status = "Error"
            return

        last_log_time = 0
        while not self.stop_event.is_set():
            try:
                state = state_client.get_robot_state()
                if state.battery_states:
                    pct = state.battery_states[0].charge_percentage.value
                    status = state.battery_states[0].Status.Name(state.battery_states[0].status)
                    status_short = "DISCHG" if "DISCHARGING" in status else status
                    self.battery_status = f"{pct:.1f}% ({status_short})"
                else:
                    self.battery_status = "No State"
            except Exception:
                pass

            # Output battery status based on current run state
            if not self.task_running:
                # Menu state: Use ANSI escape code to render battery on the first line left (bold green), and clear the rest of that line
                try:
                    print(f"\033[s\033[1;1H\033[1;32m[Spot Battery: {self.battery_status}]\033[0m\033[K\033[u", end="", flush=True)
                except Exception:
                    pass
            else:
                # Task running state: log battery every 30 seconds to prevent scrolling logs from cluttering output
                now = time.time()
                if now - last_log_time >= 30:
                    logger.info(f"[Battery Monitor] Current Battery: {self.battery_status}")
                    last_log_time = now

            # Update every 5 seconds
            for _ in range(50):
                if self.stop_event.is_set():
                    break
                time.sleep(0.1)


# ── 1. Basic Tasks ──────────────────────────────────────────────────

def _initial_check_reset() -> None:
    from src.spot_agent import SpotAgent
    from src.missions import run_reset
    with SpotAgent(stream=False, navigation=False) as agent:
        run_reset(agent)


def _step_in_place() -> None:
    from src.spot_agent import SpotAgent
    from src.missions import run_stepping_in_place
    with SpotAgent(stream=False, navigation=False) as agent:
        run_stepping_in_place(agent)


def _stand_and_view() -> None:
    from src.spot_agent import SpotAgent
    from src.missions import run_stand_and_view
    with SpotAgent(stream=False, navigation=False) as agent:
        run_stand_and_view(agent)


def _arm_height_view() -> None:
    from src.spot_agent import SpotAgent
    from src.missions import run_arm_height_view
    with SpotAgent(stream=True, navigation=False) as agent:
        run_arm_height_view(agent)


def _sit_down_rest() -> None:
    from src.spot_agent import SpotAgent
    from src.missions import run_sit_down_rest
    with SpotAgent(stream=False, navigation=False) as agent:
        run_sit_down_rest(agent)


# ── 2. Direct Control ──────────────────────────────────────────────

def _wasd_teleop() -> None:
    from src.spot_agent import SpotAgent
    from src.missions import run_wasd_teleop
    with SpotAgent(stream=False, navigation=False) as agent:
        run_wasd_teleop(agent)


# ── 3. GraphNav ──────────────────────────────────────────────────

def _go_home() -> None:
    from src.spot_agent import SpotAgent
    from src.missions import run_go_home
    waypoint = os.getenv("GO_HOME_WAYPOINT", "waypoint_0")
    with SpotAgent(stream=True, navigation=True) as agent:
        run_go_home(agent, waypoint)


def _fetch_local_coord() -> None:
    from src.spot_agent import SpotAgent
    from src.missions import run_guard_no_graphnav
    with SpotAgent(stream=True, navigation=False) as agent:
        run_guard_no_graphnav(agent)


def _fetch_one_point() -> None:
    from src.spot_agent import SpotAgent
    from src.missions import run_fetch_one_point
    target = os.getenv("FETCH_TARGET_POINT", "default")
    home = os.getenv("FETCH_HOME_POINT", "waypoint_32")
    with SpotAgent(stream=True, navigation=True) as agent:
        run_fetch_one_point(agent, target, home)


def _fetch_one_point_loop() -> None:
    from src.spot_agent import SpotAgent
    from src.missions import run_fetch_one_point_loop
    target = os.getenv("FETCH_TARGET_POINT", "default")
    home = os.getenv("FETCH_HOME_POINT", "waypoint_32")
    with SpotAgent(stream=True, navigation=True) as agent:
        run_fetch_one_point_loop(agent, target, home)


# ── Menu ──────────────────────────────────────────────────────────────────────

_MENU = [
    Separator("── 1. Basic Tasks ──────────────────"),
    Choice("Initial Check/Reset",               value=_initial_check_reset),
    Choice("Step in Place",                     value=_step_in_place),
    Choice("Stand",                             value=_stand_and_view),
    Choice("Arm 1m Height & View",              value=_arm_height_view),
    Choice("Sit down/Rest",                     value=_sit_down_rest),

    Separator("── 2. Direct Control ───────────────"),
    Choice("WASD Control",                      value=_wasd_teleop),

    Separator("── 3. GraphNav ───────────────────"),
    Choice("Go Home",                           value=_go_home),
    Choice("Fetch local coord",                 value=_fetch_local_coord),
    Choice("Fetch one point",                   value=_fetch_one_point),
    Choice("Fetch one point loop",              value=_fetch_one_point_loop),

    Separator("─────────────────────────────────────────────"),
    Choice("Exit",                              value="exit"),
]


def main() -> None:
    monitor = BatteryMonitor()
    monitor.start()

    try:
        while True:
            _clear_screen()
            # Reserve first two lines for battery status, so the menu/art renders starting from the third line to avoid overlap
            print("\n\n")
            # RMIT Red logo ASCII art with subtitle
            print("\033[1;31m" + """ ██████╗ ███╗   ███╗██╗████████╗
 ██╔══██╗████╗ ████║██║╚══██╔══╝
 ██████╔╝██╔████╔██║██║   ██║   
 ██╔══██╗██║╚██╔╝██║██║   ██║   
 ██║  ██║██║ ╚═╝ ██║██║   ██║   
 ╚═╝  ╚═╝╚═╝     ╚═╝╚═╝   ╚═╝   
 
  Centre for Future Construction""" + "\033[0m\n")
            handler = questionary.select(
                "Spot Robot Control — select operation:",
                choices=_MENU,
            ).ask()

            if handler is None or handler == "exit":
                logger.info("Exiting.")
                break

            try:
                # Switch monitor state: task starts running, switch to logging output
                monitor.task_running = True
                handler()
                input("\nPress Enter to return to menu...")
            except KeyboardInterrupt:
                logger.info("Interrupted — returning to menu.")
                time.sleep(1.0)
            except Exception:
                logger.exception("Unexpected error — returning to menu.")
                input("\nPress Enter to return to menu...")
            finally:
                # Task finishes, switch back to menu rendering mode
                monitor.task_running = False
    finally:
        monitor.stop()


if __name__ == "__main__":
    main()
