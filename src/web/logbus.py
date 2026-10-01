"""The log panel's supply: one loguru sink, one ring buffer, one stream per browser.

The terminal console and the web console show the same run. Nothing about mission code
changes to make that true — everything already goes through loguru, so this adds a sink
beside the existing two rather than a second logging convention.

`SpotTracker`'s ``[[Running]]`` / ``[[Done]]`` lines are the exception: they are `print()`
output and no sink would ever see them, which is why the tracker grew a listener hook.
They arrive here as ``task`` entries and are what makes the panel read as a timeline of
what Spot is doing rather than a wall of INFO.
"""
import queue
import threading
from collections import deque
from datetime import datetime

from loguru import logger

from ..spot_tracker import SpotTracker

# Enough that a browser opening mid-mission still sees how it got there, small enough that
# an overnight guard loop cannot grow it without bound.
_BACKLOG = 400
# Dropping is deliberate: a browser that stopped reading must not stall the robot.
_QUEUE_DEPTH = 1000


class LogBus:
    """Fans log records out to the connected browsers, keeping a replayable backlog."""

    def __init__(self, capacity: int = _BACKLOG) -> None:
        self._entries: deque[dict] = deque(maxlen=capacity)
        self._subscribers: set[queue.Queue] = set()
        self._lock = threading.Lock()
        self._seq = 0
        self._installed = False

    # ── wiring ────────────────────────────────────────────────────────────────

    def install(self) -> None:
        """Attach to loguru and to SpotTracker. Safe to call more than once."""
        if self._installed:
            return
        self._installed = True
        logger.add(self._sink, level="INFO", format="{message}", enqueue=False)
        SpotTracker.LISTENERS.append(self._on_task)

    def _sink(self, message: object) -> None:
        """Loguru sink: one record in, one entry out."""
        record = message.record  # type: ignore[attr-defined]
        self.emit(record["level"].name, record["message"], kind="log")

    def _on_task(self, event: str, name: str, duration: float) -> None:
        """SpotTracker boundary: the timeline entries the log panel is built around."""
        if event == "start":
            self.emit("TASK", name, kind="task-start")
        elif event == "done":
            self.emit("TASK", f"{name} ({duration:.2f}s)", kind="task-done")
        else:
            self.emit("TASK", f"{name} failed ({duration:.2f}s)", kind="task-error")

    # ── publishing ────────────────────────────────────────────────────────────

    def emit(self, level: str, text: str, kind: str = "log") -> None:
        """Record one entry and hand it to every subscriber."""
        with self._lock:
            self._seq += 1
            entry = {
                "seq": self._seq,
                "time": datetime.now().strftime("%H:%M:%S"),
                "level": level,
                "text": text,
                "kind": kind,
            }
            self._entries.append(entry)
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            try:
                subscriber.put_nowait(entry)
            except queue.Full:
                # A browser that has stopped draining its queue is a browser that has
                # gone away. Losing its lines costs nothing; blocking here would stall
                # whichever mission thread happened to log.
                pass

    def backlog(self) -> list[dict]:
        """Everything still in the ring buffer, oldest first."""
        with self._lock:
            return list(self._entries)

    # ── subscriptions ─────────────────────────────────────────────────────────

    def subscribe(self) -> queue.Queue:
        """Register a new stream and return its queue."""
        subscriber: queue.Queue = queue.Queue(maxsize=_QUEUE_DEPTH)
        with self._lock:
            self._subscribers.add(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: queue.Queue) -> None:
        """Drop a stream whose browser has disconnected."""
        with self._lock:
            self._subscribers.discard(subscriber)
