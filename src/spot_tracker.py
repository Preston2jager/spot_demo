import time
import functools
import sys
import builtins
from typing import Any, Callable


class SpotTracker:
    """Wraps a task in timed, coloured terminal output and intercepts print() within scope."""

    BLUE   = '\033[94m'
    GRAY_GREEN = '\033[38;5;151m'
    RED    = '\033[91m'
    YELLOW = '\033[93m'
    BOLD   = '\033[1m'
    END    = '\033[0m'

    _REAL_PRINT = builtins.print
    _STACK: list["SpotTracker"] = []

    def __init__(self, name: str, exit_on_fail: bool = False) -> None:
        """Store task name and whether an exception should propagate."""
        self.name = name
        self.exit_on_fail = exit_on_fail
        self.line_count = 0

    @classmethod
    def _global_dispatch_print(cls, *args: Any, **kwargs: Any) -> None:
        """Route all print() calls through the active tracker stack for line counting."""
        output = " ".join(map(str, args))
        added_lines = output.count('\n') + 1
        if cls._STACK:
            for tracker in cls._STACK:
                tracker.line_count += added_lines

        import re
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        plain_output = ansi_escape.sub('', output).strip()

        if plain_output.startswith("[["):
            cls._REAL_PRINT(output, **kwargs)
        else:
            if cls._STACK:
                cls._REAL_PRINT(f"{cls.YELLOW}[log]{cls.END} {output}", **kwargs)
            else:
                cls._REAL_PRINT(output, **kwargs)

    def __enter__(self) -> "SpotTracker":
        """Push this tracker onto the stack and start the timer."""
        if self._STACK:
            for tracker in self._STACK:
                tracker.line_count += 1
        self._REAL_PRINT(f"{self.BOLD}{self.BLUE}[[Running]]{self.END} {self.name}")
        self._STACK.append(self)
        builtins.print = self._global_dispatch_print
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool | None:
        """Pop from the stack, rewrite the status line, and restore print() if the stack is empty."""
        duration = time.time() - self.start_time
        if self._STACK:
            self._STACK.pop()
        if not self._STACK:
            builtins.print = self._REAL_PRINT
        move_up = self.line_count + 1
        sys.stdout.write(f"\033[{move_up}A\r\033[K")
        if exc_type is None:
            sys.stdout.write(
                f"{self.BOLD}{self.GRAY_GREEN}[[Done]]{self.END} {self.name} "
                f"{self.GRAY_GREEN}({duration:.2f}s){self.END}\n"
            )
        else:
            sys.stdout.write(f"{self.BOLD}{self.RED}[[Error]]{self.END} {self.name}\n")
            if self.line_count > 0:
                sys.stdout.write(f"\033[{self.line_count}B")
            return not self.exit_on_fail
        if self.line_count > 0:
            sys.stdout.write(f"\033[{self.line_count}B")
        sys.stdout.flush()
        return None

    def __call__(self, func: Callable) -> Callable:
        """Use as a decorator to wrap any function inside a SpotTracker context."""
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tracker = SpotTracker(self.name, self.exit_on_fail)
            with tracker:
                return func(*args, **kwargs)
        return wrapper


# Configure Loguru to route through standard print and use [[LEVEL]] format
from loguru import logger

logger.remove()
logger.level("INFO", color="<yellow>")
logger.add(
    lambda msg: print(msg.rstrip('\n')),
    format="<level><bold>[[{level}]]</bold></level> {message}",
    colorize=True
)
