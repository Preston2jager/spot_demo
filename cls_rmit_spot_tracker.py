import time
import functools
import sys
import builtins

class SpotTracker:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BOLD = '\033[1m'
    END = '\033[0m'
    
    _REAL_PRINT = builtins.print 
    _STACK = [] 

    def __init__(self, name, exit_on_fail=False):
        self.name = name
        self.exit_on_fail = exit_on_fail
        self.line_count = 0

    @classmethod
    def _global_dispatch_print(cls, *args, **kwargs):
        if not cls._STACK:
            cls._REAL_PRINT(*args, **kwargs)
            return
        output = " ".join(map(str, args))
        added_lines = output.count('\n') + 1
        for tracker in cls._STACK:
            tracker.line_count += added_lines
        cls._REAL_PRINT(f"{cls.YELLOW}[log]{cls.END} {output}", **kwargs)

    def __enter__(self):
        if self._STACK:
            for tracker in self._STACK:
                tracker.line_count += 1
        self._REAL_PRINT(f"{self.BOLD}{self.CYAN}[[Running]]{self.END} {self.name}")
        self._STACK.append(self)
        builtins.print = self._global_dispatch_print
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        if self._STACK:
            self._STACK.pop()
        if not self._STACK:
            builtins.print = self._REAL_PRINT
        move_up = self.line_count + 1
        sys.stdout.write(f"\033[{move_up}A\r\033[K")
        if exc_type is None:
            sys.stdout.write(f"{self.BOLD}{self.GREEN}[[Done]]{self.END} {self.name} "
                             f"{self.GREEN}({duration:.2f}s){self.END}\n")
        else:
            sys.stdout.write(f"{self.BOLD}{self.RED}[[Error]]{self.END} {self.name}\n")
            sys.stdout.write(f"\033[{self.line_count}B")
            return not self.exit_on_fail
        sys.stdout.write(f"\033[{self.line_count}B")
        sys.stdout.flush()

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracker = SpotTracker(self.name, self.exit_on_fail)
            with tracker:
                return func(*args, **kwargs)
        return wrapper
