import time
import functools

class SpotTracker:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

    def __init__(self, name, exit_on_fail=False):
        self.name = name
        self.exit_on_fail = exit_on_fail
        self.indent = "   " 

    def log(self, message):
        print(f"{self.indent} {message}")

    def __enter__(self):
        print(f"\n{self.BOLD}{self.OKCYAN}[Starting]: {self.name}{self.ENDC}")
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        if exc_type is None:
            print(f"{self.OKGREEN}✅ [Succeed]: {self.name} (Duration: {duration:.2f}s){self.ENDC}")

        else:
            print(f"{self.FAIL}❌ [Failed]: {self.name}{self.ENDC}")
            print(f"{self.WARNING}⚠️ : {exc_val}{self.ENDC}")
            return not self.exit_on_fail

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper
