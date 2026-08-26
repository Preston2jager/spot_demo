from pathlib import Path

from .janitor import Janitor


def load_robot_credentials(path: str | Path | None = None) -> dict[str, str]:
    """Read and return robot connection credentials from the project-root credentials.json."""
    if path is None:
        path = Path(__file__).parent.parent / "credentials.json"
    return Janitor.load_credentials(path)
