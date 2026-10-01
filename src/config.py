"""Robot connection settings.

Environment first, ``credentials.json`` behind it. The file stays the ordinary way to
run on the laptop; the environment matters because the web console also runs *inside*
the CoreIO container, where there is no credentials file to mount and Spot's API answers
on the payload network (``192.168.50.3``) rather than on ``192.168.80.3``.
"""
import os
from pathlib import Path

from .janitor import Janitor

_ENV_KEYS = {
    "hostname": "SPOT_HOSTNAME",
    "username": "SPOT_USERNAME",
    "password": "SPOT_PASSWORD",
}


def load_robot_credentials(path: str | Path | None = None) -> dict[str, str]:
    """Return hostname/username/password, preferring the environment over the JSON file.

    The file is only read when the environment is incomplete, so a fully configured
    container never needs one — and a partly configured one still gets a clear
    ``FileNotFoundError`` naming the file it wanted.
    """
    env = {key: os.getenv(var, "").strip() for key, var in _ENV_KEYS.items()}
    if all(env.values()):
        return env

    creds = dict(Janitor.load_credentials(path))
    # Anything set in the environment still wins, so SPOT_HOSTNAME alone is enough to
    # point a stock credentials.json at the payload network.
    creds.update({key: value for key, value in env.items() if value})
    return creds
