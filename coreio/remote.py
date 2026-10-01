"""SSH helpers shared by every deploy script in this folder. Runs HERE, on the laptop.

The CoreIO has no DNS and no default route, so everything it runs arrives over one SSH
session opened with the `COREIO_*` settings from `.env` — the same connection the speaker
and the console's CoreIO clients use.

Imported by module name (`from remote import ...`): the deploy scripts are run as
`python coreio/deploy_x.py`, so this folder is what lands on sys.path.
"""
import os
import sys
from pathlib import Path

import paramiko
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).parent.parent


def connect() -> paramiko.SSHClient:
    """Open the SSH session to the CoreIO using the COREIO_* settings from .env."""
    load_dotenv(_PROJECT_ROOT / ".env")
    password = os.getenv("COREIO_PASSWORD", "")
    if not password:
        sys.exit("COREIO_PASSWORD is not set — add it to .env")
    host = os.getenv("COREIO_HOST", "192.168.80.3")
    port = int(os.getenv("COREIO_PORT", "20022"))
    user = os.getenv("COREIO_USER", "spot")

    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=host, port=port, username=user, password=password,
        timeout=15, allow_agent=False, look_for_keys=False,
    )
    print(f"[deploy] connected to {user}@{host}:{port}")
    return client


def remote_home(client: paramiko.SSHClient) -> str:
    """The remote account's home directory, so the install path follows COREIO_USER."""
    home = run(client, "echo $HOME").strip()
    if not home.startswith("/"):
        sys.exit(f"[deploy] could not resolve the remote home directory (got {home!r})")
    return home


def run(client: paramiko.SSHClient, command: str, check: bool = True) -> str:
    """Run a command on the CoreIO and return its stdout."""
    _, stdout, stderr = client.exec_command(command)
    out = stdout.read().decode(errors="replace")
    status = stdout.channel.recv_exit_status()
    if status != 0 and check:
        detail = stderr.read().decode(errors="replace").strip() or f"exit status {status}"
        sys.exit(f"[deploy] remote command failed: {command}\n         {detail}")
    return out


def upload(sftp: paramiko.SFTPClient, source: Path, remote: str, skip_if_same_size: bool = False) -> None:
    """Push one file, optionally skipping a re-upload of something already the right size.

    The size check exists for the weights: 50 MB over the robot's wifi is slow enough that
    re-sending an unchanged model on every deploy is worth avoiding.
    """
    size = source.stat().st_size
    if skip_if_same_size:
        try:
            if sftp.stat(remote).st_size == size:
                print(f"[deploy] {remote} already there at {size >> 20} MB, skipping upload")
                return
        except OSError:
            pass
    print(f"[deploy] uploading {source.name} -> {remote} ({size // 1024} KB)")
    sftp.put(str(source), remote)


def stream(client: paramiko.SSHClient, command: str, label: str, timeout: int = 3600) -> None:
    """Run a long, chatty command and echo it line by line, so a build does not look like a hang."""
    _, stdout, stderr = client.exec_command(command, timeout=timeout)
    for line in iter(stdout.readline, ""):
        if line.strip():
            print(f"[deploy]   {line.rstrip()}")
    if stdout.channel.recv_exit_status() != 0:
        detail = stderr.read().decode(errors="replace").strip()
        sys.exit(f"[deploy] {label} failed{': ' + detail if detail else ''}")
